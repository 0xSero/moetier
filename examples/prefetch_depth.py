"""N134 follow-up: how far ahead should GLM-5.3-Flash's router lookahead prefetch look?

Today's GLM53_NV_PREFETCH (k = 1) applies layer l+1's router to layer l's MoE input and reads the non-resident
part of its top-N while layer l runs. Here layer l+k's router is applied to layer l's MoE input for k = 1..4 (for
consumers c < k the deepest available source, layer 0, is used), and mixes = unions of components issued at their
own layers (e.g. k2@8+k1@8: top-8 of router(c) on z[c-2] issued at layer c-2, plus top-8 of router(c) on z[c-1]
issued at layer c-1 for keys not already in flight).

Data: the N134 capture (prerouter_train.py data/ dir: sel, req, tier) + per-layer MoE inputs z<l>.f16 + router.npz.
No training: every captured token is evaluated (54,442 tokens, 15 requests), consumers 1..41.

  engine  recall on the picks the ENGINE served from NVMe at 55 GB (captured plan-time tiers 3/4)
  LOOKA   moetier ledger at 55 / 16 GB, counters only: recall on ledger-NVMe picks, reads issued (keys NVMe-tier at issue
          time and not yet issued for that target), used reads, precision, GB/token
  PILOT   prefetch in the sim loop, 'idle' channel: prefetch reads queue at low priority; one starts only when the
          channel is free and no demand read is waiting, a started read is never interrupted (demand reads can wait
          for at most the read in flight), a queued read not started by its consumer layer is cancelled (if picked,
          the demand read takes over). Landed reads sit in a landing ring (never RAM) until their consumer layer.

  python3 examples/prefetch_depth.py --run <run dir> --zdir <z dir> --router router.npz --out docs/prefetch-depth-glm53.results.json
"""
import argparse, json, os, sys, time
from collections import deque
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from moetier import spec
from moetier.ledger import Ledger, NVME
from moetier.plan import NvmeChannel, plan_layer

L, E, K, NR = 42, 288, 8, 32
KMAX = 4


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ranks(zdir, router, T, out):
    """rank[k] [T, L, NR] int16: top-NR of sigmoid(router(c) z[c-k]) + bias, source layer max(c-k, 0); c = 0 -> -1"""
    fn = [os.path.join(out, f"depth_rank_k{k}.npy") for k in range(1, KMAX + 1)]
    if all(os.path.exists(f) for f in fn):
        return [np.load(f, mmap_mode="r") for f in fn]
    rz = np.load(router)
    G, B = rz["gate"].astype(np.float32), rz["bias"].astype(np.float32)
    Rk = [np.full((T, L, NR), -1, np.int16) for _ in range(KMAX)]
    for l in range(L - 1):
        z = np.fromfile(os.path.join(zdir, f"z{l:02d}.f16"), np.float16, count=T * 4096).reshape(T, 4096).astype(np.float32)
        for c in range(l + 1, L):
            ks = [k for k in range(1, KMAX + 1) if max(c - k, 0) == l]
            if not ks:
                continue
            s = 1.0 / (1.0 + np.exp(-(z @ G[c].T))) + B[c]
            idx = np.argpartition(-s, NR, axis=1)[:, :NR]
            o = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
            r = np.take_along_axis(idx, o, 1).astype(np.int16)
            for k in ks:
                Rk[k - 1][:, c] = r
        log(f"ranks: source layer {l}")
    for f, r in zip(fn, Rk):
        np.save(f, r)
    return Rk


def parse(specs):
    """'k2@8+k1@8' -> [(2, 8), (1, 8)]"""
    return [tuple(int(x) for x in c[1:].split("@")) for c in specs.split("+")]


def src(c, k):
    return max(c - k, 0)


def engine_recall(Rk, sel, tier, comps, cons):
    T = len(sel)
    hit = np.zeros((T, L, K), bool)
    for k, n in comps:
        P = np.asarray(Rk[k - 1][:, :, :n]).astype(np.int64)
        hit |= (P[:, :, :, None] == sel[:, :, None, :]).any(2)
    m = np.zeros(L, bool); m[cons] = True
    nv = (tier >= 3) & m[None, :, None]
    al = np.broadcast_to(m[None, :, None], tier.shape)
    return dict(recall_nvme=round(float(hit[nv].mean()), 4), recall_all=round(float(hit[al].mean()), 4),
                recall_ram=round(float(hit[((tier == 1) | (tier == 2)) & m[None, :, None]].mean()), 4),
                nvme_picks_per_tok=round(float(nv.sum() / T), 2))


def make_ledger(R, sel):
    freq = np.zeros(L * E)
    for l in range(L):
        np.add.at(freq, l * E + sel[:, l].reshape(-1).astype(np.int64), 1)
    led = Ledger(R.vram_slots, R.ram_slots, L * E, exclusive=R.policy.get("ram", "exclusive") == "exclusive",
                 ram_policy=R.policy.get("ram_evict", "lru"))
    led.seed([int(k) for k in np.argsort(-freq)])
    return led


def schedule(specs, cons):
    """issuing layer -> [(spec index, consumer, k, n)]"""
    sch = {}
    for si, comps in enumerate(specs):
        for c in cons:
            for k, n in comps:
                sch.setdefault(src(c, k), []).append((si, c, k, n))
    return sch


def looka(R, sel, Rk, specs, cons, warm=200):
    led = make_ledger(R, sel)
    nv = NvmeChannel(R.nvme_expert_bytes, R.nvme_bw_gbps, R.nvme_bw_qd1_gbps, R.nvme_latency_ms)
    T = len(sel)
    S = [dict(act=0, cov=0, reads=0, used=0) for _ in specs]
    sch = schedule(specs, cons)
    issued = {}                         # (si, t, c) -> set of read keys
    t_now, ntok = 0.0, 0
    for t in range(T):
        meas = t >= warm
        for l in range(L):
            picks = {}
            for e in sel[t, l]:
                k_ = l * E + int(e); picks[k_] = picks.get(k_, 0) + 1
            if meas and l in cons:
                act = {k_ for k_ in picks if led.tier(k_) == NVME}
                for si in range(len(specs)):
                    rd = issued.pop((si, t, l), set())
                    S[si]["act"] += len(act); S[si]["cov"] += len(act & rd); S[si]["used"] += len(rd & set(picks))
            pl = plan_layer(R, led, picks, t_now, nv, {})
            t_now += pl.ms
            if not meas:
                continue
            for si, c, k, n in sch.get(l, ()):
                if c == l:                  # c = 0 / same layer: no lookahead source
                    continue
                ks = [c * E + int(e) for e in Rk[k - 1][t, c, :n] if e >= 0]
                cur = issued.setdefault((si, t, c), set())
                new = {k_ for k_ in ks if k_ not in cur and led.tier(k_) == NVME}
                cur |= new
                S[si]["reads"] += len(new)
        if meas:
            ntok += 1
    gb = R.nvme_expert_bytes / 1e9
    out = []
    for s in S:
        out.append(dict(recall_nvme=round(s["cov"] / max(1, s["act"]), 4), nvme_picks_per_tok=round(s["act"] / max(1, ntok), 2),
                        reads_per_tok=round(s["reads"] / max(1, ntok), 2), used_per_tok=round(s["used"] / max(1, ntok), 2),
                        precision=round(s["used"] / max(1, s["reads"]), 4), read_gb_per_tok=round(s["reads"] / max(1, ntok) * gb, 3)))
    return out


class LowPrio:
    """FIFO demand channel (NvmeChannel semantics) + low-priority prefetch queue served in channel idle time; a started
    prefetch read is not interrupted, a queued one can be cancelled."""
    def __init__(self, nv):
        self.nv, self.q, self.done = nv, deque(), {}

    def advance(self, t):
        nv = self.nv
        while self.q:
            k, ti = self.q[0]
            start = max(nv.free_at, ti)
            if start >= t:
                break
            self.q.popleft()
            svc = nv.deep + (nv.lat if start > nv.free_at or nv.free_at == 0.0 else 0.0)
            nv.free_at = start + svc
            self.done[k] = nv.free_at
            nv.reads += 1

    def cancel(self, keys):
        if keys:
            self.q = deque(x for x in self.q if x[0] not in keys)


G = {}


def pilot(args):
    R, comps, cons, oracle_k, warm = args
    sel, Rk = G["sel"], G["Rk"]
    led = make_ledger(R, sel)
    nv = NvmeChannel(R.nvme_expert_bytes, R.nvme_bw_gbps, R.nvme_bw_qd1_gbps, R.nvme_latency_ms)
    ch = LowPrio(nv)
    T = len(sel)
    F = R.fixed(1)
    comps = comps or []
    sch = {}
    for c in cons:
        for k, n in (comps if not oracle_k else [(oracle_k, K)]):
            if src(c, k) != c:
                sch.setdefault(src(c, k), []).append((c, k, n))
    rings, ref = {}, {}
    t_now, t_meas, ntok, reads, used, act, cov = 0.0, 0.0, 0, 0, 0, 0, 0

    def drop(k_):
        ref.pop(k_, None); ch.done.pop(k_, None); ch.cancel({k_})

    for t in range(T):
        led.step = t
        t0 = t_now
        meas = t >= warm
        for l in range(L):
            t_now += F / L
            picks = {}
            for e in sel[t, l]:
                k_ = l * E + int(e); picks[k_] = picks.get(k_, 0) + 1
            ch.advance(t_now)
            infl = {}
            ring = rings.pop((t, l), None)
            if ring:
                for k_ in ring:
                    if k_ not in ref:
                        continue
                    if k_ in picks:
                        a = ch.done.get(k_)
                        if a is not None:
                            infl[k_] = a
                        drop(k_)
                    else:
                        ref[k_] -= 1
                        if ref[k_] <= 0:
                            drop(k_)
                if meas:
                    used += len(infl)
            if meas:
                nvk = {k_ for k_ in picks if led.tier(k_) == NVME}
                act += len(nvk); cov += len(nvk & set(infl))
            pl = plan_layer(R, led, picks, t_now, nv, infl)
            t_now += pl.ms
            for c, k, n in sch.get(l, ()):
                if oracle_k:
                    ks = [c * E + int(e) for e in sel[t, c]]
                else:
                    ks = [c * E + int(e) for e in Rk[k - 1][t, c, :n] if e >= 0]
                ring = rings.setdefault((t, c), set())
                want = [k_ for k_ in dict.fromkeys(ks) if k_ not in ring and led.tier(k_) == NVME]
                for k_ in want:
                    if k_ not in ref:
                        ch.q.append((k_, t_now))
                        if meas:
                            reads += 1
                    ref[k_] = ref.get(k_, 0) + 1
                    ring.add(k_)
        if meas:
            t_meas += t_now - t0
            ntok += 1
    gb = R.nvme_expert_bytes / 1e9
    return dict(tok_s=round(1000 * ntok / max(t_meas, 1e-9), 3), ms_per_tok=round(t_meas / max(1, ntok), 3),
                recall_nvme=round(cov / max(1, act), 4), nvme_picks_per_tok=round(act / max(1, ntok), 2),
                reads_per_tok=round(reads / max(1, ntok), 2), used_per_tok=round(used / max(1, ntok), 2),
                read_gb_per_tok=round(reads / max(1, ntok) * gb, 3), tokens=ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--zdir", required=True); ap.add_argument("--router", required=True)
    ap.add_argument("--recipe", default="glm53-rtx3090-55g-nvx4"); ap.add_argument("--budgets", default="55,16")
    ap.add_argument("--specs", default="k1@8,k1@16,k2@8,k2@16,k3@8,k3@16,k4@8,k4@16,k2@8+k1@8,k3@8+k1@8,k4@8+k1@8,"
                                       "k2@16+k1@8,k3@8+k2@8+k1@8,k4@8+k2@8+k1@8")
    ap.add_argument("--pilot-specs", default="k1@8,k1@16,k2@8,k2@16,k3@8,k3@16,k4@16,k2@8+k1@8,k3@8+k1@8,k2@16+k1@8,"
                                             "k3@8+k2@8+k1@8,k4@8+k2@8+k1@8")
    ap.add_argument("--pilot-tokens", type=int, default=0, help="PILOT on the first N tokens only (0 = all)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--looka-tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = os.path.join(a.run, "data")
    sel = np.load(os.path.join(d, "sel.npy")).astype(np.int64)
    tier = np.load(os.path.join(d, "tier.npy"))
    T = len(sel)
    cons = list(range(1, L))
    Rk = ranks(a.zdir, a.router, T, a.run)
    Rk = [np.asarray(r) for r in Rk]
    specs = a.specs.split(","); comps = [parse(s) for s in specs]
    res = {"tokens": T, "consumers": "1-41", "specs": specs, "engine_55g": {}, "looka": {}, "pilot": {}}
    for s, c in zip(specs, comps):
        res["engine_55g"][s] = engine_recall(Rk, sel, tier, c, cons)
    log("engine: " + ", ".join(f"{s} {v['recall_nvme']}" for s, v in res["engine_55g"].items()))
    json.dump(res, open(a.out, "w"), indent=1)
    reg = spec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"))
    Rs = {b: spec.resolve(reg, a.recipe, **{"budget.ram_gb": float(b)}) for b in a.budgets.split(",")}
    ps = a.pilot_specs.split(",")
    selp = sel[:a.pilot_tokens] if a.pilot_tokens else sel
    Rkp = [r[:len(selp)] for r in Rk]
    jobs, keys = [], []
    for b, R in Rs.items():
        jobs.append((R, None, cons, 0, 200)); keys.append((b, "base"))
        for ok in (1, 2, 4):
            jobs.append((R, None, cons, ok, 200)); keys.append((b, f"oracle_k{ok}"))
        for s in ps:
            jobs.append((R, parse(s), cons, 0, 200)); keys.append((b, s))
    G["sel"], G["Rk"] = selp, Rkp
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        fut = pool.map_async(pilot, jobs)
        lk = {}
        for b, R in Rs.items():
            nl = a.looka_tokens or T
            lk[b] = looka(R, sel[:nl], [r[:nl] for r in Rk], comps, cons)
            res["looka"][b] = {"ram_slots": R.ram_slots, "pred": dict(zip(specs, lk[b]))}
            log(f"LOOKA {b} GB ({time.time() - t0:.0f} s): " + ", ".join(f"{s} {v['recall_nvme']}/{v['reads_per_tok']}"
                                                                     for s, v in res["looka"][b]["pred"].items()))
            json.dump(res, open(a.out, "w"), indent=1)
        out = fut.get()
    for (b, s), r in zip(keys, out):
        res["pilot"].setdefault(b, {})[s] = r
    for b in Rs:
        base = res["pilot"][b]["base"]["tok_s"]
        for s, r in res["pilot"][b].items():
            r["vs_base"] = round(r["tok_s"] / base - 1, 4)
        k1 = res["pilot"][b].get("k1@8", {}).get("tok_s")
        log(f"PILOT {b} GB: " + ", ".join(f"{s} {r['tok_s']} ({100 * r['vs_base']:+.1f}%)" for s, r in res["pilot"][b].items()))
    json.dump(res, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
