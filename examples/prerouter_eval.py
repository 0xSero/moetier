"""N134: Edge0-style prerouter heads (arXiv 2609.18063) on GLM-5.3-Flash, evaluated with the moetier ledger.

Input: the output dir of prerouter_train.py (glm53-flash-offload, branch n134-prerouter): data/sel.npy [T, 42, 8] routes
of the N134 capture (C1 decode, capture order), data/req.npy, data/split_req.npy, data/tier.npy (engine tier per pick),
pred_<variant>_<tag>.npy [n, 42, 32] ranked ids and testidx_<variant>_<tag>.npy (the TARGET token of each row).

Variants with shift 1 (edge0, self, router_next, router_self) predict token t+1 and are issued during token t;
shift 0 (same, router_same, and the route-only predictors a / lr / lrw of examples/prefetch_predictor.py, trained here
on the train split) predict token t one layer ahead. Layer 0 has no same-token source.

  LOOKA  one replay per RAM budget, counters only (residency is predictor-independent): recall on the picks the ledger
         serves from NVMe, reads issued (predicted keys NVMe-tier at issue time), used reads, GB/token.
  PILOT  the predictor in the loop: reads for predicted NVMe-tier keys go to a short-lived landing ring (never RAM; a
         used one is served like an in-flight read, an unused one is dropped after its consumer layer).
         'fifo' = reads share the FIFO channel with demand reads; 'idle' = a two-priority channel where prefetch reads
         only use time the channel would otherwise idle (demand reads preempt queued prefetches).
  Time is measured over the test tokens only (the heads' predictions exist only there).

  python3 examples/prerouter_eval.py --run <out dir of prerouter_train.py> --tag full --out docs/prerouter-glm53.results.json
"""
import argparse, json, os, sys, time
from collections import deque
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moetier import spec
from moetier.ledger import Ledger, NVME, VRAM, RAM
from moetier.plan import NvmeChannel, plan_layer

L, E, K = 42, 288, 8
EXCL = set()
NS = (8, 12, 16, 24, 32)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def route_only(sel, req, split, test_tok):
    """prefetch_predictor's a / lr / lrw (same-token, layer-ahead) trained on the train split, predicted on the test
    target tokens. Features with history are computed per test request (resets at request starts)."""
    import prefetch_predictor as PP
    tr = split[req] == 0
    ids_tr = sel[tr].astype(np.int64)
    seg_tr = np.flatnonzero(np.r_[True, req[tr][1:] != req[tr][:-1]])
    n = len(ids_tr)
    cA = int(n * 0.7)
    tabA = PP.Tables(ids_tr[:cA])
    coefs = dict(lr=PP.fit_lr(ids_tr[cA:], None, tabA, np.array([0]), PP.LR_ROUTE))
    tab = PP.Tables(ids_tr)
    te_req = np.unique(req[test_tok])
    m = np.isin(req, te_req)
    pos = np.flatnonzero(m)
    ids = sel[m].astype(np.int64)
    seg = np.flatnonzero(np.r_[True, req[m][1:] != req[m][:-1]])
    pr = PP.predict_all(ids, None, tab, seg, coefs)
    row = {p: i for i, p in enumerate(pos)}
    rows = np.array([row[t] for t in test_tok])
    return {f"ro_{k}": pr[k][rows] for k in ("a", "b", "lr")}, {"lr_coef": dict(zip(coefs["lr"][0], coefs["lr"][1].tolist()))}


class Pred:
    """ranked predictions keyed by the LOCAL index of the target token in the test sub-stream"""
    def __init__(self, name, P, tgt, shift, local_of):
        self.name, self.P, self.shift = name, P, shift
        self.row = {local_of[int(t)]: i for i, t in enumerate(tgt) if int(t) in local_of}

    def get(self, t, c, n):
        i = self.row.get(t)
        if i is None or c in EXCL:
            return None
        p = self.P[i, c, :n]
        return [c * E + int(e) for e in p if e >= 0]


def make_ledger(R, sel):
    freq = np.zeros(L * E)
    for l in range(L):
        np.add.at(freq, l * E + sel[:, l].reshape(-1).astype(np.int64), 1)
    led = Ledger(R.vram_slots, R.ram_slots, L * E, exclusive=R.policy.get("ram", "exclusive") == "exclusive",
                 ram_policy=R.policy.get("ram_evict", "lru"))
    led.seed([int(k) for k in np.argsort(-freq)])
    return led


def owner_of(shift, c):
    """(token offset, layer) at which the prediction for (t, c) is issued"""
    if shift == 1:
        return -1, (c - 1 if c > 0 else L - 1)
    return 0, c - 1


def looka(R, sel, preds, warm=200):
    """counters only. Returns per predictor x N: recall on ledger-NVMe picks, reads/useful per test token."""
    led = make_ledger(R, sel)
    nv = NvmeChannel(R.nvme_expert_bytes, R.nvme_bw_gbps, R.nvme_bw_qd1_gbps, R.nvme_latency_ms)
    T = len(sel)
    st = {(p.name, n): dict(act=0, cov=0, reads=0, used=0) for p in preds for n in NS}
    tier_led = np.zeros((T, L, K), np.int8)                        # 0 vram 1 ram 4 nvme (ledger)
    issued = {}                                                     # (name, n, t_target, c) -> set of read keys
    # issue schedule: at (token s, layer l) issue for targets (s - off, c) whose owner is (off, l)
    sched = {}
    for p in preds:
        for c in range(L):
            if p.shift == 0 and c == 0:
                continue
            off, ol = owner_of(p.shift, c)
            sched.setdefault(ol, []).append((p, c, -off))
    t_now = 0.0
    ntest = 0
    for t in range(T):
        led.step = t
        for l in range(L):
            picks = {}
            for e in sel[t, l]:
                k = l * E + int(e); picks[k] = picks.get(k, 0) + 1
            # counters at the consumer layer, before the plan updates residency
            tiers = {k: led.tier(k) for k in picks}
            for k_i, e in enumerate(sel[t, l]):
                tt = tiers[l * E + int(e)]
                tier_led[t, l, k_i] = 0 if tt == VRAM else (1 if tt == RAM else 4)
            if t >= warm:
                act = {k for k, v in tiers.items() if v == NVME}
                for p in preds:
                    if (p.shift == 0 and l == 0) or l in EXCL:
                        continue
                    for n in NS:
                        rd = issued.pop((p.name, n, t, l), None)
                        if rd is None:
                            continue
                        s = st[(p.name, n)]
                        s["act"] += len(act); s["cov"] += len(act & rd); s["used"] += len(rd & set(picks))
            pl = plan_layer(R, led, picks, t_now, nv, {})
            t_now += pl.ms
            # issue predictions owned by this layer (residency after this layer's plan)
            for p, c, dt in sched.get(l, ()):
                tgt = t + dt
                if tgt >= T or tgt < warm:
                    continue
                for n in NS:
                    ks = p.get(tgt, c, n)
                    if ks is None:
                        continue
                    rd = {k for k in ks if led.tier(k) == NVME}
                    issued[(p.name, n, tgt, c)] = rd
                    st[(p.name, n)]["reads"] += len(rd)
        if t >= warm:
            ntest += 1
    out = {}
    gb = R.nvme_expert_bytes / 1e9
    for (name, n), s in st.items():
        out[f"{name}@{n}"] = dict(recall_nvme=round(s["cov"] / max(1, s["act"]), 4),
                                 nvme_picks_per_tok=round(s["act"] / max(1, ntest), 2),
                                 reads_per_tok=round(s["reads"] / max(1, ntest), 2),
                                 used_per_tok=round(s["used"] / max(1, ntest), 2),
                                 precision=round(s["used"] / max(1, s["reads"]), 4),
                                 read_gb_per_tok=round(s["reads"] / max(1, ntest) * gb, 3),
                                 wasted_gb_per_tok=round((s["reads"] - s["used"]) / max(1, ntest) * gb, 3))
    return out, tier_led, ntest


class PrioChannel(NvmeChannel):
    """demand reads as NvmeChannel; prefetch reads queue at low priority and run only in idle channel time"""
    def __init__(self, *a):
        super().__init__(*a)
        self.q = deque()       # (key, issue_t)
        self.done = {}         # key -> arrival

    def advance(self, t):
        while self.q:
            k, ti = self.q[0]
            start = max(self.free_at, ti)
            if start + self.deep > t:
                break
            self.q.popleft()
            self.free_at = start + self.deep
            self.done[k] = self.free_at
            self.reads += 1

    def cancel(self, keys):
        if keys:
            self.q = deque(x for x in self.q if x[0] not in keys)


def pilot(R, sel, pred, n, mode, warm=200, oracle=False):
    """predictor in the loop on the test sub-stream; returns tok/s after warm-up, reads, waste"""
    led = make_ledger(R, sel)
    a = (R.nvme_expert_bytes, R.nvme_bw_gbps, R.nvme_bw_qd1_gbps, R.nvme_latency_ms)
    nv = PrioChannel(*a) if mode == "idle" else NvmeChannel(*a)
    T = len(sel)
    F = R.fixed(1)
    shift = pred.shift if pred is not None else 1
    sched = {}
    if pred is not None or oracle:
        for c in range(L):
            if (shift == 0 and c == 0) or c in EXCL:
                continue
            off, ol = owner_of(shift, c)
            sched.setdefault(ol, []).append((c, -off))
    rings, ref, arr = {}, {}, {}
    t_now, t_meas, ntok, reads, used, act, cov = 0.0, 0.0, 0, 0, 0, 0, 0

    def drop(k):
        ref.pop(k, None); arr.pop(k, None)
        if mode == "idle":
            nv.done.pop(k, None); nv.cancel({k})

    for t in range(T):
        led.step = t
        t0 = t_now
        meas = t >= warm
        for l in range(L):
            t_now += F / L
            picks = {}
            for e in sel[t, l]:
                k = l * E + int(e); picks[k] = picks.get(k, 0) + 1
            if mode == "idle":
                nv.advance(t_now)
            infl = {}
            ring = rings.pop((t, l), None)
            if ring:
                for k in ring:
                    if k not in ref:
                        continue
                    if k in picks:
                        av = arr.get(k) if mode == "fifo" else nv.done.get(k)
                        if av is not None:
                            infl[k] = av
                        drop(k)               # consumed (lands in a tier through plan_layer) or demand-read instead
                    else:
                        ref[k] -= 1
                        if ref[k] <= 0:
                            drop(k)           # unused: dropped, never enters RAM
                if meas:
                    used += len(infl)
            if meas:
                nvk = {k for k in picks if led.tier(k) == NVME}
                act += len(nvk); cov += len(nvk & set(infl))
            pl = plan_layer(R, led, picks, t_now, nv, infl)
            t_now += pl.ms
            for c, dt in sched.get(l, ()):
                tgt = t + dt
                if tgt >= T:
                    continue
                if oracle:
                    ks = [c * E + int(e) for e in sel[tgt, c]]
                else:
                    ks = pred.get(tgt, c, n)
                    if ks is None:
                        continue
                want = [k for k in dict.fromkeys(ks) if led.tier(k) == NVME]
                if not want:
                    continue
                new = [k for k in want if k not in ref]
                for k in want:
                    ref[k] = ref.get(k, 0) + 1
                if meas:
                    reads += len(new)
                if mode == "idle":
                    nv.q.extend((k, t_now) for k in new)
                else:
                    arr.update(zip(new, nv.read(t_now, len(new))))
                rings[(tgt, c)] = set(want)
        if meas:
            t_meas += t_now - t0
            ntok += 1
    gb = R.nvme_expert_bytes / 1e9
    return dict(tok_s=round(1000 * ntok / max(t_meas, 1e-9), 3), ms_per_tok=round(t_meas / max(1, ntok), 3),
                recall_nvme=round(cov / max(1, act), 4), nvme_picks_per_tok=round(act / max(1, ntok), 2),
                reads_per_tok=round(reads / max(1, ntok), 2), used_per_tok=round(used / max(1, ntok), 2),
                read_gb_per_tok=round(reads / max(1, ntok) * gb, 3), tokens=ntok)


def engine_recall(P, tgt, tier, sel, c_mask):
    """recall on picks by ENGINE tier (captured at plan time) for target tokens tgt"""
    out = {}
    tr = sel[tgt].astype(np.int64)                     # [n, L, K]
    ti = tier[tgt]
    hr = np.full(tr.shape, 10 ** 6, np.int32)
    for r in range(P.shape[2]):
        m = P[:, :, r:r + 1].astype(np.int64) == tr
        hr[m & (hr > r)] = r
    ok = np.broadcast_to(c_mask[None, :, None], tr.shape)
    for N in NS:
        h = hr < N
        for g, gm in (("all", ok), ("nvme", ok & (ti >= 3)), ("ram", ok & ((ti == 1) | (ti == 2))), ("vram", ok & (ti == 0))):
            out[f"recall@{N}/{g}"] = round(float(h[gm].mean()) if gm.any() else -1, 4)
    out["engine_nvme_picks_per_tok"] = round(float((ti >= 3).sum() / len(tgt)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--variants", default="edge0,self,same,router_same,router_next,router_self")
    ap.add_argument("--recipe", default="glm53-rtx3090-55g-nvx4")
    ap.add_argument("--budgets", default="55,16")
    ap.add_argument("--pilot", default="", help="comma list of name@N for the PILOT sim, e.g. edge0@16,self@24")
    ap.add_argument("--modes", default="idle,fifo")
    ap.add_argument("--no-route-only", action="store_true")
    ap.add_argument("--max-tokens-sim", type=int, default=0, help="cap the test sub-stream length (speed)")
    ap.add_argument("--exclude", default="", help="consumer layers to leave out everywhere, e.g. 0,41")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global EXCL
    EXCL = {int(x) for x in a.exclude.split(",") if x}
    d = os.path.join(a.run, "data")
    sel_all = np.load(os.path.join(d, "sel.npy")).astype(np.int64)
    req = np.load(os.path.join(d, "req.npy")); split = np.load(os.path.join(d, "split_req.npy"))
    tier_all = np.load(os.path.join(d, "tier.npy"))
    te_tok = np.flatnonzero(split[req] == 2)                       # test requests, capture order
    if a.max_tokens_sim:
        te_tok = te_tok[:a.max_tokens_sim]
    local_of = {int(g): i for i, g in enumerate(te_tok)}
    sel, tier = sel_all[te_tok], tier_all[te_tok]
    T = len(sel)
    res = {"T_capture": len(sel_all), "requests": int(req.max()) + 1, "test_tokens": T,
           "test_requests": int(len(np.unique(req[te_tok]))), "looka": {}, "engine": {}, "pilot": {}}
    preds = []
    for v in a.variants.split(","):
        f = os.path.join(a.run, f"pred_{v}_{a.tag}.npy")
        if not os.path.exists(f):
            log("missing", f); continue
        P = np.load(f); tgt = np.load(os.path.join(a.run, f"testidx_{v}_{a.tag}.npy"))
        keep = np.isin(tgt, te_tok)
        P, tgt = P[keep], tgt[keep]
        sh = 0 if v in ("same", "router_same") else 1
        preds.append(Pred(v, P, tgt, sh, local_of))
        cm = np.ones(L, bool)
        if sh == 0:
            cm[0] = False
        cm[list(EXCL)] = False
        res["engine"][v] = engine_recall(P, tgt, tier_all, sel_all, cm)
        log(v, {k: res["engine"][v][k] for k in ("recall@8/all", "recall@16/nvme", "recall@32/nvme")})
    if not a.no_route_only:
        t0 = time.time()
        ro, info = route_only(sel_all, req, split, te_tok)
        res["route_only_info"] = info
        cm = np.ones(L, bool); cm[0] = False; cm[list(EXCL)] = False
        for k, P in ro.items():
            preds.append(Pred(k, P, te_tok, 0, local_of))
            res["engine"][k] = engine_recall(P, te_tok, tier_all, sel_all, cm)
            log(k, {x: res["engine"][k][x] for x in ("recall@8/all", "recall@16/nvme", "recall@32/nvme")})
        log(f"route-only predictors {time.time() - t0:.0f} s")
    reg = spec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"))
    for b in a.budgets.split(","):
        R = spec.resolve(reg, a.recipe, **{"budget.ram_gb": float(b)})
        t0 = time.time()
        lk, tl, ntest = looka(R, sel, preds)
        res["looka"][b] = {"ram_slots": R.ram_slots, "vram_slots": R.vram_slots, "tokens": ntest, "pred": lk,
                           "nvme_share_of_picks": {"ledger": round(float((tl == 4).mean()), 4),
                                                   "engine": round(float((tier >= 3).mean()), 4)}}
        np.save(os.path.join(a.run, f"ledger_tier_{b}g.npy"), tl)
        log(f"LOOKA {b} GB ({time.time() - t0:.0f} s): " + ", ".join(
            f"{k} rec {v['recall_nvme']} rd {v['reads_per_tok']}" for k, v in lk.items() if k.endswith("@16")))
        json.dump(res, open(a.out, "w"), indent=1)
        if a.pilot:
            pp = {p.name: p for p in preds}
            res["pilot"].setdefault(b, {})["base"] = pilot(R, sel, None, 0, "fifo")
            for mode in a.modes.split(","):
                res["pilot"][b][f"oracle_next_{mode}"] = pilot(R, sel, None, 0, mode, oracle=True)
                for item in a.pilot.split(","):
                    name, n = item.split("@")
                    if name in pp:
                        res["pilot"][b][f"{item}_{mode}"] = pilot(R, sel, pp[name], int(n), mode)
                log(f"PILOT {b} GB {mode}: " + ", ".join(f"{k} {v['tok_s']}" for k, v in res["pilot"][b].items()
                                                         if k.endswith(mode) or k == "base"))
                json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
