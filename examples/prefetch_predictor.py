"""Layer-ahead expert predictors for GLM-5.3-Flash decode, from routes (and router weights) only. CPU, numpy.

At layer l of token t the host knows: the picks + router weights of layer l (this token), every layer of tokens
< t. Predict layer l+1's top-8 and prefetch the non-resident part one layer ahead.

Predictors (score per candidate expert j of layer l+1, top-N kept):
  a   temporal       recency: Σ_k d^k [j ∈ S(t-k, l+1)], d=0.3 (N=8 is exactly the previous token's set)
  b   transition     Σ_{i∈S(t,l)} P(j ∈ S(l+1) | i ∈ S(l)), P learned on the train split
  bw  transition, w  Σ_i w_i P(j | i)                                     (weight-aware)
  c   a∪b split      N/2 from a (weight-ranked), the rest from b
  lr  learned mix    logistic regression over [b, b2 (two-hop l-1 -> l+1), a-features, log prior]
  lrw learned mix, w lr + [bw, previous-token weight of j]                   (weight-aware)

Evaluation: offline recall/precision on all picks and on the static cold tail (keys outside the hottest 6,300 of
the train split), then the moetier ledger in the loop (`--sim`): LOOKA = counters only (recall on the picks the
ledger actually serves from NVMe, read cost of false positives), PILOT = issue the reads (false positives occupy
the NVMe channel and are dropped after their layer).

  python3 examples/prefetch_predictor.py --n116 .../s1c_55g/nv_trace.npz --out /tmp/pp.json
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from moetier import spec, sim, plan
from moetier.ledger import NVME

L, E, K = 42, 288, 8
NS = (8, 12, 16, 24, 32)
NMAX = max(NS)
HOT = 6300


def load_n116(p):
    z = np.load(p)
    st, ly, ids, w = z["step"], z["layer"], z["ids"], z["weights"]
    S = int(st.max())
    assert len(st) == S * L and (ids >= 0).all(), "expects bsz-1 decode rows, every layer of every step"
    o = np.lexsort((ly, st))
    return ids[o].reshape(S, L, K).astype(np.int64), w[o].reshape(S, L, K).astype(np.float32)


def onehot(ids, w=None):
    """ids [T, K] -> [T, E] (indicator, or router weight)."""
    T = len(ids)
    X = np.zeros((T, E), np.float32)
    np.put_along_axis(X, ids, 1.0 if w is None else w, axis=1)
    return X


class Tables:
    """P(j ∈ S(l+1) | i ∈ S(l)) and the two-hop P(j ∈ S(l+1) | i ∈ S(l-1)), plus per-layer priors."""

    def __init__(self, ids, a=1.0):
        T = len(ids)
        self.P, self.P2, self.prior = [None] * L, [None] * L, np.zeros((L, E), np.float32)
        Xs = [onehot(ids[:, l]) for l in range(L)]
        for l in range(L):
            self.prior[l] = (Xs[l].sum(0) + 1) / (T + E / K)
        for l in range(L - 1):
            for P, src in ((self.P, l), (self.P2, l - 1)):
                if src < 0:
                    continue
                C = Xs[src].T @ Xs[l + 1]
                cnt = Xs[src].sum(0)[:, None]
                P[l + 1] = ((C + a * self.prior[l + 1][None]) / (cnt + a)).astype(np.float32)
        freq = np.stack([x.sum(0) for x in Xs]).ravel()
        self.hot = np.zeros(L * E, bool)
        self.hot[np.argsort(-freq)[:HOT]] = True


def features(ids, w, tab, m, seg_starts):
    """Per-candidate features for target layer m (1..L-1) at every token: dict of [T, E]."""
    T = len(ids)
    Xl = onehot(ids[:, m - 1])
    f = dict(b=Xl @ tab.P[m])
    if w is not None:
        f["bw"] = onehot(ids[:, m - 1], w[:, m - 1]) @ tab.P[m]
    f["b2"] = onehot(ids[:, m - 2]) @ tab.P2[m] if m >= 2 else f["b"]
    Xm = onehot(ids[:, m])
    reset = np.zeros(T, bool)
    reset[seg_starts] = True                       # no history across segment (request-stream) starts
    prev = np.zeros_like(Xm); prev2 = np.zeros_like(Xm); ema = np.zeros_like(Xm)
    prev[1:], prev2[2:] = Xm[:-1], Xm[:-2]
    acc = np.zeros(E, np.float32)
    for t in range(T):                              # ema[t] = Σ_k 0.3^(k-1) X[t-k], k>=1
        if reset[t]:
            acc[:] = 0
        ema[t] = acc
        acc = 0.3 * acc + Xm[t]
    for s0 in seg_starts:
        prev[s0] = 0
        prev2[s0:s0 + 2] = 0
    f.update(prev=prev, prev2=prev2, ema=ema, logp=np.broadcast_to(np.log(tab.prior[m]), (T, E)))
    if w is not None:
        pw = np.zeros_like(Xm)
        pw[1:] = onehot(ids[:-1, m], w[:-1, m])
        for s in seg_starts:
            pw[s] = 0
        f["prevw"] = pw
    return f, Xm


LR_ROUTE = ["b", "b2", "prev", "prev2", "ema", "logp"]
LR_W = LR_ROUTE + ["bw", "prevw"]


def scores(f, name, coef=None):
    if name == "a":
        s = f["ema"] + 0.01 * f.get("prevw", f["prev"]) + 1e-4 * f["b"]
    elif name == "b":
        s = f["b"]
    elif name == "bw":
        s = f["bw"]
    elif name in ("lr", "lrw"):
        cols, wv = coef
        s = sum(wv[i] * f[c] for i, c in enumerate(cols))
    else:
        raise KeyError(name)
    return s


def topn(s, n=NMAX):
    idx = np.argpartition(-s, n, axis=1)[:, :n]
    o = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)


def split_c(fa, fb, n):
    """a ∪ b with budget n: n//2 from a (weight-ranked), then b's best not already chosen."""
    ra, rb = topn(fa, NMAX), topn(fb, NMAX)
    ch = ra[:, :n // 2]
    dup = (rb[:, :, None] == ch[:, None, :]).any(2)
    order = np.argsort(dup, axis=1, kind="stable")
    rest = np.take_along_axis(rb, order, 1)[:, :n - n // 2]
    return np.concatenate([ch, rest], 1)


def fit_lr(ids, w, tab, seg, cols, ntok=1500, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for m in range(1, L):
        f, Xm = features(ids, w, tab, m, seg)
        sel = rng.choice(len(ids), min(ntok, len(ids)), replace=False)
        Xs.append(np.stack([f[c][sel].ravel() for c in cols], 1))
        ys.append(Xm[sel].ravel())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=300, C=1.0).fit((X - mu) / sd, y)
    wv = clf.coef_[0] / sd
    return cols, wv.astype(np.float32)


def predict_all(ids, w, tab, seg, coefs):
    """-> {name: [T, L, NMAX] ranked predictions for layers 1..L-1 (layer 0 = -1)}."""
    names = ["a", "b"] + (["bw"] if w is not None else []) + list(coefs)
    out = {n: np.full((len(ids), L, NMAX), -1, np.int16) for n in names}
    out.update({f"c{n}": np.full((len(ids), L, n), -1, np.int16) for n in NS})
    for m in range(1, L):
        f, _ = features(ids, w, tab, m, seg)
        sa = scores(f, "a")
        for n in names:
            s = sa if n == "a" else scores(f, n, coefs.get(n))
            out[n][:, m] = topn(s)
        for n in NS:
            out[f"c{n}"][:, m] = split_c(sa, f["bw"] if w is not None else f["b"], n)
    return out


def offline(ids, preds, hot):
    """Recall/precision per predictor per N on layers 1..L-1; all picks and the static cold tail."""
    T = len(ids)
    keys = np.arange(L)[None, :, None] * E + ids                     # [T, L, K]
    act_cold = ~hot[keys]
    res = {}
    for name, P in preds.items():
        for n in (NS if not name.startswith("c") else [int(name[1:])]):
            if n > P.shape[2]:
                continue
            p = P[:, 1:, :n].astype(np.int64)
            a = ids[:, 1:]
            hit = (p[:, :, :, None] == a[:, :, None, :])               # [T, L-1, n, K]
            used = hit.any(3)                                        # predicted & picked
            covered = hit.any(2)                                     # picked & predicted
            pk = np.arange(1, L)[None, :, None] * E + p
            pc = ~hot[pk]
            ac = act_cold[:, 1:]
            r = dict(recall=covered.mean(), precision=used.mean(),
                     cold_recall=(covered & ac).sum() / max(1, ac.sum()),
                     cold_reads_per_tok=pc.sum() / T, cold_useful_per_tok=(pc & used).sum() / T,
                     cold_picks_per_tok=ac.sum() / T)
            r["cold_precision"] = r["cold_useful_per_tok"] / max(1e-9, r["cold_reads_per_tok"])
            res[f"{name if not name.startswith('c') else 'c'}@{n}"] = {k: round(float(v), 4) for k, v in r.items()}
    return res


def sim_loop(R, streams, pred, n, mode, cold_budget=0, idle_ms=1.0):
    """Run moetier sim (conc 1: it replays streams[0]; pred indexes streams[0] tokens) with the predictor in the loop.
    mode 'looka': counters only (no reads). 'pilot': issue reads for predicted NVMe-tier keys (FIFO with demand).
    'idle': like pilot, but only as many reads as fit in the channel's idle time before t + idle_ms (~one layer),
    i.e. a runtime whose demand reads preempt prefetches. 'oracle': prefetch the true NVMe picks (hook check).
    n: top-n of the ranked prediction. cold_budget>0: instead take the top `cold_budget` NVMe-tier predictions.
    Also records the reuse distance (tokens since the key's last pick) of every NVMe-served pick."""
    st = dict(calls=0, act_nvme=0, covered=0, reads=0, useful=0)
    rd = dict(never=0, lt64=0, lt256=0, lt1024=0, ge1024=0)
    last = {}
    pending = {}
    T = len(streams[0])

    def hook(R_, led, nxt, t, nv, inflight, recall, salt=0):
        m = next(iter(nxt)) // E
        for ks in pending.values():                                  # unused prefetches of already-planned layers
            for k in ks:
                inflight.pop(k, None)
        pending.clear()
        tok = salt % T
        ranked = [m * E + int(e) for e in pred[tok, m] if e >= 0]
        cand = [k for k in (ranked if cold_budget else ranked[:n]) if led.tier(k) == NVME and k not in inflight]
        if cold_budget:
            cand = cand[:cold_budget]
        actn = {k for k in nxt if led.tier(k) == NVME}
        if mode == "oracle":
            cand = [k for k in actn if k not in inflight]
        if mode == "idle":
            room = t + idle_ms - max(nv.free_at, t)
            cand = cand[:max(0, int(room / nv.deep))]
        if salt >= 200:
            for k in actn:
                d = salt - last[k] if k in last else None
                rd["never" if d is None else "lt64" if d < 64 else "lt256" if d < 256 else "lt1024" if d < 1024
                   else "ge1024"] += 1
        for k in nxt:
            last[k] = salt
        if salt >= 200:                                               # sim's warm-up
            st["calls"] += 1
            st["act_nvme"] += len(actn)
            st["covered"] += len(actn & set(cand))
            st["reads"] += len(cand)
            st["useful"] += len(set(cand) & set(nxt))
        if mode in ("pilot", "idle", "oracle") and cand:
            for k, a_ in zip(cand, nv.read(t, len(cand))):
                inflight[k] = a_
            pending[m] = [k for k in cand if k not in nxt]
        return len(cand)

    orig = sim.prefetch
    sim.prefetch = hook
    try:
        r = sim.run(R, streams, conc=1)
    finally:
        sim.prefetch = orig
    toks = max(1, (st["calls"] // (L - 1)))
    r.update(pf_recall_nvme=round(st["covered"] / max(1, st["act_nvme"]), 4),
             pf_reads_per_tok=round(st["reads"] / toks, 2), pf_useful_per_tok=round(st["useful"] / toks, 2),
             pf_wasted_per_tok=round((st["reads"] - st["useful"]) / toks, 2),
             pf_act_nvme_per_tok=round(st["act_nvme"] / toks, 2),
             nvme_reuse_dist={k: round(v / max(1, sum(rd.values())), 3) for k, v in rd.items()})
    r["pf_gb_per_tok"] = round(st["reads"] / toks * R.nvme_expert_bytes / 1e9, 3)
    r["pf_wasted_gb_per_tok"] = round(r["pf_wasted_per_tok"] * R.nvme_expert_bytes / 1e9, 3)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n116", required=True, help="nv_trace.npz (step, layer, ids, weights)")
    ap.add_argument("--g002", default=os.path.expanduser("~/moetier/traces/glm53-g002-decode.npy"))
    ap.add_argument("--g002-seg", default=os.path.expanduser("~/moetier/traces/glm53-g002-decode.segments.npy"))
    ap.add_argument("--recipe", default="glm53-rtx3090-55g-nvx4")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ids, w = load_n116(a.n116)
    S = len(ids)
    cA, cB = int(S * 0.5), int(S * 0.7)
    # P from [0, cA); LR fit on [cA, cB) with that P (out-of-sample features); test [cB, S) with P from [0, cB)
    tabA = Tables(ids[:cA])
    seg0 = np.array([0])
    coefs = dict(lr=fit_lr(ids[cA:cB], None, tabA, seg0, LR_ROUTE), lrw=fit_lr(ids[cA:cB], w[cA:cB], tabA, seg0, LR_W))
    print("LR coef", {k: dict(zip(v[0], np.round(v[1], 3).tolist())) for k, v in coefs.items()}, flush=True)
    tab = Tables(ids[:cB])
    test_ids, test_w = ids[cB:], w[cB:]
    out = dict(n116=dict(steps=S, train=cB, test=S - cB, coefs={k: dict(zip(v[0], v[1].tolist())) for k, v in coefs.items()}))
    pr = predict_all(test_ids, test_w, tab, seg0, coefs)
    out["offline_n116"] = offline(test_ids, pr, tab.hot)
    print(f"[{time.time() - t0:.0f}s] n116 offline done", flush=True)
    # cross-dataset: tables from all of N116, route-only predictors on G002 (no weights)
    g = np.load(a.g002).astype(np.int64)
    gs = np.load(a.g002_seg)
    tabF = Tables(ids)
    prg = predict_all(g, None, tabF, gs[:-1], dict(lr=coefs["lr"]))
    out["offline_g002"] = offline(g, prg, tabF.hot)
    print(f"[{time.time() - t0:.0f}s] g002 offline done", flush=True)
    if a.sim:
        reg = spec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"))
        R = spec.resolve(reg, a.recipe, **{"policy.prefetch.recall": 1.0, "policy.prefetch.depth": 1})
        R0 = spec.resolve(reg, a.recipe)
        res = {}
        g0 = sim.load_trace(a.g002, a.g002_seg)                     # all segments, like the CLI (conc 1 = segment 0)
        pg0 = {k: v[gs[0]:gs[1]] for k, v in prg.items()}
        res["g002_base"] = sim.run(R0, g0, conc=1)
        res["g002_oracle"] = sim_loop(R, g0, pg0["a"], 8, "oracle")
        for name in ("b", "lr"):
            for n in NS:
                res[f"g002_idle_{name}@{n}"] = sim_loop(R, g0, pg0[name], n, "idle")
            for cb in (1, 2, 4, 8):
                res[f"g002_idle_{name}_cold{cb}"] = sim_loop(R, g0, pg0[name], 0, "idle", cold_budget=cb)
        for name in ("a", "b", "lr"):
            for n in NS:
                res[f"g002_looka_{name}@{n}"] = sim_loop(R, g0, pg0[name], n, "looka")
            for n in NS:
                res[f"g002_pilot_{name}@{n}"] = sim_loop(R, g0, pg0[name], n, "pilot")
            for cb in (1, 2, 3, 4, 6, 8):
                res[f"g002_pilot_{name}_cold{cb}"] = sim_loop(R, g0, pg0[name], 0, "pilot", cold_budget=cb)
            print(f"[{time.time() - t0:.0f}s] sim {name} done", flush=True)
        # N116 test stream with weight-aware predictor
        res["n116_base"] = sim.run(R0, [test_ids], conc=1)
        res["n116_oracle"] = sim_loop(R, [test_ids], pr["a"], 8, "oracle")
        for name in ("lr", "lrw"):
            for n in (8, 16, 24, 32):
                res[f"n116_idle_{name}@{n}"] = sim_loop(R, [test_ids], pr[name], n, "idle")
            for cb in (2, 4, 8):
                res[f"n116_idle_{name}_cold{cb}"] = sim_loop(R, [test_ids], pr[name], 0, "idle", cold_budget=cb)
        for name in ("a", "lr", "lrw"):
            for n in (8, 16, 24):
                res[f"n116_looka_{name}@{n}"] = sim_loop(R, [test_ids], pr[name], n, "looka")
                res[f"n116_pilot_{name}@{n}"] = sim_loop(R, [test_ids], pr[name], n, "pilot")
            for cb in (2, 4):
                res[f"n116_pilot_{name}_cold{cb}"] = sim_loop(R, [test_ids], pr[name], 0, "pilot", cold_budget=cb)
            print(f"[{time.time() - t0:.0f}s] n116 sim {name} done", flush=True)
        out["sim"] = res
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[{time.time() - t0:.0f}s] wrote {a.out}")


if __name__ == "__main__":
    main()
