"""HOM-272: C1 decode latency analysis from the nv2 per-layer-call dump (layer_dump.npz) and, optionally, an Nsight
Systems sqlite export of ~20 steady tokens.

  python3 examples/latency_c1.py --dump layer_dump.npz [--lat lat.json] [--nsys c1_trace.sqlite] [--probe probe_report.json]
         --out docs/latency-glm53-c1/analysis.json

Dump rows are device-clock (globaltimer ns) timestamps of each decode MoE layer call; host times are already mapped to
device time by the engine (offset = min(host notice - device publish)). Model: GLM-5.3-Flash, 45 layers, dense MLP
layers 0-2, MoE layers 3-44 (nv2 li = layer - 3), full (DSA) attention at layers 3,7,...,43, KDA linear attention
elsewhere. The gap after MoE call li is the non-MoE work of model layer li+4 (its attention + norms + router + shared
expert of li); the gap after li = 41 is the step boundary (final norm, lm_head, sampling, scheduler, embedding, dense
layers 0-2 incl. their attention, layer 3 attention).
"""
import argparse, json, os, sqlite3, statistics
import numpy as np

FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}
COLS = None


def q(x, p):
    x = np.asarray(x, np.float64)
    return float(np.percentile(x, p)) if len(x) else float("nan")


def load_dump(path):
    z = np.load(path)
    cols = [str(c) for c in z["cols"]]
    return z["rows"].astype(np.int64), {c: i for i, c in enumerate(cols)}


def steps_from_dump(rows, C, L=42, idle_ms=50.0):
    """group consecutive layer calls into decode steps (li 0..L-1, single token); keep complete steps only"""
    rows = rows[np.argsort(rows[:, C["seq"]])]
    steps, cur = [], []
    for r in rows:
        if r[C["ntok"]] != 1:
            cur = []; continue
        if r[C["li"]] == 0:
            cur = []
        if cur and r[C["seq"]] != cur[-1][C["seq"]] + 1:
            cur = []
        cur.append(r)
        if r[C["li"]] == L - 1:
            if len(cur) == L:
                steps.append(np.array(cur))
            cur = []
    # drop steps whose boundary gap is idle (end of request) or whose next call is not decode
    out = []
    for s in steps:
        last = s[-1]
        gap = (last[C["t_next_pub"]] - last[C["t_comb1"]]) / 1e6
        if last[C["next_is_decode"]] and gap < idle_ms:
            out.append(s)
    return out


def layer_components(r, C):
    """ms per component for one layer call (critical-path decomposition, sums to t_next_pub - t_pub)"""
    g = lambda k: float(r[C[k]])
    pub, seen, c0, c1, g0, g1, nxt = (g(k) for k in ("t_pub", "t_seen", "t_copy0", "t_copy1", "t_comb0", "t_comb1", "t_next_pub"))
    if not (g0 >= c1 and g1 >= g0 and g1 <= nxt):
        g0 = g1 = c1
    cp = c1 - c0
    cpnv = min(cp, g("copy_nvme_wait_ns") + max(0.0, g("fix_ns") - 10000))
    over = g1 - g0
    ovnv = min(over, g("cpu_land_ns")) if r[C["ncpu"]] > 0 else 0.0
    gap = nxt - g1
    lh = g("t_next_launch")
    bub = max(0.0, min(gap, lh - g1)) if lh > 0 else 0.0
    ms = lambda x: x / 1e6
    d = {"plan": ms(seen - pub), "bookkeeping": ms(c0 - seen), "copy": ms(cp - cpnv), "nvme_wait": ms(cpnv + ovnv),
         "gpu_moe": ms(g0 - c1), "cpu_overrun": ms(over - ovnv), "fixed_gpu": ms(gap - bub), "bubble": ms(bub)}
    lanes = {"gpu_lane": ms(g0 - c0), "cpu_lane": ms(g("t_cpu1") - g("t_cpu0")) if r[C["ncpu"]] > 0 else 0.0,
             "cpu_start_after_seen": ms(g("t_cpu0") - seen) if r[C["ncpu"]] > 0 else 0.0,
             "cpu_end_vs_gpu_end": ms(g("t_cpu1") - g0) if r[C["ncpu"]] > 0 else float("nan"),
             "wall": ms(nxt - pub), "admit_jobs": float(r[C["admit_jobs"]]), "ncpu": float(r[C["ncpu"]]), "nnv": float(r[C["nnv"]])}
    return d, lanes


def gap_type(li, L=42):
    if li == L - 1:
        return "step_boundary"
    nxt_layer = li + 1 + 3
    return "dsa_attention" if nxt_layer in FULL_ATTN else "kda_linear_attention"


def analyze_dump(path, L=42):
    rows, C = load_dump(path)
    steps = steps_from_dump(rows, C, L)
    per_tok, per_type, lane_rows, per_li = [], {}, [], {}
    for s in steps:
        tot = {}
        for r in s:
            d, ln = layer_components(r, C)
            for k, v in d.items():
                tot[k] = tot.get(k, 0.0) + v
            ty = gap_type(int(r[C["li"]]), L)
            per_type.setdefault(ty, []).append(d["fixed_gpu"] + d["bubble"])
            per_li.setdefault(int(r[C["li"]]), []).append({**d, **ln})
            lane_rows.append(ln)
        tot["wall"] = sum(tot.values())
        per_tok.append(tot)
    keys = list(per_tok[0]) if per_tok else []
    agg = {k: {"mean": round(statistics.fmean([t[k] for t in per_tok]), 3), "p50": round(q([t[k] for t in per_tok], 50), 3),
               "p90": round(q([t[k] for t in per_tok], 90), 3)} for k in keys}
    walls = [t["wall"] for t in per_tok]
    types = {k: {"n": len(v), "p50": round(q(v, 50), 4), "p90": round(q(v, 90), 4), "mean": round(float(np.mean(v)), 4)} for k, v in per_type.items()}
    lanes = {}
    for k in ("gpu_lane", "cpu_lane", "cpu_start_after_seen", "cpu_end_vs_gpu_end", "admit_jobs", "ncpu", "nnv"):
        v = [x[k] for x in lane_rows if not (isinstance(x[k], float) and np.isnan(x[k]))]
        lanes[k] = {"p50": round(q(v, 50), 4), "p90": round(q(v, 90), 4), "mean": round(float(np.mean(v)), 4)}
    cpu_last = float(np.mean([x["cpu_end_vs_gpu_end"] > 0.005 for x in lane_rows if not np.isnan(x["cpu_end_vs_gpu_end"])]))
    return {"steps": len(steps), "tok_s_from_wall": round(1000 / np.mean(walls), 2) if walls else None,
            "wall_ms": {"mean": round(float(np.mean(walls)), 3), "p50": round(q(walls, 50), 3), "p90": round(q(walls, 90), 3),
                        "std": round(float(np.std(walls)), 3)},
            "per_token_ms": agg, "gap_by_next_layer_type_ms": types, "lanes_per_layer_ms": lanes, "cpu_lane_last_frac": round(cpu_last, 3),
            "per_li_wall_p50_ms": {li: round(q([x["wall"] for x in v], 50), 4) for li, v in sorted(per_li.items())}}, steps, C


def waterfall(step, C, nsys_rows=None):
    """one token as text: per layer, offsets (ms) from the step start of each phase on the GPU stream and the CPU lane"""
    t0 = float(step[0][C["t_pub"]])
    ms = lambda x: (float(x) - t0) / 1e6
    lines = ["li | GPU: pub->seen | book | copy (nvme) | MoE | wait CPU | non-MoE gap  ||  CPU lane [start, end]  | crit"]
    for r in step:
        d, ln = layer_components(r, C)
        crit = "CPU" if ln["cpu_end_vs_gpu_end"] > 0.005 else "GPU"
        if d["nvme_wait"] > 0.5 * max(1e-9, d["copy"] + d["nvme_wait"] + d["cpu_overrun"]):
            crit = "NVMe"
        lines.append(f"{int(r[C['li']]):2d} | {ms(r[C['t_pub']]):7.3f} +{d['plan']:.3f} | +{d['bookkeeping']:.3f} | "
                     f"+{d['copy'] + d['nvme_wait'] - (min(d['nvme_wait'], 1e9) if False else 0):.3f} ({d['nvme_wait']:.3f}) | +{d['gpu_moe']:.3f} | "
                     f"+{d['cpu_overrun']:.3f} | +{d['fixed_gpu'] + d['bubble']:.3f} (bub {d['bubble']:.3f}) || "
                     f"[{ms(r[C['t_cpu0']]) if r[C['ncpu']] else float('nan'):7.3f}, {ms(r[C['t_cpu1']]) if r[C['ncpu']] else float('nan'):7.3f}] "
                     f"x{int(r[C['ncpu']])} | {crit}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
def nsys_kernels(path):
    """kernel / memcpy / NVTX rows from an nsys sqlite export"""
    db = sqlite3.connect(path)
    cur = db.cursor()
    names = dict(cur.execute("select id, value from StringIds").fetchall())
    k = cur.execute("select start, end, streamId, demangledName, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    kern = [(s, e, st, names.get(sn, str(sn))) for s, e, st, dn, sn in k]
    try:
        mc = cur.execute("select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    except sqlite3.OperationalError:
        mc = []
    try:
        nv = cur.execute("select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null").fetchall()
        nv = [(s, e, t if t is not None else names.get(ti, ""), g) for s, e, t, ti, g in nv]
    except sqlite3.OperationalError:
        nv = []
    return kern, mc, nv


KCLASS = [("nv_pub", "nv2 publish"), ("nv_step", "nv2 step (reply wait + admission)"), ("nv_copy", "nv2 admission copy"),
          ("nv_combine", "nv2 combine (CPU partial)"), ("nv_restore", "nv2 restore"),
          ("exl3_moe|moe|mgemm|bc_|block_sparse", "routed MoE (EXL3)"), ("routing|topk|ds3", "router"),
          ("kda|chunk_gated|gated_delta|fused_recurrent|short_conv|causal_conv", "KDA linear attention"),
          ("flash|attn|attention|dsa|indexer|sparse_mla|mla", "DSA / full attention"),
          ("rms|norm|layernorm", "norms"), ("hc|hyper", "hyper-connections"), ("gemm|gemv|exl3|matmul|cutlass|sm80", "dense GEMM/GEMV (EXL3 linears)"),
          ("argmax|sample|softmax|topp|topk_sampling", "sampling"), ("elementwise|vectorized|reduce|copy|fill|cat|index", "torch elementwise / glue")]


def kclass(name):
    import re
    n = name.lower()
    for pat, lab in KCLASS:
        if re.search(pat, n):
            return lab
    return "other"


def analyze_nsys(path, tok_bounds=None):
    kern, mc, nv = nsys_kernels(path)
    if not kern:
        return {"kernels": 0}
    kern.sort()
    t0, t1 = kern[0][0], kern[-1][1]
    # GPU busy (union of kernel intervals) and per-class time
    busy, cs, ce = 0, None, None
    for s, e, *_ in kern:
        if cs is None or s > ce:
            if cs is not None:
                busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    by = {}
    for s, e, st, n in kern:
        c = kclass(n)
        b = by.setdefault(c, [0, 0])
        b[0] += e - s; b[1] += 1
    top = {}
    for s, e, st, n in kern:
        top.setdefault(n[:80], [0, 0]); top[n[:80]][0] += e - s; top[n[:80]][1] += 1
    span = t1 - t0
    nvsum = {}
    for s, e, t, g in nv:
        key = (t or "").split("|")[-1] if "|" in (t or "") else (t or "")
        x = nvsum.setdefault(key, [0, 0]); x[0] += e - s; x[1] += 1
    mcs = {}
    for s, e, st, b, kind in mc:
        x = mcs.setdefault(int(kind), [0, 0, 0]); x[0] += e - s; x[1] += 1; x[2] += b
    return {"window_ms": round(span / 1e6, 2), "kernels": len(kern), "gpu_busy_frac": round(busy / span, 3),
            "gpu_idle_ms": round((span - busy) / 1e6, 2),
            "kernel_time_ms_by_class": {k: {"ms": round(v[0] / 1e6, 2), "n": v[1]} for k, v in sorted(by.items(), key=lambda x: -x[1][0])},
            "top_kernels_ms": {k: {"ms": round(v[0] / 1e6, 2), "n": v[1]} for k, v in sorted(top.items(), key=lambda x: -x[1][0])[:25]},
            "nvtx_ms": {k: {"ms": round(v[0] / 1e6, 2), "n": v[1]} for k, v in sorted(nvsum.items(), key=lambda x: -x[1][0])[:40]},
            "memcpy": {str(k): {"ms": round(v[0] / 1e6, 2), "n": v[1], "MB": round(v[2] / 1e6, 1)} for k, v in mcs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--nsys")
    ap.add_argument("--lat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--waterfall-out")
    a = ap.parse_args()
    res, steps, C = analyze_dump(a.dump)
    if a.lat:
        lat = json.load(open(a.lat))
        rates = []
        for r in lat["requests"]:
            t = r["times"]
            if len(t) > 60:
                rates.append((len(t) - 41) / (t[-1] - t[40]))
        res["client_steady_tok_s"] = [round(x, 2) for x in rates]
    if a.nsys and os.path.exists(a.nsys):
        res["nsys"] = analyze_nsys(a.nsys)
    if steps and a.waterfall_out:
        mid = steps[len(steps) // 2]
        open(a.waterfall_out, "w").write(waterfall(mid, C) + "\n")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_li_wall_p50_ms",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
