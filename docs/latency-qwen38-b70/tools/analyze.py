#!/usr/bin/env python3
"""HOM-272 B70 profiling analysis. usage: analyze.py <run dir> -> <run dir>/analysis.json + printed tables."""
import glob, json, os, sys
import numpy as np

D = sys.argv[1]
J = lambda p: [json.loads(l) for l in open(p) if l.strip()]
rows = []
for p in glob.glob(os.path.join(D, "steps.*.jsonl")):
    rows += J(p)
rows.sort(key=lambda r: r["w"])
load = J(os.path.join(D, "load.jsonl")) if os.path.exists(os.path.join(D, "load.jsonl")) else []
samp = J(os.path.join(D, "sampler.jsonl")) if os.path.exists(os.path.join(D, "sampler.jsonl")) else []
host = J(os.path.join(D, "hostmon.jsonl")) if os.path.exists(os.path.join(D, "hostmon.jsonl")) else []
A = {}


def st(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if not len(x):
        return None
    return dict(mean=round(float(x.mean()), 3), p50=round(float(np.percentile(x, 50)), 3),
                p90=round(float(np.percentile(x, 90)), 3), p99=round(float(np.percentile(x, 99)), 3), n=int(len(x)))


def window_rows(t0, t1, bs):
    return [r for r in rows if t0 <= r["w"] <= t1 and r.get("mode") == "DECODE" and r.get("bs") == bs and not r.get("prof")]


def step_table(ws):
    """ws: list of row lists (one per window). Per-step wall from consecutive forward starts inside each window."""
    out = {}
    K = ["g_fwd_ms", "g_tier_ms", "g_gap_ms", "g_sample_ms", "h_fwd_ms", "h_gap_ms", "tier_ms", "t_sync_ms",
         "h_sample_ms", "t_masked", "t_evicts", "t_admit_dropped", "t_nvme_reads", "t_nvme_ms", "s_admit", "s_victim",
         "s_victim_wb", "s_vram", "s_ram", "s_ram_only", "inflight0", "flush_ms"]
    acc = {k: [] for k in K + ["wall_ms"]}
    for rs in ws:
        for a, b in zip(rs, rs[1:]):
            if b["i"] != a["i"] + 1:
                continue
            acc["wall_ms"].append((b["t0"] - a["t0"]) * 1e3)
            for k in K:
                if k in ("s_admit", "s_victim", "s_victim_wb", "t_masked"):
                    acc[k].append(b.get(k))       # one-step-lagged snapshot: row i+1 describes step i
                else:
                    acc[k].append(a.get(k))
    for k, v in acc.items():
        out[k] = st(v)
    return out


def busy(t0, t1):
    s = [x for x in samp if t0 <= x["t"] <= t1]
    res = {}
    if len(s) >= 2:
        a, b = s[0], s[-1]
        for cid, d in b["eng"].items():
            if cid not in a["eng"]:
                continue
            for k, v in d.items():
                if k.startswith("drm-cycles-"):
                    e = k[len("drm-cycles-"):]
                    tot = d.get("drm-total-cycles-" + e, 0) - a["eng"][cid].get("drm-total-cycles-" + e, 0)
                    if tot > 0:
                        res.setdefault(e, 0.0)
                        res[e] += (v - a["eng"][cid].get(k, 0)) / tot
        th = {}
        ta, tb = None, None
        for x in s:
            if "th" in x:
                if ta is None: ta = x
                tb = x
        if ta and tb and tb is not ta:
            dt = tb["t"] - ta["t"]
            for k, (comm, ticks, cpu) in tb["th"].items():
                if k in ta["th"]:
                    u = (ticks - ta["th"][k][1]) / 100 / dt
                    if u > 0.03:
                        th.setdefault(comm, 0.0); th[comm] += u
            res["threads_cpu"] = {k: round(v, 3) for k, v in sorted(th.items(), key=lambda x: -x[1])[:14]}
    h = [x for x in host if t0 <= x["t"] <= t1]
    if len(h) >= 2:
        a, b = h[0], h[-1]; dt = b["t"] - a["t"]
        cores = {}
        for c in [f"cpu{i}" for i in range(40, 48)]:
            if c in a["cpu"] and c in b["cpu"]:
                d = [y - x for x, y in zip(a["cpu"][c], b["cpu"][c])]
                tot = sum(d); idle = d[3] + d[4]
                cores[c] = round(1 - idle / tot, 3) if tot else None
        res["cores_40_47_busy"] = cores
        res["cores_40_47_mean"] = round(float(np.mean([v for v in cores.values() if v is not None])), 3)
        dk = {}
        for n in b["dk"]:
            if n in a["dk"]:
                x, y = a["dk"][n], b["dk"][n]
                dk[n] = dict(rd_GBps=round((y[1] - x[1]) * 512 / dt / 1e9, 3), rd_iops=round((y[0] - x[0]) / dt),
                             util=round((y[5] - x[5]) / 1e3 / dt, 3), wr_GBps=round((y[4] - x[4]) * 512 / dt / 1e9, 3))
        res["disk"] = dk
        try:
            res["gt_idle_frac"] = round((float(b["gt_idle_ms"]) - float(a["gt_idle_ms"])) / 1e3 / dt, 3)
        except Exception:
            pass
        res["gt_act_mhz"] = st([float(x["gt_act"]) for x in h if x.get("gt_act")])
    return res


phases = {}
for r in load:
    phases.setdefault(r["phase"], []).append(r)
A["requests"] = {ph: [dict(ct=r.get("completion_tokens"), pt=r.get("prompt_tokens"), tok_s=r.get("decode_tok_s"),
                         ttft=r.get("ttft_s"), finish=r.get("finish")) for r in rs] for ph, rs in phases.items()}
# C1: each request's decode span
if "c1" in phases:
    ws = [window_rows(r["first"] + 0.5, r["end"] - 0.2, 1) for r in phases["c1"] if r.get("first")]
    A["c1_steps"] = step_table(ws)
    A["c1_busy"] = [busy(r["first"] + 0.5, r["end"] - 0.2) for r in phases["c1"] if r.get("first")]
    A["c1_client_tok_s"] = [r.get("decode_tok_s") for r in phases["c1"]]
if "c4" in phases:
    rs = phases["c4"]
    # sglang admits only 3 of the 4 (KV reservation without max_tokens): steady window = all early streams decoding
    rs2 = sorted(rs, key=lambda r: r["first"])
    early = rs2[:-1] if rs2[-1]["first"] > min(r["end"] for r in rs2[:-1]) else rs2
    t0 = max(r["first"] for r in early) + 0.5; t1 = min(r["end"] for r in early) - 0.2
    A["c4_window_s"] = round(t1 - t0, 2); A["c4_window_bs"] = len(early)
    A["c4_steps"] = step_table([window_rows(t0, t1, len(early))])
    A["c4_busy"] = busy(t0, t1)
    A["c4_agg_tok_s"] = round(sum(r["completion_tokens"] for r in rs) / (max(r["end"] for r in rs) - min(r["first"] for r in rs)), 2)
    # prefill window: from first request start to the last first token
    A["c4_prefill_busy"] = busy(min(r["t0"] for r in rs), max(r["first"] for r in rs))
# prefill of C1 requests
if "c1" in phases:
    A["c1_prefill_busy"] = [busy(r["t0"] + 0.3, r["first"]) for r in phases["c1"] if r.get("first")]
# ngram stats deltas over c1
ng = [(r["w"], r["ng"]) for r in rows if r.get("ng")]
A["ngram_samples"] = len(ng)
if len(ng) >= 2 and "c1" in phases:
    t0 = phases["c1"][0]["t0"]; t1 = phases["c1"][-1]["end"]
    s = [x for x in ng if t0 <= x[0] <= t1]
    if len(s) >= 2:
        a, b = s[0][1], s[-1][1]
        A["ngram_c1_delta"] = {k: b[k] - a[k] for k in b if isinstance(b.get(k), (int, float)) and isinstance(a.get(k), (int, float))}
json.dump(A, open(os.path.join(D, "analysis.json"), "w"), indent=1)
print(json.dumps(A, indent=1)[:12000])
