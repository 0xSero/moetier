#!/usr/bin/env python3
"""Build a registry run record from an arm directory (sweep.json + probe_report.json + stats + quality files).

  python3 loop/mkrun.py <arm_dir> --id <run id> --recipe <recipe> --change "..." [--source host:path] [--note "..."]

Standard table: 8k C1/C2/C4 decode (aggregate; per-stream for C2/C4) + 8k prefill, 32k C1 decode + 32k prefill.
utilization / chokepoints come from `moetier probe report` (probe_report.json); quality from score.json (panel),
decode_kl.json (paired decode KL, may live in a sibling KL arm: --kl <dir>), verify_*.json.
"""
import argparse, glob, json, os, sys, time


def med(d):
    return d["median"] if isinstance(d, dict) and "median" in d else d


def table(sw):
    pf, dc = sw.get("prefill", {}), sw.get("decode", {})
    row = lambda c: dc.get(f"C{c}", {})
    t = []
    if row(1):
        t.append({"prefill": 8192, "conc": 1, "decode": med(row(1)["aggregate"]), "prefill_tok_s": med(pf.get("8192", {}))})
    for c in (2, 4):
        if row(c):
            t.append({"prefill": 8192, "conc": c, "decode": med(row(c)["aggregate"]), "per_stream": med(row(c)["per_stream_mean"])})
    if dc.get("C1@32k"):
        t.append({"prefill": 32768, "conc": 1, "decode": med(dc["C1@32k"]["aggregate"]), "prefill_tok_s": med(pf.get("32768", {}))})
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--id", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--change", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--kl", help="dir with decode_kl.json (paired decode-KL arm)")
    ap.add_argument("--note", default="")
    ap.add_argument("--title", required=True, help="perf-table title, e.g. 'GLM-5.3-Flash · 1x RTX 3090 · 55 GB RAM + NVMe · ...'")
    ap.add_argument("--kv-cache", default="131k (~1.5 GiB est.)")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--contended", nargs="*", default=[], help="other OWNERS.md slots busy during the run, e.g. B")
    ap.add_argument("--out", help="default: registry/runs/<id>.json")
    a = ap.parse_args()
    j = lambda p: json.load(open(p)) if os.path.exists(p) else None
    sw = j(os.path.join(a.arm, "sweep.json")) or {}
    pr = j(os.path.join(a.arm, "probe_report.json")) or {}
    st = j(os.path.join(a.arm, "stats_end.json")) or {}
    nv2 = st.get("nv2", {})
    q = {}
    sc = j(os.path.join(a.arm, "score.json"))
    if sc:
        q["panel_top1"], q["panel_kl"] = round(sc["top1_agreement"], 5), round(sc["mean_kl_top20"], 6)
    kl = j(os.path.join(a.kl or a.arm, "decode_kl.json"))
    if kl:
        q["decode_kl_paired"] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                     if kk in ("positions", "kl_mean", "kl_p99", "top1_agree", "cpu_experts_per_layer_call")}
                                 for k, v in kl.items() if not k.startswith("_")}
        q["decode_kl_method"] = "same process, CPU lane off (exact GPU path) greedy natural-EOS reference vs forced decode of the same tokens with the CPU lane on"
    ver = [j(p) for p in sorted(glob.glob(os.path.join(a.arm, "verify_*.json")))]
    if ver:
        q["slot_bytecheck"] = "; ".join(f"{v.get('vram_checked')} VRAM + {v.get('ram_checked')} RAM, bad {v.get('vram_bad')}+{v.get('ram_bad')}, tables {v.get('table_bad')}" for v in ver if v)
    mem = {}
    try:
        cur, peak = [int(x) for x in open(os.path.join(a.arm, "memcg_end.txt")).read().split()[:2]]
        mem = {"cgroup_max_gb": 55, "peak_gib": round(peak / 2 ** 30, 2), "end_gib": round(cur / 2 ** 30, 2)}
    except (OSError, ValueError):
        pass
    if nv2:
        mem.update({"ram_tier_gib": nv2.get("ram_gib"), "ram_slots": nv2.get("ram_slots")})
    rec = {"id": a.id, "title": a.title, "kv_cache": a.kv_cache, "gpu_count": a.gpu_count, "recipe": a.recipe, "change": a.change, "measured": time.strftime("%Y-%m-%d"), "table": table(sw),
           "per_token": {k: v for k, v in (nv2.get("decode_per_token") or {}).items()},
           "quality": q, "memory": mem,
           "utilization": pr.get("utilization"), "chokepoints": pr.get("chokepoints"),
           "engine_critical_path_end": nv2.get("critical_path"),
           "source": a.source, "note": a.note}
    if a.contended:
        rec["contended"] = a.contended
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry", "runs", a.id + ".json")
    json.dump(rec, open(out, "w"), indent=1)
    print(out, json.dumps(rec["table"]))


if __name__ == "__main__":
    main()
