#!/usr/bin/env python3
"""Compare screening arms: bench numbers + decode critical-path attribution per phase (from probe_report.json).

  python3 loop/armcmp.py <arm_dir> [<arm_dir> ...]
"""
import json, os, sys

B = ("fixed_gpu", "gpu_moe", "cpu_moe", "nvme_stall", "copy", "overhead")


def load(d):
    j = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}
    return j("sweep.json"), j("probe_report.json"), (j("stats_end.json").get("nv2") or {})


def med(x):
    return x.get("median") if isinstance(x, dict) else x


def main():
    rows = []
    for d in sys.argv[1:]:
        sw, pr, nv = load(d)
        name = os.path.basename(d.rstrip("/"))
        dec = sw.get("decode", {})
        pf = sw.get("prefill", {})
        print(f"== {name}: C1 {med(dec.get('C1', {}).get('aggregate'))}  C4 {med(dec.get('C4', {}).get('aggregate'))}  "
              f"8k pf {med(pf.get('8192'))}  read_ms {nv.get('read_ms_mean')}  ram_evict {nv.get('ram_evict', {}).get('policy')}")
        for ph, c in (pr.get("chokepoints", {}).get("phases") or {}).items():
            a = c.get("decode")
            if not a or a["tokens"] < 200:
                continue
            q = a["quality"]
            print(f"   {ph:14s} wall {a['wall_ms_per_token']:6.1f} | " + " ".join(f"{k} {a['per_token_ms'][k]:5.1f}" for k in B) +
                  f" | last c/n {a['lane_last']['cpu']:.2f}/{a['lane_last']['nvme']:.2f} | cpu_end_seen {q.get('cpu_end_to_seen_ms')}"
                  f" wb {q.get('cpu_end_to_seen_wb_busy_ms')}/{q.get('wb_busy_frac_of_waits')} | nvme/step {a['per_step']['nvme_picks']}")
            p = c.get("prefill")
        for ph, c in (pr.get("chokepoints", {}).get("phases") or {}).items():
            p = c.get("prefill")
            if p and p["tokens"] >= 8000:
                print(f"   {ph:14s} prefill {p['tokens']} tok, per fwd {p['ms_per_forward']}")


if __name__ == "__main__":
    main()
