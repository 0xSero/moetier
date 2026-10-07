"""HOM-272 candidate (2): frequency-aware RAM eviction vs LRU, replayed through the same ledger + plan_layer.

NVMe misses on the 55 GB tier are mostly evicted cold experts that come back (G3 study). LRU drops an expert that
was not touched recently even if it is picked often over a longer window. 'lfu:<window>:<halflife>' evicts, among the
<window> least recently used RAM entries, the one with the lowest decayed pick frequency (sampled LFU with recency,
cheap to run in the host engine: a short walk from the LRU tail). Budgets match the measured S3b engine (1312 VRAM
slots, 4831 RAM slots).

  python3 examples/ram_policy.py [--conc 1 4]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from moetier import spec, sim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S3B = {"budget.vram_expert_slots": 1312, "budget.ram_gb": 4831 * 9437184 / 1e9 + 4.0 + 2.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--policies", nargs="*", default=["lru", "lfu:8:0", "lfu:32:0", "lfu:128:0", "lfu:32:200",
                                                     "lfu:32:1000", "lfu:128:1000", "lfu:512:1000"])
    ap.add_argument("--out")
    a = ap.parse_args()
    reg = spec.load(os.path.join(ROOT, "registry"))
    streams = sim.load_trace(os.path.join(ROOT, "traces/glm53-g002-decode.npy"), os.path.join(ROOT, "traces/glm53-g002-decode.segments.npy"))
    rows = []
    for pol in a.policies:
        for c in a.conc:
            R = spec.resolve(reg, "glm53-rtx3090-55g-nvx4", **S3B, **{"policy.ram_evict": pol})
            t = time.time()
            r = sim.run(R, streams, conc=c)
            rows.append({"policy": pol, "conc": c, "tok_s": r["tok_s"], "nvme_per_tok": r["nvme_per_tok"],
                         "cpu_experts_per_tok": r["cpu_experts_per_tok"], "vram_hit": r["vram_hit"],
                         "ms_cpu_moe": r["ms_cpu_moe"], "ms_gpu_moe": r["ms_gpu_moe"], "s": round(time.time() - t, 1)})
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
