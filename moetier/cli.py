"""moetier show | sim | table   — read records, run the scheduler, print the standard table."""
import argparse, json, os, sys
from . import spec, sim

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry")


def _ovr(items):
    out = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moetier")
    ap.add_argument("cmd", choices=["show", "sim", "table"])
    ap.add_argument("recipe")
    ap.add_argument("--trace", help="npy [tokens, layers, topk]")
    ap.add_argument("--segments")
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--set", nargs="*", help="override, e.g. budget.ram_gb=55 policy.prefetch.recall=0.7")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    reg = spec.load(a.root)
    R = spec.resolve(reg, a.recipe, **_ovr(a.set))
    if a.cmd == "show":
        print(json.dumps(dict(recipe=R.id, vram_slots=R.vram_slots, ram_slots=R.ram_slots,
                              nvme_only=R.keys - R.vram_slots - R.ram_slots, lanes={k: vars(v) for k, v in R.lanes.items()},
                              policy=R.policy), indent=2))
        return
    streams = sim.load_trace(a.trace, a.segments)
    res = [sim.run(R, streams, conc=c) for c in a.conc]
    if a.cmd == "sim":
        for r in res:
            print(json.dumps(r))
        return
    pf = {p: sim.prefill(R, p) for p in (8192, 32768)}
    print(f"recipe {R.id}: vram {R.vram_slots} / ram {R.ram_slots} / nvme {R.keys - R.vram_slots - R.ram_slots} experts")
    print("| prefill length | concurrency | decode tok/s | prefill tok/s | bound |")
    print("|---|---|---|---|---|")
    for r in res:
        b = "cpu" if r["ms_cpu_moe"] > r["ms_gpu_moe"] else "gpu"
        print(f"| 8k | {r['conc']} | {r['tok_s']} | {pf[8192]['tok_s']} | fixed {r['ms_fixed']} + {b}-moe "
              f"(gpu {r['ms_gpu_moe']} / cpu {r['ms_cpu_moe']}) ms/step |")
    print(f"| 32k | 1 | (~8k C1 x 0.95) | {pf[32768]['tok_s']} | prefill {pf[32768]['bound']}-bound |")


if __name__ == "__main__":
    main()
