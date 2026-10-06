"""What does it take to reach 50 tok/s? Stack levers on the GLM-5.3-Flash 3090 recipe and print the standard rows."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moetier import spec, sim

ROOT = os.path.join(os.path.dirname(__file__), "..", "registry")
T = os.path.join(os.path.dirname(__file__), "..", "traces")
reg = spec.load(ROOT)
streams = sim.load_trace(f"{T}/glm53-g002-decode.npy", f"{T}/glm53-g002-decode.segments.npy")
R0 = "glm53-rtx3090-55g-nvx4"
steps = [
    ("all experts in RAM (G067, 218 GB)", {"budget.ram_gb": 230}),
    ("55 GB RAM + NVMe tail", {}),
    ("+ layer-ahead NVMe prefetch (recall 0.7)", {"policy.prefetch.recall": 0.7}),
    ("+ persistent CPU workers (handoff 0.145 -> 0.03 ms/layer)", {"lane_overrides": {"cpu": {"per_layer_ms": 0.03}}}),
    ("+ fused non-MoE kernels + graphs (fixed 14 -> 9 ms)", {"calibration.fixed_ms": {"1": 9.0, "2": 13.0, "4": 20.0}}),
    ("+ 400 more VRAM slots (fp8 KV, staging reuse)", {"budget.vram_expert_slots": 1910}),
]
acc = {}
print("| lever (cumulative) | C1 | C2 | C4 | per-token ms at C1 (fixed + max(gpu, cpu)) | NVMe/token |")
print("|---|---|---|---|---|---|")
for name, ov in steps:
    if name.startswith("all"):
        cfg = dict(ov)
    else:
        acc.update(ov)
        cfg = dict(acc)
    R = spec.resolve(reg, R0, **cfg)
    rs = [sim.run(R, streams, conc=c) for c in (1, 2, 4)]
    r1 = rs[0]
    print(f"| {name} | {rs[0]['tok_s']} | {rs[1]['tok_s']} | {rs[2]['tok_s']} | "
          f"{r1['ms_fixed']} + max({r1['ms_gpu_moe']}, {r1['ms_cpu_moe']}) | {r1['nvme_per_tok']} |", flush=True)
