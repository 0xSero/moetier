"""B70 second expert tier for GLM-5.3-Flash on omarchy: sim the 'b70' lane next to the 3090 + CPU + NVMe lanes.

Rows: 55 GB and full-RAM configs, without / with one Arc Pro B70 (48:00.0) holding a static frequency-ranked expert
set, handoff sensitivity, and the all-4-GPU stretch (3 B70s + the 3090). Prints a markdown table and writes JSONL.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moetier import spec, sim

ROOT = os.path.join(os.path.dirname(__file__), "..", "registry")
T = os.path.join(os.path.dirname(__file__), "..", "traces")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "b70-tier-glm53.sim.jsonl")
reg = spec.load(ROOT)
streams = sim.load_trace(f"{T}/glm53-g002-decode.npy", f"{T}/glm53-g002-decode.segments.npy")
BASE, B70 = "glm53-rtx3090-55g-nvx4", "glm53-rtx3090-b70-55g-nvx4"
FULL = {"budget.ram_gb": 230}
NO_B70 = {"b70.cards": []}
rows = [
    ("55 GB: 3090 + CPU + NVMe (today)", B70, NO_B70),
    ("55 GB: + B70 3000 slots, handoff 0.12 ms", B70, {}),
    ("55 GB: + B70, handoff 0.05 ms", B70, {"b70.handoff_ms": 0.05}),
    ("55 GB: + B70, handoff 0.25 ms", B70, {"b70.handoff_ms": 0.25}),
    ("55 GB: + B70, handoff 0.50 ms", B70, {"b70.handoff_ms": 0.50}),
    ("55 GB: + B70 3300 slots, handoff 0.12 ms", B70, {"b70.cards": [3300]}),
    ("55 GB: reference, 3000 more slots on the 3090 itself", B70, {**NO_B70, "budget.vram_expert_slots": 4510}),
    ("full RAM: 3090 + CPU (G067 layout)", B70, {**FULL, **NO_B70}),
    ("full RAM: + B70 3000, handoff 0.12 ms", B70, FULL),
    ("stretch 55 GB: 3090 + 3x B70 x 3000", B70, {"b70.cards": [3000, 3000, 3000]}),
    ("stretch 55 GB: 3090 + 3x B70 x 3200", B70, {"b70.cards": [3200, 3200, 3200]}),
    ("stretch full RAM: 3090 + 3x B70 x 3200", B70, {**FULL, "b70.cards": [3200, 3200, 3200]}),
    ("stretch 55 GB: 3x B70 x 3200, handoff 0.25 ms", B70, {"b70.cards": [3200, 3200, 3200], "b70.handoff_ms": 0.25}),
]
only = sys.argv[1:]  # optional row-name substrings
print("| config | C | decode tok/s | 3090 VRAM hit | B70 hit | GPU hit | CPU experts/tok | NVMe/tok | zero-copy/tok "
      "| ms/step fixed + max(3090, CPU, B70) |")
print("|---|---|---|---|---|---|---|---|---|---|")
with open(OUT, "w") as f:
    for name, rid, ov in rows:
        if only and not any(o in name for o in only):
            continue
        R = spec.resolve(reg, rid, **ov)
        for c in (1, 2, 4):
            r = sim.run(R, streams, conc=c)
            r["config"] = name
            f.write(json.dumps(r) + "\n")
            bh = r.get("b70_hit", 0.0)
            print(f"| {name} | {c} | {r['tok_s']} | {r['vram_hit']} | {bh} | {round(r['vram_hit'] + bh, 3)} | "
                  f"{r['cpu_experts_per_tok']} | {r['nvme_per_tok']} | {r['zerocopy_per_tok']} | {r['ms_fixed']} + max("
                  f"{r['ms_gpu_moe']}, {r['ms_cpu_moe']}, {r.get('ms_b70_moe', 0.0)}) |", flush=True)
