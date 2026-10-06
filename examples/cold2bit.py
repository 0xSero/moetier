"""What do 2-bit cold tiers buy on the GLM-5.3-Flash 3090 box? (docs/vllm-moet-lessons.md section 6)

All rows start from the last cumulative lever of examples/levers.py (prefetch 0.7, persistent CPU workers,
fixed 9 ms, 1910 VRAM slots). VRAM always holds 3.05bpw experts; 'lowbit' = share of routed picks NOT served
from VRAM, i.e. computed from a 2-bit copy, when RAM/NVMe hold 2-bit records (an upper bound for row 2,
see the doc). Time is modeled; nothing here is measured.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moetier import spec, sim

ROOT = os.path.join(os.path.dirname(__file__), "..", "registry")
T = os.path.join(os.path.dirname(__file__), "..", "traces")
reg = spec.load(ROOT)
streams = sim.load_trace(f"{T}/glm53-g002-decode.npy", f"{T}/glm53-g002-decode.segments.npy")
LEVERS = {"policy.prefetch.recall": 0.7, "calibration.fixed_ms": {"1": 9.0, "2": 13.0, "4": 20.0},
          "budget.vram_expert_slots": 1910}
CPU = lambda ms: {"cpu": {"per_layer_ms": 0.03, "per_expert_ms": ms}}
B2, B25, B267 = 6328320, 7077888, 8425472     # EXL3 K=2; 2-bit LUT + e8m0/32 (2.25 bpw); EXL3 (3,3,2)
rows = [
    ("all levers, 3.05bpw everywhere (levers.py last row)", "glm53-rtx3090-55g-nvx4", {"lane_overrides": CPU(0.104)}),
    ("+ NVMe tail stored at 2.0bpw EXL3 (RAM stays 3.05)", "glm53-rtx3090-55g-nvx4",
     {"lane_overrides": CPU(0.104), "tiers": {"nvme_expert_bytes": B2}}),
    ("RAM+NVMe 2.0bpw EXL3, CPU decode-bound (0.104 ms)", "glm53-rtx3090-55g-nvx4-cold2b",
     {"lane_overrides": {**CPU(0.104), "zerocopy": {"per_expert_ms": 0.254}}}),
    ("RAM+NVMe 2.0bpw EXL3, CPU -15% (cheaper K=2 bit extract)", "glm53-rtx3090-55g-nvx4-cold2b",
     {"lane_overrides": {**CPU(0.088), "zerocopy": {"per_expert_ms": 0.254}}}),
    ("RAM+NVMe 2.0bpw EXL3, CPU bytes-proportional (0.069, bound)", "glm53-rtx3090-55g-nvx4-cold2b",
     {"lane_overrides": {**CPU(0.069), "zerocopy": {"per_expert_ms": 0.254}}}),
    ("RAM+NVMe (3,3,2) EXL3 2.67bpw, CPU 0.104", "glm53-rtx3090-55g-nvx4",
     {"lane_overrides": {**CPU(0.104), "zerocopy": {"per_expert_ms": 0.338}}, "tiers": {"ram_expert_bytes": B267, "nvme_expert_bytes": B267}}),
    ("RAM+NVMe vLLM-Moet 2-bit LUT planes (2.25bpw), CPU DRAM-bound 0.060", "glm53-rtx3090-55g-nvx4",
     {"lane_overrides": {**CPU(0.060), "zerocopy": {"per_expert_ms": 0.284}}, "tiers": {"ram_expert_bytes": B25, "nvme_expert_bytes": B25}}),
    ("3.05bpw everywhere, CPU at a hypothetical 0.060 (reference)", "glm53-rtx3090-55g-nvx4", {"lane_overrides": CPU(0.060)}),
]
print("| scenario | RAM slots | C1 | C2 | C4 | C1 ms: fixed + max(gpu, cpu) | CPU experts/tok | NVMe/tok (GB) | lowbit picks C1 |")
print("|---|---|---|---|---|---|---|---|---|")
for name, rid, ov in rows:
    R = spec.resolve(reg, rid, **{**LEVERS, **ov})
    rs = [sim.run(R, streams, conc=c) for c in (1, 2, 4)]
    r = rs[0]
    low = "0" if R.ram_expert_bytes == R.expert_bytes and R.nvme_expert_bytes == R.expert_bytes else (
        f">={r['nvme_per_tok'] / (R.layers * R.topk):.3f}" if R.ram_expert_bytes == R.expert_bytes else f"{1 - r['vram_hit']:.3f}")
    print(f"| {name} | {R.ram_slots} | {rs[0]['tok_s']} | {rs[1]['tok_s']} | {rs[2]['tok_s']} | "
          f"{r['ms_fixed']} + max({r['ms_gpu_moe']}, {r['ms_cpu_moe']}) | {r['cpu_experts_per_tok']} | "
          f"{r['nvme_per_tok']} ({r['nvme_per_tok'] * R.nvme_expert_bytes / 1e9:.2f}) | {low} |", flush=True)

# Speculative decoding (lossless): window = k+1 verified tokens of one stream, accept = mean tokens per step.
# Acceptance numbers are the GB10 fork's GLM-5.3-Flash measurements (docs/models/glm-5.3-flash.md); draft cost and
# fixed(window) are assumptions (fixed() extrapolates the 1/2/4-sequence calibration, conservative for one sequence).
print()
print("| speculative what-if (C1, 3.05bpw everywhere, all levers) | window | accept | draft ms | VRAM slots | tok/s | C1 ms/step: fixed + max(gpu, cpu) | CPU experts/step |")
print("|---|---|---|---|---|---|---|---|")
for name, win, acc, dms, slots in [
        ("none (levers.py last row)", 1, 1.0, 0.0, 1910),
        ("MTP k=2 (GB10 acc 2.81-2.95)", 3, 2.85, 3.0, 1910),
        ("MTP k=3 (GB10 acc 3.62-4.0)", 4, 3.6, 4.0, 1910),
        ("DFlash2 k=7, code (GB10 acc 5.5), drafter 2.6 GB in VRAM", 8, 5.5, 4.0, 1635),
        ("DFlash2 k=7, prose (acc ~2.4, inferred from 12.4-13.5 vs 29.4 tok/s)", 8, 2.4, 4.0, 1635)]:
    R = spec.resolve(reg, "glm53-rtx3090-55g-nvx4", **{**LEVERS, "lane_overrides": CPU(0.104), "budget.vram_expert_slots": slots})
    r = sim.run(R, streams, conc=1, window=win, accept=acc, draft_ms=dms)
    print(f"| {name} | {win} | {acc} | {dms} | {slots} | {r['tok_s']} | {r['ms_fixed']} + max({r['ms_gpu_moe']}, {r['ms_cpu_moe']}) | "
          f"{r['cpu_experts_per_tok'] * acc:.1f} |", flush=True)
