# Queue (top = next)

## G1 GLM S1: exact NVMe tier (GLM53_MODE=nvme)  [DONE: runs glm53-s1d/s1c; exact; 55g C1 7.26, prefill 515/762]
Panel + greedy equality, then the standard table under --memory 50g/55g. Record run glm53-s1-*.

## G2 GLM S2+S3: device-side stall (libaio worker + per-expert landed flags, no per-layer host sync), exclusive RAM via victim ring, prefill read-ahead 2-3 layers, short prompts via staging, CPU tier on RAM-resident experts overlapped per layer  [running: builder agent N119]
moetier lever study: 55 GB base 23.4 C1 -> 27 with persistent CPU workers. Move the runtime onto moetier plan_layer.

## G3 GLM route capture with router weights + layer inputs -> next-layer predictor recall  [running offline: agent N121 on S1 nv_trace.npz routes]
Unlocks layer-ahead prefetch (+2-4 tok/s C1 in the lever study). Doc 20 section 11 has the capture spec.

## G4 GLM VRAM slots: fp8 KV (exactness check), staging reuse in decode, +400 slots  [todo]

## G5 GLM non-MoE fixed time 14 -> 9 ms: fused decode kernels + CUDA graphs  [todo]

## G6 GLM MTP k=2 (verify windows share expert reads)  [prep running offline: agent N120 (exllamav3 -mtp path, arm scripts in runs/N120-glm53-mtp/); GPU run after G2 releases the 3090; modeled C1 32.4 -> ~46 tok/s]
First measure MTP acceptance on the 3.05bpw model + verify time at 3 tokens; MTP layer's 288 experts need a home (VRAM/RAM). Then rerun the spec rows in moetier.

## G7 vLLM-Moet transfer  [researched: docs/vllm-moet-lessons.md]
Verdict: CPU lane is trellis-instruction-bound (not DRAM), so 2-bit cold tiers barely help at EXL3 quality; reject RAM/NVMe 2-bit and delta-on-2bit. Keep: their router-lookahead predictor (recall 71.6% measured on GLM-5.2) for G3; optional 2-bit NVMe-tail copy behind a KLD gate (needs a fresh stock 2.0bpw quant; the local 2.0bpw is incomplete/TR3).

## Q1 Qwen B70: victim ring srv27 matrix, GDN SYCL test, MTP stage (a), decode kernels  [paused]

## R1 Registry upkeep  [every tick]
Every new measurement -> runs/<id>.json + index. Keep recipes' `serve` blocks pointing at the exact image/env.
