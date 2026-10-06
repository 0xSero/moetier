# Queue (top = next)

## G1 GLM S1: exact NVMe tier (GLM53_MODE=nvme)  [running: builder agent N116 owns the 3090]
Panel + greedy equality, then the standard table under --memory 50g/55g. Record run glm53-s1-*.

## G2 GLM S3: CPU tier on the exclusive RAM tier, overlapped inside each layer with GPU + copy engine + NVMe  [todo]
moetier lever study: 55 GB base 23.4 C1 -> 27 with persistent CPU workers. Move the runtime onto moetier plan_layer.

## G3 GLM route capture with router weights + layer inputs -> next-layer predictor recall  [todo]
Unlocks layer-ahead prefetch (+2-4 tok/s C1 in the lever study). Doc 20 section 11 has the capture spec.

## G4 GLM VRAM slots: fp8 KV (exactness check), staging reuse in decode, +400 slots  [todo]

## G5 GLM non-MoE fixed time 14 -> 9 ms: fused decode kernels + CUDA graphs  [todo]

## G6 GLM gated MTP k=1-2 (experts shared inside a verify window)  [todo; model it in moetier sim first]

## G7 vLLM-Moet transfer: 2-bit cold tiers + precision delta for hot experts  [research running: agent N118]

## Q1 Qwen B70: victim ring srv27 matrix, GDN SYCL test, MTP stage (a), decode kernels  [paused]

## R1 Registry upkeep  [every tick]
Every new measurement -> runs/<id>.json + index. Keep recipes' `serve` blocks pointing at the exact image/env.
