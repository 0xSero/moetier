| lever (cumulative) | C1 | C2 | C4 | per-token ms at C1 (fixed + max(gpu, cpu)) | NVMe/token |
|---|---|---|---|---|---|
| all experts in RAM (G067, 218 GB) | 27.49 | 32.92 | 36.31 | 14.0 + max(16.66, 21.71) | 0.0 |
| 55 GB RAM + NVMe tail | 23.37 | 30.06 | 33.42 | 14.0 + max(14.63, 28.33) | 20.2 |
| + layer-ahead NVMe prefetch (recall 0.7) | 25.55 | 32.69 | 38.25 | 14.0 + max(16.58, 24.58) | 20.6 |
| + persistent CPU workers (handoff 0.145 -> 0.03 ms/layer) | 26.99 | 33.75 | 39.08 | 14.0 + max(13.75, 22.49) | 20.7 |
| + fused non-MoE kernels + graphs (fixed 14 -> 9 ms) | 30.99 | 37.61 | 42.29 | 9.0 + max(13.19, 22.74) | 20.5 |
| + 400 more VRAM slots (fp8 KV, staging reuse) | 32.44 | 39.44 | 45.25 | 9.0 + max(12.81, 21.12) | 16.9 |
