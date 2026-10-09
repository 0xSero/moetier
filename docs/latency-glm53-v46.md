# GLM-5.3-Flash on the RTX 3090, v4.6-nvme: where the time goes (cross-stack profile)

Status: **55 GB done (per-layer dump + probe + what-if + nsys kernel split); 16 GB and all-RAM fast mode pending** (this page is
extended as each lands).

What was profiled: image v4.6-nvme (`ghcr.io/sybil-solutions/glm53-flash-offload@sha256:aa74200f`, glm53-flash-offload
0fbc0c5). In `GLM53_MODE=nvme` it turns on N135's CPU-lane forward (`GLM53_NV_CPU_KERN=1`) and N136's decode switches
(`GLM53_LA`, `GLM53_NV_PUBFAST`, `GLM53_NV_PFSIDE`, `GLM53_LA_BTTRIM`). The profile runs that exact configuration on
the instrumented engine copy `hom272/glm53v46` (N136's glm53r = HOM-272 dump/NVTX hooks + the four switches, plus
N135's kernel patch); the hooks do not change the computation. Same-session uninstrumented speed of the image:
`registry/runs/glm53-n129-v46-full-55g` (8k C1 19.47, C4 19.75, 32k C1 18.30) vs v4.4 17.32 / 19.27.

Method and tools: the same as `docs/latency-glm53-c1.md` (per-layer device `%globaltimer` stamps, host stamps mapped
onto the device clock, `examples/latency_c1.py`; utilization from `moetier probe`). Three natural-EOS single-stream chat
requests (3,147 / 2,605 / 4,690 tokens; client steady 18.44 / 20.34 / 16.55 tok/s), 10,442 steady decode steps.
Run: `registry/runs/glm53-n129-p55-dump` (omarchy `hom272/runs/P55_dump`), 2026-10-09 13:49-14:02 CEST, nothing else
on the 3090 (the skills gym was paused; B70 48 profiled separately).

## 55 GB RAM + NVMe RAID0 (4x 9100 PRO)

### One token as a waterfall (median step, 54.8 ms)

Full 42-layer waterfall: `docs/latency-glm53-v46/waterfall_55g.txt`. Excerpt (ms from the first MoE layer's publish):

```text
li | GPU: pub->seen | book | copy (nvme) | MoE | wait CPU | non-MoE gap  ||  CPU lane [start, end]  | crit
 0 |   0.000 +0.006 | +0.016 | +0.527 (0.000) | +0.172 | +0.008 | +0.210 (bub 0.000) || [  0.045,   0.524] x3 | GPU
 1 |   0.940 +0.024 | +0.019 | +0.701 (0.000) | +0.169 | +0.970 | +0.212 (bub 0.000) || [  1.002,   2.856] x5 | CPU
 7 |  10.189 +0.023 | +0.022 | +1.744 (1.695) | +0.473 | +0.007 | +0.321 (bub 0.000) || [ 10.250,  10.794] x4 | NVMe
 9 |  14.440 +0.014 | +0.017 | +0.720 (0.019) | +0.755 | +0.659 | +0.214 (bub 0.000) || [ 14.494,  16.641] x5 | CPU
14 |  24.648 +0.032 | +0.017 | +0.002 (0.000) | +0.225 | +0.361 | +0.217 (bub 0.000) || [ 24.718,  25.316] x3 | CPU
20 |  29.069 +0.010 | +0.014 | +0.473 (0.000) | +0.172 | +0.007 | +0.212 (bub 0.000) || [ 29.118,  29.601] x3 | GPU
```

Launch bubbles are gone (lookahead): the next step is enqueued before the host reads the token. The step boundary
(final norm, lm_head, sampling, embedding, dense layers 0-2, layer-3 attention) is now 1.9 ms of GPU work with no host
gap (it was 0.7 ms of work + 2.1 ms of bubble).

### Critical-path table (ms per token, 10,442 steady steps)

| component | v4.0 (L1_pfon, 2026-10-08) | **v4.6** | p50 | p90 | share |
|---|---|---|---|---|---|
| admission copy RAM → VRAM (SM gather, excl. NVMe waits) | 17.8 | **17.3** | 17.3 | 22.7 | 31% |
| fixed non-MoE GPU (attention, norms, shared expert, router; incl. step boundary) | 11.2 + 0.7 | **11.4** | 11.2 | 11.7 | 20% |
| · before KDA layers (31/token) | 0.237/layer | 0.213/layer | | | |
| · before DSA layers (10/token) | 0.310/layer | 0.289/layer | | | |
| · step boundary | 0.7 (+2.1 bubble) | 1.9 (bubble 0) | | | |
| routed MoE kernel (VRAM experts + zero-copy NVMe picks) | 9.0 | **10.1** | 9.8 | 12.3 | 18% |
| CPU-lane overrun (GPU waits for the CPU partial) | 9.4 | **9.8** | 9.2 | 14.3 | 18% |
| NVMe wait (copy kernel + CPU job waiting for landings) | 10.1 | **5.8** | 5.2 | 11.0 | 10% |
| host plan + reply | 1.9 | **0.7** | 0.7 | 1.0 | 1% |
| device CLOCK bookkeeping | 0.6 | **0.65** | 0.65 | 0.73 | 1% |
| launch bubbles | 2.2 | **0.03** | 0 | 0 | 0% |
| **wall** | **62.3** | **55.8** | 54.8 | 68.9 | (17.9 tok/s) |

Lane that ended each layer call: **CPU lane 46 %, GPU lane 43 %, NVMe landing 11 %** (v4.0: 54 / 33 / 13).
CPU lane per layer: 3.8 experts, 0.56 ms p50 / 1.65 ms p90 job; it starts 0.06 ms after the reply.

### Idle / utilization during C1 decode (moetier probe, means over the 3 requests)

| resource | used | ceiling | idle / headroom | on the critical path? |
|---|---|---|---|---|
| GPU SMs | real work 38.8 ms/token (copy 17.3 + MoE 10.1 + non-MoE 11.4) = **70 %** of wall; NVML shows 100 % (spin-waits count) | 100 % | **30 % spinning** (waiting for the CPU partial 9.8, NVMe 5.8, plan/book 1.4 ms/token) | all real work is serial on the layer chain |
| CPU lane (22 cores) | job time 60 % of wall (33.9 ms/token busy); /proc 99 % (workers spin) | 100 % | **40 %** | 9.8 ms/token of overrun |
| PCIe H2D | 7.8 GB/s | 28 GB/s | **72 %** | admission copies block the MoE kernel |
| PCIe D2H (victim write-back) | 7.4 GB/s | 28 GB/s | 74 % | no (copy engine) |
| NVMe RAID0 | 9.9 GB/s | 26 GB/s | **62 %** | 5.8 ms/token of waits |
| DDR (derived: CPU-lane reads + PCIe + NVMe DMA) | 52 GB/s | 138 GB/s | 62 % | carries the CPU lane |
| VRAM controller busy (time) | 24 % | 100 % | 76 % | - |
| host threads (main + controller) | spin; other host CPUs 3 % | - | - | plan 0.7 ms/token |
| GPU power | 279 W | 350 W | 20 % | - |

Nothing is saturated: the token is a chain of 42 dependent layer stages and each stage waits on its slowest lane.

### What-if per layer call (v4.6; model reproduces the measured 55.8 within 2 %)

| scenario | ms/token | tok/s | saves |
|---|---|---|---|
| measured (model) | 56.8 | 17.6 | - |
| host plan + bookkeeping = 0 | 55.4 | 18.0 | 1.4 |
| CPU job starts with the GPU lane | 55.9 | 17.9 | 0.9 |
| NVMe waits = 0 | 53.1 | 18.8 | 3.7 |
| admission copy at 25 GB/s (copy engine) | 53.3 | 18.8 | 3.6 |
| **CPU experts at 98 GB/s** (native-layout DDR floor) | **47.9** | **20.9** | **8.9** |
| routed MoE kernel at VRAM roofline | 52.4 | 19.1 | 4.4 |
| non-MoE GPU at VRAM roofline | 51.8 | 19.3 | 5.0 |
| all scheduling items together | 50.5 | 19.8 | 6.3 |
| every component at its floor (per-layer serialization kept) | 24.1 | 41.4 | 32.7 |

Data: `docs/latency-glm53-v46/analysis_55g.json`, `whatif_55g.json`, `waterfall_55g.txt`.

### Kernel split from Nsight Systems (55 GB, 95 steady tokens, NVTX on)

Capture: `GLM53_PROF=150:100` (n136_prof opens a cudaProfilerApi window at decode iteration 150), `nsys profile
--cuda-graph-trace=node`, NVTX module ranges installed after K101 fusion (arm `P55_nsys`, 14:08-14:09). With NVTX the
step is 58.0 ms (3 % slower than the dump arm). GPU idle (no kernel resident) is only **0.58 ms/token**: the device is
never starved by the host now; every wait shows up as a spinning nv2 kernel.

| class | ms/token | roofline ms | achieved | note |
|---|---|---|---|---|
| nv_copy (admission gather + spin on NVMe landings) | 24.5 | - | - | 17.3 copy + ~5.8 NVMe wait + spin (dump split) |
| nv_combine (spin until the CPU partial is published) | 10.6 | - | - | = CPU-lane overrun + flag latency |
| routed + shared MoE kernels (exl3_moe_coop a + b) | 10.3 | - | - | |
| **KDA attention** (31 layers) | **5.83** | 3.31 | 531 GB/s (57 % of 936) | int8 GEMV projections 4.0 ms, gated-delta recurrence 0.37, conv/gates/norm 0.3 |
| **DSA attention** (11 layers) | **2.40** | 1.21 | 471 GB/s (50 %) | int8 GEMV 1.24, MLA unfold/absorb/decode-split/combine 0.89 |
| router (61 + 22 calls) | 0.91 | 0.11 | 112 GB/s | 7.3 us routing GEMV + 3.7 us top-k per layer: launch-latency bound |
| hc / norm sites (K101 fused) | 0.73 | 0.15 | 193 GB/s | 4 us kernels, latency bound |
| head (lm_head GEMM + norm) | 0.61 | 0.51 | 775 GB/s | at roofline |
| dense MLP layers 0-2 | 0.37 | 0.24 | 612 GB/s | |
| nv_step / nv_pub (host plan handshake on the device) | 1.24 / 0.33 | - | - | PUBFAST: nv_pub 8 us |
| sampler → next kernel | p50 9.4 us | - | - | lookahead |

Non-MoE GPU total ≈ 10.9 ms/token against a 5.5 ms roofline: the attention GEMVs run at 50-57 % of VRAM bandwidth
(int8 activations, small N), and ~1.6 ms/token is launch-latency-bound tiny kernels (router, hc/norm sites).
Data: `docs/latency-glm53-v46/nsys_55g.json` (N136's `nsys_an.py`).

### Top 5 fixes for 55 GB (expected ms/token saved)

| # | sink | fix | expected saving |
|---|---|---|---|
| 1 | CPU-lane overrun 9.8 ms; 0.196 ms/expert in-engine vs 0.096 floor | block-contiguous RAM slots (the fast mode's CPU tier reads its swizzled copy at 0.125 ms/expert in-engine): swizzle on NVMe landing / D2H write-back, unswizzle in the admission copy kernel; plus E2 (fewer routed picks) | 4-6 (what-if ceiling 8.9) |
| 2 | admission copy 17.3 ms (SM gather ~14 GB/s, 0.8 admissions/layer) | copy-engine admission (25 GB/s) + fewer admissions (admit-on-second-touch, larger VRAM cache from a smaller non-expert footprint) | 3-6 |
| 3 | non-MoE GPU 11.4 ms (≈1.8x the 6.4 ms roofline; N136 nsys: KDA 5.84 vs 3.31, DSA 2.69 vs 1.21) | fused / graph-captured KDA and DSA decode paths | up to 5 |
| 4 | routed MoE kernel 10.1 ms (0.17 ms/layer, 0.47 ms on layers with a zero-copy NVMe pick) | route NVMe-only picks to the CPU lane after landing instead of zero-copy in the GPU kernel; MoE kernel at roofline | 2-4 |
| 5 | NVMe waits 5.8 ms (p90 11.0) | deeper / earlier prefetch (N134 prerouter: 2-4 layers ahead) | ~3 |

(16 GB and all-RAM sections follow.)
