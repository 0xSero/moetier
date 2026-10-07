# GLM-5.3-Flash C1 decode on the RTX 3090 tier: where every millisecond goes

Configuration: S3b on the 55 GiB cap. That is the 4x NVMe RAID0 store, the nv2 CPU lane on 22 cores, `-ambs 4`,
image `bb633b0b`, and engine code `hom272/glm53n`. glm53n is the S3b code plus an NVTX hook and a raw per-layer dump;
neither changes the computation.

Two arms ran in one session (2026-10-08 00:43-01:29 CEST, slot B down, nothing else on the 3090). Each served three
single-stream chat requests that ran to natural EOS (no output cap): 2,400-4,200 tokens each, 9,288 and 9,418 steady
decode steps.

| arm | layer-ahead NVMe prefetch | client steady tok/s (3 requests) | wall per step: mean / p50 / p90 / p99 ms | NVMe misses / token |
|---|---|---|---|---|
| **L1_pfon** | on (`GLM53_NV_PREFETCH=1`) | **16.89 / 17.67 / 14.73** | **62.3 / 55.5 / 83.8 / 172.9** | 21.9 |
| L1_pfoff | off | 14.66 / 15.20 / 13.21 | 70.4 / 63.5 / 91.8 / 189.3 | 40.2 |

**In the same session, prefetch on is 13% faster than off.** The afternoon "prefetch off +35%" (`glm53-u1-pfoff`)
was session drift. Prefetch on is the current best and is used below. The 58 GB table
(`registry/runs/glm53-58g-pfoff.json`) was measured with prefetch off, so it understates what the 58 GB cap can do.

Source data:
- per-layer-call dumps: `omarchy:~/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272/runs/L1_{pfon,pfoff}/layer_dump.npz`
- analysis JSON: `docs/latency-glm53-c1/`
- tools: `examples/latency_c1.py` and `examples/c1_ceiling.py`

**Status: the Nsight Systems pass is still pending.** Two attempts failed for harness reasons: the entrypoint was not
executable, then an `nsys launch` flag was rejected. A third was stopped by the 01:36 B70 c3 bus drop (guard trip).
Everything below comes from the engine's own device timestamps (`%globaltimer`, per layer call) and host stamps mapped
onto the device clock.

What the dump cannot see yet:
- the kernel-level split inside the non-MoE gap (attention vs norms vs shared expert vs router);
- copy-engine activity.

The nsys trace will add both.

## 1. One token as a waterfall (L1_pfon, median-length step, 67.0 ms)

Times are ms from the first MoE layer's publish. Columns per MoE layer (li 0-41 = model layers 3-44):

- **GPU stream:** publish → reply seen (host plan) | CLOCK bookkeeping | admission copy, with its NVMe-landing wait
  in brackets | routed MoE kernel | wait for the CPU partial | non-MoE gap to the next layer.
- **CPU lane:** [start, end] of the job and its expert count.
- **crit:** which lane ended the layer.

```text
li | GPU: pub->seen | book | copy (nvme) | MoE | wait CPU | non-MoE gap  ||  CPU lane [start, end]  | crit
 0 |   0.000 +0.045 | +0.015 | +2.964 (2.916) | +0.473 | +0.000 | +0.233 || [ 0.100,  3.434] x5 | NVMe
 1 |   3.731 +0.025 | +0.019 | +1.101 (0.063) | +0.170 | +0.000 | +0.237 || [ 3.811,  5.096] x5 | CPU
 3 |   7.075 +0.043 | +0.020 | +1.141 (0.000) | +0.170 | +0.008 | +0.345 || [ 7.173,  8.099] x5 | GPU
 8 |  17.130 +0.029 | +0.022 | +1.070 (0.000) | +0.168 | +0.460 | +0.237 || [17.214, 18.928] x6 | CPU
12 |  25.060 +0.044 | +0.016 | +2.623 (2.575) | +0.474 | +0.009 | +0.236 || [25.159, 25.911] x4 | NVMe
15 |  30.703 +0.010 | +0.010 | +0.002 (0.000) | +0.172 | +0.419 | +0.348 || [30.768, 31.368] x3 | CPU
17 |  32.498 +0.025 | +0.010 | +4.580 (4.578) | +0.171 | +0.500 | +0.239 || [32.578, 37.838] x3 | NVMe
28 |  50.648 +0.011 | +0.010 | +2.036 (2.034) | +0.171 | +0.506 | +0.233 || [50.715, 53.429] x3 | NVMe
35 |  59.523 +0.005 | +0.010 | +0.002 (0.000) | +0.170 | +0.375 | +0.341 || [59.582, 60.137] x3 | CPU
41 |  66.584 +0.006 | +0.010 | +0.002 (0.000) | +0.212 | +0.175 | +2.875 (bubble 2.151) || [66.646, 67.039] x2 | CPU
```

The full 42-layer waterfall is in `docs/latency-glm53-c1/waterfall_pfon.txt`. Across all 9,288 steps, the lane that
ended the layer was:

| lane ending the layer | share of layer calls |
|---|---|
| CPU lane (GPU waits on the combine) | 54% |
| GPU lane | 33% |
| NVMe (copy or CPU job waiting for a landing) | 13% |

What one layer looks like:

- **Typical layer.** Plan (5-45 µs) and bookkeeping (10-20 µs) come first. Then two lanes run in parallel:
  - GPU lane: about 1 ms of admission copy (one 9.4 MB expert over PCIe at ~14.5 GB/s, SM gather), then a
    0.17 ms MoE kernel. The kernel takes 0.47 ms when it also reads an un-admitted NVMe expert over PCIe.
  - CPU lane: 3-6 experts in 0.6-1.3 ms.

  The layer ends with the non-MoE gap: 0.21-0.24 ms before a KDA layer, 0.29-0.35 ms before a DSA layer.
- **NVMe layers.** In about 13% of layers a picked expert is on NVMe only, and the copy kernel or the CPU job waits
  1.5-4.6 ms for it to land. These layers make up the heavy tail: p90 84 ms, p99 173 ms per token.
- **Step boundary.** After layer 41: final norm, lm_head, sampling, the scheduler, embedding, dense layers 0-2 and
  layer 3's attention. That takes 2.8 ms p50, of which about 2.1 ms is the Python thread not having launched the
  next step yet (a launch bubble).

## 2. Wall time per token, split along the critical path

L1_pfon, ms per token over 9,288 steady steps. "fixed GPU" is non-MoE GPU work (the gaps), split by the attention
type of the next model layer.

| component | mean | p50 | p90 | share of mean |
|---|---|---|---|---|
| admission copy, RAM → VRAM (SM gather, excluding NVMe waits) | 17.8 | 17.8 | 23.3 | 28% |
| fixed non-MoE GPU, total | 11.2 | 11.1 | 11.5 | 18% |
| · before KDA linear-attention layers (31 per token) | 7.3 | p50 0.237, p90 0.241 ms per layer | | |
| · before DSA full-attention layers (10 per token) | 3.1 | p50 0.310, p90 0.345 ms per layer | | |
| · step boundary (lm_head, sampling, dense 0-2, embed, layer-3 attention; bubble excluded) | 0.7 | | | |
| NVMe wait (copy kernel + CPU job waiting for landings) | 10.1 | 6.1 | 18.8 | 16% |
| CPU-lane overrun (GPU waits for the CPU partial) | 9.4 | 7.1 | 11.4 | 15% |
| routed MoE kernel on the GPU (VRAM experts + zero-copy NVMe picks) | 9.0 | 8.8 | 10.5 | 14% |
| host plan + reply (device waits for plan_layer) | 1.9 | 0.7 | 1.0 | 3% |
| launch bubbles (upper bound; nearly all at the step boundary) | 2.2 | 2.1 | 2.3 | 3% |
| device CLOCK bookkeeping | 0.6 | 0.6 | 0.7 | 1% |
| **wall** | **62.3** | **55.5** | **83.8** | std 24.5 |

Layer-call wall time by type, p50 / p90:

| layer type | p50 ms | p90 ms |
|---|---|---|
| KDA | 1.145 | 1.834 |
| DSA | 1.209 | 1.825 |
| last layer + step boundary | 3.71 | 4.37 |

Most of the variance between tokens comes from NVMe misses. In the prefetch-off run, a linear fit of token time costs
each miss about 0.39 ms on the critical path, and misses run 21.9-40.2 per token.

## 3. Resources during active decode: busy, and whether that busy time is on the critical path

From `moetier probe`, means over the three requests of L1_pfon, plus the dump:

| resource | busy | of which on the critical path | idle / spinning |
|---|---|---|---|
| GPU SMs | NVML says 98%, but that includes spin-waiting kernels | 40.2 ms of real work per token (copy 17.8 + MoE 9.0 + non-MoE 13.4) = 65% of wall, all of it on the critical path | 21.5 ms per token (35%) spinning: plan wait, NVMe landed waits, CPU-partial wait |
| CPU lane, 22 cores | 41.2 ms of job time per token (65% duty); /proc/stat shows 98% because workers spin | 15.5 ms per token on the critical path (overrun incl. its NVMe waits) | 25.6 ms hidden under the GPU lane; idle 35% of wall |
| PCIe H2D | 7.1 GB/s (25% of the 28 GB/s link) | admission copies are critical (they block the MoE kernel) | 75% |
| PCIe D2H (victim write-backs) | 7.2 GB/s (26%) | off the critical path (copy engine, vring) | 74% |
| NVMe RAID0 | 8.9 GB/s (34% of 26 GB/s) | 10.1 ms per token of NVMe wait is critical; the rest (prefetch) is hidden | 66% |
| DDR (derived) | 48 GB/s (35% of 138 GB/s practical) | carries the CPU lane, copies and NVMe DMA | 65% |
| Host main thread (cpu 24), controller (cpu 25) | 100% (spin) | plan 1.9 ms per token; launch bubbles 2.2 ms per token | - |

**No resource is saturated; every one is idle 35-75% of the time.** The token is a chain of dependent per-layer stages,
and each stage waits on one lane while the others idle.

## 4. What if a component were free, or cut to its hardware floor

Each scenario is evaluated per layer call: `plan + book + max(GPU lane, CPU lane end) + residual + gap`. The model
reproduces the measured run within 2% (63.7 vs 62.3 ms). Floors are:

- PCIe 25 GB/s (copy engine);
- CPU lane 98 GB/s, the native-layout cap;
- VRAM 936 GB/s;
- non-MoE weights of 6.0 GB per token at VRAM roofline, which is 6.4 ms.

| scenario (L1_pfon, prefetch on) | kind | ms/token | tok/s | saves |
|---|---|---|---|---|
| measured (model) | - | 63.7 | 15.7 | - |
| host plan + bookkeeping = 0 | scheduling | 61.1 | 16.4 | 2.6 |
| launch bubbles = 0 | scheduling | 61.5 | 16.3 | 2.2 |
| CPU job starts with the GPU lane (no reply wait) | scheduling | 62.5 | 16.0 | 1.2 |
| **NVMe waits = 0** (misses landed before use) | scheduling / prefetch | **55.3** | 18.1 | **8.3** |
| admission copy at 25 GB/s (copy engine) | faster transfer | 62.0 | 16.1 | 1.7 |
| **CPU experts at 98 GB/s** (native-layout DDR floor) | faster kernel | **53.7** | 18.6 | **10.0** |
| routed MoE kernel at VRAM roofline | faster kernel | 60.7 | 16.5 | 3.0 |
| **non-MoE GPU at VRAM roofline** (fused kernels, graphs) | faster kernels | **57.8** | 17.3 | **5.9** |
| **all scheduling items together** | scheduling | **49.1** | **20.4** | 14.6 |
| **every component at its floor, per-layer serialization kept** | ceiling | **23.5** | **42.6** | 40.2 |

The same table for prefetch off: all scheduling → 52.4 ms (19.1 tok/s); all floors → 23.2 ms (43.1 tok/s).

Bandwidth bounds per token, assuming perfect overlap across layers, which the layer chain does not allow:

| resource | bound ms/token | tok/s bound |
|---|---|---|
| DDR: CPU-lane reads + H2D reads + NVMe DMA writes + write-backs, at 138 GB/s | 19.3 | 52 |
| PCIe H2D | 15.6 | 64 |
| CPU lane at 98 GB/s | 15.5 | 65 |
| NVMe | 8.6 | 116 |
| GPU: non-MoE + VRAM experts at roofline | 7.8 | 129 |

**The C1 ceiling on this box.**

- **Scheduling alone** (overlap, no NVMe stalls, no host waits): about 20 tok/s.
- **Hardware floors on every lane,** with today's bytes per token and today's per-layer serialization: about 43 tok/s.
- **Above that, DDR binds.** Today each token moves about 2.7 GB through DRAM: 161 CPU-lane experts, 35 admissions,
  22 NVMe records and their write-backs. That puts the floor at ~19 ms (52 tok/s), even with perfect overlap.

**So 50 tok/s at C1 is not reachable by scheduling, and not by faster kernels with the same bytes.** It needs fewer
slow-tier bytes per token: a higher VRAM hit rate (more VRAM slots, a smaller non-expert footprint) or fewer bytes per
cold expert (a lower-bit cold tier behind a KLD gate). MTP does not reduce bytes per token at C1: verify tokens share
few experts. What MTP amortizes is the per-layer fixed and latency costs, roughly the 14.6 ms of scheduling plus
6 ms of non-MoE work.

## 5. Top 5 ms sinks, with achievable savings

| # | sink (ms/token, L1_pfon) | what removes it | achievable saving |
|---|---|---|---|
| 1 | CPU-lane time: 9.4 critical + 25.6 hidden; 0.22 ms per expert vs a 0.096 ms DDR floor | faster AVX2 kernel, better small-job parallelism (3-6 experts over 22 threads, 4 barriers per job), placement | up to 10 ms (to 98 GB/s); realistically 4-6 ms |
| 2 | NVMe waits, 10.1 (p90 18.8) | better prefetch recall and earlier issue (two layers ahead); NVMe-tier experts routed to the GPU lane when the CPU lane is long | up to 8.3 ms |
| 3 | admission copy, 17.8 (SM gather at ~14.5 GB/s) | copy-engine admission is worth only 1.7 ms alone, because the CPU lane hides most of the GPU lane; it pays off once sink 1 shrinks | 1.7 now, ~5 combined with sink 1 |
| 4 | non-MoE GPU, 11.2 + 0.7 boundary (≈2× the 6.4 ms roofline) | fused decode kernels, CUDA graphs for the KDA/DSA blocks | up to 5.9 ms |
| 5 | host overheads: plan 1.9 + bubbles 2.2 + bookkeeping 0.6 + CPU start 0.06/layer | earlier CPU dispatch; graph-captured step boundary; faster plan (C++, already ~20 µs; the rest is notice latency) | about 4-5 ms |

## 6. Cross-check against the moetier simulator

`moetier sim` with S3b's budgets (1312 VRAM slots, 4831 RAM slots) on the G002 trace predicts 22.4 tok/s with prefetch
off and 23.9 with prefetch at recall 0.5. Measured is 14.2 and 16.0. The model is 50-60% too optimistic, for these
reasons:

| what the simulator assumes | what was measured |
|---|---|
| VRAM hit 0.46-0.49. The VRAM tier is seeded from the trace's own frequencies, an oracle. | 0.40 (134.7 of ~336 picks per token): CLOCK, seeded from stats_own_dec |
| NVMe misses 25 per token, with or without prefetch. Prefetch only pre-reads in the model; it does not change misses. | 40 per token without prefetch, 22 with |
| CPU lane 0.145 + 0.165 ms per expert | 0.22 ms per expert, plus 0.14 ms per layer waiting on NVMe, plus a 0.06 ms start latency |
| An NVMe read costs 0.4-0.5 ms of channel time and is FIFO-ordered | Mean read latency 1.2-1.4 ms (2.3-3.5 ms in the afternoon session). The copy kernel waits 1.5-4.6 ms in NVMe layers. |
| Admission is a 0.38 ms GPU-lane cost per expert | About 0.5 ms per admitted expert (SM gather at ~14.5 GB/s under DRAM load) |
| No host plan latency, CLOCK bookkeeping, launch bubbles or step-boundary scheduler cost | About 5 ms per token |
| Per layer it takes max(sum of means) | E[max] over variable lanes is larger: the NVMe tail and CPU jitter add several ms per token |

`fixed_ms = 14` (calibrated on G067) matches: non-MoE gaps 11.9 + bubbles 2.2 = 14.1 ms.

Fixes for `sim.py`:
- a VRAM seed from the engine's warm scores instead of trace frequencies;
- prefetch that reduces misses at a measured recall;
- an NVMe latency distribution instead of a FIFO channel;
- per-expert CPU cost of 0.22 ms;
- an admission cost of about 0.5 ms;
- a host overhead term;
- per-layer jitter.

## Session effects

Same configs measured on the same day varied by up to about 17%. Afternoon (15:46) was fast; 22:00-02:00 was slow.
Every component moved together and GPU clocks were identical. The probe now samples CPU frequency: CPU-lane cores ran
at about 3,660 MHz in the evening. Only same-session A/B comparisons are trustworthy here.
