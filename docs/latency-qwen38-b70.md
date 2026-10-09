# Qwen3.8-Flash-Next decode on one Arc Pro B70 (nvme16): where every millisecond goes

**Status: partial.** The box was shut down by the user during the pass (2026-10-09 ~14:30 CEST). Two configs were
measured. The kernel-level split inside the decode graph and the Edge0 routing-width arms were not run.

## Setup

- **Config A (`A-raid`): nvme16 exactly as recommended.**
  - Image `ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691…`, `QWEN_B70_MODE=nvme16`.
  - `--memory 16g --memory-swap 16g --shm-size 8g`, cpuset 40-47.
  - Serve args as in `~/skillsgym/ref_config.json`: decode XPU graphs bs 1/2/4, `--max-running-requests 4`,
    `--max-total-tokens 131072`, fp8 KV.
  - Store on the 4x 9100 PRO RAID0 (`/mnt/nvx/n104`, read-only).
  - Arc Pro B70 `0000:48:00.0` only.
- **Config B (`B-cap7`):** the same, plus `--device-read-bps /dev/md127:7000000000` (single-drive emulation).

**Instrumentation.** Observation only; the computation is unchanged.

- `sglang_plugin.py` is mounted over the image copy. The only change is an env-gated `activate()` wrapper
  (`tools/sglang_plugin.b70prof.diff`, 15 lines).
- That wrapper loads `tools/b70prof.py`, which adds:
  - host `perf_counter` stamps around every `ModelRunner.forward`, `ModelRunner.sample` and `NvTierAsync.step`;
  - three XPU events per step: before the graph replay, after the replay is enqueued, and after the tier step. They
    are resolved lazily with `query()`, so no syncs are added;
  - per-step deltas of the tier counters;
  - a diff of the VRAM-slot / RAM-flag snapshot that the tier step consumes. This gives:
    - admissions: RAM-tier experts read over xe SVM and written through into VRAM;
    - victims: VRAM experts evicted and written back to RAM.
- Samplers:
  - xe fdinfo engine cycles and per-thread CPU ticks, read inside the container;
  - `/proc/stat` cores 40-47, `/proc/diskstats` (md127, its 4 members, and the root 990 PRO), and GT idle
    residency / frequency, read on the host.
- `torch.profiler` (CPU + XPU through PTI) ran over 60 decode steps at C1 and at C4.

**Workload.**

- 2 short warm-ups.
- 3 single-stream chat requests at 8k context, each with a distinct prompt (no prefix-cache hits), running to natural
  EOS. No `max_tokens` was sent.
- One C4 window.
- Profiler requests were separate and are excluded from every number below.

**Contention.** Docker samples every 60 s showed no other GPU container during A or B: `"contended": []`. The 3090
agent was waiting on `~/nvx_bench.lock`. Guard: clean before, during and after every session (0 fault lines, engine
resets 1/1 baseline, corrected AER 0 in 10 min).

## 0. Standard table

| config | prefill size | prefill speed (tok/s) | decode speed (tok/s) | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|---|
| A-raid | 8k | 1,064 (867 / 1,259 / 1,066) | **27.4** (27.6 / 26.9 / 27.7) | 1 | 131k fp8 | 1 |
| A-raid | 8k | 1,036 (3 admitted) | **30.2** agg (12.5 / 12.7 / 12.7 / 25.1) | 4 | 131k fp8 | 1 |
| B-cap7 | 8k | 975 (828 / 1,036 / 1,061) | **26.6** (28.0 / 24.1 / 27.6) | 1 | 131k fp8 | 1 |
| B-cap7 | 8k | 1,028 (3 admitted) | **29.5** agg (12.1 / 12.4 / 12.3 / 26.3) | 4 | 131k fp8 | 1 |

Completions ran 231-498 tokens, all ending in `stop`. The README says 29.2 C1 and 52.6 C4. This session ran about 6%
slower at C1. The C4 gap is mostly the admission issue below.

**C4 is really C3 + C1.** SGLang logged `#running-req: 3, #queue-req: 1` for the whole C4 window. With no
`max_tokens`, every running request reserves `(context_len - input) x new_token_ratio` tokens of the 131,072-token KV
budget, so the fourth 8k request waits until the first three finish (its TTFT is 56-60 s). The steady 3-stream window
runs at 62.4 ms per step, which is **48.1 tok/s** aggregate. The reported C4 aggregate (30.2) is that window plus a
single-stream tail.

**The 7 GB/s cap never binds.** Prefill reads 4.0-5.1 GB/s from md127 and decode reads 0.4-1.7 GB/s, so B equals A
within session noise. Everything below is A-raid unless marked.

## 1. One token as a waterfall (A-raid, C1, mean step 36.3 ms)

These are step-level stamps, not per-layer stamps: the decode forward is one XPU graph replay, and PTI does not see
kernels inside SYCL graphs (see section 6).

```text
t (ms)   host (scheduler thread)                                     | GPU (compute queue)
 0.0     run_batch: input prep, graph replay enqueued (~0.8-1.8 ms)    | idle until replay starts
 ~1      NvTierAsync.step: D2H snapshot copies enqueued, then          | graph: 48 layers (attention / GDN / HC / PLE / n-gram,
         ce.query() on the eviction-clear event BLOCKS (zeEventQuery-   |   routed MoE incl. RAM-tier SVM reads and victim
         Status ~29 ms) until the graph finishes                        |   write-backs)  31.6 ms mean (p50 30.7, p90 36.2)
 ~32.6   tier step tail: numpy bookkeeping, pin_memory, flag uploads    | snapshot copies, then idle      1.47 ms
 ~34.1   sample (0.40 ms host), copy_result_to_cpu, process_batch_result,| argmax + small copies (~0.16 ms), otherwise idle
         recv/process requests, get_next_batch_to_run (~1.5 ms)          |                                3.30 ms
 36.3    next run_batch
```

From the profiled steps: the long `zeEventQueryStatus` sits directly under `scheduler.run_batch`, right after
the snapshot copies. It is `ce.query()` in the nvme16 admission step (`nvtier._async_step_admit`, the `_clears` loop).
On this Level Zero stack, querying an event recorded behind the in-flight graph waits for the graph to finish. So the
host cannot prepare step n+1 while step n runs. SGLang's overlap schedule is on (`disable_overlap_schedule: False`),
but the blocking tier step defeats it.

## 2. Wall time per token, split along the critical path

**Method.**

- Device times come from the XPU events: graph = e0→e1, tier tail = e1→e2, gap = e2(n-1)→e0(n).
- The graph is split by regressing per-step graph time on that step's admissions and victim write-backs:
  - snapshots lag one step, so they are aligned;
  - A-raid C1: 1,291 steps, R² 0.90 → graph = 27.42 + 0.162 x admits + 0.906 x victims ms;
  - B-cap7: R² 0.97 → 27.48 + 0.138 / 0.921.
  - The steady VRAM-full subset gives 0.117 ms per admit and 0.974 ms per victim. The N109 engine record's
    victim write-back is 0.9 ms.
- Where the regression intercept is split into MoE and non-MoE, the MoE part is **estimated** from the N124
  microbench, not measured in this run: 48 x 0.0746 ms cached M=1 all-hit = 3.6 ms.

| component (C1, ms per token) | A-raid mean | B-cap7 mean | share (A) | how measured |
|---|---|---|---|---|
| non-MoE GPU (attention / QSA, GDN, hyper-connections, PLE, norms, n-gram gather + its NVMe wait, lm_head) | **~23.8** | ~23.9 | 66% | regression intercept minus estimated MoE |
| MoE kernel on VRAM experts, including masked picks remapped to the safe expert | ~3.6 | ~3.6 | 10% | microbench estimate (not split in-run) |
| RAM-tier SVM reads + write-through (7.4 admits per token; B 9.4) | **1.19** | 1.30 | 3% | regression, 0.16 ms per admit |
| victim write-backs, VRAM → punched RAM pages (3.3 per token; B 4.6) | **2.97** | 4.26 | 8% | regression, 0.91-0.97 ms per victim |
| NVMe waits | **0** | 0 | 0% | by design: a non-resident pick is masked (weight 0) and read in the background |
| host tier-step tail (GPU idle; the host finishes `NvTierAsync.step` after the blocking query) | **1.47** | 1.49 | 4% | XPU events |
| inter-step gap (sampling 0.16 ms GPU / 0.40 ms host, result copy, scheduler, next launch) | **3.30** | 3.54 | 9% | XPU events |
| **wall** | **36.34** (p50 35.5, p90 41.3, p99 51.1) | **38.06** | | |

Quality cost of zero NVMe waits: **23.6 masked picks per token** (p50 12, p90 56, p99 204), 4.9% of the 480 routed
picks. B-cap7 masks 24.6. Each masked pick costs the MoE kernel a VRAM read of the safe expert, so the masks are not
free either.

**C3 window (3 streams, A-raid, 62.4 ms per step = 20.8 ms per token):**

| component | ms per step |
|---|---|
| graph intercept | 42.0 |
| admit + victim pairs, ~0.98 ms per pair (admits and victims are collinear once VRAM is full) | 14.6 (14.9 per step) |
| tier tail | 1.90 |
| inter-step gap | 3.87 |
| **wall** | **62.4** |

Masked picks run 47.6 per step (3.3% of 1,440).

## 3. Resources during active decode: busy vs idle (A-raid, C1 unless noted)

| resource | busy | of which on the critical path | idle |
|---|---|---|---|
| B70 compute (graph executing, XPU events) | 31.6 of 36.3 ms = **87%** (C3: 91%) | all of it. 4.2 ms per token of it is PCIe- or fault-bound tier traffic (SVM reads + victim write-backs) | **4.8 ms per token = 13%**: tier tail 1.5 + inter-step gap 3.3 |
| B70 engine residency (xe fdinfo `drm-cycles-ccs` / `-bcs`) | 99.5% ccs, 99.6% bcs | not informative: it counts context residency, not work | - |
| B70 clock / GT idle | 2,800 MHz flat; GT C6 residency 0 | - | - |
| CPU cores 40-47 (`/proc/stat`) | 16-20% mean (C3: 18.5%); the sglang scheduler process uses 1.0-1.5 cores | main thread spin-blocked ~29 ms per step in `zeEventQueryStatus`; real host work ≈ 4.8 ms per step, all on the critical path | ~80%; 6+ of 8 cores do nothing useful |
| PCIe H2D (derived: admits x 1.86 MB) | 0.38 GB/s (1.4% of 26.7) | yes: SVM reads stall the MoE kernel, 0.16 ms per expert vs 0.07 ms at the link floor | 98% |
| PCIe D2H (derived: victims x 1.86 MB) | 0.17 GB/s (2% of the 7.7 GB/s D2H ceiling) | yes: 0.9-1.0 ms per victim inside the graph, about 4x the copy-rate floor (page faults on punched memfd pages) | 98% |
| NVMe RAID md127 (diskstats) | decode 0.42-1.58 GB/s (util 7-14%); prefill 4.2-5.1 GB/s (16-20% of 26 GB/s) | decode: no (masked); prefill: background staging, GPU-bound | 80-95% |
| root NVMe 990 PRO (n-gram table, dm-crypt) | ~350 read IOPS, ~2 MB/s, util 50-77% | unknown: the per-step n-gram lookup (~0.8 ms host I/O per request) overlaps layer 0, and the GPU-side `nvme_wait` time is not measured | latency-bound |
| DDR (derived: NVMe DMA + fill memmove read and write + SVM + victims) | ~3.1 GB/s (~2% of ~138 practical) | no | 98% |

**No resource is saturated.**

- The GPU is the busy lane at 87%.
- About 66% of the token is non-MoE GPU time, at roughly 3.7x the bandwidth floor (6.4 ms, section 4).
- Every other lane is idle 80-98% of the time.

## 4. What if a component were free, or cut to its hardware floor (A-raid, C1)

The model is wall = graph intercept + 0.162 x admits + 0.906 x victims + tier tail + gap. It reproduces the measured
36.35 ms.

Floors:
- VRAM 608 GB/s;
- non-MoE weights + state of ~3.9 GB per token (GDN 1.32 + hyper-connection mixers 1.26 + attention 0.38 + lm_head
  0.40 + shared expert 0.15 + router 0.13 + PLE projections 0.07 + KV / GDN state ~0.2 GB) → 6.4 ms;
- routed experts 480 x 1.86 MB = 0.89 GB → 1.47 ms;
- PCIe H2D 26.7 GB/s;
- D2H 7.7 GB/s.

| scenario | kind | ms per token | tok/s | saves |
|---|---|---|---|---|
| measured (model) | - | 36.35 | 27.5 | - |
| tier step non-blocking (tail = 0) | scheduling | 34.88 | 28.7 | 1.47 |
| inter-step gap at ~0.3 ms (overlapped scheduler, captured sampling) | scheduling | 33.35 | 30.0 | 3.0 |
| **all scheduling waits = 0** (tail 0, gap 0.3) | scheduling | **31.88** | **31.4** | **4.47** |
| victim write-back off the critical path (victim ring / copy engine, or no write-back) | scheduling / tier | 33.38 | 30.0 | 2.97 |
| victim write-back at D2H copy rate, still inline | faster transfer | 34.17 | 29.3 | 2.18 |
| RAM-tier SVM reads at the PCIe floor (0.07 vs 0.16 ms per expert) | faster transfer | 35.67 | 28.0 | 0.68 |
| MoE kernel on VRAM experts at roofline (1.47 vs ~3.6 ms; estimate) | faster kernel | 34.24 | 29.2 | ~2.1 |
| **non-MoE GPU at VRAM roofline** (6.4 vs ~23.8 ms; estimate) | faster kernels | **18.9** | **52.9** | **~17.4** |
| all scheduling + victims off the path | scheduling | 28.9 | 34.6 | 7.4 |
| **every component at its floor** (today's bytes, masking kept) | ceiling | **~8.7** | **~115** | 27.6 |

**C1 ceiling on this card.**

- Scheduling and tier fixes alone, with no kernel work, reach about 35 tok/s.
- Beyond that the non-MoE GPU work dominates: about 24 ms per token at ~3.7x its bandwidth floor. Splitting it by
  kernel is the next measurement (section 6).
- Unlike the GLM 3090 path, the expert tiers are not the main cost here: VRAM serves 93.5% of picks, and the cold
  tiers move only ~0.55 GB/s.

## 5. Top 5 fixes, ranked by expected ms per token saved (C1)

| # | fix | expected saving | confidence |
|---|---|---|---|
| 1 | Non-MoE decode kernels (attention / QSA, GDN recurrent, the hyper-connection mixers that read 1.26 GB per token, PLE, n-gram gather): fuse, cut launches inside the graph, check the HC mixer path is bandwidth-efficient | up to ~17 ms (of ~23.8); realistically 5-10 | medium: the size is an inference (intercept minus microbench MoE), the split is unmeasured |
| 2 | Overlap host scheduling with the graph: sample, result copy and next-batch prep take 3.3 ms of GPU idle per step | ~3.0 ms | high (event-measured); needs fix 4 first |
| 3 | Take victim write-backs off the critical path: the N111 victim ring in `experimental/n111-victim-ring` (async D2H via a VRAM ring), or keep RAM-tier pages pre-faulted so the kernel's write-back does not fault on punched memfd pages | 2.2-3.0 ms at C1; ~7 ms per step at C3 | high (regression R² 0.90-0.97, 0.9-1.0 ms per victim, matches N109) |
| 4 | Make `NvTierAsync` admission step non-blocking: replace `ce.query()` on the clear events with a check against the already-synced snapshot event of the previous step (or defer punches one more step) | 1.47 ms direct, and it unlocks fix 2 | high: profiler shows one ~29 ms `zeEventQueryStatus` per step under `run_batch` |
| 5 | RAM-tier SVM read cost: 0.16 ms per admit vs 0.07 at the PCIe floor. Pre-map decode fills (`EXL3_NVTIER_DECODE_PF`, the N111 svm prefetch of landed pages) | ~0.7 ms | medium |

**Throughput, not ms per token: C4 admission.** Raise `--max-total-tokens` (KV is fp8 and 131k uses ~1.5 GiB) or
lower `--schedule-conservativeness`, so the fourth 8k request is admitted. That turns the measured 30.2 aggregate into
the ~48 tok/s steady 3-stream rate, or more with 4 streams.

## 6. What could not be measured, and why

- **Kernel-level split inside the decode graph** (attention vs GDN vs HC vs MoE vs n-gram wait).
  - PTI / `torch.profiler` records only kernels outside the SYCL graph (0.16 ms per step: argmax, copies, QSA
    metadata). The 48-layer graph replay is opaque.
  - An eager pass (`A2-eager`) used `--disable-cuda-graph`, but `--cuda-graph-backend-decode full` still captured
    decode graphs, so that session measured graph mode again.
  - The corrected pass (`--disable-decode-cuda-graph`, `A3-eager`) was queued, then cancelled for the shutdown.
  - The MoE-vs-non-MoE line in section 2 is therefore an estimate from the N124 microbench.
- **n-gram NVMe stall.** The GPU spin counter of the n-gram tier is in device spin iterations, not time. Only host I/O
  time is known: ~0.8 ms per lookup request, about 1 per step.
- **PCIe and DDR counters.** There is no PCM / AMD DF access without root, and no `perf` binary. Both are derived from
  bytes moved.
- **GPU busy from the kernel driver.** xe fdinfo cycles report context residency (99.5%), not activity. The xe PMU
  `engine-active-ticks` needs `perf_event_paranoid` ≤ 0 or CAP_PERFMON, and sudo is not available.
- **No unitrace / onetrace / VTune / xpu-smi / intel_gpu_top** on the host or in the image.
- **Edge0 routing-width arms** (native K = 10; K = 8 / 6 / 4; gate-mass tau 0.02 / 0.05 / 0.10). Not run: the
  session (`C-route`) was stopped in the lock queue before it started.
  - The harness is ready: `tools/b70prof.py` with `B70_ROUTE=1`, and `tools/route.py` + `tools/route_an.py`.
  - The transform is graph-safe. Dropped picks become (safe expert 0, weight 0), and kept weights are rescaled to the
    original sum. K and tau live in device tensors, so one server switches arms at a step boundary.
  - Note: Qwen3.8-Flash-Next routes **top-10**, not top-8, so the true control is K = 10.

## Data

- On omarchy: `~/b70prof/runs/{A-raid,A2-eager,B-cap7}`. Contents:
  - `steps.81.jsonl`: per-step rows;
  - `load.jsonl`: requests with per-chunk timestamps;
  - `sampler.jsonl`, `hostmon.jsonl`;
  - `trace{1,2}.81.json`: PTI traces, C1 and C4;
  - `server.log`, `guard.log`.
- Analysis JSON: `docs/latency-qwen38-b70/analysis_{A-raid,B-cap7}.json`.
- Tools (mounted, never baked into an image): `docs/latency-qwen38-b70/tools/`.
- Run records: `registry/runs/qwen38fn-b70-prof-raid.json`, `registry/runs/qwen38fn-b70-prof-cap7.json`.
