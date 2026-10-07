# GLM-5.3-Flash S3b on the RTX 3090: utilization and chokepoints

Configuration: S3b. That is one RTX 3090, a 55 GiB container cap, the 4x NVMe RAID0 tier, nv2 with the AVX2 CPU lane,
`-ambs 4`, and image `bb633b0b` with the s3b2_55g code. The run adds two measurements on top: HOM-272 instrumentation
and `moetier probe` sampling at 100 ms. Method: `docs/utilization.md`. Run record: `registry/runs/glm53-s3b-util.json`
(arm `omarchy:~/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272/runs/u0_s3b`). Slot B was idle throughout.

## Standard table (natural completions)

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 548 tok/s | 13.04 tok/s | C1 | 131k (~1.5 GiB est.) | 1 |
| 8k | | 14.21 (7.62/stream) | C2 | | 1 |
| 8k | | 15.56 (4.38/stream) | C4 | | 1 |
| 32k | 764 tok/s | 11.88 tok/s | C1 | | 1 |

Instrumentation cost was checked against an uninstrumented arm run the same day (`u0_base`, s3b2 code, slot B idle).
That arm measured C1 14.09, C4 15.92 and 8k prefill 570. A second instrumented arm (`u1_ctrl`) measured C1 13.18,
C4 16.23 and prefill 580. Run-to-run spread is about ±5%, so the overhead is within the noise: C1 is about 6% lower,
C4 is not. Today's box is also slower than the 04:00 S3b record (14.85 / 16.27 / 583). The ship agent's
uninstrumented A/B at 12:26 got C1 12.16 with slot B busy. Compare arms only when they ran under the same conditions.

## Where the time goes per decoded token

Decode critical path, in ms per token, with each bucket's share of wall time:

| bucket | C1 | C2 | C4 | C1 @ 32k |
|---|---|---|---|---|
| fixed non-MoE GPU (attention, shared expert, lm_head, sampling) | 14.0 (18%) | 8.3 (12%) | 5.0 (8%) | 14.4 (17%) |
| GPU MoE (VRAM experts + zero-copy picks) | 11.4 (15%) | 11.9 (17%) | 10.5 (16%) | 11.7 (14%) |
| CPU MoE overrun (GPU waits for the CPU partial) | 11.1 (14%) | 3.1 (4%) | 2.3 (4%) | 11.2 (13%) |
| NVMe stall (landed waits in the copy + inside the CPU job) | 14.3 (19%) | 20.3 (29%) | 22.7 (35%) | 20.0 (24%) |
| admission copy RAM -> VRAM (SM gather, ~14.5 GB/s) | 20.2 (26%) | 19.3 (28%) | 17.8 (28%) | 21.4 (25%) |
| overhead (host plan wait, device bookkeeping, launch bubbles) | 6.2 (8%) | 7.4 (10%) | 6.3 (10%) | 6.0 (7%) |
| **wall** | **77.3** | **70.2** | **64.7** | **84.7** |
| lane finishing last (gpu / cpu / nvme) | 0.00 / 0.86 / 0.14 | 0.00 / 0.82 / 0.18 | 0.02 / 0.79 / 0.19 | 0.00 / 0.83 / 0.16 |

Per MoE layer at C1, the GPU lane (copy + NVMe wait + MoE kernel) takes 0.93 ms. The CPU lane takes 1.02 ms of job
time. On average it starts 0.07-0.15 ms after the device sees the reply, and 0.16 ms of the job is spent waiting for
NVMe landings. **The CPU lane finishes last in 86% of layer calls.** The GPU then waits 0.43 ms per layer for the
CPU partial.

An earlier diagnostic suggested the device saw the CPU's done flag ~0.28 ms late. It was wrong: it measured from CPU
end, not from max(GPU arrival, CPU end). The corrected visibility lag, measured in `u3_vis`, is -0.004 ms, i.e. none.

## What each resource does during active decode and prefill

| resource (ceiling) | decode C1 | decode C4 | prefill 8k | prefill 32k |
|---|---|---|---|---|
| GPU kernel-active (100%, spin counts) | 88% | 90% | 94% | 92% |
| VRAM memory-controller busy | 19% | 15% | 28% | 40% |
| PCIe H2D (NVML, 28 GB/s saturation) | 5.6 GB/s (20%) | 10.1 (36%) | **26.6 (95%)** | 25.3 (90%) |
| PCIe D2H (victim write-backs) | 5.3 GB/s (19%) | 9.2 (33%) | 2.8 | 2.7 |
| NVMe read (26 GB/s) | 6.7 GB/s (26%) | 10.3 (40%) | 10.8 (42%) | 8.9 (34%) |
| CPU-lane duty (useful CPU job time / wall) | 55% | 57% | 0% | 0% |
| CPU-tier cores busy (/proc/stat, spin included) | 96% | 92% | 11% | 3% |
| CPU-lane weight reads | 19 GB/s | 22 GB/s | 0 | 0 |
| DDR derived (138 GB/s practical) | 37 GB/s (27%) | 52 (38%) | 40 (29%) | 37 (27%) |
| RAM (cgroup, 55 GiB cap) | 50.7 GiB | 50.4 GiB | 48.5 | 49.7 |

Per-core load during decode:

- cpus 2-23 (CPU-lane workers, one per physical core) are 96% busy, mostly spinning between layers.
- cpu 24 (the Python main thread) and cpu 25 (the controller) are both at 100%.
- The reader threads on the SMT siblings 26-39 are at about 8%.

## Binding chokepoint per cell

- **Decode C1, and C1 @ 32k: the CPU lane.** It finishes last in 86% of layers. The largest single bucket on the
  critical path is the admission copy (26%). The copy runs on the GPU lane in parallel with the CPU lane, so making the
  copy faster alone only helps if the planner also moves work from the CPU to the GPU.
- **Decode C2 and C4: NVMe stall**, 29-35% of wall time. Streams share few experts, so each step makes 38 (C2) to 73
  (C4) NVMe reads, and the copy kernel waits on their landed flags. The NVMe array itself is at only 40% of its
  bandwidth. Mean read latency (submit to landed) is 3.4-4.0 ms for one 9.4 MB record, against 0.75 ms measured
  alone. 70% of all reads are layer-ahead prefetches.
- **Prefill, full 8k chunks: GPU compute, with PCIe H2D saturated at the same time.** One chunk takes 8.25 s of GPU
  time and only 28 ms of staging stall. Staging is overlapped with compute but uses the whole link.
- **Prefill tail: PCIe.** GLM-5.3-Flash has recurrent (KDA) layers, so exllamav3 prefills the last partial page as a
  separate forward to checkpoint the recurrent state at a page boundary. That 255-token forward re-streams every
  non-VRAM expert. It takes 4.6 s, of which 3.4 s is staging stall: 32% of an 8k prefill. Gaps between forwards add
  0.65-0.7 s each. Short chat prompts of 200-700 tokens take the same staged path and cost about 4.3 s TTFT.

Nothing is at its bandwidth ceiling during decode. GPU, PCIe, NVMe, DDR and CPU lane are each idle 45-80% of active
time. Decode is limited by latency and serialization inside each layer: plan, then copy and CPU job in parallel, then
combine wait, then the next layer's non-MoE work. It is not limited by throughput.

## Scheduling fixes (Part 3)

Each change ran as its own screening arm: 8k prefill, C1, C4. Results are in `docs/utilization-glm53-s3b-fixes.md`.
