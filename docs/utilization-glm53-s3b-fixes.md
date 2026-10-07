# GLM-5.3-Flash S3b on the RTX 3090: scheduling fixes, measured

This page continues `docs/utilization-glm53-s3b.md`. Each fix was one change, measured with the probe running and
the screening sweep: 8k prefill (3 reps), decode C1 and C4 (2 rounds each, natural completions). Arms live under
`omarchy:~/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272/runs/`, and each has a run record `registry/runs/glm53-u*.json`.
Engine code variants are in `hom272/glm53{u,w,c,x,lfu}`. All of them keep the shipped v4-nvme modes untouched: new
code directories, separate build directories, and every knob defaults to the old behavior.

## Results (screening)

| arm | change | 8k prefill | C1 | C4 | slot B | verdict |
|---|---|---|---|---|---|---|
| u0_base | uninstrumented S3b, same day | 570 | 14.09 | 15.92 | idle | reference |
| u1_ctrl | instrumented control (prefetch on, LRU) | 580 | 13.18 | 16.23 | idle | control |
| u3_vis | control + visibility diagnostics | 573 | 14.07 | 15.46 | idle | control |
| u1_pflo | prefetch reads on the low-priority reader queue | 537 | 13.02 | 15.48 | idle | no effect, dropped |
| u4_ldcv | `ld.global.cv` flag polls | - | - | - | idle | hangs on the reply poll, reverted |
| **u1_pfoff** | **layer-ahead NVMe prefetch off** (`GLM53_NV_PREFETCH=0`) | **651** | **17.84** | **17.67** | busy | **+27-35% C1, +9-14% C4** |
| u2_lfu | sampled-LFU RAM eviction (`lfu:2048:500`), prefetch on | 647 | 16.89 | 18.11 | busy | +20-28% C1, +12-17% C4 (see caveat) |
| u5_cpudiag | prefetch off + CPU-job timers | 640 | 16.95 | (lost) | busy until the B70 drop | confirms pfoff (incomplete) |
| u6_lfu_pfoff | prefetch off + LFU | - | - | - | - | stopped at start (3090 handed to the user) |

The controls spread ±5% across the day: C1 13.2-14.1, C4 15.5-16.2. Both winners beat every control by far more than
that. They ran while slot B (a vLLM XPU recipe on B70 84) was busy, so their records carry `"contended": ["B"]`. The
numbers must be re-measured with slot B idle before they ship.

## Chokepoint attribution before and after (decode, ms per token)

C1:

| arm | wall | fixed GPU | GPU MoE | CPU MoE | NVMe stall | copy | overhead | lane last cpu/nvme | NVMe reads/tok | read latency |
|---|---|---|---|---|---|---|---|---|---|---|
| u1_ctrl | 76.0 | 13.7 | 11.1 | 10.4 | 15.3 | 19.6 | 5.9 | 0.86/0.14 | 19.6 | 3.5 ms |
| u3_vis | 71.2 | 13.7 | 11.0 | 10.1 | 11.0 | 19.7 | 5.7 | 0.87/0.13 | 18.5 | 3.4 ms |
| u1_pfoff | **55.9** | 10.2 | 8.7 | 11.0 | 8.5 | 13.4 | 4.1 | 0.82/0.17 | 34.3 | 2.3 ms |
| u2_lfu | 60.4 | 11.1 | 8.8 | 10.1 | 9.2 | 16.8 | 4.6 | 0.90/0.10 | 18.1 | 3.4 ms |

C4:

| arm | wall | fixed GPU | GPU MoE | CPU MoE | NVMe stall | copy | overhead | lane last cpu/nvme | NVMe reads/tok |
|---|---|---|---|---|---|---|---|---|---|
| u1_ctrl | 61.3 | 4.6 | 10.1 | 2.2 | 20.9 | 17.7 | 5.9 | 0.82/0.17 | 24.2 |
| u1_pfoff | 56.4 | 4.1 | 10.0 | 3.4 | **24.6** | 9.7 | 4.7 | 0.65/**0.35** | 44.4 |
| u2_lfu | 55.5 | 3.6 | 8.7 | 2.0 | 20.1 | 15.9 | 5.2 | 0.86/0.13 | 22.7 |

## What moved and why

- **The layer-ahead prefetch cost more than it saved.** It predicts the next layer's picks by running that layer's
  router on this layer's input, then reads the predicted NVMe-tier experts. In the controls, 70% of all NVMe reads
  were prefetches, and the mean read latency (submit to landed) was 3.5 ms.
  - Turning it off doubles the demand misses (19.6 to 34.3 per token at C1), but read latency falls to 2.3 ms.
  - The admission copy drops from 19.6 to 13.4 ms per token, and NVMe stall from 15.3 to 8.5. PCIe and DDR are no
    longer shared with speculative traffic, and fewer RAM slots are churned.
  - Fixed GPU plus overhead drop by about 5 ms per token. This is the per-layer prediction kernel and its launches.
  - At C4 the NVMe lane is now last in 35% of layers, and NVMe stall (24.6 ms) is the binding bucket. The prefetch
    was hiding misses there, but it was too expensive to keep.
  - Prefill is unaffected by design. The 651 vs 580 difference is within the prefill spread.
- **LFU eviction**, prefetch on, gave a large gain but barely changed the miss count (23.3 to 21.9 per token). Part of
  its gain also appears in buckets that LFU cannot touch, such as fixed GPU, so some of it may be session drift.
  - It needs a same-session control.
  - The `u6_lfu_pfoff` combination is the next arm. If LFU is real, it should cut the misses that pfoff added, which
    is exactly where pfoff left C4 bound. The simulator predicts -15% misses (`docs/ram-policy-glm53.sim.jsonl`).
- **Not chokepoints:**
  - The done-flag polls: visibility lag is about 0 (u3_vis).
  - Prefetch priority (u1_pflo).
  - `ld.cv` polls.
- **Prefill has two parts:**
  - Big chunks are compute-bound with PCIe H2D saturated. No scheduling fix is possible without fewer bytes or more
    VRAM.
  - The recurrent last-page tail forward is 4.6 s, 32% of an 8k prefill. It is still open. Options:
    - drop the checkpoint split. This is opt-in only, because multi-turn prefix reuse would fall back to the previous
      chunk boundary.
    - run the tail through a CPU+GPU split path, with RAM experts on the CPU lane and only NVMe-tier experts streamed.

## Where the chokepoint is now (u1_pfoff)

- C1, 55.9 ms per token:
  - The CPU lane is still last in 82% of layers.
  - The biggest buckets: copy 13.4, CPU MoE overrun 11.0, fixed GPU 10.2, GPU MoE 8.7, NVMe 8.5.
  - The CPU job starts 0.07-0.15 ms after the device sees the reply (u3_vis), and spends about 0.16 ms per layer
    waiting for NVMe landings.
- C4, 56.4 ms per token: NVMe stall (44%), from twice the misses.
- Next, in measured order:
  1. pfoff + LFU (u6, queued).
  2. Reader pool: 32 threads with 1 MB pieces, to cut the 2.3 ms read latency (u7, queued).
  3. CPU-lane start latency and phase breakdown (u5 timers; the breakdown was lost in the B70 drop).
  4. Copy-engine admission with a launch-queue cap, for the 13.4 ms copy.
  5. The prefill tail.

## Not yet done

- The full standard table (8k C1/C2/C4, 32k C1/C2), the paired decode-KL check and the panel, for the best arm with
  slot B idle. Prefetch-off does not change any computation: same lanes, exact residency. Its decode KL should equal
  S3b's 0.0047, and that still has to be measured.
- CPU-job phase breakdown (u5 diagnostics).
- Copy-engine admission and the prefill tail. Neither was started.

## Correction, 2026-10-08

Arms later in the evening (`glm53-u6a/u6b/u6-lfu-pfoff/u7-rd32`, C1 14.5-14.9) were all slower than the afternoon by
about 17%. A same-session A/B was then run (`glm53-c1lat-pfon` vs `-pfoff`, 9,300 steady C1 tokens each, three
natural-EOS requests). It reverses the prefetch result: **prefetch on is 13% faster** (16.04 vs 14.21 tok/s from the
device timeline; client 14.7-17.7 vs 13.2-15.2). NVMe misses drop 40.2 → 21.9 per token.

The "prefetch off +27-35%" above was session drift between arms run hours apart. LFU and the 32-reader pool are
neutral in same-session repeats. Keep prefetch on. Full latency breakdown: `docs/latency-glm53-c1.md`.
