# B70 as a second expert tier for GLM-5.3-Flash on omarchy (N137)

HOM-272, 2026-10-08. One Arc Pro B70 (32 GB, slot `0000:48:00.0`) holds about 3,000 GLM-5.3-Flash experts next to
the RTX 3090's VRAM cache. Without it those experts sit in RAM or on NVMe. Stretch design: three B70s plus the 3090
(~120 GB of VRAM) hold all 12,096 experts (114.6 GB).

## Design

Each MoE layer call in decode works like this.

1. The 3090 publishes the layer's picks and its MoE input rows (fp16, 8 KB per token) to the mapped request ring. This
   step is unchanged from nv2 (`kernels/nv2`).
2. The host controller (`nv2_host.cpp` `plan_and_reply`) assigns each unique pick to a lane: `vram` (3090), `b70`,
   `cpu`/`zerocopy` (RAM tier) or `nvme->cpu`/`nvme->gpu`. The order is: VRAM-resident, then B70-resident (a static
   key set), then RAM, then NVMe. A B70 pick is masked on the 3090 (id -1, weight 0) the same way a CPU pick is, so
   the device-side code path is the CPU lane's.
3. The CPU worker, which owns the job for that layer, posts the B70 picks first: expert ids, routing weights, and the
   x rows copied from `hx` go into a shared host ring (`/dev/shm`, mapped by both containers). It then computes its
   own CPU picks, waits for the B70's landed flag (seq), adds the B70's fp32 partial into `hout`, and signals
   `HC_CDONE`. The 3090's `nv_combine` adds `hout` as it does today.
4. The B70 expert server is a persistent process on the B70. It spins on the ring and does an H2D of x, runs
   `moe_forward` (`_moe_n128.so`, swiglu clamp 10) over the slot pointer table of its resident experts, does a D2H of
   the partial, and writes the seq flag.

The three lanes overlap. The 3090 runs its VRAM experts, the CPU runs its RAM experts, and the B70 runs its own
experts, all at the same time. A layer costs `max(gpu, cpu, b70)`.

Placement is static. After the 3090's warm VRAM set, the B70 takes the next frequency ranks (warm-start scores). It is
exclusive of the 3090 cache and of the RAM tier. Its keys are never admitted to the 3090 and never warmed into RAM, so
RAM holds the next ranks after them. Prefill keeps today's staged path. B70 keys are read from NVMe there like any
other non-RAM expert (N129 measured 16 GB prefill at 654 tok/s vs 662 at 55 GB, so prefill barely depends on how many
experts RAM holds).

Cost model, b70 lane per layer: `handoff + 0.010 + 0.0215 x experts + 0.020 x extra tokens`. The last three terms are
the exl3xpu-arc GLM model lane from the b70-microbench. `handoff` covers the ring write, the B70 server noticing the
request, the submit, the H2D of 8 KB per token, the D2H of the partial and the flag. The design estimate is 0.12 ms:
three submits at 7 µs, an 11 µs flag round trip, and 81 µs of flag lag after a kernel on an idle queue. N137 measures
it.

Implementation: `plan.py` has a `b70` lane (one lane per card, cards in parallel). The ledger has a static `b70` tier
(rank round-robin over cards). The recipe is `glm53-rtx3090-b70-55g-nvx4` (`b70.cards`, `b70.handoff_ms`). The script
is `examples/b70_tier.py`, with raw output in `b70-tier-glm53.sim.jsonl`.

## Expected (sim, G002 decode trace, 9,250 tokens)

| config | C1 | C2 | C4 | GPU hit (3090 + B70) | CPU experts/tok | NVMe/tok | C1 ms/step: fixed + max(3090, CPU, B70) |
|---|---|---|---|---|---|---|---|
| 55 GB today: 3090 + CPU + NVMe | 23.37 | 30.06 | 33.42 | 0.512 | 134.3 | 20.2 | 14.0 + max(14.6, 28.3, -) |
| 55 GB + 1 B70 x 3000, handoff 0.12 | **31.56** | 41.71 | 52.19 | 0.441 + 0.292 = 0.733 | 84.6 | 3.4 | 14.0 + max(11.0, 16.4, 7.0) |
| same, handoff 0.05 | 31.62 | 41.75 | 52.19 | 0.733 | 84.6 | 3.4 | 14.0 + max(11.0, 16.4, 4.4) |
| same, handoff 0.25 | 30.81 | 41.49 | 52.16 | 0.733 | 84.6 | 3.4 | 14.0 + max(11.0, 16.4, 11.9) |
| same, handoff 0.50 | 25.81 | 39.35 | 51.92 | 0.733 | 84.6 | 3.4 | 14.0 + max(11.0, 16.4, 21.4) |
| 55 GB + 1 B70 x 3300 | 32.46 | 43.13 | 54.13 | 0.756 | 78.1 | 2.7 | 14.0 + max(10.8, 15.1, 7.2) |
| reference: 3000 more slots on the 3090 itself | 31.15 | 41.13 | 50.63 | 0.758 | 77.0 | 3.3 | 14.0 + max(13.3, 15.4, -) |
| full RAM (G067 layout): 3090 + CPU | 27.49 | 32.92 | 36.31 | 0.508 | 150.5 | 0 | 14.0 + max(16.7, 21.7, -) |
| full RAM + 1 B70 x 3000 | **33.15** | 43.85 | 52.72 | 0.728 | 89.4 | 0 | 14.0 + max(11.1, 14.9, 7.0) |
| stretch: 3090 + 3 B70 x 3000, 55 GB | **39.81** | 58.30 | 80.09 | 0.996 | 1.6 | 0 | 14.0 + max(10.3, 0.4, 7.5) |
| stretch: 3090 + 3 B70 x 3200 (55 GB or full RAM) | 39.83 | 58.30 | 80.09 | 0.999 | 0.5 | 0 | 14.0 + max(10.3, 0.1, 7.5) |
| stretch, handoff 0.25 | 36.56 | 55.36 | 77.58 | 0.999 | 0.5 | 0 | 14.0 + max(10.3, 0.1, 12.9) |

How to read it:
- One B70 moves the 55 GB config from CPU-bound (28.3 ms of CPU MoE per token) to a near tie between the CPU (16.4)
  and the fixed non-MoE time. NVMe reads per token fall from 20.2 to 3.4, because the B70 frees RAM for colder
  experts. C1 gains about +35%. The B70 lane is only 7 ms per token, so it is not the bottleneck while handoff stays
  at or below ~0.25 ms per layer. At 0.5 ms it becomes the bound and most of the C1 gain is lost.
- The B70 is worth the same as 3,000 more slots on the 3090 itself (31.56 vs 31.15 at C1), and more at C4. It is a
  third lane in parallel, not a longer 3090 queue.
- C2/C4 gain more than C1 (+39%, +56%) because the CPU lane stops growing with concurrency once fewer experts are
  on it.
- Stretch: with every expert in GPU memory the CPU and NVMe lanes disappear. C1 is then bound by the fixed 14 ms of
  non-MoE time plus the 3090's own MoE (10.3 ms). The next lever is the fixed time (fused kernels and graphs: 14 -> 9 ms
  gives about 50 tok/s in this sim).

Calibration caveat: this sim reads 23.4 for the 55 GB config, but v4.2-nvme measures 17.3 at C1 and about 18 at C4.
For the full-RAM layout it reads 27.5 against a measured 28.2 (G067). The 55 GB gap comes from NVMe stall and sync
effects that the sim underprices. The B70 removes most of the NVMe traffic (20 -> 3.4 per token), so the measured
result should land between sim x 0.74 (about 23 tok/s) and the sim (31.6) at C1.

## Status

- Step 1 (this doc, the sim): done.
- Step 2 (B70 expert server + ring + 3090-side client in a copy of the engine, `omarchy:~/freetoken-exl3/runs/N137-b70tier/`;
  numerics vs the CUDA path on real layers): see below.
- Step 3 (integration + measurement): waits until slot A (3090) and B70 48:00.0 are both free.
