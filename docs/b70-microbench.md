# Arc Pro B70 kernel microbench (N124, 2026-10-07)

Run record: `registry/runs/b70-microbench-2026-10-07.json`. Raw JSON: `docs/b70-microbench/raw/`. Scripts, runner,
guard and kernel patch: `docs/b70-microbench/bench/`.

These numbers calibrate the `exl3xpu-arc` engine lanes and the `omarchy-arcb70-nvx4` hardware ceilings. Slot A (the
RTX 3090) was busy for the whole run (`"contended": ["A"]`). H2D measured 23.9 GB/s against the idle 26.7 GB/s
(N106), so every link-bound number here is about 10% low. Re-measure with slot A idle before any of these numbers
ships.

## Setup

| item | value |
|---|---|
| device | Arc Pro B70 `0000:84:00.0` only. Only `renderD130` (from `/dev/dri/by-path/pci-0000:84:00.0-render`) is passed in, with `ZE_AFFINITY_MASK=0`; one xpu device is visible in the container. Held under the xpu lock `xpu-0000_84_00_0` (`freetoken-exl3/bench/xpu_run.sh`) |
| container | image `24c872759256` (sglang-exl3-xpu-flashnext, torch 2.13.0+xpu). `--cpuset-cpus 44-47 --memory 32g --memory-swap 32g --device-read-bps /dev/md127:6gb --network none`. SVM keys on: `NEOReadDebugKeys`, `EnableSharedSystemUsmSupport`, `EnableRecoverablePageFaults` |
| NVMe cap | docker writes `io.max` = `9:127 rbps=6442450944` (md127 itself). blk-throttle does apply to the md device: 30 GB O_DIRECT reads run at 6.49 GB/s. Transfers shorter than about 1 s exceed the cap (7.2 GB/s over 8 GB) because of slice credit |
| libs | `_moe_a9.so` (production) for the Qwen shapes. `_moe_n124.so` (a9 plus power-of-two k-splits, `bench/n124-splits.patch`) for the GLM shapes |
| timing | XPU events around back-to-back calls. The calls rotate through 2 layers of experts (1,024 Qwen or 576 GLM), so no call finds its weights in the 24 MB L2. The default decode dispatch is used: pair mode for M <= 4, grouped with MR=2 for M=8 |
| guard | `journalctl -k` checked before and after every job: no Completion-Wait timeout, Link Down, Card not present, reboot-needed, pciehp, non-corrected AER, or xe reset/wedge line for 84:00.0; no D-state khugepaged/kcompactd. The background corrected-AER rate (~100-130/min on c0:03.x, NVMe c5-c8 and 81-83) was the same before and during the run |

## 1. Ingest

H2D and D2H use `X.memcpy_async` on the compute queue. "single" is one copy with submit + sync (median). "batch" is
8-256 copies back-to-back.

| dir | host memory | 1.86 MB (Qwen expert) single ms / batch GB/s | 9.47 MB (GLM expert) | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| H2D | pinned USM | 0.127 / 22.1 | 0.447 / 23.8 | 2.88 / 23.9 | 11.35 / 23.9 |
| H2D | anon THP (sysptr) | 0.226 / 14.1 | 0.690 / 19.3 | 3.23 / 22.5 | 12.70 / 22.6 |
| H2D | memfd THP (sysptr) | 0.213 / 14.0 | 0.715 / 19.2 | 3.24 / 23.0 | 12.22 / 23.6 |
| D2H | pinned USM | 0.242 / 7.3 | 1.285 / 7.7 | 9.09 / 8.5 | 32.04 / 7.2 |
| D2H | anon THP | 0.524 / 4.2 | 1.786 / 7.2 | 8.85 / 7.8 | 36.38 / 7.7 |
| D2H | memfd THP | 0.461 / 4.5 | 1.533 / 7.3 | 9.39 / 7.7 | 31.27 / 8.2 |

The device kernel reading host RAM (zero-copy) runs at 21.5-25.6 GB/s. That figure comes from the exl3xpu MoE kernel
in section 2 with every pick host-resident. Pinned USM and memfd THP read through xe SVM are the same within ±5%.

NVMe pipeline: O_DIRECT reads (thread pool) land in a memfd THP ring, and the H2D copy into VRAM is issued as each
record lands. All runs are under the 6 GiB/s slot cap.

| store | record | volume | threads | H2D | GB/s | read ms p50 / p90 |
|---|---|---|---|---|---|---|
| GLM | 9.47 MB | 30 GB | 32 | no | **6.49** | 30.7 / 91.3 |
| GLM | 9.47 MB | 30 GB | 32 | yes | **5.77** | 30.1 / 87.0 |
| GLM | 9.47 MB | 8 GB (burst) | 16 | yes | 6.64 | 5.2 / 53.4 |
| GLM | 9.47 MB | 8 GB (burst) | 16 | yes, via USM bounce | 4.97 | 4.6 / 49.3 |
| GLM | 9.47 MB | 1.5 GB | 1 | no | 4.38 | 1.68 / 1.79 |
| Qwen | 1.86 MB | 30 GB | 32 | no | 3.99 | 5.8 / 9.3 |
| Qwen | 1.86 MB | 30 GB | 32 | yes | 2.64 | 6.1 / 14.1 |
| Qwen | 1.86 MB | 1.5 GB | 1 | no | 3.32 | 0.52 / 0.66 |

Findings:
- O_DIRECT into Level Zero USM host memory (`sycl::malloc_host`) fails with EFAULT. Records have to land in anon or
  memfd pages, then reach the GPU either through an SVM read or through a sysptr H2D copy. A CPU bounce into USM costs
  25% (6.64 -> 4.97 GB/s).
- 9.47 MB records saturate the cap with the copy overlapped (5.77 of 6.44 GB/s). The 1.86 MB rate is bounded by the
  Python pool on 4 cores, not by the device.

## 2. Expert compute (exl3xpu decode MoE kernel)

Shapes: Qwen3.8-Flash-Next 3.05bpw (H 2560, I 640, E 512, top-10, 1,862,400 B blob, real blobs from the NVMe store)
and GLM-5.3-Flash 3.05bpw (H 4096, I 2048, E 288, top-8, 9,474,048 B blob, real layer-3 experts packed with
`pack_expert` and cycled over the slots). The VRAM, USM and SVM tiers give bit-identical outputs for both models.

### GLM shapes need a kernel build

The stock `_moe_a8.so` and `_moe_a9.so` cannot run H 4096 / I 2048:
- gate|up needs `(H/16)/P % 8 == 0`;
- down needs `(I/16) % P == 0` and `((I/16)/P) % RB == 0`;
- the instantiated P values are only {5, 10, 20, 40}.

With P=5 on H 4096, gate|up silently reads 4 rows past the gate|up region, and down fails its TORCH_CHECK.
`_moe_n124.so` adds GU P 4/8/16 and DN P 4/16 instances.

Checked against `exl3xpu_C.linear` per projection (gate, up, SiLU(g)*u, down) on 4 real experts with M=4: relative
L2 error is **0.0012** at gu/dn = 8/16, 4/4 and 16/16.

| GLM split gu/dn | M1 ms | M4 ms | M8 ms |
|---|---|---|---|
| 8/16 (used for the rows below) | 0.1753 | 0.7092 | 1.3906 |
| 4/4 | 0.1696 | 0.6698 | 1.3323 |
| 8/4 | 0.1812 | 0.7036 | 1.3786 |
| 16/16 | 0.1832 | 0.7323 | 1.4764 |

Two gaps remain before GLM can be served from this kernel:
- The `swiglu_limit` 10 clamp on gate/up is missing.
- `routed_scaling_factor` 2.5 has to be applied. It can be folded into `topk_w`, but the sglang plugin rejects it today.

### Per layer call (ms), uniform routing; "unique" = distinct experts in the call

| model | M | unique | VRAM moe_forward | VRAM cached (b2b) | zero-copy USM | zero-copy SVM |
|---|---|---|---|---|---|---|
| Qwen | 1 | 10 | 0.0616 | 0.0746 | 0.753 | 0.826 |
| Qwen | 2 | 19.8 | 0.1223 | 0.1265 | 1.605 | 1.658 |
| Qwen | 4 | 38.9 | 0.2024 | 0.2065 | 3.166 | 3.334 |
| Qwen | 8 | 74.9 | 0.3517 | 0.3547 | 6.218 | 6.128 |
| GLM | 1 | 8 | 0.1681 | 0.1810 | 3.240 | 2.964 |
| GLM | 2 | 15.9 | 0.3451 | 0.3435 | 5.982 | 5.960 |
| GLM | 4 | 30.8 | 0.6886 | 0.6833 | 13.64 | 12.88 |
| GLM | 8 | 58.1 | 1.3326 | 1.3085 | 23.52 | 24.11 |

### Same experts for every token (m = M tokens per expert), ms

| model | tier | M1 | M2 | M4 | M8 | M2/M4/M8 grouped (pair_max_m 0, MR = M) |
|---|---|---|---|---|---|---|
| Qwen | VRAM | 0.0616 | 0.1073 | 0.1769 | 0.2211 | 0.0734 / 0.0971 / 0.1229 |
| Qwen | USM | 0.828 | 1.515 | 3.197 | 2.772 | 0.833 / 0.819 / 0.842 |
| GLM | VRAM | 0.1717 | 0.3317 | 0.6551 | 0.6783 | 0.1899 / 0.2365 / - |
| GLM | USM | 3.482 | 6.373 | 13.67 | 10.49 | 3.398 / 3.225 / - |

The default pair mode reads an expert's weights once per (token, expert) pair. An extra token on a shared expert
therefore costs about as much as a new expert: Qwen VRAM +0.0046 / +0.0038 ms (M2 / M4), and zero-copy re-streams the
whole blob over PCIe each time. Grouped dispatch with MR = M reads the weights once. Shared zero-copy at M8 then costs
the same as M1. On VRAM the extra-token cost drops to ~0.001 (Qwen) or ~0.0025 (GLM) ms. Grouped is slower for spread
routing, though (Qwen M8 random 0.452 vs 0.352 ms), so it only pays where experts are shared. Those cases are MTP
verify windows and zero-copy experts that several tokens pick.

### Mixed calls (M=1, j of the picks host-resident) and the cached path

| model | tier | j=0 | 20% | 50% | 100% | slope per host expert |
|---|---|---|---|---|---|---|
| Qwen | USM | 0.0617 | 0.186 | 0.439 | 0.859 | 0.081 |
| Qwen | SVM | 0.0618 | 0.181 | 0.423 | 0.828 | 0.078 |
| GLM | USM | 0.1732 | 0.881 | 1.718 | 3.332 | 0.398 |
| GLM | SVM | 0.1698 | 0.849 | 1.586 | 3.150 | 0.374 |

On the `moe_forward_cached` server path, an all-hit call costs +0.013 ms per layer over `moe_forward` for Qwen M=1
(route and LRU bookkeeping). An all-miss call with `max_fill 0` (zero-copy, no fill) costs 0.077 ms per Qwen expert.
With write-through into the LRU slot it costs 0.073 ms per miss over the all-hit call, so write-through is ~free.

When each call is preceded by small state copies (a drained queue), the all-hit Qwen M=1 call measures 0.17-0.20 ms
instead of 0.075 ms. The difference is launch latency, which XPU graph replay hides.

The commit check is false once a call has more than one miss, the same as N109 (pre-existing). It does not affect
the timings.

## 3. Launch / sync (µs)

| case | median | p90 |
|---|---|---|
| empty kernel submit (host side, 1-element `add_`) | 6.84 | 10.9 |
| back-to-back empty kernels | 7.57 per kernel | - |
| submit + `synchronize` | 47.7 | 50.8 |
| `synchronize` on an idle queue | 28.3 | 32.3 |
| event record + `query()` spin | 29.0 | 33.1 |
| 4 B copy + sync (H2D / D2H) | 40.0 / 39.8 | 42.0 / 41.5 |
| landed-flag round trip: host writes seq to USM, 4 B H2D + 4 B D2H, host spins (no sync) | **11.2** | 15.0 |
| device-written flag -> 4 B D2H -> host spin | **10.7** | 13.3 |
| flag lag after a 1 ms GEMM (wall - kernel time, idle queue) | 81 | 91 |

Polling a landed flag in USM is 2.5-4x cheaper than `synchronize` or event polling. An idle-queue launch costs about
80 µs end to end, so per-layer host handoffs have to be queued ahead or captured in graphs.

## What changed in the records

| record | field | before | after |
|---|---|---|---|
| engine/exl3xpu-arc | gpu per_layer / per_expert / per_extra_token ms | 0 / 0.0063 / 0 | 0.031 / 0.0043 / 0.0042 (cached b2b fit; per_extra_token = pair mode) |
| engine/exl3xpu-arc | zerocopy per_expert / per_extra_token ms | 0.052 / 0 | 0.080 / 0.075 (link-bound; the old value sits below the 0.070 ms PCIe floor and was L2-assisted) |
| engine/exl3xpu-arc | model_lanes.glm-5.3-flash-exl3-3.05bpw | - | gpu 0.010 / 0.0215 / 0.020; zerocopy 0.41 / 0.40 |
| engine/exl3xpu-arc | overheads (launch/sync µs), calibration_runs | - | added |
| hardware/omarchy-arcb70-nvx4 | ceilings | none | vram 608 (kernel 461 measured), h2d 26.7 idle / 23.9 contended, d2h 8.5, zero-copy 23.3, nvme slot cap 6.44 (6.49 measured), launch µs, host cpu/ddr |
| moetier/spec.py | resolve | - | applies `engine.model_lanes[<recipe model>]` over the engine lanes |

What this means for a GLM-5.3-Flash expert lane on the B70:
- VRAM-resident experts cost 0.0215 ms each. The B70's 32 GB holds ~3,000 of the 12,096 experts.
- RAM-resident experts read through zero-copy cost 0.41 ms each, which is PCIe-bound and about equal to the 3090's
  zero-copy lane (0.38).
- NVMe-fed experts are limited to ~5.8 GB/s by the slot cap, i.e. ~610 experts/s.

So the B70 is useful for GLM as a second VRAM tier: about 3,000 more resident experts, each costing 0.0215 ms. It is
not useful as a zero-copy lane.
