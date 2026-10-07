# Utilization and chokepoints: method

Speed tables say how fast a configuration is. They do not say why, or what to change next. Every moetier run
record therefore carries two measured sections, `utilization` and `chokepoints`. This page defines how they are
measured and what each field means. The schema summary is in the README under "Run records".

## Two instruments

| instrument | where | what it answers |
|---|---|---|
| `moetier probe record` | host, beside the server, ~100 ms samples | how busy each resource is, as % of its ceiling |
| engine critical-path counters | inside the engine, every MoE layer call | where each token's wall time goes, and which lane finished last |

`moetier probe report` joins the two over the bench's phase windows. It reads the probe samples, the bench's wall-clock
`marks`, and the hardware record's `ceilings`.

## Probe samples (host, no root)

| field | source | caveat |
|---|---|---|
| `gpu_util_pct` | NVML `utilization.gpu` (ctypes on `libnvidia-ml.so.1`) | share of time any kernel was resident. A device kernel that spin-waits on a host flag counts as busy. |
| `vram_busy_pct` | NVML `utilization.memory` | share of time the memory controller was busy. This is time, not bytes. It is an upper bound on use of the 936 GB/s. |
| `pcie_h2d_gbps` / `pcie_d2h_gbps` | NVML PCIe throughput, RX / TX from the device's view, 20 ms window | single 20 ms window per sample; average over many samples |
| `nvme_read_gbps` | `/proc/diskstats`, md array | |
| `cpu_tier_cores_busy_pct`, per-core | `/proc/stat` over the server cpuset | CPU-lane workers spin about 2 ms between layers, so spinning shows as busy |
| `cpu_lane_duty_pct` | engine counter `cpu_busy_ns` / wall | useful CPU-lane work: time a CPU job was running |
| `cpu_lane_weight_gbps` | engine `cpu_experts` x expert bytes / wall | weight bytes the CPU lane read |
| `ddr_derived_gbps` | CPU-lane weight bytes + PCIe rx + PCIe tx + NVMe read bytes | **derived**: Zen 3 DF/UMC counters are not exposed without root, so this omits activations and page-table traffic |
| `ram_gib` | cgroup `memory.current` | |

The engine writes a live counters file (one 4 KiB page of int64, about every 20 ms, plus `<file>.json` with the
field names). Each sample interval is classified from it:

- **decode**: decode layer calls advanced, no prefill.
- **prefill**: staged prefill layers or prefill tokens advanced, no decode.
- **mixed**: both. Mixed intervals are excluded from per-kind statistics.
- **idle**: neither.

Statistics are taken over active time only: mean, p50, p90, % of ceiling, and `idle_frac`, the share of active time
the resource was below 5% of its ceiling.

## Engine critical path (decode)

One decode MoE layer call `s` on the device's compute stream, timed with `%globaltimer` and written into mapped host
memory:

```text
A  nv_pub        request published (unique picks + rows for the CPU job)
C  nv_step       host reply seen (plan_layer lanes)
E  nv_copy       admission copies start (RAM -> VRAM; NVMe sources wait for their landed flag)
F                copies done
G  nv_combine    entry: routed MoE kernels done = GPU lane end
H                CPU partial seen = CPU lane end, if the CPU lane was last
N  nv_pub(s+1)   next layer call (non-MoE GPU work of the next layer lies in between)
```

The host CPU job records its own start and end (CLOCK_REALTIME) and its wait for NVMe landings. A pure-CPU
attribution thread maps host time to device time with `off = min(host notice - tpub)`, windowed. It never makes a
CUDA call, which avoids the launch-queue deadlock. It then books every layer call into these buckets:

| bucket | interval | meaning |
|---|---|---|
| `overhead` | C-A + E-C + bubbles | host plan + reply, device CLOCK bookkeeping, and launch bubbles: an upper bound on the time the next `nv_pub` had not yet been launched by the Python thread after H |
| `copy` | F-E minus NVMe waits | admission copies (SM gather over PCIe) |
| `nvme_stall` | NVMe landed waits inside the copy + the NVMe share of the CPU wait | exact-mode stall on NVMe reads |
| `gpu_moe` | G-F | routed expert kernels on VRAM-resident experts (and the fused shared expert) |
| `cpu_moe` | H-G minus its NVMe share | CPU lane overrun: the GPU waited for the CPU partial |
| `fixed_gpu` | N-H minus bubbles | non-MoE GPU work. At a step boundary this also covers lm_head, sampling and the scheduler. A boundary gap over 50 ms counts as idle and is excluded. |

The buckets sum to the decode wall time (`wall_ms_per_token`). They are normalised per token, where a step carries
`tokens_per_step` tokens at C2/C4. `binding` is the largest bucket.

Per layer call, the counters also record:

- `lane_last`: which lane finished last. The CPU lane counts as last if the GPU waited over 5 µs for it; NVMe counts as last if NVMe waits made up most of that wait.
- the slack of the other lane.
- per-layer-index means of: GPU lane (copy + MoE), CPU lane busy, copy, NVMe wait, CPU-critical, and gap.

They are exposed in the server's `/stats` as `nv2.critical_path`, and as `a_*` fields in the live file.

## Prefill

Staged prefill forwards are timed with CUDA events at the cross-stream wait for each layer's staging buffer. For each
forward the counters record:

- `staging_stall`: the time the GPU waited for staging H2D from RAM or the NVMe ring.
- `gpu_compute`: the first-to-last MoE-layer span on the GPU, minus the stall.
- `host_gap`: host time from the previous forward's last staged layer to this forward's first.

Each forward is booked when the next one begins (one forward lag).

## Quality of the measurement itself

`chokepoints.*.decode.quality` reports how much of the measurement can be trusted:

- `lost`: debug-ring entries overwritten before they were read.
- `bad_order`: non-monotonic timestamps.
- `no_combine`: layer calls without a combine kernel, so the CPU lane end is unknown.
- `launch_bound_frac`: share of layer calls whose `nv_pub` ran within 30 µs of its host launch, i.e. the GPU was waiting for Python.
- `launch_ring_miss`: launch-time ring misalignment, e.g. graph replays.

Instrumentation overhead is a handful of 8-byte mapped stores per layer plus one 2 ms polling thread. It is checked
against the uninstrumented run's table: see `docs/utilization-glm53-s3b.md`.
