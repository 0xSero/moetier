# moetier


# ATTRIBUTION: THIS IS ALL BASED OFF OF https://github.com/kacper-daftcode/vllm-Moet THIS WORK IS MY SLOP INTERPRETATION OF THAT REPO. IF YOU HAVE RTX PRO 6000S/5090S I WOULD RECOMMEND VLLM-MOET OVER THIS

A minimal standard for running big MoE models on small machines: **where every expert lives, who computes it, and
when its bytes move.** The scheduling layer is the same for every model; the model, hardware and engines are data.

Like the [local-ai-registry](https://github.com/0xSero/local-ai-registry), it is data first: records under
`registry/`, one recipe that assembles them, and a tiny core that reads them.

```text
recipe/<id>.json
  model     -> model/<id>.json       expert geometry: layers, experts, top-k, bytes per expert, store layout
  hardware  -> hardware/<id>.json    tiers and links: VRAM, RAM, NVMe (bw, latency), PCIe
  engines[] -> engine/<id>.json      lanes: where an expert can be computed and what it costs
  budget                             RAM cap, VRAM expert slots
  policy                             placement, exactness (stall | mask), overlap, prefetch
  calibration                        measured non-MoE time per step, prefill constants
runs/<id>.json                       measured tables (evidence)
```

## The model of the problem

Per decode step, every MoE layer routes each token to `top-k` experts. Each picked expert is in exactly one tier:

| tier | holds | computed by (lane) |
|---|---|---|
| `vram` | hottest experts (clock) | `gpu` |
| `ram` | next tier, exclusive of VRAM | `cpu` in place, or `zerocopy` (GPU reads the host copy over PCIe) |
| `nvme` | everything (packed 4K-aligned record store) | read first, then `cpu` (lands in RAM) or `gpu` (pushed to VRAM) |

The scheduler, `plan_layer`, runs once per layer and decides which lane computes each pick. The lanes run
concurrently: the GPU works on VRAM experts, the CPU works on RAM experts, and the copy engine and the NVMe reader
move bytes. So a layer costs `max(lanes)`, and the split is chosen to minimise that max. NVMe misses either stall
(exact) or are masked (lossy, counted). Prefetch issues the next layer's predicted NVMe reads one layer ahead.

## The contract (what an engine adapter implements)

| hook | meaning |
|---|---|
| `route(layer) -> {key: tokens}` | routed picks of the current step |
| `compute(lane, keys)` | run those experts on that lane (GPU kernel / zero-copy kernel / CPU kernel) |
| `fill(keys)` | async NVMe record -> RAM slot (O_DIRECT, deep queue) |
| `push(keys)` | async RAM -> VRAM slot copy (copy engine) |
| `victims() / drain()` | device evictions land in a VRAM ring and are drained to RAM off the critical path |
| `landed(handle)`, `fence()` | completion by polling sequence numbers; flags observed by queued device work |

One rule is not negotiable: **only the host marks an expert resident, and only after its bytes have landed.** A
device kernel that set residency itself caused zeroed pages flagged valid (a race we hit in practice). See
`moetier/ledger.py`.

`moetier/transport.py` ships a portable reference for the NVMe side: an O_DIRECT thread-pool reader over the
record store, and a 2 MiB-aligned RAM slot pool. Device copies are engine specific (see `adapters/`).

## Quick start

```bash
python3 -m moetier show glm53-rtx3090-55g-nvx4
python3 -m moetier table glm53-rtx3090-55g-nvx4 --trace traces/glm53-g002-decode.npy \
    --segments traces/glm53-g002-decode.segments.npy
python3 -m moetier sim glm53-rtx3090-55g-nvx4 --trace ... --set budget.ram_gb=55 policy.prefetch.recall=0.7
python3 examples/levers.py        # what it takes to reach 50 tok/s
```

Only the standard library is needed, plus numpy for traces. `sim` replays a real routing trace through the same
ledger and `plan_layer` a runtime uses. Time is modeled from the records.

## Showcase: GLM-5.3-Flash on one RTX 3090 with 55 GB of DDR4 and a 4× NVMe RAID0

The recipe is `registry/recipe/glm53-rtx3090-55g-nvx4.json`. 55 GB holds 1,510 VRAM, 5,087 RAM and 5,499 NVMe-only
experts (9.47 MB each).

**Calibration:** with every expert in RAM, the simulator gives 27.5 / 32.9 / 36.3 tok/s against the measured
28.15 / 31.57 / 33.98 (G067). That is −2% / +4% / +7%.

Cumulative levers (decode tok/s; C2 and C4 are totals across streams):

| lever (cumulative) | C1 | C2 | C4 | per-token ms at C1: fixed + max(gpu, cpu) |
|---|---|---|---|---|
| all experts in RAM (G067, 218 GB) | 27.5 | 32.9 | 36.3 | 14 + max(16.7, 21.7) |
| 55 GB RAM + NVMe tail | 23.4 | 30.1 | 33.4 | 14 + max(14.6, 28.3) |
| + layer-ahead NVMe prefetch (recall 0.7) | 25.6 | 32.7 | 38.3 | 14 + max(16.6, 24.6) |
| + persistent CPU workers (handoff 0.145 → 0.03 ms/layer) | 27.0 | 33.8 | 39.1 | 14 + max(13.8, 22.5) |
| + fused non-MoE kernels and graphs (fixed 14 → 9 ms) | 31.0 | 37.6 | 42.3 | 9 + max(13.2, 22.7) |
| + 400 more VRAM slots | 32.4 | 39.4 | 45.3 | 9 + max(12.8, 21.1) |

What the planner says:
- With 55 GB the box can **beat its 218 GB all-RAM result** once scheduling and kernels improve.
- The binding constraint is then the **CPU lane**: ~134 RAM-resident experts per token at ~91 GB/s, close to the
  DDR4 limit.
- Exact 50 tok/s at C1 therefore needs fewer RAM bytes per token. The options are MTP (tokens in a verify window share
  experts), a higher VRAM hit rate, or lossy top-k.

## Layout

| path | what |
|---|---|
| `registry/` | records: `model/`, `hardware/`, `engine/`, `recipe/`, `runs/` |
| `moetier/spec.py` | load records, resolve a recipe (with dotted overrides) |
| `moetier/ledger.py` | residency: VRAM clock, exclusive RAM LRU, host-owned flags |
| `moetier/plan.py` | the scheduler: `plan_layer`, NVMe channel, prefetch |
| `moetier/transport.py` | transport contract + O_DIRECT record reader + RAM slot pool |
| `moetier/sim.py` | trace replay and prefill model |
| `adapters/` | how existing engines implement the contract |
| `examples/levers.py` | the lever study above |

## Status

v0.1. The scheduler, ledger and simulator are complete and calibrated against one measured config. The adapters
describe the existing implementations (exllamav3 on CUDA for GLM-5.3-Flash, exl3xpu on Intel Arc for
Qwen3.8-Flash-Next); moving them onto `plan_layer` is the next step. Speculative decoding is a what-if only
(`sim.run(window, accept, draft_ms)`), and per-tier record sizes (`recipe.tiers`) model e.g. 2-bit cold tiers; see
`docs/vllm-moet-lessons.md` and `examples/cold2bit.py`. Not modeled yet: prefill/decode interleaving.
