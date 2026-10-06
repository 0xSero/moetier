# Adapters

An adapter maps one engine onto the moetier contract (`route`, `compute`, `fill`, `push`, `victims/drain`, `landed`,
`fence`). These two implementations exist today and are the reference for moving onto `plan_layer`.

## exllamav3 on CUDA: GLM-5.3-Flash, RTX 3090 (github.com/0xSero/glm53-flash-offload)

| hook | today |
|---|---|
| `route` | exllamav3 block_sparse_mlp router (monkeypatched) |
| `compute(gpu)` | exllamav3 fused MoE kernel over the VRAM expert cache (elastic, `glm53/expert_cache.py`) |
| `compute(zerocopy)` | same kernel with host pointers (pinned host arena) |
| `compute(cpu)` | AVX2 EXL3 trellis kernel (`glm53/cpu_tier.py`, 22 threads); split by the ft_split cost model |
| `fill` | none yet: all experts are pinned in RAM (218 GiB). The NVMe tier (N116, GLM53_MODE=nvme) adds a record store and RAM slot pool |
| `push` | prefill staging 2 × 2.6 GB on a copy stream |
| `victims/drain` | n/a in all-RAM mode |

## exl3xpu on Intel Arc: Qwen3.8-Flash-Next, Arc Pro B70 (github.com/0xSero/qwen38-flash-next-b70-offload)

| hook | today |
|---|---|
| `route` | SGLang router → `Exl3XpuMoEMethod.apply` |
| `compute(gpu)` | `moe_forward_cached` (device-managed VRAM slot cache, masking) |
| `compute(zerocopy)` | the same kernel reading the memfd RAM tier through xe SVM |
| `compute(cpu)` | none: the decode tier masks misses and fills them in the background |
| `fill` | `nvtier.py` O_DIRECT thread pool → 2 MiB memfd slots (`fill_many`) |
| `push` | staged prefill: NVMe → anonymous THP buffer → copy engine → VRAM double buffer, 4 layers ahead |
| `victims/drain` | `experimental/n111-victim-ring`: VRAM ring, pinned staging drain, host-set flags |
| `landed/fence` | `NvTierAsync`: lagged event-ordered snapshots, punch only after flag-clear events complete |
