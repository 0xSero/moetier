# vLLM-Moet: what transfers to GLM-5.3-Flash on one RTX 3090 + 55 GB DDR4 + NVMe RAID0

HOM-266, 2026-10-06. This was a read-only study. The four public repos were shallow-cloned as untrusted data into the
session scratchpad (`moet-src/`) and nothing from them was built or run. The only things executed were the moetier
simulator (CPU, on the Mac) and metadata reads on omarchy. No GPU jobs were run.

| repo | HEAD read |
|---|---|
| kacper-daftcode/vLLM-Moet ("upstream") | `f7cc6d9`, 2026-10-02 |
| lrozewicz/vLLM-Moet-GB10 ("GB10") | `8da6cf9`, 2026-09-12 |
| 0xSero/moet | `b0e2085`, 2026-07-13 |
| 0xSero/sglang-moet (= `~/Documents/moet`) | `cd7377b`, 2026-07-10 |

Labels used below:
- **[M]** measured by the repo's authors. These are their numbers; we did not re-run them.
- **[C]** claimed or imported, without a run artifact.
- **[S]** our simulator (modeled, not measured).
- **[inf]** inference.

Patch paths are files inside `patch/vllm-moet-v0.24.0.patch`. All of them live under
`vllm/model_executor/layers/quantization/utils/` unless the path says otherwise.

## 0. Answer (ranked for our C1/C2/C4 decode, at EXL3-class quality)

| # | idea from vLLM-Moet / GB10 | our verdict | modeled effect on the lever table (C1) [S] | quality |
|---|---|---|---|---|
| 1 | **Speculative decoding as the way to amortize expert reads.** GB10 GLM-5.3-Flash: MTP k=2 acceptance 2.81-2.95; 2-bit MTP k=3 3.62-4.0; DFlash2 k=7 5.5 on code [M] | **Adopt (research lever; check the serving rules first).** It is the only lever that lifts the CPU-lane ceiling without changing bytes or precision | 32.4 -> **46.2** (MTP k=2), 47.0 (MTP k=3). DFlash2 k=7: 41.0 on code, 17.9 on prose. Wide windows lose on our CPU lane | lossless |
| 2 | Router-lookahead prefetch (`moe_w2_looka.py`): layer L+1's router applied to layer L's input recalls **71.6 %** of the true top-8 on GLM-5.2, vs 41.3 % for the previous token's experts [M, measured on "colibri", a CPU engine] | **Confirms our lever** (we assumed recall 0.7). Build it with this exact predictor | already in the table: 23.4 -> 25.6 | lossless |
| 3 | 2-bit copy of the **NVMe tail only**; RAM and VRAM stay at 3.05 | **Try, behind a KLD gate** | 32.4 -> **34.3** (C2 39.4 -> 42.2, C4 45.3 -> 47.5) | at least 5 % of picks at 2 bits (more once NVMe fills linger in RAM); est. KLD 0.068 -> ~0.09 [inf] |
| 4 | "2-bit cold + precision for hot", i.e. RAM and NVMe at 2 bits, VRAM at full precision (vLLM-Moet's base cache + FP4 pool, inverted) | **Reject for EXL3-class quality** | 32.4 -> 37.8 (CPU stays decode-bound) or 41.6 (if bytes-proportional) | **45-55 % of all picks at 2 bits**; est. KLD ~0.25 vs 0.068 [inf]. Our only GLM-5.3 2.0bpw receipt is KLD 0.438 / top-1 77.4 % |
| 5 | vLLM-Moet's 2-bit LUT planes (`{-4,-1,1,4}` × UE8M0/32) as a **CPU-cheap** format | Speed is real: the CPU lane becomes DRAM-bound | 32.4 -> **43.1** (C4 60.8) | same 52 % of picks at 2 bits as #4, and RTN 2-bit is worse than trellis 2-bit. Quality-blocked |
| 6 | Residual Δ in EXL3 trellis (upstream "Δ-pool": base (2,2,1) + Δ (2,2,3) on the residual, summed pre-activation) | **Feasible, but no win on our tiers.** It doubles decode work (our CPU lane is decode-bound), and base+Δ at 3 bpw is worse than direct 3.05 | negative on the CPU lane; neutral at best on the GPU | below a direct quant at equal bits |
| 7 | Smaller ideas: persisted heat + preheat on boot; scan-resistant prefill (prefill must not evict the decode hot set); per-step KPI line; decode-once-per-union-expert at M ≤ 16; capture-safe fixed group shapes | **Adopt the policies.** The kernel idea we already have on CPU (`tile_accum_i16<M>`) | small or indirect: fewer cold-start NVMe misses, no prefill-induced slumps | lossless |

Bottom line:
- vLLM-Moet's core trick (2-bit planes + an FP4 pool for hot experts) is a **capacity trick for GPUs whose decode is
  HBM-bound**. It accepts that most routed traffic runs at 2 bits.
- Our binding constraint is a **trellis-decode-issue-bound AVX2 CPU lane** (`glm53/CPU_TIER.md`; research/14: about
  2.1 cycles per 8 weights, the same whether the expert sits in L3 or DRAM). Fewer bytes per expert do not make
  that lane cheaper unless the format's decode is also cheaper.
- The exact-quality path to 50 tok/s goes through **more tokens per expert read (MTP/speculation)**, not lower bits.
- Lower bits are worth it only for the NVMe tail, and only if a KLD gate passes.

## 1. The method

### 1.1 2-bit "planes"

**Scope.** Routed experts only: gate, up and down (`w13`, `w2`). Attention, dense layers, shared expert, router,
embeddings and lm_head keep the checkpoint's precision (README "How it fits"; GB10 doc
`docs/models/glm-5.3-flash.md` "1. Size").

**Codebook.** Sign-symmetric `{-4,-1,+1,+4}`, codes 0..3 = -4, -1, +1, +4. There is no zero level. Each block of 32
weights along K has a **UE8M0 power-of-two scale** (`moe_w2_planes.py` docstring, `_NIBBLE_TO_CODE`,
`reference_dequant`). That is 2.25 bpw including scales.

**Quantizer** (`moe_w2_planes._f64_to_codes_scales`, RTN, load-time, f64):
1. Scale per block: `exp = round(log2(amax/6))`.
2. Snap to the e2m1 grid with midpoints `[.25,.75,1.25,1.75,2.5,3.5,5]`.
3. Map magnitudes {0,.5,1,1.5,2} -> ±1 and {3,4,6} -> ±4, keeping the sign bit.
4. Exact zeros take ±1 by sign. If more than 95 % of a tensor's zeros share one sign, the sign alternates by
   k-parity instead. This fixes Kimi's all-+0 export, which otherwise injects 3x the bias that breaks GLM.

**Sources.** Native MXFP4 nibbles are remapped directly (`mxfp4_to_codes`). FP8 block-128 checkpoints
(`fp8_block_to_codes_scales`) and modelopt NVFP4 checkpoints (`nvfp4_to_codes_scales`) are dequantized to f64 and
re-quantized. Upstream docs call this "f64-exact vs the reference pipeline".

**Why symmetric** (`docs/quality.md`, DS4 QUANT_PROBE) [M]:
- The optimal-L2 2-bit codebook is sign-asymmetric. It collapsed DS4 to loops: MTP acceptance 1.00, coherence 0/12.
- The symmetric codebook at the same L2 error scored acceptance 2.73 vs 2.68 for the official FP4 experts, and
  12/12 coherent.
- 33,023 of 33,024 tensors chose it.
- On GLM-5.2 the asymmetric bias was -0.042, 99 % negative.

**GPTQ variant** (GB10 only, `tools/glm53/gptq_build.py`) [M]:
- Same grid and scales. Only the gate/up codes are re-chosen: one Hessian per layer, act-order, block 128,
  damp 0.01, 64 × 4096 tokens of code.
- Gate/up output error falls by 22 % on average. Down stays RTN.
- KLD between GPTQ and RTN is 0.110 / 0.196 / 0.175 (code / English / Polish), against noise floors of
  0.019 / 0.018 / 0.010.

### 1.2 The FP4 "delta" is not a residual

It is the **checkpoint's own e2m1 FP4 nibbles** (the "baseline" the 2-bit codes were snapped from), kept per expert
and sharing the same UE8M0 block-32 scales (`moe_w2_delta.py` docstring: "Hot routed experts get their FULL e2m1
nibble planes cached").

Two pool encodings exist:

| encoding | kernel | slot contents | size | status |
|---|---|---|---|---|
| full nibble plane | `moe_w4_mm` | e2m1 nibbles | 4 bit; a GLM slot is 12.6-12.75 MiB | default |
| "split" / quintal | `moe_w4q_mm` (`VLLM_MOE_W2_DELTA_SPLIT=1`) | a radix-5 refinement plane at **2.5 bit/elem**; the resident 2-bit base code picks the small or big magnitude class and one base-5 digit picks the magnitude inside it | 5/8 of a nibble slot, so 1.6x experts/GiB | opt-in |

For the quintal encoding, e2m1 is reconstructed **bit-exactly**, zeros included (`moe_w2_planes.pack_quintal_fragment_major`,
`quintal_dequant`; `kernels/MANIFEST.md` "Split FP4"). Its predecessor `moe_w4s` used a 2-bit refinement that
merged |0| into 0.5; that measurably decayed GSM8K on DS4, where zeros are 11.6 % of weights, so it was retired.

### 1.3 How the delta cache works (`moe_w2_delta.DeltaTier`)

**Layout**
- Host store: the FP4 planes of **every** expert, in pinned RAM or an NVMe pack.
- GPU pool: `VLLM_MOE_W2_DELTA_GB` (or `auto` = free VRAM after KV minus a 3 GiB reserve) worth of slots.
- Slot table: `int32 [layers, E]` on the GPU, -1 meaning 2-bit. The desc-build kernel reads it inside CUDA graphs.

**Manager thread**
- Runs one pass per step boundary. A free-running 5 ms poll cost GLM TP2 40 % of decode.
- Reads the forward's "seen" flags and promotes up to `DELTA_PROMOTE` = 8 per pass.
- Evicts only slots that have been cold for 2 or more passes.
- Table updates race graph replay on purpose: the worst case is one step reading the old tier, and both tiers hold
  valid weights.

**Policies** (`_POLICY`, chosen offline with `tools/delta_sim.py` trace replay):

| policy | behaviour |
|---|---|
| `freq` (default) | promote the hottest |
| `need` | only the confidence gate fills the pool; hot ≠ needs precision, and FP4 is twice the bytes |
| `lru` | promote in order, evict coldest |

GB10 adds promotion hysteresis 1.25, worth +15 % decode [M].

**Pool persistence.** Ownership survives restarts (`POOL_HEAT`, preloaded before graph capture).

**Confidence gate** (`moe_w2_gate.py`)
- If the step's `max_prob <= τ`, it force-promotes that step's routed experts (capped at 64 per fire), replays the
  graph, and re-decides. It iterates to a fixed point, at most 3 times.
- Offline [M, `moe_w2_gate.py` docstring]: at τ 0.67, ~30 % of tokens fire and recover ~90 % of the 2-bit -> FP4
  top-1 gap and ~61 % of the PPL gap. AUROC is 0.916.
- At τ 0.60: fires on 16 % of steps, 46 % precision, 68 % recall, against a 10.8 % base disagreement.
- Arming it costs ~10 % single-stream; τ 0.60 replays were throughput-neutral on DS4 (`docs/v024-port.md`).
- Uncapped fires on GLM-5.2 measured 200-1400 promotes per fire, about 6 GiB of H2D, and decode fell from 56 to
  3 tok/s.

### 1.4 Precision recovered

| model / config | metric | result | label |
|---|---|---|---|
| DS4 1x PRO 6000, bare 2-bit vs FP4 | next-token agreement | 89 % (README "Quality") | [M] |
| DS4, 2-bit + FP4 delta 2 GiB (170 slots) | arithmetic `17×23−100` | fixed | [M] |
| DS4, same | decode | 151 -> 143 tok/s (≈ -0.6 ms/step) | [M] |
| DS4, same | MTP acceptance | 2.725 vs 2.546 for a 2-card FP4 control | [M] |
| DS4 TP2 "max-quality" | GSM8K | 95.5 % (-1.5 pp vs native) | [C]; README says it must be re-measured (SwiGLU-clamp bug, THINK metadata) |
| DS4 TP2 "max-quality" | GPQA | 73.2 % (-1.0 pp, +10.6 % tokens) | [C], same caveat |
| DS4 (`kernels/MANIFEST.md`) | GSM8K-200 | native 97.0 % / 116 tokens vs W2 97.0 % / 122 tokens | [M] |
| DS4, prefill-only truncated KL vs FP8 (`0xSero/moet` `benchmarks/evals/kld/README.md`) | KL / top-1 | pure 2-bit **0.641 / 73.5 %**; with delta+gate 0.643 / 73.4 % | [M]; the gate never fires in prefill, so this only shows that bare 2-bit is far from the source |
| GLM-5.2 TP4 | decode | 105 tok/s bare 2-bit + MTP, 83-85 with delta + gate | [M] |
| GLM-5.3-Flash (GB10) | HumanEval / HumanEval+ | GPTQ planes + 2 GiB FP4 pool 97.0 % / 93.9 %, against the original on the z.ai API at 95.1 % / 89.6 % (McNemar p 0.45 / 0.09) | [M] |
| GLM-5.3-Flash (GB10) | HumanEval ablations | RTN no pool 92.1 %; RTN + pool 95.7 % | [M] |
| GLM-5.3-Flash (GB10) | KLD vs the original | none (no logprobs from the API) | — |
| GLM-5.3-Flash (GB10) | loops | bare 2-bit: Polish prose broke in 4 of 6 samples; pool off brought loops back (32 %, 54 %); 3 GiB pool 10 of 10 clean | [M] |

**Speed effect of the FP4 tier.** It is always a cost, because FP4 reads twice the bytes:
- DS4: -5 %.
- GLM-5.2 TP4: -20 % with the gate.
- GB10 GLM-5.3: decode is 26 % faster without the pool (32.8 vs 29.4 tok/s) [M].

## 2. The kernels (`kernels/MANIFEST.md`, `kernels/sass/`, `docs/sm120-alu-subunits.md`)

**What each kernel does**

| kernel | operation | notes |
|---|---|---|
| `moe_w2_mm` | 2-bit MoE GEMM | codes decoded in-register with one `PRMT` byte-LUT (`PRMT_LUT_WORD 0x4838B8C8` = e4m3 bytes for -4, -1, 1, 4), then `QMMA.SF` block-scaled tensor-core MMA with the UE8M0 scales fed per k32. Regcount 64, 4 CTA/SM; per-K cubins for 512-7168. Op rel-err 1-3e-3, deterministic. Activations are FP8 with group 128 UE8M0 (per-32 groups lost GSM8K: 95.5 % vs 97.0 %) |
| `moe_w4_mm` / `moe_w4q_mm` | FP4 pool GEMMs | `w4q` decodes radix-5 with a magic divide `(x*0x334)>>12` |
| AFRAG (`mc4afrag`) | prefill variant | fragment-major activations, one `LDG.128` per A fragment. 1.30x / 1.27x on the GEMM, +12 % end-to-end prefill [M] |
| EXL3 decode-wave canons (`kernels/sass/exl3-wave-m8/`) | upstream's **EXL3 trellis** path | decode each union expert's trellis once per step, then HMMA over 8 or 16 token rows. Fixed groups of 6 experts, group count `G = ceil(k·T/6)`; G = T overflowed at top-8. Empty-group guard prologue for CUDA-graph capture. Layer cost is almost flat in M: M = 1 -> 16 is +10 %; per token 13.88 µs at M = 8, 7.34 µs at M = 16. GLM-5.2 TP4 C4 + MTP: 59-60 -> 194-203 tok/s [M] |
| x4 prefill canon | EXL3 prefill | 4 slabs × 16 rows per decode: 2.09-2.21x vs `exl3_moe_dual`; cold prefill 870 -> 2051-2152 tok/s [M] |

**The toolchain.** The cubins exist only because the authors built their own SM120 SASS ISA database
(`blackwell-isa`) and assembler (`cubit`); ptxas cannot express the hand scheduling. Their own finding is that the
M = 1 GEMVs are **ALU-issue-bound**: on SM120 the LOP3/PRMT/SHF "B port" is the binding unit (`sm120-alu-subunits.md`
rules 1-6). The CUDA C++ kernels for DS4.1 and Qwen3.8 (`tools/*/…_sm120.cu`, `mma.sync.m16n8k32` e4m3) are a
separate track.

**Portability**

| target | 2-bit LUT GEMV/GEMM (`{-4,-1,1,4}` × 2^e per 32) | EXL3 decode-wave (once-per-union-expert, M ≤ 16) |
|---|---|---|
| sm_86 (3090) | **Easy in CUDA C++ or Triton.** PRMT exists on every arch, and the codebook is exactly int8/int4 and fp16. Ampere has `mma.sync.m16n8k32.s8` (k = 32 = one scale block -> one IMMA per block, fp32 scale-accumulate, the same structure as their QMMA.SF loop) and `m16n8k16` fp16. No FP4 or `QMMA.SF`, so the FP4 pool would be dequant -> fp16. At M = 1 on the GPU this is memory-bound anyway | The idea ports (HMMA m16n8k16 exists on Ampere). exllamav3's own kernels are the base; the SASS does not port |
| Intel Xe2 (B70) | DPAS int8 + `vpshufb`-style byte LUT, straightforward | same idea in ESIMD/SYCL |
| AVX2 CPU (our lane) | **The decode is about one `vpshufb` per 32 codes, then `vpmaddubsw` int8.** About 0.06 ms/expert at ~118 GB/s DRAM-bound [inf], vs 0.104 for the trellis | `tile_accum_i16<M>` already decodes once for up to 8 rows; a 2nd row costs +33 % (research/14) |

SASS, cubins and the `cubit` toolchain are SM120-only. Nothing binary is reusable on sm_86, sm_121 aside (GB10 runs
the sm_120 cubins with `max_abs_diff = 0`), or Xe2.

## 3. Offload and scheduling (upstream)

**GPU is the only compute device.** Host RAM and NVMe are *storage* tiers feeding H2D. There is **no CPU compute
lane** (`moe_w2_store.py`, `moe_w2_delta.py`, `docs/v024-port.md` "BASE cache").

**Base cache** (`VLLM_MOE_W2_BASE_CACHE_GB`). The whole 2-bit base lives on the host and the GPU pool caches hot
experts.
- A miss **zeroes that pair's contribution** in-graph and bumps a counter.
- The runner then fetches every missing expert in one batched pinned H2D and replays the graph. The PRO 6000 does
  51.6 GiB/s, the 5090 only ~26.6.
- The replay iterates to a fixed point for second-order misses. Strict mode is the default; `MISS_TOL=k` is
  approximate.
- The decode KPI is "replay % of steps":

| config | coverage | token hit | decode | label |
|---|---|---|---|---|
| DS4 1x 5090, 11 GiB pool | 15.2 % | 96.5-97.7 % | 27-28 tok/s | [M] |
| DS4 1x 5090, 14 GiB pool | 19.3 % | 98.7-98.9 % | ~31 tok/s | [M] |
| GLM-5.2 | ~51 % | ~91 % | — | [M], README |
| GLM-5.2 (port doc) | ~20 % | ~89 % | — | port doc gives this instead; the two are inconsistent |

**NVMe stores** (`moe_w2_store.py`).
- Per-rank pack files with `(layer·E + expert)·stride` rows, 4 KiB-aligned, plus a JSON sidecar.
- Three backends: pinned, mmap/pread through the page cache, and **tiered** (a pinned MRU arena over the pack; hits
  are zero-copy pinned views, misses `preadv` into the arena slot; O_DIRECT is optional).
- The arena is LRU because "freq-pinning lost on live GLM traces (routing too flat)".
- The hot set persists to `<pack>.heat.json` and preheats on boot (57 GiB in ~35 s).
- **Prefill is scan-flagged.** It may fill free arena slots but never evicts the decode hot set. A batch-size
  heuristic misclassified GLM's 100+-row replay fetches and froze the arena at -66 %.

DS4 on 1x 5090 [M]:

| store | decode | RSS |
|---|---|---|
| pinned | 33.0 tok/s | 42-44 GiB |
| pack only | 25.5 tok/s | 15 GiB |
| pack + 20 GiB arena | 32.8 tok/s | 26-33 GiB |

GLM-5.2 TP2 three-tier: 28.3 tok/s strict, 31.7 at `MISS_TOL=8`; needle passes to 121K [M].

**Prefetch.**
- Draft-affinity: a token -> experts table, roughly a 41 %-recall predictor.
- `moe_w2_looka.py` router-lookahead: measured 71.6 % recall on GLM-5.2. PILOT prefetches the predicted
  non-resident experts on a side stream.

**Speculation.** MTP everywhere (acceptance 2.3-3.0 on GLM-5.2), Eagle3 for Kimi, DFlash/DSpark in the newer
images. MTP also runs under PP.

## 4. GB10 variant (GLM-5.3-Flash, `docs/models/glm-5.3-flash.md`, `CHANGELOG.md`)

All measurements are on one ASUS GX10 (GB10, 48 SM, 121.6 GiB unified memory), v0.3.0, 2026-09-12.

**Setup**
- Source: `zai-org/GLM-5.3-Flash` FP8 (306 GiB). The NVFP4 source was rejected (experts ~146 GiB; double
  quantization).
- Geometry: 45 layers + MTP, H 4096, 288 experts top-8, inter 2048, 34 KDA + 11 MLA layers.
- Experts: GPTQ-coded 2-bit planes. BF16 rest packed into lossless 12-bit "BF12" (-2.4 GiB). FP4 pool 2 GiB
  (~170 experts = 1.4 %). FP4 store on NVMe (146 GB). Gate **off**: it would need a ~12.8 GiB pool.

**Speed and quality** [M]

| | |
|---|---|
| decode, code | 29.4 tok/s, DFlash2 k=7, acceptance 5.5 |
| decode, `effort` high | 24.6 tok/s |
| decode, prose | 12.4-13.5 tok/s |
| bare 2-bit, no speculation | 11.4 tok/s |
| bare 2-bit + MTP k=2 | 22.4 tok/s |
| prefill | 632-643 tok/s, flat to 198K |
| context | 204,800; a 198K prompt has TTFT 313 s |
| HumanEval / HumanEval+ | 97.0 / 93.9 % (vs the original 95.1 / 89.6 on the z.ai API) |
| loop probe | 0 of 16 |

**Memory**
- Model 99.6 GiB; KV 6.2-7.5 GiB; planned 70.9 GiB of 2-bit experts.
- Unified-memory specifics:
  - `torch.cuda.mem_get_info` counts page cache as used: 49.4 GiB "free" vs 117 GiB MemAvailable.
  - `drop_page_cache.py` (`posix_fadvise DONTNEED`) recovered +43.6 GiB.
  - `hostguard.py` kills the container from `/proc/meminfo`, because docker `mem_limit` does not protect the host.
  - Streaming per-layer FP8 -> W2 build.
  - Plain safetensors loader: the Run:ai streamer reads out of order and pushed memory to 121 GiB.
- **NVMe expert streaming was rejected.** They estimate ~620 MiB of distinct experts per token, about 12 GB/s at
  20 tok/s, against a 3-6 GB/s drive. NVMe feeds only FP4 promotions. With `DELTA_PROMOTE=8` the pool turned over
  every 32 ticks, reading 41.7 GiB in 2 minutes [M].

**No GLM-5.3 routing-locality curve** is in the repo. `cache_hit_frac.py` is about the prefix cache.

## 5. The user's repos

| repo | what it is | state |
|---|---|---|
| **0xSero/sglang-moet** = `~/Documents/moet` (same HEAD `cd7377b`; the local copy has uncommitted docs, experiments, `lossless.py`, artifacts) | SGLang port of vLLM-Moet W2 for **4x RTX PRO 6000 (sm_120), GLM-5.2 NVFP4 first**. `moet-kernels` vendors 25 upstream cubins pinned to upstream `cf029984` / cubit `5912400` (`tools/import_vllm_moet.py`, `_vendor/sm120/provenance.json`). 9,891-line SGLang patch on `504570f`: MoE runner, quant method, streaming loader, TileLang DSA indexer for SM120, teacher-replay logprobs, GLM-5.2 NextN | Only **resident W2** works: TP4, graphs, MTP. Host-W2, FP4 promotion and the gate are **not built**. Graph/eager divergence after row 124 (`STATUS.md`). Pure-Python/torch references exist (`w2.py`, `torch/conversion.reference_dequant`), but there is **no non-SM120 compute path**; `cuda_driver.py` fails closed. The 3090 is explicitly excluded (`README.md`, `DESIGN.md`). Local results: DS4 1x PRO 6000 GPQA non-think 69.2 % (vs 71.2 official), 131.5 tok/s; GLM-5.2 GPQA run collapsed to loops (0/198 valid) |
| **0xSero/moet** (TypeScript, Bun) | Engine-agnostic serve and bench harness (GPQA, needle, KLD, TB2.1, DeepSWE, speed sweeps) for vLLM-Moet or sglang-moet on sm120 / 5090 / sm121 profiles | Verified only against a mock server; results directories empty. Two useful items: the DS4 truncated-KL table in §1.4, and `benchmarks/scripts/damage-rank.py` (per-expert W2 damage ranking for static FP4 residency) |

**Neither is a 3090, GLM-5.3 or CPU/NVMe-offload project.** For this box they contribute reference format code and
quality harness ideas, nothing else.

## 6. Transfer to GLM-5.3-Flash on the 3090 box

### 6.0 What our box has that changes the math

- **The CPU lane is decode-issue-bound, not DRAM-bound.**
  - `ft_mul1` I16 runs ~2.1 cycles per 8 weights, the same with L3-resident or DRAM experts (research/14).
  - It reaches 83-91 GB/s against a 136-140 GB/s roof (`glm53/CPU_TIER.md`).
  - A K=2 trellis has the **same number of weights to decode**. Only the bit extraction is cheaper (an estimated
    0-15 % of the uops) [inf].
  - So "fewer bytes per expert" barely moves the CPU lane unless the **format** changes.
- **Quality anchors for GLM-5.3-Flash** (KLD vs BF16):
  - turboderp's chart (research/21): EXL3 3.05bpw 0.0680, EXL3 4.05bpw 0.0345, AWQ int4 0.0565, NVFP4 0.0452;
    noise floor 0.0138.
  - Our only 2-bit receipt: omarchy `GLM-5.3-Flash-EXL3-2.0bpw/results`, a rank-sliced TP4 MCG K2 selective quant,
    KLD **0.438**, top-1 **0.774**, PPL +41 %, quality gate **FAIL**.

### 6.1 The local 2.0bpw checkpoint is not usable

Checked on omarchy with hard timeouts:

| item | value |
|---|---|
| on disk | `~/models/GLM-5.3-Flash-EXL3-2.0bpw` = 34.0 GB |
| index | references 133 files (111.27 GB) |
| **missing** | **all 84 expert shards** `layers/layer-NN-part-{0,1}.safetensors`; 77.44 GB of quantized expert bytes absent. Only `retained/` (33.8 GB BF16 non-expert) and per-layer JSON are present |
| format | `glm53-selective-exl3-tp4-v1`: each projection is split into `rank0..3` trellis/suh/svh/mcg. It is not stock exllamav3, so the 3.05 runtime cannot load it as a drop-in |
| `RELEASE_STATUS.json` | load, serve and MTP *pending*, quality *fail* |

**A 2-bit cold tier therefore needs a fresh stock-layout EXL3 K=2 expert quant** (turboderp pipeline; experts only,
6,328,320 B/expert) before any KLD gate can run.

### 6.2 Simulator results (moetier, `examples/cold2bit.py`) [S]

Every row starts from the last cumulative lever of `examples/levers.py`: prefetch recall 0.7, persistent CPU
workers, fixed time 9 ms, 1910 VRAM slots. VRAM always holds 3.05bpw.

Record sizes and lane costs used:
- 2.0bpw EXL3 = 6,328,320 B (0.668x).
- (3,3,2) = 8,425,472 B.
- vLLM-Moet LUT planes = 7,077,888 B (2.25 bpw).
- Zero-copy cost is scaled by bytes, because it is PCIe-bound.
- The CPU cost is stated per row.

| scenario | RAM slots | C1 | C2 | C4 | C1 ms: fixed + max(gpu, cpu) | CPU experts/tok | NVMe/tok (GB) | picks at 2 bits (C1) |
|---|---|---|---|---|---|---|---|---|
| all levers, 3.05bpw everywhere | 5087 | **32.44** | 39.44 | 45.25 | 9.0 + max(12.81, 21.12) | 149.3 | 16.9 (0.16) | 0 |
| + NVMe tail stored at 2.0bpw EXL3 (RAM stays 3.05) | 5087 | **34.25** | 42.15 | 47.49 | 9.0 + max(13.22, 19.58) | 157.3 | 17.6 (0.11) | ≥ 0.052 |
| RAM+NVMe 2.0bpw EXL3, CPU decode-bound (0.104 ms) | 7616 | 37.77 | 44.07 | 47.16 | 9.0 + max(14.16, 16.07) | 138.1 | 3.7 (0.02) | **0.454** |
| RAM+NVMe 2.0bpw EXL3, CPU -15 % (cheaper K=2 extract) | 7616 | 39.39 | 46.97 | 51.08 | 9.0 + max(12.48, 15.58) | 157.4 | 3.7 (0.02) | 0.495 |
| RAM+NVMe 2.0bpw EXL3, CPU bytes-proportional (0.069, upper bound) | 7616 | 41.63 | 51.44 | 57.33 | 9.0 + max(11.24, 14.18) | 180.0 | 3.7 (0.02) | 0.553 |
| RAM+NVMe (3,3,2) EXL3 2.67bpw, CPU 0.104 | 5720 | 34.09 | 41.60 | 46.81 | 9.0 + max(13.02, 19.72) | 154.2 | 12.4 (0.10) | 0.509 (at 2.67 bits) |
| RAM+NVMe vLLM-Moet 2-bit LUT planes (2.25bpw), CPU DRAM-bound 0.060 | 6809 | **43.12** | 52.61 | 60.83 | 9.0 + max(10.92, 12.60) | 167.9 | 6.5 (0.05) | 0.520 (RTN 2-bit) |
| reference: 3.05 everywhere with a hypothetical 0.060 ms CPU kernel | 5087 | 37.18 | 44.64 | 52.02 | 9.0 + max(11.15, 16.08) | 146.8 | 16.7 (0.16) | 0 |

How to read it:
- Even with an unchanged CPU cost, 2-bit RAM+NVMe gains about 5 tok/s at C1. That gain is almost all **removed NVMe
  waits**: NVMe falls from 16.9 to 3.7 experts per token as the RAM tier grows from 5,087 to 7,616 slots. Fewer
  bits do not make the CPU lane faster.
- The price is that **about half of all routed picks run at 2 bits.**
- The 2-bit-cold idea depends on what share of picks VRAM serves. Ours is only 45-55 % at 1,510-1,910 slots,
  versus vLLM-Moet's 96-99 % token hit on DS4. That is why their recipe is acceptable for them and not for us.
- KLD estimate for half the picks at 2 bits [inf]: 0.068 + 0.5 × (0.2 to 0.44 - 0.068) ≈ **0.13-0.25**, which is
  2-4x EXL3 3.05 and not "EXL3-class".
- NVMe-tail-only stays at 5-15 % of picks at 2 bits [inf]. The 5.2 % is the lower bound; NVMe fills linger in RAM
  as 2-bit copies until they are evicted or re-read at 3.05. Estimated ΔKLD ≈ +0.02-0.05.

Speculative decoding (lossless), C1, 3.05 everywhere, all levers [S]:
- New sim knobs: `window` = k+1 verified trace tokens, `accept` = mean tokens per step, `draft_ms`.
- `fixed(window)` reuses the 1/2/4-sequence calibration, which is pessimistic for one sequence.
- CPU cost per extra token on an expert is 0.04 ms (C004 / C060a).

| speculative what-if | window | accept | draft ms | VRAM slots | tok/s | ms/step: fixed + max(gpu, cpu) | CPU experts/step |
|---|---|---|---|---|---|---|---|
| none | 1 | 1.0 | 0 | 1910 | 32.44 | 9.0 + max(12.81, 21.12) | 149 |
| MTP k=2 (GB10 acceptance 2.81-2.95) | 3 | 2.85 | 3 | 1910 | **46.15** | 16.5 + max(33.4, 41.9) | 318 |
| MTP k=3 (GB10 acceptance 3.62-4.0, 2-bit drafter) | 4 | 3.6 | 4 | 1910 | **47.03** | 20.0 + max(45.0, 52.1) | 401 |
| DFlash2 k=7, code (acceptance 5.5), 2.6 GB drafter in VRAM | 8 | 5.5 | 4 | 1635 | 41.03 | 34.0 + max(89.8, 95.4) | 708 |
| DFlash2 k=7, prose (acceptance ~2.4, inferred) | 8 | 2.4 | 4 | 1635 | 17.9 | 34.0 + max(89.8, 95.4) | 708 |

- The expert union per 3-token window is 318 CPU experts vs 3 × 149. Reads are shared, but routing is flat enough
  that k=2-3 is the sweet spot on our CPU lane.
- Wide windows (k=7) **lose** on prose and only tie on code. GB10 profits from k=7 because its GPU reads planes at
  HBM speed.
- Caveats: the acceptance figures come from GB10's 2-bit target. Our MTP layer (layer 45) is itself a 288-expert
  MoE layer whose experts need placement. `draft_ms` is assumed, and a single-sequence verify fixed time is
  probably below `fixed(T)`.
- **MTP k=2 is the one modeled lever that clears 45 tok/s at C1 with exact output.** Combined with fused non-MoE
  (already in the base) and a better VRAM hit rate, it puts 50 tok/s in reach without lossy changes [S].
- research/14 noted DSpark is excluded by the DSV41 serving rules. Check whether the same applies to GLM-5.3 MTP.

### 6.3 A residual (Δ) format on top of EXL3 trellis (6b)

**Feasible: upstream did exactly this.** It is the "Δ-pool, pack v3" (`moe_w2_cubit.py` comments on
`_EXL3_DELTA_PACK`; `moe_w2_exl3.delta_slot_geom`, `forward_topk_dual`, `forward_topk_unified`; tests
`tools/test_moe_w2_exl3_delta.py`):
- **Base:** EXL3 (2,2,1), i.e. gate K2, up K2, down K1 (1.67 bpw), 3INST codebook.
- **Δ:** an independent EXL3 quant of the **residual W - Ŵbase** per expert at (2,2,3), MUL1 codebook, with its own
  Hadamard `suh`/`svh`.
- **Combination:** summed pre-activation, `(Wb+Δ)x = Wb x + Δ x`, then SwiGLU. Non-pooled pairs point their Δ `svh`
  at a zero vector, which zeroes the Δ contribution without a kernel flag.

Their numbers [M, DS4 vs its FP4 checkpoint, block output rel-err]:

| config | rel-err | note |
|---|---|---|
| base EXL3 at 2.01 bpw | 0.26-0.30 | |
| base + Δ | ~0.06-0.11 | |
| Δ slot | 6.03 MiB | vs 12.75 MiB for an FP4 slot |
| `exl3_decode_wave_dual` decode | 2.0x over six mgemm calls | |

The GLM-5.2 serving recipe (`bench/recipes/glm-5.2-exl3/pro6000x4-tp4-m16.yaml`) ships the **base only**
(`DELTA_GB=0`, gate off). In-serving quality evidence is GSM8K-200 93.0 -> 93.5 % (route on vs off) and a
prompt-logprob KLD at the cross-boot floor (0.2887). That is route parity, not quality vs the BF16 model.

**For us it does not pay:**
1. Every base+Δ expert decodes **two trellises**. Our CPU lane and the 3090 EXL3 kernels are decode-issue-bound, so
   cost roughly doubles for the experts that get the Δ.
2. At equal total bits, a two-stage residual quant is worse than a direct quant: 2.0 + 1.05 bits < 3.05 direct.
   EXL3 also has integer K per tensor, so "2.0 + Δ to reach 3.05" means a Δ of ~1 bpw at K=1, the weakest trellis
   rate.
3. Its only advantage is **progressive upgrade without re-reading the base**. That helps when the base is already
   resident where the Δ is applied. On our box a VRAM-hot expert at 3.05 direct is both better and cheaper than
   2.0 base + Δ, and RAM-tier experts would have to stream the base over PCIe or decode it twice on the CPU.

Verdict: **no**, unless a future GPU lane becomes bandwidth-bound and VRAM holds the 2-bit base of *all* experts.
That does not fit: 12,096 × 6.33 MB = 76.6 GB.

### 6.4 Kernel and scheduling ideas we lack (6c)

1. **Router-lookahead predictor, done their way.** Feed layer L's MoE input (post-attention LN) to layer L+1's
   router in-graph. They measured **71.6 % vs 41.3 %** recall. Our prefetch lever assumes 0.7 recall: build this
   predictor, then measure recall on our trace with the same counter design (LOOKA = counters only, PILOT = act).
2. **Scan-resistant prefill.** Prefill fills free RAM/VRAM slots but never evicts the decode hot set; mark it with a
   caller flag, not a batch-size heuristic. Our full-layer prefill staging and RAM LRU should get the same rule.
3. **Persist and preheat the hot set.** Write VRAM clock and RAM LRU ownership to a heat file and preload before
   graph capture. That cuts the cold-start replay/NVMe phase after every restart.
4. **A per-step KPI line** in the serve log (replay %, missing pairs per step, RAM-hit %). Upstream calls pool size
   "the dominant knob" and sizes from this line.
5. **Union-expert decode-once at M ≤ 16** for the GPU lane under MTP verify and C2/C4. The layer is nearly flat in M
   (+10 % from M = 1 to 16), and a fixed-shape group builder keeps it graph-capturable (watch the top-8
   `G = ceil(k·T/6)` overflow). Check what exllamav3's MoE kernel does at M = 3-12 on the 3090 before writing
   anything.
6. **If we ever accept lossy tiers: a need-based precision policy plus a confidence gate.** Promote only on
   low-confidence tokens, cap promotions per fire at 64, and iterate to a fixed point. On our box a fire costs a
   full extra step plus NVMe reads of 3.05 records (~134 × 9.47 MB ≈ 1.3 GB ≈ 53 ms). At a 16 % fire rate that is
   +8 ms per token, which eats the 2-bit-cold gain. Not worth it here [inf].
7. **A CPU-native cheap-decode format** (vLLM-Moet LUT planes, or similar) is the only way to make the CPU lane
   DRAM-bound: 0.104 -> ~0.06 ms/expert, a C1 ceiling of 37-43 tok/s [S]. At 2 bits it fails quality. A 3-bit
   symmetric LUT + GPTQ would be ~10.2 MB/expert, more bytes than EXL3 at worse fidelity. Not worth pursuing
   without a KLD study.
8. **GB10 operational notes that apply to omarchy:**
   - `mem_get_info` counts page cache as used; budget from MemAvailable.
   - Use `posix_fadvise(DONTNEED)` for stale checkpoint pages.
   - A host-memory guard that does not rely on cgroup limits.

### 6.5 Recommended next steps

1. Add MTP to the lever table properly:
   - Measure GLM-5.3-Flash MTP k=2 acceptance on our 3.05 target. It needs the layer-45 experts placed; budget
     ~288 × 9.47 MB.
   - Measure verify fixed time at T = 3 for one sequence.
   - Rerun `examples/cold2bit.py` speculative rows with those numbers.
2. Implement the router-lookahead counters (LOOKA) in the 3090 engine and report recall.
3. Only if the 50 tok/s target is still short: quantize **stock-layout EXL3 K=2 experts** (experts only; the
   omarchy 2.0bpw directory is unusable) and run the NVMe-tail-at-2-bit variant through the forced-token decode-KL
   gate (C052i method; noise floor KL 0.01-0.024).
4. Do not pursue RAM-tier 2-bit, the EXL3 Δ format, or the FP4 delta tier on this box.

## 7. moetier changes made for this study

Minimal, documented, and backward compatible: the baseline sim is unchanged, 23.37 / 30.06 / 33.42 before and
after.

- `moetier/spec.py`: optional `recipe.tiers.{ram_expert_bytes, nvme_expert_bytes}` (default: the model's
  `expert_bytes`). RAM slots are counted at `ram_expert_bytes`; VRAM keeps `expert_bytes`.
- `moetier/plan.py`: an NVMe miss pushed to VRAM moves the NVMe record (`nvme_expert_bytes`).
- `moetier/sim.py`:
  - The NVMe channel reads `nvme_expert_bytes`; prefill link and NVMe terms use per-tier sizes.
  - `run(..., window, accept, draft_ms)` is a speculative-decoding what-if: `window` consecutive trace tokens per
    stream form one step (their expert union, m tokens per expert), `accept` tokens are credited per stream, and
    `draft_ms` is added per step.
- `registry/recipe/glm53-rtx3090-55g-nvx4-cold2b.json`: a what-if recipe (2.0bpw RAM and NVMe records).
- `examples/cold2bit.py`: the tables in §6.2. Runtime is about 3 minutes on the Mac.

Lane costs for 2-bit formats go through `lane_overrides`, explicitly. Whether a lane scales with bytes depends on
the kernel's bound, and for the AVX2 trellis it does not.
