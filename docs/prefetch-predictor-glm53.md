# Layer-ahead expert prediction for GLM-5.3-Flash decode (route-only predictors)

Question: can the host predict layer l+1's routed experts from routes (and router weights) alone, well enough to
drive layer-ahead NVMe prefetch on the 3090 + 55 GB + NVMe RAID0 tier? And is a hidden-state router-lookahead
worth a GPU run on top?

Measured offline on CPU (this Mac), 2026-10-07, track HOM-272. No GPU or container was touched.
Script: `examples/prefetch_predictor.py`. Raw results: `docs/prefetch-predictor-glm53.results.json`.

## Verdict

- **Route-only predictors cannot drive NVMe prefetch.** On all picks the best one (a learned mix, `lrw`) recalls
  0.40 of the true top-8 at N = 8, and 0.62 at N = 24. On the picks the moetier ledger actually serves from NVMe,
  the same predictor recalls **0.07 at N = 8 and 0.20 at N = 24**. Only 4-9 % of the reads it issues are used.
- In the simulator, prefetch through the FIFO NVMe queue **loses speed at every budget**: C1 drops from 21.4 to
  21.1 (N = 8) and to 14.4 (N = 24) on the N116 test stream. If prefetch reads only use idle channel time, it
  gains **+0.05 to +0.2 tok/s**. Do not build it with these predictors.
- The reason: 90-95 % of NVMe misses are re-references 64-1,023 tokens apart. Short-range history
  can't see them, and transition tables mostly predict hot experts that are already resident.
- **Hidden-state router-lookahead is the only predictor worth testing**, as an in-engine counter (LOOKA) during
  a GPU run that is already scheduled, not as a separate capture campaign. The simulator gives the lever at most
  **+3.7 tok/s at C1** (23.4 -> 27.1 at recall 1.0), and +2.2 at the README's assumed recall of 0.7. Build the
  prefetch path (PILOT) only if LOOKA shows recall of at least ~0.4 on NVMe-served picks (≥ +1.0 tok/s in the sim).
- The README lever row "+ layer-ahead NVMe prefetch (recall 0.7)" is an **assumption that holds only for a
  hidden-state predictor**. Route-only predictors reach about 0.05 at an affordable read budget.

## Data

| trace | content | used as |
|---|---|---|
| N116 `s1c_55g/nv_trace.npz` (omarchy) | 1,362,312 rows = 32,436 decode steps × 42 layers; `ids` [rows, 8] int16, `weights` [rows, 8] f32 (sum 2.5 per row = routed scaling; not sorted); `step`, `layer`, `model_layer_offset=3`; bsz 1 (C2/C4 ran serially in exact mode) | train on the first 70 % of steps, test on the last 30 % (9,731 steps) |
| N116 `s1d_50g/nv_trace.npz` | 33,954 steps. **77 % of its steps are identical to s1c steps** (same prompts, greedy) | not used: it would leak the test set |
| G002 `traces/glm53-g002-decode.npy` | [9,250, 42, 8], 4 segments, no weights | independent test set. Tables are trained on all of N116 s1c, so this tests generalization across prompts |

Duplicates inside s1c: 27.7 % of steps repeat an earlier step (C2/C4 replays). They all fall in the train part.
**0 % of the test steps duplicate a train step.**

## Predictors

All of them are computed at layer l of token t from what the host already has: layer l's picks and weights for
this token, and every layer of earlier tokens.

| id | score for candidate j of layer l+1 | host cost per layer |
|---|---|---|
| a | temporal: Σ_k 0.3^(k-1)·[j ∈ S(t-k, l+1)], ties broken by weight. N = 8 is exactly the previous token's set | update a 288-float EMA |
| b | transition: Σ_{i ∈ S(t,l)} P(j ∈ S(l+1) \| i ∈ S(l)), P learned on the train split (288×288 per layer, add-1 prior smoothing) | sum 8 table rows |
| bw | as b, with each row weighted by i's router weight | as b |
| c | union with budget N: N/2 from a, the rest from b | a + b |
| lr | logistic mix of b, two-hop b2 (l-1 -> l+1), prev, prev2, EMA, log prior. P comes from the first 50 % of steps and the mix is fit on 50-70 % (out of sample), 1,500 tokens × 41 layers | ~16 row gathers + 288-wide axpy, ~10 µs |
| lrw | lr + bw + the previous token's weight of j | as lr |

Fitted `lrw`: EMA 10.7, prev −9.0, prev2 −2.3 (so "seen within the last ~3 tokens" is the main signal), bw +8.7
with b −1.2 (weights help the table term), b2 +0.7.

## 1. Recall and precision per budget N (offline)

Recall = share of the true layer l+1 top-8 covered. Precision = share of the N predictions that are picked.
Layers 1-41, since layer 0 has no layer-ahead source. "Cold tail" = picks outside the 6,300 hottest keys of the
train split.

N116 test (9,731 tokens, weights available):

| predictor | N=8 rec / prec | N=12 | N=16 | N=24 | N=32 | cold-tail recall N=8 / 24 |
|---|---|---|---|---|---|---|
| a temporal | 0.325 / 0.325 | 0.398 / 0.265 | 0.441 / 0.220 | 0.509 / 0.170 | 0.563 / 0.141 | 0.301 / 0.475 |
| b transition | 0.249 / 0.249 | 0.315 / 0.210 | 0.369 / 0.184 | 0.451 / 0.150 | 0.514 / 0.129 | 0.022 / 0.070 |
| bw transition, weighted | 0.263 / 0.263 | 0.330 / 0.220 | 0.383 / 0.192 | 0.466 / 0.155 | 0.528 / 0.132 | 0.032 / 0.093 |
| c a ∪ b split | 0.347 / 0.347 | 0.438 / 0.292 | 0.501 / 0.251 | 0.596 / 0.199 | 0.656 / 0.164 | 0.217 / 0.411 |
| lr learned mix | 0.386 / 0.386 | 0.469 / 0.313 | 0.528 / 0.264 | 0.612 / 0.204 | 0.670 / 0.168 | 0.239 / 0.429 |
| **lrw learned mix, weighted** | **0.396 / 0.396** | 0.479 / 0.319 | 0.536 / 0.268 | 0.620 / 0.206 | 0.676 / 0.169 | 0.245 / 0.437 |

G002 cross-set (tables and mix fit on N116, no weights):

| predictor | N=8 | N=12 | N=16 | N=24 | N=32 |
|---|---|---|---|---|---|
| a | 0.336 | 0.407 | 0.448 | 0.513 | 0.567 |
| b | 0.201 | 0.257 | 0.303 | 0.378 | 0.439 |
| c | 0.329 | 0.419 | 0.481 | 0.574 | 0.634 |
| lr | 0.385 | 0.467 | 0.522 | 0.602 | 0.657 |

Router weights add about 1 point (lr -> lrw). The learned mix transfers across prompt sets without loss
(0.386 -> 0.385). The best route-only recall at N = 8 is 0.39-0.40, against 0.716 measured for vLLM-Moet's
hidden-state lookahead on GLM-5.2.

**The static cold tail is the wrong proxy.** It shows temporal recall of 0.30-0.38. Those picks are mostly
RAM-LRU hits, because experts recently used are resident. Table 2 uses the real tiers instead.

## 2. Recall on the picks the ledger serves from NVMe, read cost, and C1 effect

Setup: moetier's `sim.run` with the predictor in the loop (`prefetch` hook), recipe `glm53-rtx3090-55g-nvx4`
(1,510 VRAM / 5,087 RAM slots, exclusive RAM LRU, exact stall), conc 1. Modes:
- **LOOKA**: counters only, so the ledger evolves exactly as in the baseline.
- **FIFO PILOT**: predicted NVMe-tier keys are read through moetier's FIFO channel, ahead of the next layer's
  demand reads. Unused reads are dropped after their layer and never pollute RAM.
- **idle PILOT**: only as many reads as fit in the channel's idle time within ~1 ms (one layer). This models a
  runtime where demand reads preempt prefetch, so false positives cost ~0 time.

Hook check: prefetching the true NVMe picks ("oracle") gives 27.09 on G002, which exactly matches the CLI at
`policy.prefetch.recall=1.0`.

N116 test stream (baseline 21.37 tok/s, 29.4 NVMe reads/token, oracle 26.16):

| predictor | N | recall, NVMe picks | NVMe reads/token | used/token | read GB/token (wasted) | C1 FIFO | C1 idle-only |
|---|---|---|---|---|---|---|---|
| a | 8 / 16 | 0.000 / 0.000 | 0 | 0 | 0 | 21.37 | - |
| lr | 8 | 0.050 | 20.4 | 1.4 | 0.19 (0.18) | 21.05 | 21.41 |
| lr | 16 | 0.113 | 63.6 | 3.2 | 0.60 (0.57) | 19.03 | 21.49 |
| lr | 24 | 0.182 | 130.2 | 5.2 | 1.23 (1.19) | 14.12 | 21.54 |
| lrw | 8 | 0.066 | 21.2 | 1.9 | 0.20 (0.18) | 21.06 | 21.43 |
| lrw | 16 | 0.131 | 63.8 | 3.7 | 0.60 (0.57) | 19.08 | 21.51 |
| lrw | 24 | 0.200 | 126.8 | 5.7 | 1.20 (1.15) | 14.44 | 21.57 |
| lrw | top-2 NVMe-tier per layer | 0.143 | 74.1 | 4.1 | 0.70 (0.66) | 20.71 | 21.59 |

G002 segment 0 (the CLI's C1 stream; baseline 23.37, 20.2 NVMe reads/token, oracle 27.09):

| predictor | N | recall, NVMe picks | NVMe reads/token | used/token | read GB/token | C1 FIFO | C1 idle-only |
|---|---|---|---|---|---|---|---|
| a | 8 / 16 / 32 | 0.000 / 0.000 / 0.021 | 0 / 0 / 41 | 0 / 0 / 0.4 | 0 / 0 / 0.39 | 23.36 / 23.36 / 20.71 | - |
| b | 8 | 0.122 | 107.7 | 2.4 | 1.02 | 16.97 | 23.43 |
| b | 32 | 0.306 | 435.7 | 6.1 | 4.13 | 5.54 | 23.47 |
| lr | 8 | 0.048 | 21.5 | 1.0 | 0.20 | 23.06 | 23.43 |
| lr | 16 | 0.113 | 78.3 | 2.2 | 0.74 | 19.42 | 23.44 |
| lr | 32 | 0.241 | 250.6 | 4.8 | 2.37 | 9.09 | 23.46 |
| lr | top-1 / top-2 NVMe-tier per layer | 0.070 / 0.113 | 39.3 / 75.9 | 1.4 / 2.2 | 0.37 / 0.72 | 22.71 / 22.49 | 23.41 / 23.46 |

At the read budgets the idle channel can absorb (~40 reads/token), effective recall is 0.04-0.05.

Why this fails: reuse distance of each NVMe-served pick, i.e. tokens since that key was last picked:

| stream | never seen | < 64 | 64-255 | 256-1,023 | ≥ 1,024 |
|---|---|---|---|---|---|
| N116 test | 1.4 % | 1.0 % | 74.1 % | 20.9 % | 2.6 % |
| G002 seg 0 | 9.2 % | 0.0 % | 59.4 % | 30.6 % | 0.8 % |

A pick re-used within ~64 tokens is still in RAM by construction, so temporal predictors never point at an NVMe
key. Misses are experts last used 64-1,000 tokens earlier and then evicted. Cross-layer tables rank by
co-occurrence, which favours the hot set. Among non-resident candidates they are barely better than chance:
2-5 % of reads are used.

## 3. In moetier lever terms

Planner prefetch (`policy.prefetch.recall`, no false-positive cost), CLI on G002, conc 1:

| recall | 0 | 0.05 | 0.12 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 tok/s | 23.37 | 23.49 | 23.66 | 23.87 | 23.98 | 24.14 | 24.42 | 24.77 | 25.13 | 25.55 | 26.05 | 27.09 |

The gain is about +0.37 tok/s per 0.1 of recall. The whole lever is worth +3.7 tok/s (+16 %) at C1.

| predictor | effective recall (NVMe picks) | read budget | false-positive cost | C1 |
|---|---|---|---|---|
| route-only, best (lrw/lr), idle bandwidth | 0.04-0.05 | ~0.37-0.43 GB/token issued, ~97 % wasted | ~0 time if prefetch is preemptible, else large | 23.4 -> 23.46 (N116: 21.37 -> 21.59) |
| route-only, FIFO, N = 8 | 0.05-0.07 | 0.2 GB/token, 92-95 % wasted | +~8 ms/token of channel time | **−0.3** |
| route-only, FIFO, N = 24 | 0.18-0.20 | 1.2-1.5 GB/token | channel saturates (24 GB/s => 50-60 ms/token) | **−7 to −10.5** |
| hidden-state lookahead (assumed recall 0.7 and NVMe-read precision 0.7) | 0.7 | G002: ~20 reads/token = 0.19 GB (~6 wasted, 0.06 GB). N116: ~29 reads = 0.28 GB (~9 wasted) | 2-3.5 ms/token of channel at FIFO; ~0 when preemptible | 23.4 -> **25.55** (sim without FP cost); FIFO FP cost ~ −0.1 |

False-positive bandwidth. Every wasted read costs 9.47 MB, or 0.39 ms of the 24 GB/s channel:
- A FIFO queue puts wasted reads in front of the next layer's demand reads. This is why every FIFO row above
  loses speed.
- A preemptible queue limits the waste to bandwidth the channel has idle: about 35 ms/token at C1, ~90 reads.
- Wasted reads must land in a short-lived ring (like S2's 32-slot ring), not in the RAM LRU. Otherwise they evict
  experts that are re-used.

## 4. Recommendation

1. **Do not implement a route-only predictor for NVMe prefetch.** Its best case is +0.05-0.2 tok/s with
   preemptible reads, and it is net negative with FIFO reads.
   - If the runtime needs a host-side predictor for something else, take `lrw`: a 288×288 fp32 table per layer
     (13.6 MB total), 8 weighted row gathers, a two-hop term and a 288-float EMA, about 10 µs per layer. It
     recalls 0.40 at N = 8 and 0.54 at N = 16 of all picks.
   - One possible use is a RAM -> VRAM push or zero-copy hint for RAM-tier experts. The simulator does not model
     that lever, so this is unmeasured.
2. **Build the prefetch plumbing once, with counters, and feed it the hidden-state predictor.** That means
   per-layer predicted ids, demand-first preemptible reads, a drop-after-layer landing ring, and LOOKA counters for
   recall on NVMe-served picks and used/issued reads.
3. **Hidden-state router-lookahead: yes, as an in-engine LOOKA counter during the next scheduled N119/S-series GPU
   run, not a dedicated capture.**
   - Method: layer l+1's router (288×4096) applied to layer l's post-attention MoE input, on the GPU. That is one
     GEMV of ~2.4 MB per layer, a few µs. Its top-N goes back on the per-layer host sync the engine already does.
   - Gate: it must recall at least 0.4 on NVMe-served picks. That is worth ≥ +1.0 tok/s C1 in the sim (+1.4 at
     0.5, +2.2 at 0.7), or ~4 ms/token of the measured S2 `nvme_wait` of 10.85 ms/token.
   - Risk: 0.716 is an all-pick number from GLM-5.2. NVMe misses are rare, low-margin experts, and route-only recall
     fell from 0.39 on all picks to 0.05 on misses. Hidden-state recall will fall less, because it is
     token-specific, but by an unknown amount. That is exactly what LOOKA measures.
4. The miss stream is mostly a capacity and replacement effect (reuse distance 64-1,000 tokens), not an
   unpredictable one. A frequency-aware RAM policy instead of pure LRU may cut misses directly. That is untested
   and a separate study.

## Caveats

- Tiers come from moetier's ledger: frequency seed, VRAM clock, exclusive RAM LRU, 1,510 VRAM slots. The live
  runtime differs. S1 measured 49.7 NVMe reads/token and S2 33.4, against the sim's 20-29.
- Only C1, one-layer lookahead. Layer 0 is never prefetched: it could be predicted from layer 41 of the previous
  token, but that was not measured.
- The N116 stream is one concatenation of requests, and history is not reset at request boundaries (a few hundred
  boundaries in ~32k steps). G002 resets history at segment starts.
- In the idle-only model, a prefetch read may start only if it finishes within 1 ms of channel idle time. That is
  generous to the predictor, because real preemption has granularity.

## Reproduce

```bash
scp omarchy:freetoken-exl3/runs/N116-glm53-nvme/s1/s1c_55g/nv_trace.npz /tmp/s1c/
python3 examples/prefetch_predictor.py --n116 /tmp/s1c/nv_trace.npz --sim --out docs/prefetch-predictor-glm53.results.json
# ~25 min on an M-series Mac (offline part ~3 min); needs numpy + scikit-learn
python3 -m moetier sim glm53-rtx3090-55g-nvx4 --trace traces/glm53-g002-decode.npy \
    --segments traces/glm53-g002-decode.segments.npy --set policy.prefetch.recall=0.05 --conc 1
```
