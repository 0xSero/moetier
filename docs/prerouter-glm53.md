# Edge0 prerouter on GLM-5.3-Flash: next-token expert prediction for NVMe prefetch (N134, phase B gate)

Question: does an Edge0-style prerouter (arXiv 2609.18063, github.com/Edge0-AI/Edge0, Apache-2.0) predict the
GLM-5.3-Flash experts that our offload stack serves from NVMe well enough to drive an exact prefetch one token ahead?
Gate: recall on NVMe-served picks of at least 0.4, at an acceptable number of extra reads.

**Gate result: FAIL.** The next-token heads recall 0.09 (N = 8), 0.16 (N = 16) and 0.29 (N = 32) of the picks the
engine served from NVMe at 55 GB. At the 16 GB ledger budget they recall 0.12 / 0.21 / 0.33. At N = 16 they issue
100 reads per token (55 GB) and 320 reads per token (16 GB), and 92-88 % of those reads are wasted. Phase C
(integration) is not started.

The same data shows that today's same-token router lookahead (`GLM53_NV_PREFETCH=1`) already recalls 0.55 (N = 8)
and 0.75 (N = 16) of the NVMe-served picks. That is far above what the route-only study assumed.

Track HOM-272, 2026-10-08/09. Code: glm53-flash-offload branch `n134-prerouter` (`prerouter/`), this repo
`examples/prerouter_eval.py`. Raw numbers: `docs/prerouter-glm53.results.json`.

## What Edge0 does (from the paper and `python/src/edge0/prerouter`)

- The head owned by layer N runs at token t. Its input is layer N's MoE input (the post-attention norm output)
  concatenated with two one-hots: the top-k layer N routed at t and at t-1. It predicts layer N+1's routing at
  token t+1, so the SSD reads for t+1 overlap token t.
- Head = fc1 (512) -> exact erf GELU -> fc2, plus a bias-free linear path on the same features. The linear path is
  warm-started from the next layer's router. Output = router logits.
- Edge0 uses the prediction AS the routing (lossy), and recovers quality with a LoRA trained on the student path plus
  on-policy distillation.
- The public repo has no training code. Phase 1 is described only as "distil the heads from the next layer's true
  router".
- Their own numbers: adjacent tokens share about 25 % of a layer's experts. Every same-token variant they tried did
  worse than plain LRU in their engine, because of per-layer synchronization.

## Our port

- **Heads.** fc1 512, erf GELU, fc2 initialised at 0, and `linear_init` = [W_router(c) | 0 | 0] on the 4096 + 2·288
  features. Selection = top-k of sigmoid(logit) + `e_score_correction_bias`, as GLM's noaux_tc router does.
- **Loss.** BCE between sigmoid(head) and sigmoid(true router logits of the consumer at the target token), plus a
  listwise CE of the true top-8 under the selection score with τ = 0.05. AdamW 1e-3, one-cycle schedule, 6 epochs,
  batch 512, best validation epoch kept.
- **Variants.** `edge0` is the paper's head (owner c-1, target t+1). `self` uses owner c (features of the consumer
  layer itself at t, target t+1). `same` is a trained same-token head (owner c-1, target t), i.e. Pre-gated MoE.
- **Baselines.** `router_same` = router(c) on z[t, c-1], which is today's `GLM53_NV_PREFETCH`. `router_next` = the
  same router applied to t+1 (the head at initialization). `router_self` = router(c) on z[t, c] for t+1, a temporal
  baseline. `ro_*` are the route-only predictors from `prefetch_predictor.py`, trained on the train split here.
- **Capture.** A copy of the hom272 `glm53a` engine with a capture hook (`prerouter/capture_*.patch`).
  - Per decode token and MoE layer it records the MoE input row (fp16), the routed top-8 ids and weights, and the
    tier each pick had when the host planned it (0 VRAM, 1 RAM, 2 VRAM ring, 3 in flight, 4 new NVMe read).
  - Config: shipped `nvme3` config, `GLM53_NV_PREFETCH=0` (pure demand residency), 55 GB, C1, temperature 1.0,
    top-p 0.95, natural-length completions, 161 prompts covering code, chat, reasoning, writing, knowledge,
    multilingual, structured, multi-turn, long context and the panel.
  - Checks: tier records align 1:1 with decode calls (layer match 1.000, key match 1.000). The fp32 router on the
    captured rows reproduces 99.9 % of the captured picks.
- **Data actually used.** 54,442 decode tokens (18 GB). That is 15 requests, stopped early on the coordinator's
  request.
  - The split is by prompt: train 11 prompts (36.8k tokens incl. val), test 4 prompts (9,680 tokens: knowledge, code,
    and 2 panel prompts).
  - Consumers 0 and 41 are excluded. omarchy went unreachable before layer 41's rows were copied off.
  - The 16 GB engine-tier capture did not run. 16 GB numbers come from the moetier ledger
    (`budget.ram_gb=16`: 971 RAM slots).
- **Training ran off-box on an M1 Max (MPS).** The per-layer rows were streamed from the RAID, about 30 min for all
  42 layers × 6 variants.

## Results (test prompts, consumers 1-40)

Recall on picks the ENGINE served from NVMe (55 GB, captured tiers; 13.1 % of picks, 42.9 per token):

| predictor | lead | rec@8 all | rec@8 NVMe | rec@16 NVMe | rec@32 NVMe | rec@16 RAM |
|---|---|---|---|---|---|---|
| edge0 (paper head) | 1 token | 0.277 | 0.087 | 0.165 | 0.286 | 0.347 |
| self (owner = consumer) | 1 token | 0.290 | 0.085 | 0.166 | 0.292 | 0.362 |
| router_next (head at init) | 1 token | 0.225 | 0.029 | 0.092 | 0.203 | 0.271 |
| router_self (temporal) | 1 token | 0.251 | 0.000 | 0.095 | 0.226 | 0.305 |
| same (trained, same token) | 1 layer | 0.629 | 0.486 | 0.680 | 0.826 | 0.778 |
| **router_same (today's prefetch)** | 1 layer | **0.654** | **0.548** | **0.748** | **0.876** | 0.807 |
| ro_lr (route-only learned mix) | 1 layer | 0.323 | 0.051 | 0.115 | 0.245 | 0.402 |
| ro_b (route-only transition) | 1 layer | 0.234 | 0.144 | 0.224 | 0.333 | 0.319 |

Moetier ledger, LOOKA (counters only), test stream. Reads = predicted keys that are NVMe-tier when the prediction is
issued:

| budget | predictor | N | recall NVMe | reads / token | used / token | precision | read GB / token |
|---|---|---|---|---|---|---|---|
| 55 GB (34 NVMe picks/token) | edge0 | 8 / 16 / 32 | 0.082 / 0.155 / 0.271 | 39 / 100 / 256 | 4.7 / 7.7 / 12.2 | 0.12 / 0.08 / 0.05 | 0.37 / 0.95 / 2.42 |
| | router_same | 8 / 16 / 32 | 0.546 / 0.746 / 0.874 | 44 / 125 / 331 | 18.6 / 25.3 / 29.7 | 0.42 / 0.20 / 0.09 | 0.42 / 1.19 / 3.13 |
| 16 GB (130 NVMe picks/token) | edge0 | 8 / 16 / 32 | 0.119 / 0.209 / 0.335 | 140 / 319 / 726 | 25 / 40 / 58 | 0.18 / 0.12 / 0.08 | 1.33 / 3.03 / 6.88 |
| | router_same | 8 / 16 / 32 | 0.596 / 0.775 / 0.888 | 143 / 338 / 782 | 77 / 100 / 115 | 0.54 / 0.30 / 0.15 | 1.36 / 3.21 / 7.41 |

PILOT (predictor in the sim loop, test stream; a landing ring that never enters RAM). `idle` = prefetch reads use
only idle channel time, so demand reads preempt them. `fifo` = prefetch reads share the FIFO channel.

| budget | base | oracle next-token (perfect) | edge0@8 / @16 / @32 idle | edge0@16 fifo | router_same@8 idle / fifo |
|---|---|---|---|---|---|
| 55 GB | 20.31 | 25.56 (+26 %) | 20.61 / 20.78 / 20.81 (+2.3 % best) | 15.66 | 20.31 / 19.66 |
| 16 GB | 12.45 | 16.27 (+31 %) | 12.96 / 12.99 / 12.92 (+4.3 % best) | 6.73 | 12.45 / 10.39 |

- The sim's single-FIFO channel does not reproduce the measured +13 % of today's same-token prefetch. One-layer
  lead leaves no idle time in the `idle` model. The `fifo` model charges every guess at 24 GB/s in series. The
  absolute PILOT numbers for same-token prefetch are therefore pessimistic.
- The next-token rows are less affected, because their reads have a whole token to land.

## Why the next-token heads fail here

- **Information.** With a one-token lead, the hidden state of token t says little about which cold experts token t+1
  will pick. Recall on all picks is 0.28 at N = 8, close to Edge0's "about a quarter of experts shared by adjacent
  tokens".
- **The misses are the hardest picks.** NVMe-served picks are cold, low-frequency experts. On them, recall falls to
  0.09 at N = 8. The extra signal from the MLP over the warm-started router is real (0.029 -> 0.087 at N = 8,
  0.20 -> 0.29 at N = 32) but far from 0.4.
- **Read cost is the binding constraint, not lead time.** Reaching 0.29 needs N = 32 and 256 reads/token at 55 GB
  (2.4 GB/token). At 16 GB it needs 726 reads/token (6.9 GB/token), over 2.5x what the RAID can move in one token.
- **Data size.** 37k training tokens from 11 prompts is small. Validation recall@8 was flat after 2-3 epochs and the
  per-layer curves are similar across layers (0.10-0.26 at N = 16 on NVMe picks).
  - More data could move these numbers by some points. It cannot plausibly move 0.16 to 0.4: the router itself on
    t's state (`router_next`) is at 0.09, and the gap to same-token (0.75) is information, not fit.
  - A second capture window is **not** needed to decide this gate.

## What this changes

1. **Do not build the next-token prefetch (phase C) on these heads.** At the budgets that fit the channel (N ≤ 16)
   it is worth +0.3-0.5 tok/s in the preemptible model at 55 GB, and it loses speed with FIFO reads.
2. **Today's router lookahead is the strong predictor.**
   - It recalls 0.55-0.75 of NVMe-served picks at 42-44 % precision (N = 8). The earlier route-only study assumed
     0.05-0.07 for any predictor.
   - The README lever "layer-ahead prefetch (recall 0.7)" is roughly what `GLM53_NV_PREFETCH=1` achieves at
     N = 8-16.
   - Its limit is lead time (one layer, ~1.3 ms), not recall.
3. **The remaining lever is lead time.** Read layer l+1's predicted experts earlier, using the router lookahead
   from two or more layers back.
   - This was not measured here. Capture rows for z[t, c-2] are on the RAID.
   - Alternatively, make the current prefetch preemptible: a demand-first reader queue (`GLM53_NV_PF_LO`) plus a
     landing ring, so wasted reads stop competing at 16 GB.
4. **The next-token oracle bounds what any one-token-ahead scheme can win at exact routing:** +26 % (55 GB) and
   +31 % (16 GB) in the sim.

## Phase D: prediction as routing (lossy). Estimate only, nothing trained

- **Speed.** If the head's top-8 IS the routing, every NVMe read is known a token ahead. That is the oracle row:
  about +26 % at 55 GB and +31 % at 16 GB in the sim (20.3 -> 25.6 and 12.4 -> 16.3). The measured engine has
  CPU-lane and copy costs the sim does not model, so expect less.
- **Quality.** Recall@8 on all picks is 0.28, so about 72 % of routed experts would be replaced at each layer.
  - Edge0 reports that distillation-only heads under student routing produce "repetitive, collapsed text". Their
    fix is a LoRA SFT on about 2M rows plus on-policy distillation, and it still costs 3.9 points on average (6.1 on
    AIME).
  - For GLM-5.3-Flash, expect the same collapse without SFT. With SFT, expect a multi-point benchmark loss.
  - This needs a training campaign (LoRA on attention / shared experts, student path with streamed experts) and
    quality gates (KL panel, GPQA, AIME-style).
- **Not started.** It needs the user's OK.

## Reproduce

```bash
# capture (omarchy, slot A under ~/gpu3090.lock + ~/nvx_bench.lock): glm53-flash-offload n134-prerouter/prerouter
MEM=55g TARGET=200000 ./run_cap.sh cap55            # -> /mnt/nvx/n134/cap55/cap/{z,sel,w,meta,tier}.bin
# train + per-variant predictions (any torch device; --zdir streams per-layer z files)
python3 prerouter/prerouter_train.py --cap cap55 --router router.npz --requests requests.jsonl --out run55 \
    --zdir zl --delete-z --device mps
# ledger evaluation (this repo)
python3 examples/prerouter_eval.py --run run55 --exclude 0,41 \
    --pilot edge0@8,edge0@16,edge0@32,self@16,router_same@8,router_same@16,same@16 --out docs/prerouter-glm53.results.json
```

## Caveats

- **Test set.** 4 test prompts and 9.7k tokens. The CIs on the recalls are a few points, which does not change a
  0.16-vs-0.4 gate.
- **Tiers.** The engine tiers are 55 GB only. The 16 GB tiers come from the ledger (41 % of picks are NVMe-tier
  there).
- **Capture effects.** The capture hook's pinned ring (0.16 GiB) and per-layer D2D copies ran during the capture.
  Decode was 15-17.7 tok/s at C1 with prefetch off.
- **Separate finding (serve.py).** With `GLM53_MAX_RQ_TOKENS=4096`, `usage.completion_tokens` under-reports long
  completions: one request decoded 21,503 tokens but reported 7,198.
