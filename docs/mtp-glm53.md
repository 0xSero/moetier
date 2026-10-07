# MTP self-speculative decoding for GLM-5.3-Flash on the 1x3090 NVMe/CPU-tier stack (N120, HOM-272)

Status: **prepared, not run.** No GPU job was started; the 3090 belonged to N119 (S3b) the whole time. Everything below is
read from code (exllamav3 1.5.1 extracted from the serving image `ghcr.io/0xsero/glm53-flash-offload@sha256:bb633b0b…`,
the N119 tree `omarchy:~/freetoken-exl3/runs/N116-glm53-nvme/s2/glm53`), from earlier measured runs (G063, G063b, G063c,
S3a), or modeled in `examples/mtp.py`. Tags: [M] measured, [C] read from code, [S] simulated, [E] estimate.

## Summary

- **Acceptance on this checkpoint is lower than the GB10 numbers the earlier projection used.**
  - Measured with exllamav3's built-in MTP head, greedy [M]:
    - k=1: **1.79** tokens/round (G063b, 18,175 rounds, sweep prompts).
    - k=2: **2.27** (G063c, 4,687 rounds, sweep prompts).
    - k=3: 2.71 on the smoke prompts (G063). That is about 2.54 on sweep prompts [E].
  - GB10's 2-bit target measured 2.81-2.95 at k=2. Those are the numbers behind "32.4 -> 46 tok/s" in
    `vllm-moet-lessons.md`. With 2.27 the same model gives **36.6**.
- **A verify window shares expert fetches inside one forward, but routing is flat, so there is little to share.**
  - In our stack each unique expert per layer is read from NVMe once, copied to VRAM once, and decoded once on the CPU
    for all of its rows [C].
  - The 3-row union is still about 2.6x one token. Expert work per *generated* token rises about 16 % at k=2 [S].
  - What MTP really buys is amortizing the per-step costs: about 28 ms of the 65 ms S3a step is non-MoE time plus
    per-layer handoff [S, fitted].
- **Expected gains** (table below):
  - Today's S3 config: C1 15.3 -> **18-20 tok/s** (+15-29 %), with k=1 ≈ k=2.
  - Fully-levered design: 32.4 -> **35-38** (k=1 best). k=2 gives 32.5-36.6.
  - C4: no gain. The 8-row cap forces k=1, and the union is already wide.
  - MTP does not reach 50 tok/s on its own. At k=2 that would need more than 3.0 tokens/round, which is impossible.
- **Recommended placement: M1.** The MTP layer's 288 experts join the nv2 pool as store layer 42.
  - The N116 store already packs them there (`mtp_layer_li = 42`).
  - It costs about 150-210 VRAM slots, against 380-440 for stock M0 (all 288 in VRAM).
- Ready-to-run arms are in `omarchy:~/freetoken-exl3/runs/N120-glm53-mtp/`.
  - No shared file is modified. A wrapper imports `serve.py` unchanged.
  - An offline CPU-only self-test passed: `SELFTEST_OK`.

## 1. How exllamav3 1.5.1 does MTP [C]

**Loading** (`model_init.py`)
- `-mtp` sets `draft_model_dir = model_dir` and builds the `mtp` component, `Glm5NextMTPModel`
  (`architecture/glm5_next_mtp.py`). Its layers:
  - an input layer: enorm(embedding) ‖ hnorm(target post-final-norm hidden) -> `eh_proj`, 2H -> H;
  - one TransformerBlock (layer 45) with MLA + DSA indexer (`indexer_mode="full"`);
  - a `BlockSparseMLP` with its own 288 experts, top-8, and a shared expert;
  - `shared_head.norm`.
- It borrows the target's embedding and its 6-bit lm_head.
- caps: `mtp_draft`, `attach_target`, `default_draft_size = 3`.
- The draft model loads **before** the target and gets its own MLA cache, the same size as the main cache.
- `-ndt` is the draft length k. `-dds` (dynamic draft) stops drafting early with an online confidence calibrator.
- `-dmcl` is stock CPU offload of the MTP experts. It cannot be used with our tier.

**One decode round** (`generator.py`)
1. `iterate_draftmodel_mtp_gen` runs **k sequential MTP forwards**. Each is one row per job, ends in an argmax through
   the target lm_head, and feeds its own output state back as the next `target_hidden`.
2. `iterate_gen` runs **one target forward over `[jobs, 1+k]` rows**:
   - with `recurrent_history=True`: KDA keeps per-row states so it can rewind;
   - with `export_state_norm_keys`: it exports the post-norm hidden state for MTP.
3. Acceptance is greedy-match position by position.
   - `reject_remainder` rewinds the KDA recurrent state and the page positions.
   - A recurrent checkpoint boundary stops acceptance.
4. If A > 1 tokens are accepted, a **catch-up MTP prefill** runs over the accepted positions (A-1 rows). It rewrites the
   draft KV from target states. `mtp_last_hidden` is then carried to the next round.
5. MTP forwards per round = k + P(A > 1). That is about k + 0.79.

**Prompt prefill**
- The target runs `forward(last_tokens_only=1)` instead of `prefill`, because it needs post-norm states for every token.
- Then the MTP head prefills the same chunk. Its MoE sees chunk x 8 picks and takes the prefill path.

**MoE view of verify rows** (`block_sparse_mlp.py`)
- `y` is `[jobs*(1+k), H]` in one forward call, and routing is per row.
- At **≤ 8 rows** (`MAX_BSZN = 8`), the fused `run_bszN` kernels run **every (token, expert) slot separately, with no
  dedup**. A 9.4 MB expert does not fit the 6 MB L2, so every row re-reads its weights. This is cheap from VRAM.
- Above 8 rows, the sorted/grouped path computes each expert once over its rows.

**Batched decode and `-ambs`**
- `-ambs` (`autosplit_max_batch_size`) is a load-time reserve. It is also the cache's `max_batch_size`, which is the
  number of **recurrent-state slots**. The generator caps concurrency at that number.
- This is why exact modes, which run with no `-ambs`, serialize C2/C4.
- It works with MTP: drafting batches all decode jobs. Only CFG/multi-sequence jobs are refused.
- The cost is the KDA state, `ambs x (max_history+1) x 34 layers x 4 MiB`:
  - stock MTP raises `max_history` from 0 to max(3, ndt), which is **+1.6 GB VRAM at -ambs 4**;
  - `serve_mtp.py` sets it to `-ndt`: +0.57 GB per draft token.

**Graph capture**
- exllamav3 1.5.1 has no whole-model CUDA graph. Graphs exist only inside modules:
  - GatedMLP at bsz 1, q_len 1;
  - BC single-expert graphs;
  - the BC_GatedMLP multi-row graph for shared experts.
- Verify rows (q_len > 1) take the non-graph GatedMLP path for the dense layers 0-2. That is part of the T > 1 fixed
  cost.
- nv2's device-side bounded spin on the host reply is not capturable. Nothing captures it today.
- `model.warmup` skips draft models. The MTP head's Triton kernels compile on the first request, and the smoke step
  absorbs that. The target warmup already covers "multi-token step with recurrent history" shapes.
- `k_hcfuse` fuses up to R ≤ 8 rows. The MTP block has no mHC.
- If whole-step graphs are added later (the "fused non-MoE + graphs" lever), each verify width T = 1..8 needs its own
  graph, and the nv2 reply wait has to stay outside the graph.

## 2. What the NVMe + CPU-tier build (N119 nv2, S2/S3) needs for MTP [C]

**What already works with T rows**
- `expert_cache.attach_modules` wraps `routing_fn`, and `nv2.layer` -> `nv_pub` dedups the picks: unique keys plus
  `cnt[u]` tokens per key (`nv2_shared.h`).
- `plan_and_reply` prices the CPU lane `ca + cb + ctok*(cnt-1)`. This is moetier `plan_layer`.
- An NVMe miss for any number of rows is **one** read with **one** landed flag.
  - The device-side stall (`nv_step`) and the CPU's wait-for-landing are per key.
  - The S1 path `nv_tier.ensure` also dedups.
- CPU job (`run_cpu_job` -> `ft_core.h moe_forward`): jobs are grouped per expert, up to `MAXM = 8` rows, and one
  trellis decode serves all of its rows. The self-test already checks the kernel at m = 1 and m = 3.
- Admission, the exclusive RAM tier and the victim ring are unchanged. The GPU lane is exllamav3's own kernels on the
  live pointer tables.

**Hard limits.** Above them a layer drops to nv2's `small` path: no CPU lane, no admission, RAM experts read
zero-copy. That is slow but still exact.
- CPU lane only when `ntok ≤ MAXB = 8` and `picks ≤ MAXP = 64`. Decode admission only when picks ≤
  `GLM53_EC_ADMIT_MAX = 64`. Prefetch only at bsz ≤ 8.
- So the verify batch must stay at **≤ 8 rows**: C1 k ≤ 7, C2 k ≤ 3, **C4 k = 1**.
- `serve_mtp.py` enforces this per round (`GLM53_MTP_ROWCAP=8`: k_eff = min(k, 8 // jobs - 1)).
- Raising the cap means recompiling `nv2_shared.h` (MAXB 16, MAXP 128) and `ADMIT_MAX=128`. At 9-16 rows exllamav3
  also switches to its grouped path. Not needed for C1/C2.

**The MTP layer's 288 experts (2.73 GB)**

| option | VRAM | host RAM | draft forward cost | work |
|---|---|---|---|---|
| **M1: in the nv2 pool as layer 42** (store records 42 x 288..) | 0.13 GB dense + 0.16 GB draft KV + KDA history; its hot experts compete in CLOCK (about 60 slots assumed) | none extra (RAM tier is a cache) | 8 picks per draft through the same lanes: VRAM hit, else CPU (0.21 ms/expert live) or NVMe stall | `serve_mtp.py GLM53_MTP_POOL=1`: drop `mtp_layer_li` from the manifest the loader sees, add the draft's MoE to the pool, disable layer-ahead prefetch for the last trunk layer |
| M0: stock, all in VRAM | 2.86 GB + 0.16 + history = 380-440 slots fewer | none | all VRAM, about 1.3 ms | none (zero-code control) |
| M2: pinned in RAM, CPU-only | 0.3 GB | +2.73 GB = -288 RAM slots | 8 x 0.21 ms + handoff, serial | not built; M1 adapts to this when MTP experts go cold |

**Details checked in code for M1**
- The MTP trellis shapes equal the trunk's: gate/up `[256,128,48]` I16, down `[128,256,48]`, K=3, so the record size
  is the same. The Pool asserts it.
- The pool's layer order is the store order: trunk 0..41 (model layers 3..44), then MTP at 42. `nv2.Nv2.__init__`
  asserts it.
- CPU-tier scale copies and `BCProxy` cover all pool layers.
- Staged prefill: layer 41 already submits staging for layer 42, and the MTP prefill that follows the target chunk
  uses that buffer (`_stage_hook_nv2`: 42 > last_li 41, same forward).
- The elastic prefill hook is on the target class only. The MTP prefill runs inside the target's elastic window.
- nv2's `decode_per_step` divides by L = 43. With MTP the MTP layer is called k+1 times per round, so use the deltas
  that `collect_mtp.py` computes, not that field.

## 3. Ready-to-run measurement: `omarchy:~/freetoken-exl3/runs/N120-glm53-mtp/`

| file | role |
|---|---|
| `serve_mtp.py` | Wrapper: imports `serve.py` unchanged and adds env-gated patches: M1 pool placement; history = `-ndt`; row cap; acceptance counters (Job.draft_stats); CUDA-event timing of draft / verify / catch-up and host wall time per (jobs x window). New endpoints: `GET /mtp_stats`, `POST /mtp_stats_reset` |
| `prep_snapshot.sh` | Freezes `s2/glm53` + `serve_mtp.py` into `snap/`. Derives `entrypoint_mtp.sh` (S2 entrypoint + `GLM53_SCRIPT` override; diff-checked). Copies the compiled nv/nv2 builds. Checks that the store has li 42 = layer 45 |
| `run_mtp.sh <tag> <nvme3/nvme2> <k> <M1/M0/->` | One arm in container `n120-<tag>`. Same mounts, caps (55g, cpus 2-39) and S3b env as `run_s2.sh`. Refuses to start if GPU0 has >1.5 GB used or any compute app, or if kcheck trips. Steps: smoke, panel, greedy, verify, sweep (`bench/sweep.py --template glm --conc 1 2 4 --prefill 8192 32768`, natural completions, no max_tokens) |
| `greedy_mtp.py` | The 5 greedy_eq prompts (or the `--ref` file's own prompts), temperature 0, natural end. Reports equality plus the first differing character and tokens/round per prompt |
| `collect_mtp.py` | Arm table: decode C1/C2/C4/C1@32k, prefill, tokens/round, P(>=i), draft + verify (catch-up) / wall ms, nv2 per-token deltas over the sweep, pool slots, greedy |
| `queue_mtp.sh` | `q_smoke` (k=2 M1, fail-fast) -> `e0`/`e1`/`e2` (exact nvme2: k=0 vs s2a_55g ref, then k=1/2 vs e0) -> `a0` (S3 baseline) -> `m1k1`, `m1k2`, `m0k2` (full) -> optional `m1k3d` (`-dds`). About 3.7 h. `WAIT=1` polls for an idle GPU0 |
| `selftest_mtp.py` | CPU-only container test of the patches and checkpoint geometry. Passed on `snap_selftest/` |

Launch on omarchy once N119 releases the GPU:

```bash
cd ~/freetoken-exl3 && nohup bash runs/N120-glm53-mtp/queue_mtp.sh > runs/N120-glm53-mtp/queue.log 2>&1 &
```

The queue snapshots the current S2 code at launch, so it measures whatever S3 is by then.

**Greedy-equality expectations**
- Exact path (e1/e2 vs e0):
  - Speculation is lossless relative to the verify forward.
  - But the T-row kernels (bszN at bsz = k+1, non-graph dense path) can round differently from the 1-row path. A late
    near-tie divergence is acceptable; an early one is a bug.
- S3 arms: the CPU lane is already approximate (decode KL 0.0047, greedy 0/5 vs exact). Equality there is not
  expected. Use the panel (prefill path, must stay top-1 1.0 / KL 0) and coherence.

**Go criteria**
- m1k1 or m1k2 C1 at least +10 % over a0, with panel 1.0 / 0, the exact-path greedy check clean, and `nv_verify` 0 bad.
- Then recalibrate `examples/mtp.py`: put the measured draft ms in `d`, and fit the per-row slope `s` from
  verify_ms(k) - verify_ms(0).

## 4. Expected C1/C2/C4 (`python3 examples/mtp.py`) [S]

Model and scenarios
- Acceptance is measured (k=3 estimated). Draft cost = (k + 0.79) forwards x d ms.
- Bounds:
  - **Fo**: +2 ms of non-MoE time per extra row; CPU +0.04 ms per extra token; GPU re-reads free; d = 1.3 (M0) /
    1.6 (M1).
  - **Fb**: +6 ms per row (now) or +4 (levers); CPU +0.08 / +0.06; GPU and zero-copy +0.023 ms per extra row on a
    shared expert; d = 2.0 / 2.5.
- C2 runs k ≤ 3 and C4 runs k = 1 (row cap).
- **now** is calibrated to measured S3a (C1 15.30 [M]):
  - The 4-chat trace is about 10 % more local than served traffic, so the 1,376 real slots are modeled as 900 to match
    the measured VRAM hit of 0.45.
  - CPU lane 0.21 ms/expert, as measured live.
  - Fitted per-step overhead of 14.4 ms on top of 14 ms non-MoE.
  - C2/C4 at k=0 are modeled: S3a ran them serialized.
- **levers** is the `levers.py` end state: 32.44 at k=0, the same as before.

**now (S3 config at --memory 55g)**

| k | MTP experts | VRAM slots | tokens/round | draft ms | C1 Fo / Fb | C2 Fo / Fb | C4 Fo / Fb |
|---|---|---|---|---|---|---|---|
| 0 | - | 1376 | 1.00 | 0 | **15.3** (measured) | 20.4 / 20.4 | 23.7 / 23.5 |
| 1 | M1 | 1225 | 1.79 | 2.9 | **19.6 / 18.3** | 23.6 / 21.7 | 24.6 / 22.5 |
| 2 | M1 | 1165 | 2.27 | 4.5 | **19.7 / 17.6** | 22.2 / 19.5 | 24.6 / 22.5 |
| 3 | M1 | 1105 | 2.54 (est) | 6.1 | 17.9 / 15.6 | 19.2 / 16.6 | 24.6 / 22.5 |
| 1 | M0 | 996 | 1.79 | 2.3 | 19.2 / 18.0 | 22.7 / 20.9 | 24.6 / 22.6 |
| 2 | M0 | 936 | 2.27 | 3.6 | 19.1 / 17.2 | 20.7 / 18.4 | 24.6 / 22.6 |

**levers (target design: fixed 9 ms, prefetch 0.7, persistent CPU workers, 1,910 slots)**

| k | MTP experts | VRAM slots | tokens/round | C1 Fo / Fb | C2 Fo / Fb | C4 Fo / Fb |
|---|---|---|---|---|---|---|
| 0 | - | 1910 | 1.00 | 32.4 | 39.4 / 39.5 | 45.3 / 45.0 |
| 1 | M1 | 1759 | 1.79 | **38.1 / 34.9** | 44.4 / 40.7 | 46.2 / 42.6 |
| 2 | M1 | 1699 | 2.27 | 36.6 / 32.5 | 41.2 / 36.3 | 45.8 / 42.3 |
| 3 | M1 | 1639 | 2.54 (est) | 32.9 / 28.5 | 35.9 / 30.9 | 45.3 / 42.0 |
| 2 | M0 | 1470 | 2.27 | 35.8 / 32.1 | 40.1 / 35.7 | 43.8 / 40.6 |

How to read it
- At k=2, CPU experts per round go from 115 to 303 (now) and from 149 to 331 (levers): the union of 3 rows is
  2.6-2.2x one token. MTP therefore cannot lower the CPU lane's work per token. It amortizes the per-step fixed time.
  - That is why k=1 ≈ k=2 on today's config, which has a large 28 ms per-step fixed time.
  - On the levers design (9 ms) k=1 wins.
- Break-even acceptance for k=2 to reach the old 46 tok/s on levers: about 2.86. This checkpoint measured 2.27 on
  sweep prompts and 2.38 on smoke prompts.
- Code-heavy prompts may accept more. GB10 measured on code. The arms log tokens/round per greedy prompt.
- The G063b/G063c rejection (-10 % / -22 %) was on the GPU-only zero-copy path, where PCIe bytes per token were the
  binding lane and grew with the union. S3's binding lane is the CPU, with large per-step overheads, so MTP now pays.
  It is still about +20 %, not +40 %.

## 5. Changes made in moetier

- `moetier/plan.py`: the GPU and zero-copy lanes honour `per_extra_token_ms` (default 0). This prices exllamav3's
  no-dedup bsz ≤ 8 kernels. Baseline sims are unchanged: 24.53 / 29.17 before and after on the reference probe, and
  levers 32.44 is reproduced.
- `examples/mtp.py`: the what-if above. Its acceptance constants are the measured values for this checkpoint.
