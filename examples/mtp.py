"""MTP self-speculative decoding what-if for GLM-5.3-Flash EXL3 3.05bpw on the 3090 box (docs/mtp-glm53.md).

Acceptance is MEASURED on this checkpoint with exllamav3 1.5.1's built-in MTP head, greedy (runs on omarchy):
  k=1  1.79 tokens/round  G063b, 18,175 rounds, sweep prompts
  k=2  2.27               G063c, 4,687 rounds, sweep prompts
  k=3  2.54 (estimate)    G063 smoke prompts gave 2.71 (P>=1/2/3 = .80/.58/.33); scaled by the k=2 sweep/smoke ratio
Everything else is modeled: one verify step = window k+1 consecutive trace tokens per stream (expert union, m tokens
per expert), `accept` tokens credited per stream, draft cost = (k + P(>=1 accepted)) MTP forwards per round (k drafts +
the catch-up prefill exllamav3 runs after an accepted round), each `d` ms.

Two scenarios:
  now     calibrated to the measured S3a arm (C1 15.30 tok/s): VRAM hit 0.45 needs 900 sim slots for the 1,376 real ones
          (the 4-chat G002 trace is ~10 % more local than served traffic), CPU lane 0.21 ms/expert as measured live
          (38.1 ms busy for 151.7 experts/token), per-step handoff overhead H fitted so k=0 C1 = 15.30
  levers  the levers.py target (fixed 9/13/20 ms, prefetch recall 0.7, persistent CPU workers, 1,910 VRAM slots)
Bounds per scenario: Fo = optimistic (one sequence's T rows cost +2 ms of non-MoE time per row, CPU +0.04 ms per extra
token, GPU re-reads free), Fb = pessimistic (+6 ms/row now, +4 levers; CPU +0.08 / +0.06; GPU and zero-copy re-read a
shared expert per extra row, +0.023 ms, as exllamav3's fused bsz<=8 kernels do; draft forwards slower).
Placement of the MTP layer's 288 experts: M1 = in the nv2 pool as store layer 42 (N120 serve_mtp.py GLM53_MTP_POOL=1),
M0 = stock exllamav3 load, all 288 in VRAM (2.73 GB). Both pay the KDA state history (-ambs 4: 0.57 GB per draft token).
Row cap: nv2's CPU lane and exllamav3's fused decode kernels stop at 8 rows (MAXB / MAX_BSZN), so C2 runs k<=3 and C4
runs k=1 (serve_mtp.py GLM53_MTP_ROWCAP=8). Runtime: a few minutes on the Mac.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moetier import spec, sim

ROOT = os.path.join(os.path.dirname(__file__), "..", "registry")
T = os.path.join(os.path.dirname(__file__), "..", "traces")
reg = spec.load(ROOT)
streams = sim.load_trace(f"{T}/glm53-g002-decode.npy", f"{T}/glm53-g002-decode.segments.npy")
RID = "glm53-rtx3090-55g-nvx4"
SLOT_GB = 9437184 / 1e9
ACC = {0: (1.0, 0.0), 1: (1.79, 0.79), 2: (2.27, 0.78), 3: (2.54, 0.78)}   # k -> (tokens/round, P(>=1 accepted))
ROWCAP = 8
QUICK = int(os.environ.get("MTP_QUICK_STEPS", "0"))   # >0: cap sim steps (smoke the script)

SC = {
    "now": dict(slots=1376, scale=900 / 1376, ram_gb=52.58, recall=0.5, H=None, N={1: 14.0, 2: 20.0, 4: 30.0},
                cpu=(0.145, 0.21), Fo=dict(s=2.0, ctok=0.04, gx=0.0, d={"M0": 1.3, "M1": 1.6}),
                Fb=dict(s=6.0, ctok=0.08, gx=0.023, d={"M0": 2.0, "M1": 2.5}), measured_c1=15.30),
    "levers": dict(slots=1910, scale=1.0, ram_gb=55, recall=0.7, H=0.0, N={1: 9.0, 2: 13.0, 4: 20.0},
                   cpu=(0.03, 0.104), Fo=dict(s=2.0, ctok=0.04, gx=0.0, d={"M0": 1.3, "M1": 1.6}),
                   Fb=dict(s=4.0, ctok=0.06, gx=0.023, d={"M0": 2.0, "M1": 2.5})),
}


def mtp_slots(k, place):
    """VRAM expert slots the MTP head costs (real slots): weights + draft MLA KV + KDA state history + M1 hot set."""
    if k == 0:
        return 0
    w = 2.86 if place == "M0" else 0.13          # mtp.safetensors (2.73 GB experts + 0.13 dense) vs dense only
    gb = w + 0.16 + 0.57 * k                     # draft KV (131k ctx MLA + indexer), 34 KDA x 4 MiB x 4 seqs per history slot
    return int(gb / SLOT_GB) + (60 if place == "M1" else 0)


def run(sc, bound, k, place, conc):
    P, B = SC[sc], SC[sc][bound]
    k_eff = min(k, ROWCAP // conc - 1) if k else 0
    acc, p1 = ACC[k_eff]
    win = k_eff + 1
    real = P["slots"] - mtp_slots(k, place)      # the VRAM cost is paid at load for the configured k
    F = P["H"] + P["N"][conc] + B["s"] * (win - 1) * conc
    tab = {str(n): P["H"] + P["N"][n] for n in (1, 2, 4)}
    tab[str(conc * win)] = F
    ov = {"budget.vram_expert_slots": int(real * P["scale"]), "budget.ram_gb": P["ram_gb"],
          "policy.prefetch.recall": P["recall"], "calibration.fixed_ms": tab,
          "lane_overrides": {"cpu": {"per_layer_ms": P["cpu"][0], "per_expert_ms": P["cpu"][1], "per_extra_token_ms": B["ctok"]},
                             "gpu": {"per_extra_token_ms": B["gx"]}, "zerocopy": {"per_extra_token_ms": B["gx"]}}}
    R = spec.resolve(reg, RID, **ov)
    dms = (k_eff + p1) * B["d"][place] if k_eff else 0.0
    r = sim.run(R, streams, conc=conc, window=win, accept=acc, draft_ms=dms, max_steps=QUICK)
    r.update(k_eff=k_eff, acc=acc, draft_ms=round(dms, 2), real_slots=real)
    return r


# fit the per-step handoff overhead H of the measured S3a arm (k=0, C1 = 15.30 tok/s)
SC["now"]["H"] = 0.0
for _ in range(2):
    r0 = run("now", "Fo", 0, "M1", 1)
    SC["now"]["H"] += 1000.0 / SC["now"]["measured_c1"] - r0["ms_per_step"]
print(f"now: fitted per-step overhead H = {SC['now']['H']:.1f} ms (non-MoE 14 ms + H + MoE lanes = {1000 / 15.30:.1f} ms/token)")

ROWS = [(0, "-"), (1, "M1"), (2, "M1"), (3, "M1"), (1, "M0"), (2, "M0")]
for sc in ("now", "levers"):
    print()
    print(f"### {sc}")
    print("| k | MTP experts | VRAM slots (real) | tokens/round | draft ms/round | C1 Fo | C1 Fb | C2 Fo | C2 Fb | C4 Fo | C4 Fb | C1 Fo ms/step: fixed + max(gpu, cpu) | CPU experts/round C1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, place in ROWS:
        pl = place if k else "M1"
        res = {(b, c): run(sc, b, k, pl, c) for b in ("Fo", "Fb") for c in (1, 2, 4)}
        a = res[("Fo", 1)]
        print(f"| {k} | {place if k else '-'} | {a['real_slots']} | {a['acc']} | {a['draft_ms']} | "
              f"{res[('Fo', 1)]['tok_s']} | {res[('Fb', 1)]['tok_s']} | {res[('Fo', 2)]['tok_s']} | {res[('Fb', 2)]['tok_s']} | "
              f"{res[('Fo', 4)]['tok_s']} | {res[('Fb', 4)]['tok_s']} | "
              f"{a['ms_fixed']} + max({a['ms_gpu_moe']}, {a['ms_cpu_moe']}) | {a['cpu_experts_per_tok'] * a['acc']:.0f} |", flush=True)
