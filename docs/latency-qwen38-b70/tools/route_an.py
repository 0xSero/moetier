#!/usr/bin/env python3
"""Edge0 routing arms: speed vs quality. usage: route_an.py <C-route dir> [ref_panel.json]"""
import json, math, os, sys
D = sys.argv[1]
R = [json.loads(l) for l in open(os.path.join(D, "route.jsonl")) if l.strip()]
ref_gym = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
ctl = R[0]["panel"]


def agree(o, r):
    a, b = o["tokens"], r["tokens"]
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n, len(b)


def kl_top(o, r, n):
    """KL(ctl || arm) over the control's top-20 at positions < first divergence (shared context). Tokens missing from
    the arm's top-20 get the arm's 20th logprob (an upper bound on their probability)."""
    ks = []
    for i in range(n):
        p, q = r["top"][i], o["top"][i]
        if not p or not q:
            continue
        floor = min(q.values())
        z = sum(math.exp(v) for v in p.values())
        kl = 0.0
        for t, lp in p.items():
            pp = math.exp(lp) / z
            kl += pp * (math.log(pp) - min(q.get(t, floor), 0.0))
        ks.append(kl)
    return ks


out = []
for rec in R:
    P = rec["panel"]
    ag = [agree(o, r) for o, r in zip(P, ctl)]
    kls = []
    for o, r, (n, _) in zip(P, ctl, ag):
        kls += kl_top(o, r, n)
    row = dict(arm=rec["arm"], K=rec["K"], tau=rec["tau"], applied=rec["applied"],
               c1=rec["c1"]["decode_tok_s"], c1_ct=rec["c1"]["completion_tokens"], c1_finish=rec["c1"]["finish"],
               pf8k=round(rec["c1"]["prompt_tokens"] / rec["c1"]["ttft_s"], 1),
               c4_agg=rec["c4"]["agg"], c4_per=rec["c4"]["per"], c4_finish=rec["c4"]["finish"], c4_ct=rec["c4"]["ct"],
               c4_prefill=rec["c4"]["prefill_tok_s"],
               chat=rec["chat"]["finish"], chat_chunks=rec["chat"]["chunks"],
               panel_exact=sum(1 for o, r in zip(P, ctl) if o["tokens"] == r["tokens"]),
               panel_prefix=round(sum(n / max(1, m) for n, m in ag) / len(ag), 3),
               panel_all_stop=all(o["finish"] == "stop" for o in P),
               kl_top20_mean=round(sum(kls) / max(1, len(kls)), 4), kl_n=len(kls))
    if ref_gym:
        ag2 = [agree(o, r) for o, r in zip(P, ref_gym)]
        row["gymref_prefix"] = round(sum(n / max(1, m) for n, m in ag2) / len(ag2), 3)
        row["gymref_exact"] = sum(1 for o, r in zip(P, ref_gym) if o["tokens"] == r["tokens"])
    out.append(row)
json.dump(out, open(os.path.join(D, "route_analysis.json"), "w"), indent=1)
hdr = "| arm | K | tau | C1 tok/s | C4 agg tok/s | 8k prefill tok/s (C1) | panel exact vs ctl | panel prefix vs ctl | top-20 KL vs ctl (n pos) | gym-ref prefix | chat | all EOS |"
print(hdr); print("|" + "---|" * hdr.count("|")[:-0] if False else "|" + "---|" * (hdr.count("|") - 1))
for r in out:
    eos = r["panel_all_stop"] and r["c1_finish"] == "stop" and all(f == "stop" for f in r["c4_finish"]) and r["chat"] == "stop"
    print(f"| {r['arm']} | {r['K']} | {r['tau']:g} | {r['c1']} | {r['c4_agg']} | {r['pf8k']} | {r['panel_exact']}/8 | {r['panel_prefix']} | "
          f"{r['kl_top20_mean']} ({r['kl_n']}) | {r.get('gymref_prefix')} | {r['chat']} ({r['chat_chunks']}) | {eos} |")
