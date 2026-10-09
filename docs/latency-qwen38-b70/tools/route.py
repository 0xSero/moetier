#!/usr/bin/env python3
"""HOM-272 Edge0 round 2 step 1 (Qwen nvme16): routing-width / gate-mass arms on one server (B70_ROUTE=1).
No max_tokens is ever sent. usage: route.py <port> <outdir> [arms 'K:tau,K:tau,...']"""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PORT, OUT = int(sys.argv[1]), sys.argv[2]
ARMS = [tuple(a.split(":")) for a in (sys.argv[3] if len(sys.argv) > 3 else "10:0,8:0,6:0,4:0,10:0.02,10:0.05,10:0.1,10:0").split(",")]
sys.argv = [sys.argv[0], str(PORT), OUT, "route", ""]
import load  # noqa: E402  (reuses stream_chat / long_prompt)
LOG = open(os.path.join(OUT, "route.jsonl"), "a", buffering=1)
PANEL = [
    "Explain in two short paragraphs why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List the planets of the solar system in order from the Sun, one per line.",
    "Translate into French: 'The library opens at nine and closes at six on weekdays.'",
    "What is 17 multiplied by 23? Show the steps briefly.",
    "Give three tips for writing clear technical documentation.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Write a haiku about a quiet server room at night.",
]

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())

def panel():
    outs = []
    for p in PANEL:
        j = post({"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": p}], "logprobs": True,
                  "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}})
        ch = j["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        outs.append(dict(tokens=[t["token"] for t in lp], lp=[t["logprob"] for t in lp],
                         top=[{x["token"]: x["logprob"] for x in (t.get("top_logprobs") or [])} for t in lp],
                         finish=ch.get("finish_reason"), completion_tokens=j["usage"]["completion_tokens"],
                         text=ch["message"].get("content") or ""))
    return outs

def chat_sanity():
    # thinking on, greedy; the client stops reading at 4096 tokens and calls that a runaway (gate, not a server cap)
    body = {"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": "Name three primary colors."}],
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    n = 0; fin = None; txt = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
            j = json.loads(line[5:].strip())
            for ch in j.get("choices") or []:
                de = ch.get("delta") or {}
                pc = (de.get("content") or "") + (de.get("reasoning_content") or "")
                if pc: n += 1; txt.append(pc)
                if ch.get("finish_reason"): fin = ch["finish_reason"]
            if n >= 4096: fin = "client_runaway_stop"; break
    return dict(finish=fin, chunks=n, runaway=fin != "stop", tail="".join(txt)[-200:])

def set_route(k, tau):
    open(os.path.join(OUT, "ROUTE"), "w").write(f"{k} {tau}\n")
    for _ in range(6):
        post({"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": "Count from 1 to 40, separated by commas."}],
              "chat_template_kwargs": {"enable_thinking": False}})
        try:
            last = open(os.path.join(OUT, "ROUTE.applied")).read().strip().splitlines()[-1].split()
            if int(last[4]) == int(k) and abs(float(last[6]) - float(tau)) < 1e-9: return True
        except Exception:
            pass
    return False

for ai, (k, tau) in enumerate(ARMS):
    k, tau = int(k), float(tau); arm = f"K{k}_tau{tau:g}_{ai}"
    ok = set_route(k, tau)
    rec = dict(arm=arm, K=k, tau=tau, applied=ok, t=time.time())
    rec["chat"] = chat_sanity()
    rec["panel"] = panel()
    r1 = load.stream_chat(load.long_prompt(8192, f"route-{arm}-c1"))
    r1["decode_tok_s"] = round((r1["completion_tokens"] - 1) / (r1["end"] - r1["first"]), 2); r1["ttft_s"] = round(r1["first"] - r1["t0"], 3)
    r1.pop("chunk_ts", None); rec["c1"] = r1
    import threading
    res = [None] * 4
    def one(i): res[i] = load.stream_chat(load.long_prompt(8192, f"route-{arm}-c4-{i}"))
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]; [t.start() for t in th]; [t.join() for t in th]
    span = max(r["end"] for r in res) - min(r["first"] for r in res)
    rec["c4"] = dict(agg=round(sum(r["completion_tokens"] for r in res) / span, 2),
                     per=[round((r["completion_tokens"] - 1) / (r["end"] - r["first"]), 2) for r in res],
                     ct=[r["completion_tokens"] for r in res], finish=[r["finish"] for r in res],
                     prefill_tok_s=round(sum(r["prompt_tokens"] for r in res) / (max(r["first"] for r in res) - min(r["t0"] for r in res)), 1))
    rec["t_end"] = time.time()
    LOG.write(json.dumps(rec) + "\n")
    print(arm, ok, rec["chat"]["finish"], rec["c1"]["decode_tok_s"], rec["c1"]["ttft_s"], rec["c4"]["agg"], flush=True)
