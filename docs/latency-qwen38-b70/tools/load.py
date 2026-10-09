#!/usr/bin/env python3
"""HOM-272 B70 profiling workload (stdlib). No max_tokens is ever sent: every completion ends naturally.
usage: load.py <port> <outdir> <arm> phases(comma: warm,c1,c4,prof_c1,prof_c4)"""
import json, os, random, sys, threading, time, urllib.request
PORT, OUT, ARM = int(sys.argv[1]), sys.argv[2], sys.argv[3]
PHASES = sys.argv[4].split(",")
MODEL = "flashnext"
WORDS = ("river stone lantern copper meadow signal harbor quartz willow engine ember orbit canyon velvet "
         "matrix pebble falcon tundra cobalt prism glacier thistle beacon saffron vortex marble cedar "
         "anchor nebula lattice harvest summit cipher compass garnet monsoon ripple timber zephyr").split()
LOG = open(os.path.join(OUT, "load.jsonl"), "a", buffering=1)

def long_prompt(target_tokens, tag):
    rnd = random.Random(f"{target_tokens}-{tag}")
    n = int(target_tokens * 0.72)
    body = " ".join(rnd.choice(WORDS) for _ in range(n))
    return (f"[document {tag} {rnd.getrandbits(64):016x}]\n{body}\n\n"
            "The text above is a list of random words. Write a short story (about 300 words) that uses ten of them.")

def stream_chat(content, thinking=False, on_first=None):
    body = {"model": MODEL, "temperature": 0, "messages": [{"role": "user", "content": content}], "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; ts = []; usage = None; finish = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices") or []:
                de = ch.get("delta") or {}
                piece = (de.get("content") or "") + (de.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                        if on_first: on_first()
                    ts.append(round(now - t0, 4)); text.append(piece)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", len(ts))
    return dict(t0=t0, first=first, end=t1, completion_tokens=ct, prompt_tokens=(usage or {}).get("prompt_tokens"),
                finish=finish, chunk_ts=ts, text_tail="".join(text)[-300:])

def run(phase, items, conc_hook=None):
    res = [None] * len(items); th = []
    firsts = threading.Event(); nfirst = [0]; lk = threading.Lock()
    def mark():
        with lk:
            nfirst[0] += 1
            if nfirst[0] == len(items): firsts.set()
    def one(i, content, thinking):
        try: res[i] = stream_chat(content, thinking, on_first=mark)
        except Exception as e: res[i] = dict(error=repr(e))
    tp = time.time()
    for i, (content, thinking) in enumerate(items):
        t = threading.Thread(target=one, args=(i, content, thinking)); t.start(); th.append(t)
    if conc_hook: conc_hook(firsts)
    [t.join() for t in th]
    for i, r in enumerate(res):
        r.update(phase=phase, idx=i, conc=len(items), arm=ARM, phase_t0=tp)
        if r.get("first"):
            span = r["end"] - r["first"]
            r["decode_tok_s"] = round((r["completion_tokens"] - 1) / span, 2) if span > 0 else None
            r["ttft_s"] = round(r["first"] - r["t0"], 3)
        LOG.write(json.dumps(r) + "\n")
    print(phase, [(r.get("completion_tokens"), r.get("decode_tok_s"), r.get("ttft_s"), r.get("finish")) for r in res], flush=True)
    return res

def prof_hook(steps, delay):
    def h(firsts):
        firsts.wait(600); time.sleep(delay)
        open(os.path.join(OUT, "PROF_ON"), "w").write(str(steps))
        print("PROF_ON written", flush=True)
    return h

for ph in PHASES:
    if ph == "warm":
        run("warm", [("Name three primary colors.", False)]); run("warm", [("Write a haiku about a quiet server room at night.", False)])
    elif ph == "c1":
        for i in range(3): run("c1", [(long_prompt(8192, f"{ARM}-c1-{i}"), False)])
    elif ph == "c4":
        run("c4", [(long_prompt(8192, f"{ARM}-c4-{i}"), False) for i in range(4)])
    elif ph == "prof_c1":
        run("prof_c1", [(long_prompt(8192, f"{ARM}-pc1"), False)], conc_hook=prof_hook(60, 3))
    elif ph == "prof_c4":
        run("prof_c4", [(long_prompt(8192, f"{ARM}-pc4-{i}"), False) for i in range(4)], conc_hook=prof_hook(60, 3))
    time.sleep(3)
