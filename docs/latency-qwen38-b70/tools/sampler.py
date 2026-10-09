#!/usr/bin/env python3
"""In-container sampler (docker exec, root): xe fdinfo engine cycles of the server's DRM clients + per-thread CPU ticks.
usage: sampler.py <out.jsonl> [period_s]"""
import json, os, sys, time
out = open(sys.argv[1], "a", buffering=1); per = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
fdcache = {}; last_scan = 0
def drm_fds(pid):
    r = []
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/dri/"): r.append(fd)
            except OSError: pass
    except OSError: pass
    return r
k = 0
while True:
    t = time.time()
    if t - last_scan > 10:
        fdcache = {}
        for p in os.listdir("/proc"):
            if p.isdigit() and int(p) != os.getpid():
                f = drm_fds(p)
                if f: fdcache[p] = f
        last_scan = t
    eng = {}; seen = set()
    for p, fds in fdcache.items():
        for fd in fds:
            try: txt = open(f"/proc/{p}/fdinfo/{fd}").read()
            except OSError: continue
            cid = None; d = {}
            for line in txt.splitlines():
                if line.startswith("drm-client-id:"): cid = line.split()[1]
                elif line.startswith("drm-cycles-") or line.startswith("drm-total-cycles-"):
                    a, b = line.split(":", 1); d[a] = int(b.split()[0])
            if cid is None or cid in seen: continue
            seen.add(cid)
            eng[f"{p}:{cid}"] = d
    rec = dict(t=t, eng=eng)
    if k % 2 == 0:
        th = {}
        for p in fdcache:
            try:
                for tid in os.listdir(f"/proc/{p}/task"):
                    s = open(f"/proc/{p}/task/{tid}/stat").read()
                    comm = s[s.index("(") + 1:s.rindex(")")]
                    f = s[s.rindex(")") + 2:].split()
                    th[f"{p}/{tid}"] = [comm, int(f[11]) + int(f[12]), int(f[36])]   # utime+stime ticks, last cpu
            except OSError: pass
        rec["th"] = th
    out.write(json.dumps(rec) + "\n"); k += 1
    time.sleep(max(0.0, per - (time.time() - t)))
