# N124 job 3: launch / sync overhead on one Arc Pro B70.
#   empty      smallest torch kernel (1-element add_) : host submit cost (no sync) and submit+sync round trip
#   copy4      4-byte X.memcpy_async H2D and D2H, submit+sync
#   sync_idle  torch.xpu.synchronize() on an idle queue
#   event      record + query-poll until complete (instead of synchronize)
#   flag_rtt   landed-flag pattern: host writes seq into pinned USM word A, enqueues device work that copies A -> device
#              word B -> host USM word C, host spins on C (no synchronize); ms from enqueue to the host seeing seq
#   flag_dev   device-produced flag: enqueue a moe-size kernel then a 4 B D2H of a seq word; host spins on the word
import os, sys, json, time, ctypes
import numpy as np, torch
sys.path.insert(0, "/n")
from exl3xpu.moe_offload import ops, s64

X = ops(); dev = torch.device("xpu")
OUT = os.environ.get("OUT", "/o/sync.json")
res = dict(meta=dict(device=torch.xpu.get_device_name(0), torch=torch.__version__), cases=[])


def dump():
    with open(OUT + ".tmp", "w") as f: json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def log(d): print(json.dumps(d), flush=True); res["cases"].append(d); dump()
def sync(): torch.xpu.synchronize()


def spin(w, v, tmax=2.0):
    t = time.perf_counter(); k = 0
    while w.value != v:
        k += 1
        if (k & 1023) == 0 and time.perf_counter() - t > tmax: raise RuntimeError(f"flag timeout v={v} got={w.value}")
    return k


def stats(name, a_ms, **kw):
    a = np.array(a_ms) * 1e3
    log(dict(case=name, n=len(a), us_median=round(float(np.median(a)), 2), us_p10=round(float(np.percentile(a, 10)), 2),
             us_p90=round(float(np.percentile(a, 90)), 2), us_p99=round(float(np.percentile(a, 99)), 2), **kw))


N = 2000
t1 = torch.zeros(1, device=dev); sync()
for _ in range(200): t1.add_(1)
sync()
# submit only (host side), back-to-back
lat = []
for _ in range(N):
    t = time.perf_counter(); t1.add_(1); lat.append((time.perf_counter() - t) * 1e3)
sync(); stats("empty_submit_host", lat)
# back-to-back device throughput of tiny kernels
sync(); t = time.perf_counter()
for _ in range(N): t1.add_(1)
sync(); log(dict(case="empty_b2b", n=N, us_per_kernel=round((time.perf_counter() - t) / N * 1e6, 2)))
# submit + sync
lat = []
for _ in range(N):
    t = time.perf_counter(); t1.add_(1); sync(); lat.append((time.perf_counter() - t) * 1e3)
stats("empty_submit_sync", lat)
lat = []
for _ in range(N):
    t = time.perf_counter(); sync(); lat.append((time.perf_counter() - t) * 1e3)
stats("sync_idle", lat)
# event poll
lat = []
for _ in range(N):
    t = time.perf_counter(); t1.add_(1); e = torch.xpu.Event(); e.record()
    while not e.query(): pass
    lat.append((time.perf_counter() - t) * 1e3)
stats("empty_event_poll", lat)

hb = X.host_alloc(4096); HP = hb.data_ptr(); hb.zero_()
db = torch.zeros(1024, dtype=torch.int32, device=dev); DP = db.data_ptr(); sync()
A, C = HP, HP + 64
wa = ctypes.c_int32.from_address(A); wc = ctypes.c_int32.from_address(C)
for name, d, s in (("copy4_h2d", DP, A), ("copy4_d2h", C, DP)):
    lat = []
    for _ in range(N):
        t = time.perf_counter(); X.memcpy_async(s64(d), s64(s), 4); sync(); lat.append((time.perf_counter() - t) * 1e3)
    stats(name + "_sync", lat)
# landed-flag round trip, host -> device -> host, no synchronize
lat = []; spins = []
for i in range(1, N + 1):
    t = time.perf_counter()
    wa.value = i
    X.memcpy_async(s64(DP), s64(A), 4); X.memcpy_async(s64(C), s64(DP), 4)
    k = spin(wc, i)
    lat.append((time.perf_counter() - t) * 1e3); spins.append(k)
sync(); stats("flag_rtt_h2d_d2h_poll", lat, spins_median=int(np.median(spins)))
# one-hop: device -> host flag (D2H 4 B) after queued work, host polls
lat = []
for i in range(1, N + 1):
    db[0:1].fill_(i + 100000)          # device writes seq (tiny kernel)
    t = time.perf_counter()
    X.memcpy_async(s64(C), s64(DP), 4)
    spin(wc, i + 100000)
    lat.append((time.perf_counter() - t) * 1e3)
sync(); stats("flag_d2h_poll_after_fill", lat)
# landed flag behind a real expert-sized kernel: measure flag visibility lag vs kernel end (events)
x = torch.randn(4096, 4096, device=dev, dtype=torch.float16); sync()
lag = []; ker = []
for i in range(1, 201):
    a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    sync(); t = time.perf_counter(); a.record(); y = x @ x; b.record(); db[1:2].fill_(i); X.memcpy_async(s64(C), s64(DP + 4), 4)
    spin(wc, i)
    tw = (time.perf_counter() - t) * 1e3; sync(); kd = a.elapsed_time(b)
    lag.append(tw - kd); ker.append(kd)
stats("flag_after_gemm_wall_minus_kernel", lag, gemm_ms_median=round(float(np.median(ker)), 3))
dump()
uid = int(os.environ.get("HOST_UID", "-1"))
if uid >= 0:
    try: os.chown(OUT, uid, uid)
    except Exception: pass
print("DONE", flush=True)
os._exit(0)                      # do not wait on stuck pool threads at interpreter exit
