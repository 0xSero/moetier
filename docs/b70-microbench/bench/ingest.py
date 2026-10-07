# N124 job 1: B70 ingest microbench (one Arc Pro B70, slot B 0000:84:00.0).
# Mounts: /n = N104 exl3xpu pkg (ro), /q = Qwen NVMe store dir (ro), /g = GLM NVMe store dir (ro), /o = out dir.
# Cases:
#   h2d / d2h      copy-engine bandwidth (X.memcpy_async on the current XPU queue) from/to pinned USM host, anon THP
#                  (system pointer, SVM keys on) and memfd THP; single-copy latency (submit + sync) and batched GB/s
#   zc_read        zero-copy read bandwidth inside a device kernel is measured by experts.py (moe kernel reading host)
#   nvme_pipe      O_DIRECT record reads (thread pool, landed into a pinned USM ring) -> H2D copy into VRAM, pipelined,
#                  at the container's blkio cap (docker --device-read-bps /dev/md127:6gb)
import os, sys, json, time, ctypes, mmap, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
sys.path.insert(0, "/n")
from exl3xpu.moe_offload import ops, s64

X = ops(); dev = torch.device("xpu")
OUT = os.environ.get("OUT", "/o/ingest.json")
TB = float(os.environ.get("TBUDGET", "480")); T0 = time.perf_counter()
QWEN, GLM = 1862400, 9474048
res = dict(meta=dict(device=torch.xpu.get_device_name(0), torch=torch.__version__,
                     svm_keys={k: os.environ.get(k) for k in ("NEOReadDebugKeys", "EnableSharedSystemUsmSupport")}),
           h2d=[], d2h=[], nvme=[], notes=[])


def dump():
    with open(OUT + ".tmp", "w") as f: json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def log(k, d):
    d["t"] = round(time.perf_counter() - T0, 1); print(json.dumps(d), flush=True); res[k].append(d); dump()


def sync(): torch.xpu.synchronize()


libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MiB2 = 2 << 20
BUF = 512 << 20          # host region per kind


def anon_thp(n):
    p = libc.mmap(None, n + MiB2, 3, 0x22, -1, 0); p = (p + MiB2 - 1) // MiB2 * MiB2
    libc.madvise(p, n, 14); ctypes.memset(p, 1, n); return p


def memfd_thp(n):
    fd = os.memfd_create("n124", 0); os.ftruncate(fd, n)
    r = libc.mmap(None, n + MiB2, 0, 0x22, -1, 0); b = (r + MiB2 - 1) // MiB2 * MiB2
    assert libc.mmap(ctypes.c_void_p(b), n, 3, 0x11, fd, 0) == b
    libc.madvise(b, n, 14); ctypes.memset(b, 2, n); return b


usm_t = X.host_alloc(BUF); usm_t.fill_(3)
hosts = {"usm": usm_t.data_ptr(), "anon_thp": anon_thp(BUF), "memfd_thp": memfd_thp(BUF)}
res["meta"]["usm_kind"] = {k: int(X.usm_kind(s64(p))) for k, p in hosts.items()}
dbuf = torch.empty(BUF, dtype=torch.uint8, device=dev); DP = dbuf.data_ptr()
sync()

SIZES = [("qwen_expert", QWEN), ("glm_expert", GLM), ("64MiB", 64 << 20), ("256MiB", 256 << 20)]


def bench_dir(direction):
    for kind, hp in hosts.items():
        for name, sz in SIZES:
            if time.perf_counter() - T0 > TB: return
            n_fit = BUF // sz
            src_dst = (lambda i: (s64(DP + (i % n_fit) * sz), s64(hp + (i % n_fit) * sz))) if direction == "h2d" else \
                      (lambda i: (s64(hp + (i % n_fit) * sz), s64(DP + (i % n_fit) * sz)))
            try:
                # warm
                for i in range(min(n_fit, 4)): X.memcpy_async(*src_dst(i), sz)
                sync()
                lat = []
                for i in range(30 if sz < (64 << 20) else 6):
                    t = time.perf_counter(); X.memcpy_async(*src_dst(i), sz); sync(); lat.append((time.perf_counter() - t) * 1e3)
                nb = max(8, min(256, int(2e9 // sz)))
                gb = []
                for rep in range(3):
                    sync(); t = time.perf_counter()
                    for i in range(nb): X.memcpy_async(*src_dst(i), sz)
                    sync(); gb.append(nb * sz / (time.perf_counter() - t) / 1e9)
                a = np.array(lat)
                log(direction, dict(host=kind, size=name, bytes=sz, single_ms_median=round(float(np.median(a)), 4),
                                    single_ms_p90=round(float(np.percentile(a, 90)), 4), single_GBps=round(sz / np.median(a) / 1e6, 2),
                                    batch_n=nb, batch_GBps=[round(v, 2) for v in gb]))
            except Exception as e:
                log(direction, dict(host=kind, size=name, err=repr(e)[:300]))


PARTS = os.environ.get("PARTS", "copy,nvme").split(",")
if "copy" in PARTS:
    bench_dir("h2d")
    bench_dir("d2h")
# exactness spot check: usm -> device -> anon
ctypes.memset(hosts["usm"], 7, QWEN); X.memcpy_async(s64(DP), s64(hosts["usm"]), QWEN); X.memcpy_async(s64(hosts["anon_thp"]), s64(DP), QWEN); sync()
res["meta"]["roundtrip_exact"] = bool(np.all(np.frombuffer((ctypes.c_char * QWEN).from_address(hosts["anon_thp"]), dtype=np.uint8) == 7))
dump()


# ------------------------------------------------------------------ NVMe O_DIRECT -> pinned USM ring -> VRAM
def nvme_pipe(path, rec, n_rec, threads, ring, copy=True, stride_recs=7, base_rec=0, bounce=False):
    """O_DIRECT into a memfd THP ring (O_DIRECT into L0 USM host memory fails with EFAULT); copy=True: H2D straight
    from the system pointer (SVM keys on); bounce=True: CPU memcpy ring -> pinned USM slot, then H2D from USM"""
    nfd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    nrec_file = os.fstat(nfd).st_size // rec
    RP = memfd_thp((ring * rec + MiB2 - 1) // MiB2 * MiB2)
    if bounce: ub = X.host_alloc(ring * rec); UBP = ub.data_ptr()
    dst = torch.empty(ring * rec, dtype=torch.uint8, device=dev); DD = dst.data_ptr()
    free = [threading.Semaphore(1) for _ in range(ring)]
    landed = []
    lk = threading.Lock(); cv = threading.Condition(lk)
    read_ms = []

    def rd(i):
        s = i % ring
        free[s].acquire()
        b = (ctypes.c_char * rec).from_address(RP + s * rec)
        t = time.perf_counter()
        n = os.preadv(nfd, [b], ((base_rec + i * stride_recs) % nrec_file) * rec)
        read_ms.append((time.perf_counter() - t) * 1e3)
        assert n == rec, n
        with cv: landed.append(i); cv.notify()

    pool = ThreadPoolExecutor(threads)
    sync(); t0 = time.perf_counter()
    futs = [pool.submit(rd, i) for i in range(n_rec)]
    done = 0; inflight = []   # (slot, event)
    while done < n_rec:
        bad = [f for f in futs if f.done() and f.exception() is not None]
        if bad: raise bad[0].exception()
        with cv:
            if not landed: cv.wait(0.0002)
            got = landed[:]; landed.clear()
        for i in got:
            s = i % ring
            if copy:
                src = RP + s * rec
                if bounce: ctypes.memmove(UBP + s * rec, src, rec); src = UBP + s * rec
                X.memcpy_async(s64(DD + s * rec), s64(src), rec)
                e = torch.xpu.Event(); e.record(); inflight.append((s, e))
            else:
                free[s].release()
            done += 1
        # release ring slots whose copies completed
        keep = []
        for s, e in inflight:
            if e.query(): free[s].release()
            else: keep.append((s, e))
        inflight = keep
    sync()
    for s, _ in inflight: free[s].release()
    dt = time.perf_counter() - t0
    for f in futs: f.result()
    pool.shutdown(); os.close(nfd)
    a = np.array(read_ms)
    return dict(rec=rec, n=n_rec, threads=threads, ring=ring, copy=copy, bounce=bounce, ring_mem="memfd_thp", GB=round(n_rec * rec / 1e9, 2), s=round(dt, 3),
                GBps=round(n_rec * rec / dt / 1e9, 2), read_ms_median=round(float(np.median(a)), 3),
                read_ms_p90=round(float(np.percentile(a, 90)), 3))


try:
    _u = X.host_alloc(1 << 22); _fd = os.open("/q/qwen_experts.bin", os.O_RDONLY | os.O_DIRECT)
    _p = (_u.data_ptr() + 4095) // 4096 * 4096
    os.preadv(_fd, [(ctypes.c_char * (1 << 20)).from_address(_p)], 0); res["notes"].append("O_DIRECT into USM host: ok")
except Exception as e: res["notes"].append(f"O_DIRECT into L0 USM host memory fails: {e!r}")
stores = [("qwen", "/q/qwen_experts.bin", 1863680), ("glm", "/g/glm53_flash_exl3_3.05bpw_experts.bin", 9474048)]
for name, path, rec in (stores if "nvme" in PARTS else []):
    if not os.path.exists(path): res["notes"].append(f"missing {path}"); continue
    tgt = float(os.environ.get("NVME_GB", "8")) * 1e9      # bytes per case
    CASES = [tuple(int(v) for v in c.split(":")) for c in os.environ.get("NVME_CASES", "1:0:0,16:0:0,16:1:0,32:1:0,16:1:1").split(",")]
    for threads, copy, bounce in [(t, bool(c), bool(b)) for t, c, b in CASES]:
        if time.perf_counter() - T0 > TB: break
        n = int(tgt // rec) if threads > 1 else int(1.5e9 // rec)
        try:
            d = nvme_pipe(path, rec, n, threads, ring=max(2 * threads, 8), copy=copy, bounce=bounce,
                          base_rec=1000 + 37 * len(res["nvme"]) * 101)
            d["store"] = name; log("nvme", d)
        except Exception as e:
            log("nvme", dict(store=name, threads=threads, copy=copy, bounce=bounce, err=repr(e)[:300]))

res["meta"]["total_s"] = round(time.perf_counter() - T0, 1); dump()
uid = int(os.environ.get("HOST_UID", "-1"))
if uid >= 0:
    try: os.chown(OUT, uid, uid)
    except Exception: pass
print("DONE", flush=True)
os._exit(0)                      # do not wait on stuck pool threads at interpreter exit
