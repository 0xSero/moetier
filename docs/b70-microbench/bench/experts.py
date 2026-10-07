# N124 job 2: exl3xpu routed-expert kernel cost on one Arc Pro B70 (decode path, M = 1..8 tokens).
# MODEL=qwen : Qwen3.8-Flash-Next 3.05bpw shapes (H 2560, I 640, E 512, top-10), real blobs from the NVMe store (/q)
# MODEL=glm  : GLM-5.3-Flash 3.05bpw shapes (H 4096, I 2048, E 288, top-8), random trellis (timing only; needs the
#              N124 lib with power-of-two k-splits: H/16 = 256 and I/16 = 128 are not divisible by 5)
# Tiers: vram (device slots), usm (pinned USM host, zero-copy over PCIe), svm (memfd THP system memory via xe SVM).
# Cases (moe_forward, pointer tables; events around back-to-back calls that rotate through 2 layers of experts so the
# 24 MB L2 never holds the next call's weights):
#   fit      M=1, k distinct experts (k = 1..topk)                     -> per_layer + per_expert * k
#   distinct M tokens x topk, all experts distinct (m = 1)               -> per_expert at scale
#   shared   M tokens on the same topk experts (m = M)                   -> per_extra_token
#   random   uniform routing                                            -> realistic mix
#   mixed    M=1/4 random routing, j of the picked experts host-resident -> zero-copy per-expert cost in a mixed call
#   cached   moe_forward_cached (server decode path), all hits vs all misses (zero-copy read + write-through)
import os, sys, json, time, ctypes
import numpy as np, torch
sys.path.insert(0, "/n")
from exl3xpu.moe_offload import ops, s64

X = ops(); dev = torch.device("xpu")
MODEL = os.environ.get("MODEL", "qwen")
OUT = os.environ.get("OUT", f"/o/experts_{MODEL}.json")
TB = float(os.environ.get("TBUDGET", "480")); T0 = time.perf_counter()
if MODEL == "qwen":
    H, I, E, TOPK, NL = 2560, 640, 512, 10, 48
else:
    H, I, E, TOPK, NL = 4096, 2048, 288, 8, 42
K = 3; BLOB = int(X.blob_bytes(H, I, K))
MiB2 = 2 << 20; HSTRIDE = (BLOB + MiB2 - 1) // MiB2 * MiB2
NLY = 2; LT = NLY + 1; S = NLY * E
MS = [1, 2, 4, 8]
X.moe_set_prefill_min_m(9)
if MODEL == "glm": X.moe_set_splits(8, 16)   # provisional (default 5/10 is invalid for H 4096 / I 2048)           # M <= 8 stays on the decode kernels (pair mode M <= 4, grouped MR=2 at M=8)
rng = np.random.default_rng(0); torch.manual_seed(0)
res = dict(meta=dict(model=MODEL, H=H, I=I, E=E, topk=TOPK, K=K, blob=BLOB, host_stride=HSTRIDE, layers_resident=NLY,
                     lib=os.environ.get("EXL3_MOE_LIB"), device=torch.xpu.get_device_name(0), torch=torch.__version__,
                     svm_keys={k: os.environ.get(k) for k in ("NEOReadDebugKeys", "EnableSharedSystemUsmSupport")}),
           splits=[], cases=[])


def dump():
    with open(OUT + ".tmp", "w") as f: json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def log(d, key="cases"):
    d["t"] = round(time.perf_counter() - T0, 1); print(json.dumps(d), flush=True); res[key].append(d); dump()


def sync(): torch.xpu.synchronize()
def ev(): return torch.xpu.Event(enable_timing=True)


# ------------------------------------------------------------------ blobs: host USM, memfd THP (SVM), VRAM slots
usm = X.host_alloc(LT * E * BLOB); UB = usm.data_ptr()
if MODEL == "qwen":
    REC = 1863680; SRC_LAYERS = [6, 30]
    nfd = os.open("/q/qwen_experts.bin", os.O_RDONLY)
    for li, L in enumerate(SRC_LAYERS):
        raw = np.empty(E * REC, dtype=np.uint8); mv = memoryview(raw); off = 0
        while off < raw.size:
            n = os.preadv(nfd, [mv[off:]], L * E * REC + off); assert n > 0; off += n
        usm[li * E * BLOB:(li + 1) * E * BLOB].copy_(torch.from_numpy(raw.reshape(E, REC)[:, :BLOB].copy()).flatten())
    res["meta"]["blobs"] = f"real, store layers {SRC_LAYERS}"
elif os.environ.get("GLM_REAL", "1") == "1":
    # real GLM-5.3-Flash experts (layer GLM_LAYER, experts 0..NREAL-1) packed with pack_expert, cycled over all slots
    from safetensors import safe_open
    from exl3xpu.moe_offload import pack_expert
    GL = int(os.environ.get("GLM_LAYER", "3")); NREAL = int(os.environ.get("NREAL", "16"))
    idx = json.load(open("/m/model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{GL}.mlp.experts."
    REAL = {}
    files = {}
    for e in range(NREAL):
        for r in ("gate", "up", "down"):
            for f in ("trellis", "suh", "svh"):
                k = f"{pre}{e}.{r}_proj.{f}"; files.setdefault(idx[k], []).append((e, r, f, k))
    for fn, lst in files.items():
        with safe_open(f"/m/{fn}", framework="pt") as sf:
            for e, r, f, k in lst: REAL.setdefault(e, {}).setdefault(r, {})[f] = sf.get_tensor(k)
    blobs = torch.stack([pack_expert(REAL[e]["gate"], REAL[e]["up"], REAL[e]["down"], K) for e in range(NREAL)])
    assert blobs.shape[1] == BLOB, (blobs.shape, BLOB)
    for k in range(S): usm[k * BLOB:(k + 1) * BLOB].copy_(blobs[k % NREAL])
    res["meta"]["blobs"] = f"real GLM layer {GL} experts 0..{NREAL - 1}, cycled over {S} slots (key k holds expert k % {NREAL})"
else:
    g = torch.Generator().manual_seed(0)
    tail = (3 * H + 3 * I) * 2
    one = torch.randint(0, 256, (E, BLOB), dtype=torch.uint8, generator=g)
    sg = (torch.randint(0, 2, (E, tail // 2), generator=g).to(torch.float16) * 2 - 1)
    one[:, BLOB - tail:] = sg.view(torch.uint8).view(E, tail)
    for li in range(NLY): usm[li * E * BLOB:(li + 1) * E * BLOB].copy_(one.flatten())
    del one
    res["meta"]["blobs"] = "random trellis, +-1 fp16 scales (timing only)"

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
SVM_SIZE = LT * E * HSTRIDE
mfd = os.memfd_create("n124", 0); os.ftruncate(mfd, SVM_SIZE)
r = libc.mmap(None, SVM_SIZE + MiB2, 0, 0x22, -1, 0); SB = (r + MiB2 - 1) // MiB2 * MiB2
assert libc.mmap(ctypes.c_void_p(SB), SVM_SIZE, 3, 0x11, mfd, 0) == SB
libc.madvise(SB, SVM_SIZE, 14)
for k in range(S): ctypes.memmove(SB + k * HSTRIDE, UB + k * BLOB, BLOB)
slots = torch.empty((S, BLOB), dtype=torch.uint8, device=dev); SLOT_BASE = s64(slots.data_ptr())
X.memcpy_async(SLOT_BASE, s64(UB), S * BLOB); sync()
res["meta"]["slot_copy_exact"] = all(torch.equal(slots[k].cpu(), usm[k * BLOB:(k + 1) * BLOB]) for k in (0, E + 7, S - 1))
res["meta"]["load_s"] = round(time.perf_counter() - T0, 1)


def addr(tier, key):
    return s64(SLOT_BASE + key * BLOB if tier == "vram" else (UB + key * BLOB if tier == "usm" else SB + key * HSTRIDE))


TABS = {t: [torch.tensor([addr(t, li * E + e) for e in range(E)], dtype=torch.int64, device=dev) for li in range(NLY)]
        for t in ("vram", "usm", "svm")}

# ------------------------------------------------------------------ splits (GLM: pick power-of-two k-splits)
x8 = torch.randn(8, H, dtype=torch.bfloat16, device=dev)


def fwd(x, ids, w, tab): return X.moe_forward(x, ids, w, tab, I, K, E)


def timeit(calls, min_ms=250.0, min_it=24, max_it=4000):
    for c in calls: c()
    sync()
    a, b = ev(), ev(); a.record(); calls[0](); b.record(); sync(); one = max(a.elapsed_time(b), 1e-3)
    n = max(len(calls), int(max(min_it, min(max_it, min_ms / one))))
    n = (n + len(calls) - 1) // len(calls) * len(calls)
    a, b = ev(), ev(); a.record()
    for i in range(n): calls[i % len(calls)]()
    b.record(); sync()
    return a.elapsed_time(b) / n, n


def ids_distinct(M, k, ncalls=48):
    """ncalls (layer, ids[M,k]) with all M*k experts distinct within a call, walking a permutation (no reuse nearby)"""
    out = []; perm = [rng.permutation(E) for _ in range(NLY)]; pos = [0] * NLY
    for c in range(ncalls):
        li = c % NLY; need = M * k
        if pos[li] + need > E: perm[li] = rng.permutation(E); pos[li] = 0
        out.append((li, perm[li][pos[li]:pos[li] + need].reshape(M, k).astype(np.int32))); pos[li] += need
    return out


def ids_shared(M, k, ncalls=48):
    return [(li, np.repeat(ids[:1], M, 0)) for li, ids in ids_distinct(1, k, ncalls)]


def ids_random(M, k, ncalls=48):
    return [(c % NLY, np.stack([rng.choice(E, k, replace=False) for _ in range(M)]).astype(np.int32)) for c in range(ncalls)]


def make_calls(sets, tier, host_frac=None):
    calls = []; info = []
    for li, ids in sets:
        M, k = ids.shape
        tab = TABS[tier][li]
        if host_frac is not None:            # mixed: j of the unique picked experts read from host tier `tier`
            u = np.unique(ids); j = int(round(host_frac * len(u)))
            hp = rng.choice(u, j, replace=False) if j else np.zeros(0, dtype=np.int64)
            t = TABS["vram"][li].clone()
            if j: t[torch.from_numpy(hp.astype(np.int64)).to(dev)] = TABS[tier][li][torch.from_numpy(hp.astype(np.int64)).to(dev)]
            tab = t; info.append(j)
        xs = x8[:M]
        di = torch.from_numpy(ids).to(dev); dw = torch.full((M, k), 1.0 / k, dtype=torch.float32, device=dev)
        calls.append(lambda xs=xs, di=di, dw=dw, tab=tab: fwd(xs, di, dw, tab))
    return calls, info


def over(): return time.perf_counter() - T0 > TB


# SVM first touch: map every memfd page on the GPU before timing
t = time.perf_counter()
for li in range(NLY):
    for c in range(0, E, TOPK):
        ids = torch.tensor([[(c + j) % E for j in range(TOPK)]], dtype=torch.int32, device=dev)
        fwd(x8[:1], ids, torch.full((1, TOPK), 0.1, device=dev), TABS["svm"][li])
sync(); res["meta"]["svm_first_touch_s"] = round(time.perf_counter() - t, 2)

if MODEL == "glm":
    cands = [(8, 16), (4, 4), (8, 4), (16, 16), (4, 16), (16, 4)]
    best = None; ref = None
    ids4 = torch.from_numpy(ids_random(4, TOPK, 1)[0][1]).to(dev); w4 = torch.full((4, TOPK), 0.125, device=dev)
    for gu, dn in cands:
        try:
            X.moe_set_splits(gu, dn)
            y = fwd(x8[:4], ids4, w4, TABS["vram"][0]).float(); sync()
            if ref is None: ref = y
            rel = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
            ms1, _ = timeit(make_calls(ids_distinct(1, TOPK), "vram")[0])
            ms4, _ = timeit(make_calls(ids_distinct(4, TOPK), "vram")[0])
            ms8, _ = timeit(make_calls(ids_distinct(8, TOPK), "vram")[0])
            d = dict(gu_p=gu, dn_p=dn, ms_m1=round(ms1, 4), ms_m4=round(ms4, 4), ms_m8=round(ms8, 4), rel_maxdiff_vs_first=rel,
                     finite=bool(torch.isfinite(y).all()))
            log(d, "splits")
            sc = ms1 + ms4 / 2 + ms8 / 4
            if best is None or sc < best[0]: best = (sc, gu, dn)
        except Exception as e:
            log(dict(gu_p=gu, dn_p=dn, err=repr(e)[:300]), "splits")
    X.moe_set_splits(best[1], best[2]); res["meta"]["splits_used"] = [best[1], best[2]]
else:
    res["meta"]["splits_used"] = "default (gu 5, dn 10 at M=1 else 5)"

# exactness across tiers (same blob bytes -> bit-identical output)
ids4 = torch.from_numpy(ids_random(4, TOPK, 1)[0][1]).to(dev); w4 = torch.full((4, TOPK), 1.0 / TOPK, device=dev)
yv, yu, ysv = (fwd(x8[:4], ids4, w4, TABS[t][1]) for t in ("vram", "usm", "svm")); sync()
res["meta"].update(exact_usm_vs_vram=bool(torch.equal(yv, yu)), exact_svm_vs_vram=bool(torch.equal(yv, ysv)),
                   out_finite=bool(torch.isfinite(yv.float()).all()))
dump()

# reference: moe kernel (VRAM, M=4 tokens, top-1 weight 1) vs exl3xpu_C.linear per projection, SiLU(g)*u, no clamp
if MODEL == "glm" and "REAL" in globals():
    from exl3xpu import ops as lops
    C = lops._get_esimd()
    def lin(x, d):
        t = d["trellis"].to(dev); n = t.shape[1] * 16
        sonb = torch.zeros(n // 128, dtype=torch.int32, device=dev)
        return torch.ops.exl3xpu_C.linear(x, t, d["suh"].to(dev).half().unsqueeze(0), d["svh"].to(dev).half(), sonb, [0, n], K, 2,
                                          lops.SMALL_M_MAX, lops.RECON_SLICE_N)
    xr = torch.randn(4, H, dtype=torch.float16, device=dev) * 0.5
    errs = []
    for e in range(4):
        g_, u_ = lin(xr, REAL[e]["gate"]).float(), lin(xr, REAL[e]["up"]).float()
        ref = lin((torch.nn.functional.silu(g_) * u_).half(), REAL[e]["down"]).float()
        ids = torch.full((4, 1), e, dtype=torch.int32, device=dev); w1 = torch.ones(4, 1, dtype=torch.float32, device=dev)
        per_split = {}
        for gu, dn in ((8, 16), (4, 4), (16, 16)):
            X.moe_set_splits(gu, dn)
            y = fwd(xr, ids, w1, TABS["vram"][0]).float(); sync()
            per_split[f"{gu}/{dn}"] = round(float((y - ref).norm() / ref.norm()), 5)
        errs.append(dict(expert=e, ref_norm=round(float(ref.norm()), 3), rel_l2_err=per_split,
                         gate_absmax=round(float(g_.abs().max()), 2), up_absmax=round(float(u_.abs().max()), 2)))
    X.moe_set_splits(8, 16)
    res["meta"]["ref_check"] = errs; print(json.dumps(errs), flush=True); dump()


def run(case, sets_fn, Ms, ks, tiers):
    for M in Ms:
        for k in ks:
            sets = sets_fn(M, k)
            nu = float(np.mean([len(np.unique(ids)) for _, ids in sets]))
            for tier in tiers:
                if over(): return
                ms, n = timeit(make_calls(sets, tier)[0])
                log(dict(case=case, M=M, k=k, tier=tier, unique=nu, pairs=M * k, ms=round(ms, 4), iters=n,
                         GBps_weights=round(nu * BLOB / ms / 1e6, 2)))


TIERS = ["vram", "usm", "svm"]
run("fit", ids_distinct, [1], sorted(set([1, 2, 4, 6, 8, TOPK])), TIERS)
run("distinct", ids_distinct, MS, [TOPK], TIERS)
run("shared", ids_shared, [2, 4, 8], [TOPK], TIERS)
run("random", ids_random, MS, [TOPK], TIERS)

# grouped (align) path with one MR-row chunk per expert: weights read once per expert instead of once per pair
MRS = [2, 4, 8] if MODEL == "qwen" else [2, 4]
for case, fn in (("shared_grouped", ids_shared), ("random_grouped", ids_random)):
    for M in MRS:
        sets = fn(M, TOPK); nu = float(np.mean([len(np.unique(ids)) for _, ids in sets]))
        for tier in ("vram", "usm"):
            if over(): break
            try:
                X.moe_set_pair_max_m(0); X.moe_set_mr(M)
                ms, n = timeit(make_calls(sets, tier)[0])
                log(dict(case=case, M=M, k=TOPK, tier=tier, mr=M, unique=nu, pairs=M * TOPK, ms=round(ms, 4), iters=n))
            except Exception as e:
                log(dict(case=case, M=M, tier=tier, mr=M, err=repr(e)[:300]))
            finally:
                X.moe_set_pair_max_m(4); X.moe_set_mr(0); sync()

for M in (1, 4):
    sets = ids_random(M, TOPK)
    for tier in ("usm", "svm"):
        for fr in (0.0, 0.2, 0.5, 1.0):
            if over(): break
            calls, info = make_calls(sets, tier, host_frac=fr)
            ms, n = timeit(calls)
            log(dict(case="mixed", M=M, k=TOPK, tier=tier, host_frac=fr, host_experts=float(np.mean(info)), ms=round(ms, 4), iters=n))

# ------------------------------------------------------------------ moe_forward_cached (server decode path)
if hasattr(X, "moe_forward_cached") and not over():
    n = LT * E
    ptrs_all = torch.zeros(n, dtype=torch.int64, device=dev)
    slot_of = torch.full((n,), -1, dtype=torch.int32, device=dev)
    slot_key = torch.full((S,), -1, dtype=torch.int32, device=dev)
    slot_last = torch.full((S,), -1, dtype=torch.int32, device=dev)
    tick = torch.ones(1, dtype=torch.int32, device=dev)
    host_base = torch.zeros(LT, dtype=torch.int64, device=dev)
    fill_all = torch.zeros(n, dtype=torch.int64, device=dev)
    fill_list = torch.zeros(1 + E, dtype=torch.int32, device=dev)
    pend = torch.zeros(0, dtype=torch.int32, device=dev); done_seq = torch.zeros(0, dtype=torch.int32, device=dev)
    z8, z32 = torch.zeros(0, dtype=torch.uint8, device=dev), torch.zeros(0, dtype=torch.int32, device=dev)
    X.moe_set_tier(z8, z32, z32); X.moe_set_victims(z32)
    DUMMY0 = NLY * E

    def configure(tier):
        if tier == "usm": X.moe_set_host_stride(0); hb = [UB + li * E * BLOB for li in range(LT)]
        else: X.moe_set_host_stride(HSTRIDE); hb = [SB + li * E * HSTRIDE for li in range(LT)]
        host_base.copy_(torch.tensor([s64(v) for v in hb], dtype=torch.int64))

    def build_state(tier, li, miss):
        pa = np.array([addr("vram", k) if k < S else addr(tier, k) for k in range(n)], dtype=np.int64)
        so = np.where(np.arange(n) < S, np.arange(n), -1).astype(np.int32)
        sk = np.arange(S, dtype=np.int32); sl = np.ones(S, dtype=np.int32)
        for j, e in enumerate(miss):
            k = li * E + int(e); d = DUMMY0 + j
            pa[k] = addr(tier, k); so[k] = -1; sk[k] = d; so[d] = k; pa[d] = addr("vram", k); sl[k] = 0
        tt = lambda a: torch.from_numpy(a).to(dev)
        return tt(pa), tt(so), tt(sk), tt(sl)

    def apply(st):
        pa, so, sk, sl = st
        ptrs_all.copy_(pa); slot_of.copy_(so); slot_key.copy_(sk); slot_last.copy_(sl); tick.fill_(1)

    def fc(li, xs, di, dw, mf=E):
        return X.moe_forward_cached(xs, di, dw, ptrs_all, li, slot_of, slot_key, slot_last, tick, host_base, SLOT_BASE,
                                    fill_all, fill_list, mf, I, K, E, pend, done_seq)

    # back-to-back (queue kept full, like a graph replay): all hits; and all misses with max_fill = 0 (zero-copy read,
    # no write-through, state unchanged between calls)
    for M in MS:
        sets = ids_random(M, TOPK, 16)
        for tier, mf in (("usm", 0.0), ("usm", 1.0), ("svm", 1.0)):
            if over(): break
            try:
                configure(tier); li0 = sets[0][0]
                apply(build_state(tier, 0, []))
                if mf:   # every picked expert of every call host-only (both layers)
                    pa = ptrs_all.cpu().numpy(); so = slot_of.cpu().numpy()
                    for li, ids in sets:
                        for e in np.unique(ids): k = li * E + int(e); pa[k] = addr(tier, k); so[k] = -1
                    ptrs_all.copy_(torch.from_numpy(pa)); slot_of.copy_(torch.from_numpy(so))
                calls = [(lambda li=li, di=torch.from_numpy(ids).to(dev), dw=torch.full((M, TOPK), 1.0 / TOPK, device=dev):
                          fc(li, x8[:M], di, dw, 0 if mf else E)) for li, ids in sets]
                ms, nit = timeit(calls)
                so = slot_of.cpu().numpy()
                still = int(sum((so[li * E + np.unique(ids)] == -1).sum() for li, ids in sets))
                log(dict(case="cached_b2b", M=M, k=TOPK, tier=tier, miss_frac=mf, max_fill=0 if mf else E, ms=round(ms, 4), iters=nit,
                         host_only_after=still))
            except Exception as e:
                log(dict(case="cached_b2b", M=M, tier=tier, miss_frac=mf, err=repr(e)[:300]))

    for M in MS:
        sets = ids_random(M, TOPK, 16)
        for tier, mf in (("usm", 0.0), ("usm", 1.0), ("svm", 1.0), ("usm", 0.5)):
            if over(): break
            configure(tier)
            per = []; nm = []
            prep = []
            for li, ids in sets:
                u = np.unique(ids); j = int(round(mf * len(u)))
                miss = np.sort(rng.choice(u, j, replace=False)) if j else []
                prep.append((li, build_state(tier, li, miss), torch.from_numpy(ids).to(dev),
                             torch.full((M, TOPK), 1.0 / TOPK, device=dev), len(miss), miss))
            ok = True
            for rep in range(4):
                evs = []
                for li, st, di, dw, nmiss, miss in prep:
                    apply(st); dc = di.clone(); a, b = ev(), ev(); a.record(); fc(li, x8[:M], dc, dw); b.record(); evs.append((a, b))
                    if rep == 0: nm.append(nmiss)
                sync()
                if rep == 0:
                    so = slot_of.cpu().numpy(); li, _, _, _, _, miss = prep[-1]
                    ok = all(so[li * E + int(e)] == li * E + int(e) for e in miss)
                if rep: per += [a.elapsed_time(b) for a, b in evs]
            a = np.array(per)
            log(dict(case="cached", M=M, k=TOPK, tier=tier, miss_frac=mf, misses=float(np.mean(nm)), ms=round(float(a.mean()), 4),
                     ms_median=round(float(np.median(a)), 4), ms_p90=round(float(np.percentile(a, 90)), 4), calls=len(per),
                     fills_landed=ok))
    configure("usm")

res["meta"]["total_s"] = round(time.perf_counter() - T0, 1); dump()
uid = int(os.environ.get("HOST_UID", "-1"))
if uid >= 0:
    try: os.chown(OUT, uid, uid)
    except Exception: pass
print("DONE", flush=True)
os._exit(0)                      # do not wait on stuck pool threads at interpreter exit
