"""HOM-272 B70 cross-stack profiling hook (B70PROF=1). Observes only; the computation is unchanged unless B70_ROUTE=1.

Per ModelRunner.forward call (one model step) it records host perf_counter stamps and XPU events (no syncs):
  e0 before the forward (graph replay), e1 at the start of the NVMe-tier host step (after the replay is enqueued),
  e2 after the forward returns. Events are resolved lazily once complete (query()), so no extra waits are added.
Per NvTierAsync.step call: duration, its snapshot wait (stats sync_ms delta), tier stat deltas, and the per-step
VRAM-slot / RAM-flag diff of the snapshot the step consumed (admissions = keys that gained a VRAM slot = RAM-tier SVM
reads written through; victims = keys that lost a slot; victims whose RAM copy was absent = write-backs).
Per ModelRunner.sample call: host time and events.
Profiler: touch $B70PROF_OUT/PROF_ON (content = number of model steps, default 40) -> torch.profiler CPU+XPU over the
next N forward calls, chrome trace + key_averages written to $B70PROF_OUT, PROF_ON renamed to PROF_DONE.<n>.
Routing (B70_ROUTE=1, experimental Edge0 arms): $B70PROF_OUT/ROUTE holds "K tau"; picks outside the top-K by weight or
with renormalised weight < tau become (safe expert 0, weight 0) and the kept weights are rescaled to the original sum.
Device tensors hold K and tau, so graphs capture the transform once and the arm switches at a step boundary.
"""
import json, os, threading, time

import torch

OUT = os.environ.get("B70PROF_OUT", "/b70out")
_inst = False
_rows = []
_pending = []
_lock = threading.Lock()
_state = dict(n=0, prof=None, prof_left=0, prof_id=0, last_e2=None, cur=None, route=(10, 0.0), route_seq=0)
_fh = None


def _w(rows):
    global _fh
    if _fh is None:
        os.makedirs(OUT, exist_ok=True)
        _fh = open(os.path.join(OUT, f"steps.{os.getpid()}.jsonl"), "a", buffering=1 << 20)
    for r in rows:
        _fh.write(json.dumps(r) + "\n")
    _fh.flush()


def _ev():
    e = torch.xpu.Event(enable_timing=True)
    e.record()
    return e


def _resolve(force=False):
    done = []
    keep = []
    for r in _pending:
        ev = r.pop("_ev", None)
        if ev is None:
            done.append(r); continue
        try:
            ok = ev["e2"].query() if not force else (ev["e2"].synchronize() or True)
        except Exception:
            ok = False
        if not ok:
            r["_ev"] = ev; keep.append(r); continue
        try:
            r["g_fwd_ms"] = ev["e0"].elapsed_time(ev["e1"]) if ev.get("e1") is not None else None
            r["g_tier_ms"] = ev["e1"].elapsed_time(ev["e2"]) if ev.get("e1") is not None else None
            r["g_total_ms"] = ev["e0"].elapsed_time(ev["e2"])
            if ev.get("prev_e2") is not None:
                r["g_gap_ms"] = ev["prev_e2"].elapsed_time(ev["e0"])     # device time between steps (idle + sample)
            if ev.get("s0") is not None:
                r["g_sample_ms"] = ev["s0"].elapsed_time(ev["s1"])
        except Exception as e:  # pragma: no cover
            r["ev_err"] = repr(e)[:120]
        done.append(r)
    _pending[:] = keep
    _rows.extend(done)
    if len(_rows) >= 256 or (force and _rows):
        t = time.perf_counter()
        _w(_rows); _rows.clear()
        if _state["cur"] is not None:
            _state["cur"]["flush_ms"] = (time.perf_counter() - t) * 1e3


def _poll_files():
    p = os.path.join(OUT, "PROF_ON")
    if _state["prof"] is None and os.path.exists(p):
        try:
            n = int(open(p).read().strip() or 40)
        except Exception:
            n = 40
        from torch.profiler import profile, ProfilerActivity
        pr = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=False, with_stack=False)
        pr.__enter__()
        _state.update(prof=pr, prof_left=n)
        _state["prof_id"] += 1
        os.rename(p, os.path.join(OUT, f"PROF_RUNNING.{_state['prof_id']}"))
    if os.environ.get("B70_ROUTE", "0") == "1":
        rp = os.path.join(OUT, "ROUTE")
        try:
            k, tau = open(rp).read().split()[:2]
            k, tau = int(k), float(tau)
        except Exception:
            k, tau = 10, 0.0
        if (k, tau) != _state["route"]:
            _route_set(k, tau)


def _prof_stop():
    pr = _state["prof"]
    _state["prof"] = None
    pr.__exit__(None, None, None)
    i = _state["prof_id"]
    try:
        pr.export_chrome_trace(os.path.join(OUT, f"trace{i}.{os.getpid()}.json"))
    except Exception as e:
        open(os.path.join(OUT, f"trace{i}.err"), "w").write(repr(e))
    try:
        ka = pr.key_averages()
        with open(os.path.join(OUT, f"kavg{i}.{os.getpid()}.txt"), "w") as f:
            f.write(ka.table(sort_by="self_xpu_time_total", row_limit=80))
    except Exception as e:
        open(os.path.join(OUT, f"kavg{i}.err"), "w").write(repr(e))
    try:
        os.rename(os.path.join(OUT, f"PROF_RUNNING.{i}"), os.path.join(OUT, f"PROF_DONE.{i}"))
    except Exception:
        pass


def _fb_info(a, k):
    fb = k.get("forward_batch") or (a[0] if a else None)
    try:
        mode = fb.forward_mode.name
    except Exception:
        mode = "?"
    try:
        bs = int(fb.batch_size)
    except Exception:
        bs = -1
    try:
        nt = int(fb.input_ids.numel())
    except Exception:
        nt = -1
    return mode, bs, nt


def _patch_runner():
    from sglang.srt.model_executor import model_runner as mr
    orig_fwd = mr.ModelRunner.forward
    orig_sample = getattr(mr.ModelRunner, "sample", None)

    def forward(self, *a, **k):
        st = _state
        st["n"] += 1
        if st["n"] % 16 == 0:
            _poll_files()
        if torch.xpu.is_current_stream_capturing():
            return orig_fwd(self, *a, **k)
        mode, bs, nt = _fb_info(a, k)
        t0 = time.perf_counter()
        r = dict(i=st["n"], t0=t0, w=time.time(), mode=mode, bs=bs, nt=nt, tid=threading.get_ident() % 100000,
                 prof=int(st["prof"] is not None), route=list(st["route"]))
        if st.get("last_t2") is not None:
            r["h_gap_ms"] = (t0 - st["last_t2"]) * 1e3
        ev = dict(e0=_ev(), prev_e2=st["last_e2"])
        st["cur"] = r
        st["cur_ev"] = ev
        out = orig_fwd(self, *a, **k)
        ev["e2"] = _ev()
        t2 = time.perf_counter()
        r["h_fwd_ms"] = (t2 - t0) * 1e3
        r["_ev"] = ev
        st["last_e2"] = ev["e2"]; st["last_t2"] = t2
        _pending.append(r)
        if st["prof"] is not None:
            st["prof_left"] -= 1
            if st["prof_left"] <= 0:
                torch.xpu.synchronize()
                _prof_stop()
        if st["n"] % 64 == 0:
            try:
                from . import ngram_host
                for tb in list(ngram_host._TABLES.values()):
                    g = st["n"] % 512 == 0
                    r["ng"] = tb.stats(gpu=g); r["ng_gpu"] = int(g)
            except Exception as e:
                r["ng_err"] = repr(e)[:100]
        _resolve()
        return out
    mr.ModelRunner.forward = forward

    if orig_sample is not None:
        def sample(self, *a, **k):
            t = time.perf_counter()
            r = _state.get("cur")
            ev = _state.get("cur_ev")
            s0 = _ev() if ev is not None else None
            out = orig_sample(self, *a, **k)
            if ev is not None:
                ev["s0"], ev["s1"] = s0, _ev()
            if r is not None:
                r["h_sample_ms"] = (time.perf_counter() - t) * 1e3
                r["t_sample"] = t
            return out
        mr.ModelRunner.sample = sample


def _patch_tier():
    import numpy as np
    from . import nvtier
    cls = nvtier.NvTierAsync
    orig = cls.step
    keys = ("masked", "evicts", "admit_dropped", "nvme_reads", "nvme_ms", "sync_ms", "fill_ms", "evict_rescued")
    prev = {}

    def step(self):
        ev = _state.get("cur_ev")
        if ev is not None and "e1" not in ev:
            ev["e1"] = _ev()
        b = {kk: self.stats.get(kk, 0) for kk in keys}
        t = time.perf_counter()
        n_inf0 = int(self.inflight.sum()) if hasattr(self, "inflight") else -1
        ret = orig(self)
        dt = (time.perf_counter() - t) * 1e3
        r = _state.get("cur")
        if r is None:
            return ret
        r["tier_ms"] = dt
        for kk in keys:
            d = self.stats.get(kk, 0) - b[kk]
            r["t_" + kk] = round(d, 4) if isinstance(d, float) else d
        r["inflight0"] = n_inf0
        j = self._k
        if self._snap_ev[j] is not None:
            slot = self._h_slot[j].numpy()
            res = self._h_res[j].numpy()
            v = slot >= 0
            if "v" in prev:
                pv, pres = prev["v"], prev["res"]
                gained = v & ~pv
                lost = pv & ~v
                r["s_admit"] = int(gained.sum())
                r["s_victim"] = int(lost.sum())
                r["s_victim_wb"] = int((lost & (pres == 0)).sum())
            r["s_vram"] = int(v.sum())
            r["s_ram"] = int(res.sum())
            r["s_ram_only"] = int(((res == 1) & ~v).sum())
            prev["v"] = v.copy(); prev["res"] = res.copy()
        return ret
    cls.step = step


_RT = {}


def _route_set(k, tau):
    _state["route"] = (k, tau)
    if _RT:
        _RT["K"].fill_(k); _RT["tau"].fill_(tau)
    try:
        open(os.path.join(OUT, "ROUTE.applied"), "a").write(f"{time.time():.3f} step {_state['n']} K {k} tau {tau}\n")
    except Exception:
        pass


def _patch_route():
    from . import sglang_plugin as sp
    sp.classes()
    M = sp.Exl3XpuMoEMethod
    orig = M.apply

    def apply(self, layer, dispatch_output):
        if not _RT:
            dev = dispatch_output.hidden_states.device
            k0, t0 = _state["route"]
            _RT["K"] = torch.full((1,), k0, dtype=torch.int32, device=dev)
            _RT["tau"] = torch.full((1,), t0, dtype=torch.float32, device=dev)
        tk = dispatch_output.topk_output
        w, ids = tk.topk_weights, tk.topk_ids
        wf = w.float()
        s = wf.sum(-1, keepdim=True)
        wn = wf / s.clamp_min(1e-20)
        # rank without sort (graph-safe): number of picks with larger weight, ties broken by position
        a_ = wn.unsqueeze(-1); b_ = wn.unsqueeze(-2)
        n = wn.shape[-1]
        tri = torch.ones(n, n, dtype=torch.bool, device=wn.device).tril(-1)
        rank = ((b_ > a_) | ((b_ == a_) & tri)).sum(-1)
        keep = ((rank < _RT["K"]) & (wn >= _RT["tau"])) | (rank == 0)
        wk = torch.where(keep, wf, torch.zeros_like(wf))
        wk = wk * (s / wk.sum(-1, keepdim=True).clamp_min(1e-20))
        ids2 = torch.where(keep, ids, torch.zeros_like(ids))
        tk2 = tk._replace(topk_weights=wk.to(w.dtype), topk_ids=ids2)
        return orig(self, layer, dispatch_output._replace(topk_output=tk2))
    M.apply = apply


def install():
    global _inst
    if _inst:
        return
    _inst = True
    try:
        _patch_runner()
    except Exception as e:
        print(f"b70prof: runner patch failed {e!r}", flush=True)
    try:
        _patch_tier()
    except Exception as e:
        print(f"b70prof: tier patch failed {e!r}", flush=True)
    if os.environ.get("B70_ROUTE", "0") == "1":
        try:
            _patch_route()
            print("b70prof: routing transform installed", flush=True)
        except Exception as e:
            print(f"b70prof: route patch failed {e!r}", flush=True)
    print(f"b70prof: installed pid {os.getpid()}", flush=True)
