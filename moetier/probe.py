"""moetier probe: resource utilisation and chokepoints of an offload engine during a bench.

  record   host-side sampler (stdlib only, no root): every ~100 ms writes one JSON line with
             gpu   NVML via ctypes (libnvidia-ml.so.1): kernel-active %, memory-controller busy %, PCIe rx/tx (20 ms
                   window), VRAM used, power, SM clock
             cpu   per-core busy fraction from /proc/stat for a cpuset
             disk  bytes read/written from /proc/diskstats (md array and members)
             mem   cgroup memory.current of the server's container
             live  the engine's live counters file (int64 page + .json field names), if the engine writes one:
                   decode/prefill step counters, CPU-lane work, NVMe bytes, critical-path attribution totals
  report   samples + phase marks (bench wall-clock windows) + the hardware record's ceilings ->
           `utilization` and `chokepoints` sections of a run record (schema: README "Run records")

DDR bandwidth: AMD Zen3 data-fabric/UMC counters are not exposed without root here (no amd_df PMU), so DDR traffic is
derived: CPU-lane weight reads (cpu experts x expert bytes) + PCIe rx (GPU DMA/zero-copy reads of host memory) +
PCIe tx (device -> host writes) + NVMe read bytes (DMA writes into RAM). Activations and page-table traffic are ignored.
Spin-waiting is counted as busy by both /proc/stat (CPU-lane workers spin between layers) and NVML (device kernels that
wait on a host flag are 'active'), so the report also carries the engine's own useful-work counters.
"""
import argparse, bisect, ctypes, json, math, os, signal, statistics, struct, subprocess, sys, time

# ---------------------------------------------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------------------------------------------


class Nvml:
    """Minimal NVML through ctypes (no pynvml needed)."""

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index):
        self.lib = None
        for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise OSError("libnvidia-ml not found")
        assert self.lib.nvmlInit_v2() == 0, "nvmlInit failed"
        self.h = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.h)) == 0
        name = ctypes.create_string_buffer(96)
        self.lib.nvmlDeviceGetName(self.h, name, ctypes.c_uint(96))
        self.name = name.value.decode()

    def sample(self):
        u, m = self.Util(), self.Mem()
        v = ctypes.c_uint()
        out = {}
        if self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0:
            out["util"], out["mem_busy"] = u.gpu, u.memory
        for k, which in (("pcie_tx_kbs", 0), ("pcie_rx_kbs", 1)):          # NVML_PCIE_UTIL_TX_BYTES / RX_BYTES (device view)
            if self.lib.nvmlDeviceGetPcieThroughput(self.h, ctypes.c_int(which), ctypes.byref(v)) == 0:
                out[k] = v.value
        if self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0:
            out["vram_used"] = m.used
        if self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(v)) == 0:
            out["power_mw"] = v.value
        if self.lib.nvmlDeviceGetClockInfo(self.h, ctypes.c_int(1), ctypes.byref(v)) == 0:   # NVML_CLOCK_SM
            out["sm_mhz"] = v.value
        return out


def cpuset(spec):
    out = []
    for part in str(spec).split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def read_stat():
    cur = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = [int(x) for x in p[1:9]]
                idle = v[3] + v[4]
                cur[int(p[0][3:])] = (sum(v) - idle, sum(v))
    return cur


def read_disks(devs):
    out = {}
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if len(p) > 9 and p[2] in devs:
                out[p[2]] = (int(p[5]) * 512, int(p[9]) * 512)
    return out


class Live:
    """The engine's live counters: one 4 KiB page of int64 (written ~every 20 ms) + <file>.json with field names."""

    def __init__(self, path):
        self.path, self.fields = path, None

    def read(self):
        try:
            if self.fields is None:
                self.fields = json.load(open(self.path + ".json"))["fields"]
            with open(self.path, "rb") as f:
                b = f.read(8 * len(self.fields))
            v = struct.unpack(f"<{len(b) // 8}q", b)
            return dict(zip(self.fields, v))
        except (OSError, ValueError, KeyError, struct.error):
            return None


def docker_cgroup(container):
    """cgroup v2 dir of a docker container (bounded docker call)."""
    try:
        cid = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", container], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return None
    d = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope"
    return d if cid and os.path.isdir(d) else None


def record(a):
    nv = None
    try:
        nv = Nvml(a.gpu)
    except Exception as ex:                                               # noqa: BLE001
        print(f"probe: NVML unavailable ({ex}); GPU fields omitted", file=sys.stderr)
    cpus = cpuset(a.cpus) if a.cpus else None
    devs = set(a.disks.split(","))
    live = Live(a.live) if a.live else None
    cg = a.cgroup or None
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    out = open(a.out, "a", buffering=1)
    out.write(json.dumps({"header": {"gpu": nv.name if nv else None, "gpu_index": a.gpu, "cpus": a.cpus, "disks": sorted(devs),
                                     "interval_s": a.interval, "live": a.live, "container": a.container,
                                     "started": time.time()}}) + "\n")
    prev_s, prev_d, prev_t = read_stat(), read_disks(devs), time.time()
    t_end = time.time() + a.duration if a.duration else None
    nxt = time.time()
    while not stop["now"] and (t_end is None or time.time() < t_end):
        if a.stop_file and os.path.exists(a.stop_file):
            break
        nxt += a.interval
        time.sleep(max(0.0, nxt - time.time()))
        if cg is None and a.container:
            cg = docker_cgroup(a.container)
        t = time.time()
        rec = {"t": round(t, 4), "dt": round(t - prev_t, 4)}
        if nv:
            rec["gpu"] = nv.sample()
        st = read_stat()
        busy = {c: (st[c][0] - prev_s[c][0]) / max(1, st[c][1] - prev_s[c][1]) for c in st if c in prev_s}
        rec["cpu"] = [round(busy.get(c, 0.0), 3) for c in (cpus or sorted(busy))]
        prev_s = st
        dk = read_disks(devs)
        rec["disk"] = {k: [dk[k][0] - prev_d[k][0], dk[k][1] - prev_d[k][1]] for k in dk if k in prev_d}
        prev_d, prev_t = dk, t
        if cg:
            try:
                rec["mem"] = int(open(cg + "/memory.current").read())
            except (OSError, ValueError):
                pass
        if live:
            v = live.read()
            if v:
                rec["live"] = v
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")
    out.close()


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------

DECODE_BUCKETS = ("fixed_gpu", "gpu_moe", "cpu_moe", "nvme_stall", "copy", "overhead")


def _q(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(xs, ceiling=None, idle_frac_of_ceiling=0.05):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.fmean(xs)
    o = {"mean": round(m, 3), "p50": round(_q(xs, 0.5), 3), "p90": round(_q(xs, 0.9), 3), "n": len(xs)}
    if ceiling:
        o["ceiling"] = ceiling
        o["pct_of_ceiling"] = {"mean": round(100 * m / ceiling, 1), "p50": round(100 * _q(xs, 0.5) / ceiling, 1),
                               "p90": round(100 * _q(xs, 0.9) / ceiling, 1)}
        o["idle_frac"] = round(sum(1 for x in xs if x < idle_frac_of_ceiling * ceiling) / len(xs), 3)
    return o


def load_samples(path):
    head, rows = None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "header" in r:
            head = head or r["header"]
            continue
        rows.append(r)
    return head, rows


def intervals(rows, slot_bytes, cpu_tier_idx, server_idx=None):
    """Per sample interval: kind (decode | prefill | mixed | idle) + resource values."""
    out = []
    for p, r in zip(rows, rows[1:]):
        dt = r["t"] - p["t"]
        if dt <= 0:
            continue
        lp, lr = p.get("live"), r.get("live")
        d = (lambda k: (lr.get(k, 0) - lp.get(k, 0)) if lp and lr else 0)
        dec, pf = d("dec_reqs") > 0, d("stage_layers") > 0 or d("pf_tokens") > 0
        kind = "mixed" if dec and pf else "decode" if dec else "prefill" if pf else ("idle" if lp and lr else "unknown")
        g = r.get("gpu", {})
        rx = g.get("pcie_rx_kbs", 0) * 1024 / 1e9                        # GB/s host -> device
        tx = g.get("pcie_tx_kbs", 0) * 1024 / 1e9
        md = sum(v[0] for k, v in r.get("disk", {}).items() if k.startswith("md")) / dt / 1e9
        cpu_all = r.get("cpu", [])
        cpu = [cpu_all[i] for i in server_idx if i < len(cpu_all)] if server_idx else cpu_all
        other = [cpu_all[i] for i in range(len(cpu_all)) if server_idx and i not in set(server_idx)]
        tier = [cpu_all[i] for i in cpu_tier_idx if i < len(cpu_all)]
        cpu_lane_gbps = d("cpu_experts") * slot_bytes / dt / 1e9 if lp else None
        v = {"gpu_util_pct": g.get("util"), "vram_busy_pct": g.get("mem_busy"), "pcie_h2d_gbps": rx, "pcie_d2h_gbps": tx,
             "nvme_read_gbps": md, "cpu_cores_busy_pct": 100 * statistics.fmean(cpu) if cpu else None,
             "cpu_tier_cores_busy_pct": 100 * statistics.fmean(tier) if tier else None,
             "host_other_cpus_busy_pct": 100 * statistics.fmean(other) if other else None,
             "cpu_lane_duty_pct": 100 * d("cpu_busy_ns") / 1e9 / dt if lp else None,
             "cpu_lane_weight_gbps": cpu_lane_gbps,
             "ddr_derived_gbps": (cpu_lane_gbps or 0) + rx + tx + md,
             "ram_gib": r["mem"] / 2 ** 30 if "mem" in r else None, "gpu_power_w": g.get("power_mw", 0) / 1000 or None}
        out.append({"t0": p["t"], "t1": r["t"], "dt": dt, "kind": kind, "v": v, "lp": lp, "lr": lr, "cpu": cpu_all})
    return out


def ceilings_of(hw):
    c = hw.get("ceilings", {})
    nv = c.get("pcie_nvml_sat_gbps")          # NVML PCIe counters include protocol bytes: saturate above copy GB/s
    return {"gpu_util_pct": 100.0, "vram_busy_pct": 100.0, "pcie_h2d_gbps": nv or c.get("pcie_h2d_gbps"),
            "pcie_d2h_gbps": nv or c.get("pcie_d2h_gbps"), "nvme_read_gbps": c.get("nvme_read_gbps"),
            "cpu_cores_busy_pct": 100.0, "cpu_tier_cores_busy_pct": 100.0, "cpu_lane_duty_pct": 100.0,
            "host_other_cpus_busy_pct": 100.0,
            "cpu_lane_weight_gbps": c.get("ddr_practical_gbps"), "ddr_derived_gbps": c.get("ddr_practical_gbps"),
            "ram_gib": None, "gpu_power_w": c.get("gpu_power_w")}


def attribution(lp, lr):
    """Decode critical-path attribution between two live snapshots (engine a_* totals)."""
    if not lp or not lr or "a_toks" not in lr:
        return None
    d = {k: lr[k] - lp.get(k, 0) for k in lr if k.startswith("a_")}
    tok = d["a_toks"]
    if tok <= 0:
        return None
    ms = lambda ns: round(ns / 1e6 / tok, 3)
    b = {"fixed_gpu": ms(d["a_fixed"] + d["a_sfixed"]), "gpu_moe": ms(d["a_gmoe"]), "cpu_moe": ms(d["a_ccrit"]),
         "nvme_stall": ms(d["a_copy_nv"] + d["a_ccrit_nv"]), "copy": ms(d["a_copy"]),
         "overhead": ms(d["a_plan"] + d["a_book"] + d["a_bubble"] + d["a_sbubble"])}
    wall = ms(d["a_wall"])
    last = {k: d[f"a_last_{k}"] for k in ("gpu", "cpu", "nvme")}
    lt = max(1, sum(last.values()))
    n = max(1, d["a_n"])
    steps = max(1, d["a_steps"])
    return {"tokens": tok, "steps": d["a_steps"], "tokens_per_step": round(tok / steps, 2), "layer_calls": d["a_n"],
            "wall_ms_per_token": wall, "tok_s_from_wall": round(1000 / wall, 2) if wall else None,
            "per_token_ms": b, "share": {k: round(v / wall, 3) for k, v in b.items()} if wall else None,
            "binding": max(b, key=b.get),
            "binding_lane": max(last, key=last.get),
            "moe_stage_ms_per_token": round(b["copy"] + b["nvme_stall"] + b["gpu_moe"] + b["cpu_moe"], 3),
            "detail_ms_per_token": {"plan_wait": ms(d["a_plan"]), "dev_bookkeeping": ms(d["a_book"]),
                                    "bubble_ub": ms(d["a_bubble"] + d["a_sbubble"]), "fixed_in_step": ms(d["a_fixed"]),
                                    "fixed_step_boundary": ms(d["a_sfixed"]), "copy_nvme": ms(d["a_copy_nv"]),
                                    "cpu_wait_nvme": ms(d["a_ccrit_nv"]), "cpu_lane_busy": ms(d["a_cpu_lane"]),
                                    "cpu_dispatch": ms(d["a_cpu_disp"]), "gpu_lane": ms(d["a_gpu_lane"])},
            "lane_last": {k: round(v / lt, 3) for k, v in last.items()},
            "slack_ms_per_layer": {"gpu_waiting_on_cpu": round(d["a_gslack"] / 1e6 / n, 4),
                                   "cpu_idle_before_gpu": round(d["a_cslack"] / 1e6 / max(1, d["a_cpu_jobs"]), 4)},
            "per_layer_ms": {"gpu_lane": round(d["a_gpu_lane"] / 1e6 / n, 4), "cpu_lane": round(d["a_cpu_lane"] / 1e6 / max(1, d["a_cpu_jobs"]), 4),
                             "copy": round((d["a_copy"] + d["a_copy_nv"]) / 1e6 / n, 4), "nvme_wait": round((d["a_copy_nv"] + d["a_ccrit_nv"]) / 1e6 / n, 4)},
            "per_step": {"cpu_experts": round(d["a_cpux"] / steps, 1), "nvme_picks": round(d["a_nvpicks"] / steps, 1)},
            "quality": {"lost": d["a_lost"], "bad_order": d["a_bad"], "no_combine": d["a_nocomb"], "idle_gaps": d["a_idle_n"],
                        "launch_bound_frac": round(d["a_lbound"] / n, 3), "launch_ring_miss": d["a_lmiss"],
                        "cpu_start_after_reply_ms": round(d.get("a_cstart", 0) / 1e6 / max(1, d["a_cpu_jobs"]), 4),
                        "cpu_end_to_seen_ms": round(d.get("a_hlag", 0) / 1e6 / max(1, d.get("a_hlagn", 0)), 4),
                        "cpu_end_to_seen_wb_busy_ms": round(d.get("a_hlagwb", 0) / 1e6 / max(1, d.get("a_hlagwbn", 0)), 4),
                        "wb_busy_frac_of_waits": round(d.get("a_hlagwbn", 0) / max(1, d.get("a_hlagn", 0)), 3)}}


def prefill_attr(lp, lr, wall_s):
    if not lp or not lr or "pf_tokens" not in lr:
        return None
    d = {k: lr.get(k, 0) - lp.get(k, 0) for k in ("pf_tokens", "pf_forwards", "pf_stall_us", "pf_gap_us", "pf_span_us",
                                                   "stage_layers", "stage_reads")}
    if d["pf_tokens"] <= 0:
        return None
    f = max(1, d["pf_forwards"])
    span, stall, gap = d["pf_span_us"] / 1e3, d["pf_stall_us"] / 1e3, d["pf_gap_us"] / 1e3
    b = {"gpu_compute": round(max(0.0, span - stall), 1), "staging_stall": round(stall, 1), "host_gap": round(gap, 1)}
    tb = sum(b.values())
    return {"tokens": d["pf_tokens"], "forwards": d["pf_forwards"], "tok_s_active": round(d["pf_tokens"] / wall_s, 1) if wall_s else None,
            "ms_total": b, "ms_per_forward": {k: round(v / f, 1) for k, v in b.items()},
            "share": {k: round(v / tb, 3) for k, v in b.items()} if tb else None,
            "binding": max(b, key=b.get), "nvme_reads": d["stage_reads"],
            "note": "PFPROF stall/span of forward k are booked when forward k+1 begins (one forward lag)"}


def report(a):
    head, rows = load_samples(a.samples)
    hw = json.load(open(a.hardware))
    ceil = ceilings_of(hw)
    marks = []
    if a.marks:
        mj = json.load(open(a.marks))
        marks = mj.get("marks", mj) if isinstance(mj, dict) else mj
    lf = json.load(open(a.live_json)) if a.live_json else {}
    slot = lf.get("slot_bytes", a.slot_bytes)
    allc = cpuset(head["cpus"]) if head and head.get("cpus") else []
    tier = set(cpuset(a.cpu_tier_cpus)) if a.cpu_tier_cpus else set()
    tier_idx = [i for i, c in enumerate(allc) if c in tier]
    srv = set(cpuset(a.server_cpus)) if a.server_cpus else None
    server_idx = [i for i, c in enumerate(allc) if c in srv] if srv else None
    iv = intervals(rows, slot, tier_idx, server_idx)
    if not marks:
        marks = [{"phase": "all", "t0": iv[0]["t0"], "t1": iv[-1]["t1"]}] if iv else []
    starts = [x["t0"] for x in iv]
    util, choke = {"method": method_text(head, a), "phases": {}}, {"phases": {}}
    for m in marks:
        lo, hi = bisect.bisect_left(starts, m["t0"]), bisect.bisect_right(starts, m["t1"])
        sel = [x for x in iv[lo:hi] if x["t1"] <= m["t1"] + 1e-6]
        if not sel:
            continue
        ph = {}
        for kind in ("decode", "prefill"):
            ks = [x for x in sel if x["kind"] == kind]
            if not ks:
                continue
            ph[kind] = {"active_s": round(sum(x["dt"] for x in ks), 1),
                        "resources": {k: _stats([x["v"][k] for x in ks], ceil.get(k)) for k in ks[0]["v"]}}
            # per-core busy (mean over active time), for placement questions
            cores = [statistics.fmean(c) for c in zip(*[x["cpu"] for x in ks if x["cpu"]])] if ks[0]["cpu"] else []
            if cores and allc:
                ph[kind]["per_core_busy_pct"] = {str(c): round(100 * b, 1) for c, b in zip(allc, cores)}
        tot = sum(x["dt"] for x in sel)
        ph["time_split"] = {k: round(sum(x["dt"] for x in sel if x["kind"] == k) / tot, 3)
                            for k in ("decode", "prefill", "mixed", "idle", "unknown")}
        util["phases"][m["phase"]] = ph
        lp, lr = sel[0]["lp"], sel[-1]["lr"]
        c = {"decode": attribution(lp, lr),
             "prefill": prefill_attr(lp, lr, sum(x["dt"] for x in sel if x["kind"] == "prefill"))}
        c["binding"] = binding(c, ph, ceil)
        choke["phases"][m["phase"]] = c
    return {"utilization": util, "chokepoints": choke}


def binding(c, ph, ceil):
    """One line per phase: the critical-path bucket that dominates, and the resource closest to its ceiling."""
    out = {}
    for kind in ("decode", "prefill"):
        at = c.get(kind)
        res = (ph.get(kind) or {}).get("resources") or {}
        near = sorted(((v["pct_of_ceiling"]["mean"], k) for k, v in res.items()
                       if v and "pct_of_ceiling" in v and k not in ("gpu_util_pct", "cpu_cores_busy_pct", "cpu_tier_cores_busy_pct", "gpu_power_w", "host_other_cpus_busy_pct")),
                      reverse=True)
        if at or near:
            out[kind] = {"critical_path": at["binding"] if at else None,
                         "lane_last": at.get("binding_lane") if at else None,
                         "critical_share": (at["share"] or {}).get(at["binding"]) if at and at.get("share") else None,
                         "resource_nearest_ceiling": {"resource": near[0][1], "pct_mean": near[0][0]} if near else None}
    return out


MD_RES = [("gpu_util_pct", "GPU util %"), ("vram_busy_pct", "VRAM busy %"), ("pcie_h2d_gbps", "PCIe H2D GB/s"),
          ("pcie_d2h_gbps", "PCIe D2H GB/s"), ("nvme_read_gbps", "NVMe GB/s"), ("cpu_lane_duty_pct", "CPU-lane duty %"),
          ("cpu_tier_cores_busy_pct", "CPU-tier cores busy %"), ("cpu_lane_weight_gbps", "CPU-lane weights GB/s"),
          ("ddr_derived_gbps", "DDR derived GB/s"), ("ram_gib", "RAM GiB")]


def markdown(res):
    """Compact tables: utilization (mean / p90 / % of ceiling) per phase and kind, decode attribution per phase."""
    U, C = res["utilization"]["phases"], res["chokepoints"]["phases"]
    out = []
    for kind in ("decode", "prefill"):
        phs = [p for p in U if kind in U[p] and U[p][kind]["active_s"] >= 5]
        if not phs:
            continue
        out.append(f"\n{kind} (active windows): mean / p90 (% of ceiling, mean)\n")
        out.append("| resource | " + " | ".join(phs) + " |")
        out.append("|---|" + "---|" * len(phs))
        for k, lab in MD_RES:
            cells = []
            for p in phs:
                v = U[p][kind]["resources"].get(k)
                if not v:
                    cells.append("-"); continue
                pc = f" ({v['pct_of_ceiling']['mean']:.0f}%)" if "pct_of_ceiling" in v and k not in ("gpu_util_pct", "vram_busy_pct", "cpu_lane_duty_pct", "cpu_tier_cores_busy_pct") else ""
                cells.append(f"{v['mean']:.1f} / {v['p90']:.1f}{pc}")
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
    phs = [p for p in C if C[p].get("decode") and C[p]["decode"]["tokens"] >= 200]
    if phs:
        out.append("\ndecode critical path, ms per token (share)\n")
        out.append("| bucket | " + " | ".join(phs) + " |")
        out.append("|---|" + "---|" * len(phs))
        for b in DECODE_BUCKETS:
            out.append(f"| {b} | " + " | ".join(f"{C[p]['decode']['per_token_ms'][b]:.1f} ({100 * (C[p]['decode']['share'] or {}).get(b, 0):.0f}%)" for p in phs) + " |")
        out.append("| **wall** | " + " | ".join(f"**{C[p]['decode']['wall_ms_per_token']:.1f}** ({C[p]['decode']['tok_s_from_wall']} tok/s)" for p in phs) + " |")
        out.append("| lane last gpu/cpu/nvme | " + " | ".join("/".join(f"{C[p]['decode']['lane_last'][k]:.2f}" for k in ("gpu", "cpu", "nvme")) for p in phs) + " |")
        out.append("| binding | " + " | ".join(f"{C[p]['decode']['binding']} (lane {C[p]['decode'].get('binding_lane')})" for p in phs) + " |")
    return "\n".join(out)


def method_text(head, a):
    return {"sampler": "moetier probe record (stdlib + NVML ctypes)", "interval_s": (head or {}).get("interval_s"),
            "active_windows": "per sample interval from the engine's live counters: decode = dec_reqs advanced, prefill = "
                              "staged layers / prefill tokens advanced, mixed = both (excluded), idle = neither",
            "gpu_util_pct": "NVML utilization.gpu: % of time a kernel was resident (spin-waiting kernels count as busy)",
            "vram_busy_pct": "NVML utilization.memory: % of time the memory controller was busy (time-based, not GB/s; "
                             "an upper bound on VRAM bandwidth use vs the 936 GB/s ceiling)",
            "pcie": "NVML PCIe throughput rx/tx (device view, 20 ms window per sample)",
            "nvme_read_gbps": "/proc/diskstats sectors read on the md array",
            "cpu": "/proc/stat busy per core over the server cpuset (spin counts as busy); cpu_lane_duty_pct = engine CPU "
                   "job time / wall (useful work)",
            "ddr_derived_gbps": "derived (no DF/UMC counters without root): CPU-lane weight bytes + PCIe rx + PCIe tx + NVMe "
                                "read bytes", "ceilings": "hardware record `ceilings`"}


# ---------------------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moetier probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="sample resources every --interval s into a JSONL file (host side)")
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=0.1)
    r.add_argument("--duration", type=float, default=0, help="seconds (0 = until SIGTERM / --stop-file)")
    r.add_argument("--stop-file")
    r.add_argument("--gpu", type=int, default=0, help="NVML index")
    r.add_argument("--cpus", help="cpuset to sample, e.g. 2-39 (default: all)")
    r.add_argument("--disks", default="md127,nvme0n1,nvme1n1,nvme2n1,nvme3n1")
    r.add_argument("--container", help="docker container name (cgroup memory.current)")
    r.add_argument("--cgroup", help="cgroup dir (instead of --container)")
    r.add_argument("--live", help="engine live counters file")
    p = sub.add_parser("report", help="samples + marks + hardware ceilings -> utilization / chokepoints JSON")
    p.add_argument("--samples", required=True)
    p.add_argument("--hardware", required=True, help="registry/hardware/<id>.json")
    p.add_argument("--marks", help="JSON with marks: [{phase, t0, t1}] (wall clock), e.g. the bench output")
    p.add_argument("--live-json", help="the engine's <live>.json sidecar (slot bytes)")
    p.add_argument("--slot-bytes", type=int, default=9437184)
    p.add_argument("--cpu-tier-cpus", help="cpuset of the CPU-lane workers, e.g. 2-23")
    p.add_argument("--server-cpus", help="server cpuset when the samples cover more CPUs (others -> host_other_cpus_busy_pct)")
    p.add_argument("--out")
    p.add_argument("--md", action="store_true", help="also print markdown tables")
    m = sub.add_parser("md", help="markdown tables from a report JSON or a run record")
    m.add_argument("report")
    a = ap.parse_args(argv)
    if a.cmd == "md":
        r = json.load(open(a.report))
        print(markdown(r))
        return
    if a.cmd == "record":
        record(a)
        return
    res = report(a)
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    else:
        print(s)
    if a.md:
        print(markdown(res))


if __name__ == "__main__":
    main()
