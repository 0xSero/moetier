"""HOM-272: C1 decode "what if X were free / at its hardware floor" calculator for GLM-5.3-Flash on the RTX 3090 tier.

Per MoE layer the critical path is   plan + bookkeeping + max(GPU lane, CPU start + CPU lane + flag latency) + non-MoE gap
(the non-MoE GPU work of the next layer), plus the step boundary (lm_head, sampling, dense layers 0-2) once per token.
Inputs are the measured per-token / per-layer means from a probe run (default: glm53-u6b-pfoff, prefetch off, 55 GB,
evening session) and the per-token byte volumes (picks x 9.44 MB records). Floors come from the hardware record:
VRAM 936 GB/s, PCIe 25 GB/s (copy engine), NVMe 23.9 GB/s, DDR 138 GB/s practical, CPU lane 98 GB/s (native layout cap).

  python3 examples/c1_ceiling.py [--run registry/runs/glm53-u6b-pfoff.json] [--json out.json]
"""
import argparse, json, os

REC = 9437184 / 1e9            # GB per expert record
NONEXP_GB = {"kda_attn": 3.148, "dsa_attn": 1.149, "shared_experts": 0.662, "lm_head": 0.476, "dense_mlp": 0.227,
             "attn_other": 0.227, "router": 0.099}       # per-token weight reads outside the routed experts (checkpoint sizes)
HW = {"vram": 936.0, "pcie": 25.0, "nvme": 23.9, "ddr": 138.0, "cpu_lane": 98.0}
L = 42


def model(m):
    """token ms from per-layer means: m has plan, book, gpu_copy, gpu_nvme, gpu_moe, cpu_start, cpu_lane, cpu_nvme,
    flag, gap, boundary, bubble (all ms per layer except boundary/bubble per token)"""
    gpu = m["gpu_copy"] + m["gpu_nvme"] + m["gpu_moe"]
    cpu = m["cpu_start"] + m["cpu_lane"] + m["flag"]
    layer = m["plan"] + m["book"] + max(gpu, cpu) + m["gap"]
    return L * layer + m["boundary"] + m["bubble"]


def measured(run):
    c = run["chokepoints"]["phases"]["decode_C1"]["decode"]
    d, pt, pl, q = c["detail_ms_per_token"], c["per_token_ms"], c["per_layer_ms"], c["quality"]
    pert = run["per_token"]
    m = {"plan": d["plan_wait"] / L, "book": d["dev_bookkeeping"] / L,
         "gpu_copy": pt["copy"] / L, "gpu_nvme": d["copy_nvme"] / L, "gpu_moe": pt["gpu_moe"] / L,
         "cpu_start": q.get("cpu_start_after_reply_ms", 0.08) or 0.08, "cpu_lane": d["cpu_lane_busy"] / L, "cpu_nvme": d["cpu_wait_nvme"] / L,
         "gap": d["fixed_in_step"] / L, "boundary": d["fixed_step_boundary"], "bubble": d["bubble_ub"]}
    # flag latency = whatever closes the measured per-layer CPU overrun beyond the CPU lane's own length
    gpu = m["gpu_copy"] + m["gpu_nvme"] + m["gpu_moe"]
    over = (pt["cpu_moe"] + d["cpu_wait_nvme"]) / L          # GPU waiting for the CPU partial, per layer
    m["flag"] = max(0.0, gpu + over - (m["cpu_start"] + m["cpu_lane"]))
    vol = {"cpu_experts": pert["cpu_experts"], "vram_hits": pert["vram_hits"], "admits": pert["zc_admits"],
           "nvme": pert["nvme"], "nvme_gpu": pert["nvme_gpu"], "nvme_cpu": pert["nvme_cpu"]}
    return m, vol, c["wall_ms_per_token"]


def scenarios(m, vol):
    out = []
    base = model(m)
    def add(name, kind, **chg):
        mm = dict(m); mm.update({k: (v(mm) if callable(v) else v) for k, v in chg.items()})
        t = model(mm)
        out.append({"scenario": name, "kind": kind, "ms_per_token": round(t, 1), "tok_s": round(1000 / t, 1), "saves_ms": round(base - t, 1)})
        return mm
    out.append({"scenario": "model of the measured run", "kind": "-", "ms_per_token": round(base, 1), "tok_s": round(1000 / base, 1), "saves_ms": 0.0})
    add("host plan + device bookkeeping = 0", "scheduling", plan=0, book=0)
    add("launch bubbles = 0", "scheduling", bubble=0)
    add("CPU-done flag latency = 0", "scheduling", flag=0)
    add("CPU job starts at publish (no reply wait)", "scheduling", cpu_start=0)
    add("NVMe waits = 0 (all misses landed in time)", "scheduling/prefetch", gpu_nvme=0, cpu_lane=lambda x: x["cpu_lane"] - x["cpu_nvme"])
    add("admission copy at 25 GB/s (copy engine)", "faster transfer", gpu_copy=lambda x: vol["admits"] * REC / HW["pcie"] * 1e3 / L)
    add("CPU experts at the DDR floor (98 GB/s native layout)", "faster kernel",
        cpu_lane=lambda x: vol["cpu_experts"] * REC / HW["cpu_lane"] * 1e3 / L + x["cpu_nvme"])
    add("GPU routed MoE at VRAM roofline", "faster kernel", gpu_moe=lambda x: (vol["vram_hits"] * REC / HW["vram"] + vol["nvme_gpu"] * REC / HW["pcie"]) * 1e3 / L)
    roof = sum(NONEXP_GB.values()) / HW["vram"] * 1e3
    add(f"non-MoE GPU at VRAM roofline ({sum(NONEXP_GB.values()):.1f} GB/token -> {roof:.1f} ms)", "faster kernels / graphs",
        gap=lambda x: roof * (x["gap"] * L / (x["gap"] * L + x["boundary"])) / L, boundary=lambda x: roof * (x["boundary"] / (x["gap"] * L + x["boundary"])))
    # everything scheduling-removable at once
    sched = dict(m); sched.update(plan=0, book=0, bubble=0, flag=0, cpu_start=0, gpu_nvme=0, cpu_lane=m["cpu_lane"] - m["cpu_nvme"])
    t = model(sched)
    out.append({"scenario": "ALL scheduling removable (plan, bubbles, flag, start, NVMe waits)", "kind": "scheduling", "ms_per_token": round(t, 1),
                "tok_s": round(1000 / t, 1), "saves_ms": round(base - t, 1)})
    floor = dict(sched)
    floor.update(gpu_copy=vol["admits"] * REC / HW["pcie"] * 1e3 / L, cpu_lane=vol["cpu_experts"] * REC / HW["cpu_lane"] * 1e3 / L,
                 gpu_moe=(vol["vram_hits"] * REC / HW["vram"] + vol["nvme_gpu"] * REC / HW["pcie"]) * 1e3 / L,
                 gap=roof * (m["gap"] * L / (m["gap"] * L + m["boundary"])) / L, boundary=roof * (m["boundary"] / (m["gap"] * L + m["boundary"])))
    t = model(floor)
    out.append({"scenario": "everything at its hardware floor, per-layer serialization kept (honest C1 ceiling, same bytes)",
                "kind": "ceiling", "ms_per_token": round(t, 1), "tok_s": round(1000 / t, 1), "saves_ms": round(base - t, 1)})
    # resource floors with perfect cross-layer overlap (bandwidth bounds)
    ddr = (vol["cpu_experts"] + vol["admits"] + vol["nvme_gpu"] + 2 * vol["nvme"] + vol["admits"]) * REC / HW["ddr"] * 1e3
    bounds = {"NVMe bytes / 23.9 GB/s": vol["nvme"] * REC / HW["nvme"] * 1e3,
              "PCIe H2D bytes (admits + zero-copy NVMe) / 25 GB/s": (vol["admits"] + vol["nvme_gpu"]) * REC / HW["pcie"] * 1e3,
              "CPU-lane bytes / 98 GB/s": vol["cpu_experts"] * REC / HW["cpu_lane"] * 1e3,
              "DDR bytes (CPU reads + H2D reads + NVMe writes + D2H write-backs + RAM->RAM) / 138 GB/s": ddr,
              "GPU: non-MoE + VRAM experts at roofline": roof + vol["vram_hits"] * REC / HW["vram"] * 1e3}
    return out, {k: {"ms": round(v, 1), "tok_s_bound": round(1000 / v, 1)} for k, v in bounds.items()}


def main():
    ap = argparse.ArgumentParser()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--run", default=os.path.join(root, "registry/runs/glm53-u6b-pfoff.json"))
    ap.add_argument("--json")
    a = ap.parse_args()
    run = json.load(open(a.run))
    m, vol, wall = measured(run)
    sc, bounds = scenarios(m, vol)
    print(f"run {run['id']}: measured {wall} ms/token; per-layer inputs " + json.dumps({k: round(v, 3) for k, v in m.items()}))
    print(f"volumes per token: " + json.dumps(vol))
    print("\n| scenario | kind | ms/token | tok/s | saves ms |\n|---|---|---|---|---|")
    for s in sc:
        print(f"| {s['scenario']} | {s['kind']} | {s['ms_per_token']} | {s['tok_s']} | {s['saves_ms']} |")
    print("\n| bandwidth bound (perfect overlap across layers) | ms/token | tok/s bound |\n|---|---|---|")
    for k, v in bounds.items():
        print(f"| {k} | {v['ms']} | {v['tok_s_bound']} |")
    if a.json:
        json.dump({"run": run["id"], "inputs_per_layer_ms": m, "volumes_per_token": vol, "scenarios": sc, "bounds": bounds}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
