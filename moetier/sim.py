"""Replay a routing trace through ledger + plan. Same scheduler as runtime; time is modeled from the records."""
import numpy as np
from .ledger import Ledger
from .plan import NvmeChannel, plan_layer, prefetch


def load_trace(path, segments=None):
    tr = np.load(path).astype(np.int32)                  # [tokens, layers, topk] expert ids
    seg = np.load(segments) if segments else np.array([0, len(tr)])
    return [tr[seg[i]:seg[i + 1]] for i in range(len(seg) - 1)]


def run(R, streams, conc=1, warm=200, max_steps=0, window=1, accept=1.0, draft_ms=0.0):
    """window/accept/draft_ms: speculative decoding what-if. Each step verifies `window` consecutive trace tokens per
    stream (their expert union, m tokens per expert), credits `accept` tokens per stream, and adds `draft_ms`."""
    L, E = R.layers, R.experts
    cat = np.concatenate(streams)
    freq = np.zeros(L * E)
    for l in range(L):
        np.add.at(freq, l * E + cat[:, l, :R.topk].reshape(-1), 1)
    led = Ledger(R.vram_slots, R.ram_slots, L * E, exclusive=R.policy.get("ram", "exclusive") == "exclusive")
    led.seed([int(k) for k in np.argsort(-freq)])
    nv = NvmeChannel(R.nvme_expert_bytes, R.nvme_bw_gbps, R.nvme_bw_qd1_gbps, R.nvme_latency_ms)
    pf = R.policy.get("prefetch", {})
    recall = pf.get("recall", 0.0) if pf.get("depth", 0) else 0.0
    # lockstep: `conc` streams each decoding their own trace segment (cycled)
    cur = [(i % len(streams), (i * 997) % max(1, len(streams[i % len(streams)]))) for i in range(conc)]
    st = dict(t=0.0, tok=0, steps=0, picks=0, vram=0, ram_cpu=0, zc=0, nvme=0, masked=0, gpu=0.0, cpu=0.0, fixed=0.0)
    inflight, step = {}, 0
    while True:
        rows = [streams[s][(i + w) % len(streams[s])] for s, i in cur for w in range(window)]   # [conc*window][L][topk]
        measure = step >= warm
        if measure:
            snap = (st["t"], nv.reads)
        t = st["t"] + draft_ms
        F = R.fixed(len(rows))
        for l in range(L):
            picks = {}
            for r in rows:
                for e in r[l, :R.topk]:
                    k = l * E + int(e)
                    picks[k] = picks.get(k, 0) + 1
            t += F / L
            p = plan_layer(R, led, picks, t, nv, inflight)
            if l + 1 < L and recall > 0:
                nxt = {(l + 1) * E + int(e) for r in rows for e in r[l + 1, :R.topk]}
                prefetch(R, led, nxt, t, nv, inflight, recall, salt=step)
            t += p.ms
            if measure:
                n = sum(picks.values())
                st["picks"] += n
                st["vram"] += sum(picks[k] for k in p.gpu)
                st["ram_cpu"] += len(p.cpu)
                st["zc"] += len(p.zerocopy)
                st["nvme"] += len(p.nvme_cpu) + len(p.nvme_gpu)
                st["masked"] += len(p.masked)
                st["gpu"] += p.gpu_ms
                st["cpu"] += p.cpu_ms
                st["fixed"] += F / L
        nv.free_at = max(nv.free_at, 0.0)
        if measure:
            st["tok"] += conc * accept
            st["steps"] += 1
            st["t_meas"] = st.get("t_meas", 0.0) + (t - st["t"])
        st["t"] = t
        step += 1
        cur = [(s, (i + window) % len(streams[s])) for s, i in cur]
        if max_steps and st["steps"] >= max_steps:
            break
        if not max_steps and step >= warm + min(len(x) for x in streams) // window:
            break
    T, n, steps = st.get("t_meas", 1e-9), max(1, st["tok"]), max(1, st["steps"])
    return dict(tok_s=round(1000 * n / T, 2), ms_per_step=round(T / steps, 2), conc=conc,
                vram_hit=round(st["vram"] / max(1, st["picks"]), 3),
                cpu_experts_per_tok=round(st["ram_cpu"] / n, 1), zerocopy_per_tok=round(st["zc"] / n, 1),
                nvme_per_tok=round(st["nvme"] / n, 1), masked_per_tok=round(st["masked"] / n, 1),
                ms_fixed=round(st["fixed"] / steps, 2), ms_gpu_moe=round(st["gpu"] / steps, 2),
                ms_cpu_moe=round(st["cpu"] / steps, 2), vram_slots=R.vram_slots, ram_slots=R.ram_slots)


def prefill(R, prompt):
    """Per chunk, every layer streams its non-VRAM experts: layer = max(compute, PCIe, NVMe share)."""
    pf = R.prefill
    C, comp, first = pf["chunk"], pf["compute_ms_per_layer"], pf.get("first_request_ms", 0.0)
    nonvram = R.experts - min(R.experts, R.vram_slots // R.layers)
    ram_share = min(1.0, R.ram_slots / max(1, R.layers * R.experts - R.vram_slots))
    link = nonvram * (ram_share * R.ram_expert_bytes + (1 - ram_share) * R.nvme_expert_bytes) / (R.h2d_gbps * 1e6)
    nvme = nonvram * (1 - ram_share) * R.nvme_expert_bytes / (R.nvme_bw_gbps * 1e6)
    per_layer = max(comp * min(C, prompt) / C, link, nvme)
    chunks = -(-prompt // C)
    ms = chunks * R.layers * per_layer + first
    return dict(prompt=prompt, tok_s=round(1000 * prompt / ms, 1),
                bound=["compute", "pcie", "nvme"][[comp, link, nvme].index(max(comp, link, nvme))],
                ms_per_layer=dict(compute=round(comp, 1), pcie=round(link, 1), nvme=round(nvme, 1)))
