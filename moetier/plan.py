"""The scheduling layer: one function decides, per MoE layer, which lane computes each routed expert.

Same code drives the simulator and a runtime adapter. Inputs per layer: the routed picks {key: tokens}, the ledger,
the NVMe channel and in-flight prefetches. Output: a LayerPlan (who computes what, what to fetch, critical time).

Lanes run concurrently when policy.overlap is true (GPU on VRAM experts, CPU on RAM experts, PCIe and NVMe moving
bytes, and a second-tier GPU lane 'b70' on the experts it holds), so the layer costs max(lanes); otherwise they
serialize. The b70 lane is one lane per card: per_layer_ms = handoff (hidden state host -> card, partials + landed flag
back) + its kernel launch, then per_expert_ms per expert it computes; cards run in parallel, so b70 = max over cards. Exactness: 'stall' waits for NVMe misses,
'mask' drops them (lossy, counted).
"""
from dataclasses import dataclass, field
from .ledger import VRAM, RAM, NVME, B70


class NvmeChannel:
    """FIFO read channel. Deep queues stream at bw; an isolated read pays latency + size/bw_qd1."""

    def __init__(self, bytes_per_read, bw_gbps, bw_qd1_gbps, latency_ms):
        self.deep = bytes_per_read / (bw_gbps * 1e6)       # ms per record at queue depth
        self.one = bytes_per_read / (bw_qd1_gbps * 1e6)
        self.lat, self.free_at, self.reads = latency_ms, 0.0, 0

    def read(self, t, n):
        """Issue n reads at time t; return their arrival times."""
        out = []
        for i in range(n):
            start = max(t, self.free_at)
            svc = self.deep if n > 1 else self.one
            if start > self.free_at or self.free_at == 0.0:
                svc += self.lat
            self.free_at = start + svc
            out.append(self.free_at)
            self.reads += 1
        return out


@dataclass
class LayerPlan:
    gpu: list = field(default_factory=list)        # keys computed from VRAM
    zerocopy: list = field(default_factory=list)   # RAM keys read by the GPU over PCIe
    cpu: list = field(default_factory=list)        # RAM keys computed by the CPU in place
    nvme_cpu: list = field(default_factory=list)   # NVMe misses: read -> RAM -> CPU
    nvme_gpu: list = field(default_factory=list)   # NVMe misses: read -> PCIe -> VRAM
    masked: list = field(default_factory=list)     # dropped picks (policy.exact == 'mask')
    b70: list = field(default_factory=list)        # keys computed by a second-tier GPU from its own VRAM
    gpu_ms: float = 0.0
    b70_ms: float = 0.0
    cpu_ms: float = 0.0
    ms: float = 0.0


def plan_layer(R, ledger, picks, t0, nvme, inflight):
    """R: Recipe. picks: {key: tokens}. inflight: {key: arrival_ms} from earlier prefetches."""
    lanes, pol, S = R.lanes, R.policy, R.nvme_expert_bytes      # an NVMe miss pushed to VRAM is the NVMe record
    gpu, cpu, zc = lanes.get("gpu"), lanes.get("cpu"), lanes.get("zerocopy")
    p = LayerPlan()
    if getattr(ledger, "policy", "lru") != "lru":
        ledger.observe(picks, getattr(ledger, "step", 0))
    V, Rm, N, B = [], [], [], []
    for k, m in picks.items():
        t = ledger.tier(k)
        (V if t == VRAM else Rm if t == RAM else B if t == B70 else N).append((k, m))
    # b70 lane(s): static expert sets, one lane per card, cards in parallel
    b70 = lanes.get("b70")
    if B:
        per_card = {}
        for k, m in B:
            c = ledger.b70[k]
            per_card[c] = per_card.get(c, b70.per_layer_ms) + b70.per_expert_ms + b70.per_extra_token_ms * (m - 1)
        p.b70, p.b70_ms = [k for k, _ in B], max(per_card.values())
    push_ms = S / (R.h2d_gbps * 1e6)
    # GPU lane: per_extra_token_ms (default 0) prices extra rows on one expert, e.g. exllamav3's fused bsz<=8 decode
    # kernels, which run every (token, expert) slot and re-read a shared expert's weights (MTP verify, C2/C4)
    gx = gpu.per_extra_token_ms if gpu else 0.0
    zx = zc.per_extra_token_ms if zc else 0.0
    g = (gpu.per_layer_ms + sum(gpu.per_expert_ms + gx * (m - 1) for _, m in V)) if V else 0.0
    c = cpu.per_layer_ms if (cpu and (Rm or N)) else 0.0
    p.gpu = [k for k, _ in V]
    # RAM picks: CPU in place vs GPU zero-copy, greedy min-max (biggest token counts first)
    for k, m in sorted(Rm, key=lambda x: -x[1]):
        ce = c + cpu.per_expert_ms + cpu.per_extra_token_ms * (m - 1) if cpu else float("inf")
        ge = g + (zc.per_expert_ms + zx * (m - 1) if zc else float("inf"))
        if max(ce, g) <= max(c, ge):
            c = ce; p.cpu.append(k)
        else:
            g = ge; p.zerocopy.append(k)
    # NVMe picks: stall (exact) or mask (lossy)
    if N:
        if pol.get("exact", "stall") == "mask":
            p.masked = [k for k, _ in N]
        else:
            need = [k for k, _ in N if k not in inflight]
            arr = dict(zip(need, nvme.read(t0, len(need)) if need else []))    # absolute arrival times
            arr.update({k: inflight[k] for k, _ in N if k in inflight})
            for k, m in sorted(N, key=lambda x: arr[x[0]]):
                a = max(0.0, arr[k] - t0)                                        # relative to layer start
                ce = max(c, a) + cpu.per_expert_ms + cpu.per_extra_token_ms * (m - 1) if cpu else float("inf")
                ge = max(g, a + push_ms) + (gpu.per_expert_ms + gx * (m - 1) if gpu else 0.0)
                if max(ce, g) <= max(c, ge):
                    c = ce; p.nvme_cpu.append(k)
                else:
                    g = ge; p.nvme_gpu.append(k)
    p.gpu_ms, p.cpu_ms = g, c
    p.ms = max(g, c, p.b70_ms) if pol.get("overlap", True) else g + c + p.b70_ms
    # residency updates (host-owned): touches, promotions of GPU-routed misses, NVMe landings in RAM
    protect = set(picks)
    for k in p.gpu:
        ledger.vram.touch(k)
    for k in p.cpu:
        ledger.ram_touch(k)
    for k in p.nvme_cpu:
        ledger.ram_put(k)
    for k in p.zerocopy + p.nvme_gpu:
        ledger.promote(k, protect)
    for k in list(picks):
        inflight.pop(k, None)
    return p


def prefetch(R, ledger, next_picks, t, nvme, inflight, recall, salt=0):
    """Issue reads for a `recall` fraction of the next layer's NVMe-tier picks (a predictor's hits)."""
    if recall <= 0:
        return 0
    want = [k for k in next_picks if ledger.tier(k) == NVME and k not in inflight
            and ((k * 2654435761 + salt) % 1000) < recall * 1000]
    for k, a in zip(want, nvme.read(t, len(want))):
        inflight[k] = a
    return len(want)
