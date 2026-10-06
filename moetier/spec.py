"""Records -> one resolved Recipe. Data first: everything a scheduler needs is in registry/*.json."""
import json, os
from dataclasses import dataclass, field

COLLECTIONS = ("model", "hardware", "engine", "recipe", "runs")


def load(root):
    reg = {c: {} for c in COLLECTIONS}
    for c in COLLECTIONS:
        d = os.path.join(root, c)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".json"):
                r = json.load(open(os.path.join(d, f)))
                reg[c][r["id"]] = r
    return reg


@dataclass
class Lane:
    name: str          # gpu | zerocopy | cpu (an engine may add more)
    reads: str         # tier it reads expert bytes from: vram | ram
    per_layer_ms: float = 0.0
    per_expert_ms: float = 0.0
    per_extra_token_ms: float = 0.0

    def cost(self, picks):
        """picks: list of token counts (one entry per expert this lane computes)."""
        if not picks:
            return 0.0
        return self.per_layer_ms + sum(self.per_expert_ms + self.per_extra_token_ms * (m - 1) for m in picks)


@dataclass
class Recipe:
    id: str
    layers: int
    experts: int
    topk: int
    expert_bytes: int
    vram_slots: int
    ram_slots: int
    lanes: dict
    h2d_gbps: float
    nvme_bw_gbps: float
    nvme_bw_qd1_gbps: float
    nvme_latency_ms: float
    policy: dict
    fixed_ms: dict
    prefill: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def keys(self):
        return self.layers * self.experts

    def fixed(self, ntok):
        """Non-MoE time per step for ntok tokens (attention, linear attention, dense, sampling), interpolated."""
        F = {int(k): v for k, v in self.fixed_ms.items()}
        ks = sorted(F)
        if ntok in F:
            return F[ntok]
        if ntok <= ks[0]:
            return F[ks[0]]
        for a, b in zip(ks, ks[1:]):
            if a < ntok < b:
                return F[a] + (F[b] - F[a]) * (ntok - a) / (b - a)
        a, b = ks[-2], ks[-1]
        return F[b] + (F[b] - F[a]) / (b - a) * (ntok - b)


def resolve(reg, recipe_id, **overrides):
    r = json.loads(json.dumps(reg["recipe"][recipe_id]))
    for k, v in overrides.items():          # dotted overrides: policy.prefetch.recall=0.7, budget.ram_gb=55
        d, parts = r, k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    m, hw = reg["model"][r["model"]], reg["hardware"][r["hardware"]]
    lanes = {}
    for eid in r["engines"]:
        for name, l in reg["engine"][eid]["lanes"].items():
            lanes[name] = Lane(name, l["reads"], l.get("per_layer_ms", 0.0), l.get("per_expert_ms", 0.0),
                               l.get("per_extra_token_ms", 0.0))
    for name, o in r.get("lane_overrides", {}).items():      # what-if: a faster kernel or handoff on one lane
        for k, v in o.items():
            setattr(lanes[name], k, v)
    b, mo = r["budget"], m["moe"]
    keys = mo["layers"] * mo["experts"]
    ram_for_experts = (b["ram_gb"] - b.get("runtime_gb", 0) - b.get("other_gb", 0)) * 1e9
    ram_slots = max(0, min(keys, int(ram_for_experts // mo["expert_bytes"])))
    return Recipe(id=r["id"], layers=mo["layers"], experts=mo["experts"], topk=mo["topk"],
                  expert_bytes=mo["expert_bytes"], vram_slots=int(b["vram_expert_slots"]), ram_slots=ram_slots,
                  lanes=lanes, h2d_gbps=hw["gpu"]["h2d_gbps"], nvme_bw_gbps=hw["nvme"]["bw_gbps"],
                  nvme_bw_qd1_gbps=hw["nvme"].get("bw_qd1_gbps", hw["nvme"]["bw_gbps"]),
                  nvme_latency_ms=hw["nvme"].get("latency_ms", 0.1), policy=r["policy"],
                  fixed_ms=r["calibration"]["fixed_ms"], prefill=r["calibration"].get("prefill", {}), raw=r)
