"""Residency ledger: where every expert lives. The host is the ONLY writer of residency.

Rule (learned the hard way, N111): a device kernel may report a victim, but only the ledger marks an expert
resident in a tier, and only after its bytes have landed. Eviction clears the flag first; memory is reused only
after the device has observed the clear (transport fences this).

Tiers: vram (clock over N slots), b70 (a static, frequency-seeded expert set per second-tier GPU, e.g. Arc Pro B70
cards; never evicted, exclusive of vram and of an exclusive ram tier), ram (exclusive LRU of experts NOT in vram/b70, or
inclusive), nvme (everything).
"""
from collections import OrderedDict

VRAM, RAM, NVME, B70 = "vram", "ram", "nvme", "b70"


class Clock:
    def __init__(self, n):
        self.n, self.owner, self.ref, self.where, self.hand = n, [None] * n, bytearray(n), {}, 0

    def __contains__(self, key):
        return key in self.where

    def touch(self, key):
        s = self.where.get(key)
        if s is not None:
            self.ref[s] = 1

    def admit(self, key, protect=()):
        """Place key, return the evicted key (or None). Keys in `protect` (this layer's picks) are not evicted."""
        if self.n == 0 or key in self.where:
            return None
        for _ in range(2 * self.n + 1):
            s = self.hand
            self.hand = (s + 1) % self.n
            o = self.owner[s]
            if o is not None and o in protect:
                continue
            if self.ref[s]:
                self.ref[s] = 0
                continue
            break
        if o is not None:
            del self.where[o]
        self.owner[s], self.where[key], self.ref[s] = key, s, 1
        return o


class Ledger:
    def __init__(self, vram_slots, ram_slots, keys, exclusive=True, ram_policy="lru", prior=None, b70_slots=()):
        """ram_policy: 'lru' (evict the least recently used), or 'lfu:<window>:<halflife>' = sampled LFU with recency:
        among the <window> least recently used RAM entries evict the one with the lowest decayed pick frequency
        (half-life in decode steps; 0 = no decay). Frequency counts every pick of the key in any tier (the host sees all
        of them), optionally starting from a prior (e.g. the warm-start scores)."""
        self.vram, self.ram, self.ram_slots, self.keys, self.exclusive = Clock(vram_slots), OrderedDict(), ram_slots, keys, exclusive
        self.all_in_ram = ram_slots >= keys
        self.policy, self.win, self.hl = "lru", 0, 0.0
        if ram_policy and ram_policy.startswith("lfu"):
            parts = (ram_policy.split(":") + ["32", "0"])[1:3]
            self.policy, self.win, self.hl = "lfu", int(parts[0]), float(parts[1])
        self.freq, self.tlast, self.now = {}, {}, 0.0
        if prior is not None:
            for k, v in prior.items():
                self.freq[k] = float(v)
        self.lfu_evictions = 0
        self.b70_slots = [int(n) for n in b70_slots]       # per second-tier card; key -> card index once seeded
        self.b70 = {}

    def _f(self, k):
        f = self.freq.get(k, 0.0)
        if self.hl and f:
            f *= 0.5 ** ((self.now - self.tlast.get(k, self.now)) / self.hl)
        return f

    def observe(self, keys, now):
        """picks of one layer call (host-visible), now = decode step index"""
        if self.policy != "lfu":
            return
        self.now = now
        for k in keys:
            self.freq[k] = self._f(k) + 1.0
            self.tlast[k] = now

    def _evict_ram(self):
        if self.policy == "lfu" and self.win > 1:
            best, bf = None, None
            for i, k in enumerate(self.ram):
                if i >= self.win:
                    break
                f = self._f(k)
                if bf is None or f < bf:
                    best, bf = k, f
            del self.ram[best]
            self.lfu_evictions += 1
            return best
        return self.ram.popitem(last=False)[0]

    def tier(self, key):
        if key in self.vram:
            return VRAM
        if key in self.b70:
            return B70
        if self.all_in_ram or key in self.ram:
            return RAM
        return NVME

    def seed(self, ranked_keys):
        """Hottest -> vram, next -> b70 cards (rank round-robin, so every layer's b70 picks spread over the cards), next
        -> ram (exclusive) or hottest -> ram (inclusive)."""
        for k in ranked_keys[: self.vram.n]:
            self.vram.admit(k)
        self.vram.ref = bytearray(self.vram.n)
        nb, left, i = sum(self.b70_slots), list(self.b70_slots), 0
        for k in ranked_keys[self.vram.n: self.vram.n + nb]:
            while left[i % len(left)] == 0:
                i += 1
            c = i % len(left)
            self.b70[k], left[c], i = c, left[c] - 1, i + 1
        if not self.all_in_ram:
            src = ranked_keys[self.vram.n + nb:] if self.exclusive else ranked_keys
            for k in src[: self.ram_slots]:
                self.ram[k] = None

    def ram_touch(self, key):
        if key in self.ram:
            self.ram.move_to_end(key)

    def ram_put(self, key):
        if self.all_in_ram or self.ram_slots == 0:
            return None
        self.ram[key] = None
        self.ram.move_to_end(key)
        if len(self.ram) > self.ram_slots:
            return self._evict_ram()
        return None

    def promote(self, key, protect=()):
        """key moves into vram (after its copy landed). Exclusive: it leaves ram, the vram victim goes to ram."""
        victim = self.vram.admit(key, protect)
        if self.exclusive and not self.all_in_ram:
            self.ram.pop(key, None)
            if victim is not None:
                self.ram_put(victim)
        return victim
