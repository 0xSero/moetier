"""Residency ledger: where every expert lives. The host is the ONLY writer of residency.

Rule (learned the hard way, N111): a device kernel may report a victim, but only the ledger marks an expert
resident in a tier, and only after its bytes have landed. Eviction clears the flag first; memory is reused only
after the device has observed the clear (transport fences this).

Tiers: vram (clock over N slots), ram (exclusive LRU of experts NOT in vram, or inclusive), nvme (everything).
"""
from collections import OrderedDict

VRAM, RAM, NVME = "vram", "ram", "nvme"


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
    def __init__(self, vram_slots, ram_slots, keys, exclusive=True):
        self.vram, self.ram, self.ram_slots, self.keys, self.exclusive = Clock(vram_slots), OrderedDict(), ram_slots, keys, exclusive
        self.all_in_ram = ram_slots >= keys

    def tier(self, key):
        if key in self.vram:
            return VRAM
        if self.all_in_ram or key in self.ram:
            return RAM
        return NVME

    def seed(self, ranked_keys):
        """Hottest -> vram, next -> ram (exclusive) or hottest -> ram (inclusive)."""
        for k in ranked_keys[: self.vram.n]:
            self.vram.admit(k)
        self.vram.ref = bytearray(self.vram.n)
        if not self.all_in_ram:
            src = ranked_keys[self.vram.n:] if self.exclusive else ranked_keys
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
            return self.ram.popitem(last=False)[0]
        return None

    def promote(self, key, protect=()):
        """key moves into vram (after its copy landed). Exclusive: it leaves ram, the vram victim goes to ram."""
        victim = self.vram.admit(key, protect)
        if self.exclusive and not self.all_in_ram:
            self.ram.pop(key, None)
            if victim is not None:
                self.ram_put(victim)
        return victim
