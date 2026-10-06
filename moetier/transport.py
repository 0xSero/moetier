"""Transport contract: how bytes move between tiers. An adapter implements these for its engine.

    fill(keys, dst)    async reads NVMe records -> RAM slots (O_DIRECT, deep queue); returns a handle
    push(keys)         async RAM -> VRAM slot copies on a copy stream/queue; returns a handle
    victims()          device-reported VRAM evictions (written to a VRAM ring, never straight to host)
    drain(victims)     async VRAM ring -> pinned staging -> RAM slot (copy engine, off the critical path)
    landed(handle)     non-blocking completion check (poll a sequence number, never block on a device event)
    fence()            all queued device work has observed the latest residency flag changes

Reference implementation below: a portable O_DIRECT reader pool for the NVMe store (Linux, libaio-free:
os.preadv on a thread pool). Device copies are engine specific (adapters/).
"""
import ctypes, mmap, os
from concurrent.futures import ThreadPoolExecutor


class RecordStore:
    """Packed expert store: record i at offset i * record_bytes (4K aligned)."""

    def __init__(self, path, record_bytes, threads=32):
        assert record_bytes % 4096 == 0, "records must be 4K aligned for O_DIRECT"
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        self.rec, self.pool = record_bytes, ThreadPoolExecutor(threads)

    def read_into(self, key, addr):
        """Read record `key` into a 4K-aligned host address (an anonymous or registered RAM slot)."""
        buf = (ctypes.c_char * self.rec).from_address(addr)
        n = os.preadv(self.fd, [buf], key * self.rec)
        if n != self.rec:
            raise IOError(f"short read {n} for key {key}")
        return key

    def fill(self, keys, addr_of):
        """Async fill; addr_of(key) -> destination address. Returns futures (landed() = f.done())."""
        return [self.pool.submit(self.read_into, k, addr_of(k)) for k in keys]


class Slots:
    """Anonymous, 2 MiB-aligned RAM slot pool (THP). Counted by the cgroup; register with the GPU runtime if the
    engine reads it zero-copy (cudaHostRegister / USM import)."""

    def __init__(self, n, slot_bytes, align=2 << 20):
        self.n, self.slot = n, -(-slot_bytes // 4096) * 4096
        self.m = mmap.mmap(-1, self.n * self.slot + align)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.m))
        self.base = (base + align - 1) // align * align
        self.free = list(range(n - 1, -1, -1))

    def addr(self, i):
        return self.base + i * self.slot
