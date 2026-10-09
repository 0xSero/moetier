#!/usr/bin/env python3
"""Host sampler (user sero, stdlib): /proc/stat per-core (40-47 + all), md127 + member NVMe diskstats, B70-48 GT idle
residency / act freq, MemAvailable. usage: hostmon.py <out.jsonl> [period_s]"""
import json, sys, time
out = open(sys.argv[1], "a", buffering=1); per = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
DISKS = {"md127", "nvme0n1", "nvme1n1", "nvme2n1", "nvme4n1", "nvme3n1", "dm-0", "dm-1"}
G = "/sys/class/drm/card4/device/tile0/gt0"
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
while True:
    t = time.time(); cpu = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu"):
            f = line.split(); n = f[0]
            if n == "cpu" or (n[3:].isdigit() and 40 <= int(n[3:]) <= 47) or (n[3:].isdigit() and int(n[3:]) in range(0, 40)):
                cpu[n] = [int(x) for x in f[1:9]]
    dk = {}
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] in DISKS: dk[f[2]] = [int(f[3]), int(f[5]), int(f[6]), int(f[7]), int(f[9]), int(f[12]), int(f[13])]  # rd, rd_sect, rd_ms, wr, wr_sect, io_ms, wtime
    mem = {}
    for line in open("/proc/meminfo"):
        if line.startswith(("MemAvailable", "MemFree", "Dirty")): a, b = line.split(":"); mem[a] = int(b.split()[0])
    out.write(json.dumps(dict(t=t, cpu=cpu, dk=dk, mem=mem, gt_idle_ms=rd(G + "/gtidle/idle_residency_ms"),
                              gt_act=rd(G + "/freq0/act_freq"), gt_cur=rd(G + "/freq0/cur_freq"))) + "\n")
    time.sleep(max(0.0, per - (time.time() - t)))
