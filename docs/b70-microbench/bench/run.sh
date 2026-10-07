#!/bin/bash
# N124 B70 microbench runner. usage (omarchy): ~/freetoken-exl3/bench/xpu_run.sh 0000:84:00.0 n124-<tag> ~/n124/run.sh <tag> <script.py> [lib] [ENV=VAL...]
set -u
D=$HOME/n124; TAG=${1:?tag}; SCRIPT=${2:?script}; LIB=${3:-n124}; shift 3 || shift $#
[ "${DSV41_XPU_PCI:-}" = 0000:84:00.0 ] || { echo "refuse: not 84:00.0 (${DSV41_XPU_PCI:-})"; exit 9; }
[ "$DSV41_XPU_RENDER" = "$(readlink -f /dev/dri/by-path/pci-0000:84:00.0-render)" ] || { echo "refuse: render node mismatch"; exit 9; }
case $LIB in n124) LIBV=(-v $D/build/_moe_n124.so:/lib124/_moe.so:ro); LIBP=/lib124/_moe.so;; a9) LIBV=(); LIBP=/n/exl3xpu/_moe_a9.so;; *) echo bad lib; exit 2;; esac
since=$(date '+%Y-%m-%d %H:%M:%S')
$D/guard.sh "2026-10-07 12:05:00" | tee $D/out/guard_pre_$TAG.txt
grep -q "GUARD OK" $D/out/guard_pre_$TAG.txt || { echo "PRE GUARD FAIL - not starting"; exit 3; }
envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
timeout 660 docker run --rm --name n124-$TAG --device "$DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 \
  --cpuset-cpus 44-47 --memory 32g --memory-swap 32g --device-read-bps /dev/md127:6gb --ulimit memlock=-1:-1 --network none \
  -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=1 -e EnableRecoverablePageFaults=1 \
  -v $HOME/freetoken-exl3/runs/N104-nvtier/src:/n:ro -v /mnt/nvx/n104:/q:ro -v /mnt/nvx/glm53:/g:ro -v $HOME/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/m:ro -v $D:/o "${LIBV[@]}" \
  -e EXL3_MOE_LIB=$LIBP -e OUT=/o/out/$TAG.json -e HOST_UID=$(id -u) "${envs[@]}" \
  --entrypoint bash 24c872759256 -c "ls /dev/dri; cat /sys/fs/cgroup/io.max; python3 -c 'import torch;print(\"xpu devices\", torch.xpu.device_count())'; python3 /o/$SCRIPT" > $D/out/log_$TAG.txt 2>&1
rc=$?
echo "docker rc=$rc" | tee -a $D/out/log_$TAG.txt
timeout 30 docker stop n124-$TAG >/dev/null 2>&1
$D/guard.sh "2026-10-07 12:05:00" | tee $D/out/guard_post_$TAG.txt
exit $rc
