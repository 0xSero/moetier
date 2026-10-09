#!/bin/bash
# HOM-272 B70-48 profiling session (v2: EXTRA_ARGS env appended to the serve args). usage: session.sh <arm> <readbps|-> <route 0|1> <client cmd...>
# Locks: ~/skillsgym/b70.lock (one B70 job) + ~/nvx_bench.lock (RAID reads) held for the whole session.
set -uo pipefail
ARM=$1; RBPS=$2; ROUTE=$3; shift 3
B=$HOME/b70prof; T=$B/tools; O=$B/runs/$ARM; mkdir -p "$O"; chmod 777 "$O"
NAME=b70prof-$ARM; PORT=30420
IMG=ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691d6ab890b95f7c6179e4b677a0d759018e39b20e93bf0efdb8780450c
M=turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5; P=/opt/trellis-serve/xpu/exl3xpu
log() { echo "[$(date '+%F %T')] $*" | tee -a "$O/session.log"; }
teardown() { timeout 120 docker stop -t 90 "$NAME" >/dev/null 2>&1; timeout 60 docker rm "$NAME" >/dev/null 2>&1; }
exec 8>"$HOME/skillsgym/b70.lock"; log "waiting b70.lock"; flock -w 1800 8 || { log "b70.lock timeout"; exit 3; }
exec 9>"$HOME/nvx_bench.lock"; log "waiting nvx_bench.lock"; flock -w 3600 9 || { log "nvx lock timeout"; exit 3; }
log "locks held"
[ -f $B/TRIP ] && { log "TRIP file present, refusing"; exit 2; }
SINCE=$(date '+%F %T')
timeout 150 $T/guard.sh "$SINCE" "pre-$ARM" >> "$O/guard.log" 2>&1 || { log "pre-guard TRIPPED"; echo "pre $ARM" >> $B/TRIP; exit 2; }
RENDER=$(readlink -f /dev/dri/by-path/pci-0000:48:00.0-render); [ -e "$RENDER" ] || { log "no render"; exit 2; }
RB=(); [ "$RBPS" != "-" ] && RB=(--device-read-bps "/dev/md127:$RBPS")
teardown
timeout 120 docker run -d --init --name "$NAME" --device "$RENDER:$RENDER:rwm" --memory 16g --memory-swap 16g --shm-size 8g \
  --ulimit memlock=-1:-1 --cpuset-cpus 40-47 "${RB[@]}" -p 127.0.0.1:$PORT:30000 \
  -e QWEN_B70_MODE=nvme16 -e B70PROF=1 -e B70PROF_OUT=/b70out -e B70_ROUTE=$ROUTE \
  -v "$HOME/models/$M:/models/$M:ro" -v /mnt/nvx/n104:/nvx:ro -v "$O:/b70out" \
  -v $T/sglang_plugin.py:$P/sglang_plugin.py:ro -v $T/b70prof.py:$P/b70prof.py:ro -v $T/sampler.py:/b70tools/sampler.py:ro \
  "$IMG" python3 -m sglang.launch_server --model-path /models/$M --quantization exl3 --trust-remote-code --device xpu \
  --host 0.0.0.0 --port 30000 --served-model-name flashnext --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3 \
  --context-length 65536 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --max-total-tokens 131072 \
  --max-running-requests 4 --dtype bfloat16 --max-mamba-cache-size 32 --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder ${EXTRA_ARGS:-} > "$O/docker_run.txt" 2>&1 \
  || { log "docker run failed: $(cat $O/docker_run.txt)"; exit 1; }
log "container started $(date +%s)"; echo $(date +%s) > "$O/t_start"
( while sleep 60; do
    timeout 150 $T/guard.sh "$SINCE" "mid-$ARM" >> "$O/guard.log" 2>&1 || { echo "mid $ARM $(date)" >> $B/TRIP; log "MID GUARD TRIPPED"; pkill -f "load.py $PORT" ; pkill -f "route.py $PORT"; teardown; exit; }
    echo "$(date +%s) $(timeout 20 docker ps --format '{{.Names}}' | tr '\n' ' ')" >> "$O/contended.txt"
  done ) & GP=$!
ok=0; for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q flashnext && { ok=1; break; }
  [ "$(timeout 15 docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || break
  [ -f $B/TRIP ] && break; sleep 5; done
log "ready=$ok after $(( $(date +%s) - $(cat $O/t_start) ))s"
if [ $ok = 1 ]; then
  nohup python3 $T/hostmon.py "$O/hostmon.jsonl" 0.5 > /dev/null 2>&1 & HP=$!
  timeout 20 docker exec -d $NAME python3 /b70tools/sampler.py /b70out/sampler.jsonl 0.25
  log "client: $*"
  timeout 5400 "$@" > "$O/client.out" 2>&1; log "client rc=$?"
  kill $HP 2>/dev/null
fi
timeout 60 docker logs $NAME > "$O/server.log" 2>&1
teardown; log "torn down"
kill $GP 2>/dev/null; wait $GP 2>/dev/null
timeout 150 $T/guard.sh "$SINCE" "post-$ARM" >> "$O/guard.log" 2>&1 || { echo "post $ARM" >> $B/TRIP; log "POST GUARD TRIPPED"; }
log "session done"
