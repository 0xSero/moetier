#!/bin/bash
# N124 guard: new fault lines since SINCE (default: N124 start)
SINCE="${1:-2026-10-07 12:05:00}"
J=$(timeout 40 journalctl -k --since "$SINCE" --no-pager 2>/dev/null)
bad=$(echo "$J" | grep -E "Completion-Wait loop timed out|Link Down|Card not present|reboot is needed|pciehp|wedged" ; \
      echo "$J" | grep -E "Error [0-9]+, type:" | grep -v "type: corrected"; \
      echo "$J" | grep -E "xe 0000:84:00.0" | grep -iE "reset|wedge|hang|timeout|fail")
d=$(ps -eo stat,comm | awk '$1 ~ /^D/ && ($2 ~ /khugepaged|kcompactd/)')
ncorr=$(echo "$J" | grep -c "type: corrected")
n84=$(echo "$J" | grep -c "0000:84:00")
if [ -n "$bad$d" ]; then echo "GUARD TRIP"; echo "$bad" | tail -20; echo "$d"; exit 1; fi
echo "GUARD OK since=$SINCE corrected_aer=$ncorr lines_84=$n84 $(date +%T)"
