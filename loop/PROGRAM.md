# moetier research + validation loop

Goal: push MoE offload configs on omarchy to their limits with moetier as the standard, keep the registry full, and
ship every validated best (repo + image + recipe). Primary target: GLM-5.3-Flash on 1x RTX 3090 + 55 GB DDR4 + 4x NVMe
RAID0, 50 tok/s decode. Secondary: Qwen3.8-Flash-Next on Arc B70 (paused while the 3090 work runs).

## Each tick (bounded ~25 min of real work)
1. Guard: ssh omarchy (hard timeouts). `journalctl -k -b` must have no "Completion-Wait loop timed out", "Link Down",
   "Card not present", "reboot is needed", non-corrected Hardware Error lines; no D-state khugepaged/kcompactd. If tripped:
   stop GPU work, record it, ask the user before any reboot (BMC path: memory omarchy-pcie-faults-and-hangs).
2. Ownership: ONE heavy GPU job on omarchy at a time (marginal PCIe). If a background agent owns a GPU (see OWNERS.md),
   do not start GPU work; check on it instead.
3. Advance the top running/todo item in QUEUE.md by one concrete step that ends in a measurement, a working artifact,
   or a clear negative result.
4. Every measurement -> a registry run record ~/moetier/registry/runs/<id>.json with the standard table
   (8k C1/C2/C4, 32k C1/C2: decode per stream + total, prefill tok/s) + quality evidence; rebuild registry/index.json.
5. If a run beats the recipe's best AND passes its quality gate (exact path: panel top-1/KL vs reference + greedy
   equality; lossy: KLD vs reference within the recipe's stated budget, no runaway completions > 5k tokens):
   - update that model's repo (README tables, results/, banner numbers) and push;
   - deploy the image (bake the mounted code into an image, push to ghcr via the local-ai-images flow, pin digest);
   - update the local-ai-registry recipe as `candidate` with the evidence (registry-publish skill; promotion to
     `validated` only with acceptance runs on the pinned digest).
6. Append one line to LEDGER.jsonl and `track log HOM-266 "..."`. Reply with one line: item, step, result.

## Rules
- Never set max_tokens / output caps. Natural completions only.
- Quoted heredocs only; never bare `tmux kill-session` or self-matching `pkill -f`; stop only containers you started and
  only by name when the lock owner names your tag.
- No secrets in logs, commits or records. Sudo / BMC credentials: never echo.
- The Qwen27B B70 service (omarchy-local-ai-*, 03:00.0) must keep running. DSV4.1 stays stopped unless the user asks.
