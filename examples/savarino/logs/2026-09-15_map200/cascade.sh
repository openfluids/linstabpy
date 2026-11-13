#!/bin/bash
# Self-chaining cascade: run arm N, then submit this script for arm N+1 gated (afterany) on every job arm N created.
set -euo pipefail
ARMS=(cpu_farm cpu_lines gpu_farm gpu_lines)
idx="${1:?arm index}"
name="${ARMS[$idx]}"
log=$HOME/rc9runs/map200/cascade.log
echo "=== cascade arm $idx ($name) $(date -Is)" >> "$log"
out=$(bash $HOME/rc9runs/map200/arm.sh "$name" 2>&1); rc=$?
echo "$out" >> "$log"
[ $rc -ne 0 ] && { echo "arm $name generator FAILED rc=$rc" >> "$log"; exit $rc; }
ids=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+" | paste -sd: -)
echo "arm $name job ids: $ids" >> "$log"
next=$((idx+1))
if [ "$next" -lt "${#ARMS[@]}" ] && [ -n "$ids" ]; then
  sbatch --dependency=afterany:$ids --partition=LocalQ --time=00:20:00 --job-name="cascade_$next" \
    --output=$HOME/rc9runs/map200/cascade_step%j.log --wrap "bash $HOME/rc9runs/map200/cascade.sh $next" >> "$log" 2>&1
fi
echo "DONE step $idx" >> "$log"
