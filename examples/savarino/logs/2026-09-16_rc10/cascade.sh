#!/bin/bash
# Self-chaining rerun of every case on the released tree v1.0.0rc10.
# Step N submits its jobs, then queues step N+1 gated (afterany) on all of them.
# Steps 0-13 are timed: the node runs one step at a time. Step 14 is the untimed block.
set -euo pipefail
R=$HOME/rc10runs
log=$R/cascade.log
idx="${1:?step index}"
run_step() {
  case "$1" in
    0)  bash $R/arm.sh map200_cpu_farm ;;
    1)  # The line driver compares each gain with the farm record. Write today's farm
        # into the tree first, so the comparison is against this campaign and not a
        # file from an earlier one.
        $HOME/openfluids/linstabpy-rc5/.venv/bin/python $R/tree/examples/savarino/collect_campaign.py \
          $R/map200_cpu_farm c9 > $R/tree/examples/savarino/savarino_map3_200pt.csv
        wc -l < $R/tree/examples/savarino/savarino_map3_200pt.csv
        bash $R/arm.sh map200_cpu_lines ;;
    2)  bash $R/arm.sh map200_gpu_farm ;;
    3)  bash $R/arm.sh map200_gpu_lines ;;
    4)  bash $R/arm.sh map317_cpu_farm ;;
    5)  bash $R/arm.sh sub47_cpu_farm ;;
    6)  bash $R/arm.sh sub47_cpu_lines ;;
    7)  bash $R/arm.sh sub47_gpu_farm ;;
    8)  bash $R/arm.sh sub47_gpu_lines ;;
    9)  local a b c d
        mkdir -p $R/mpiscal/n4 $R/mpiscal/n8 $R/mpiscal/n16 $R/mpiscal/gpu
        a=$(sbatch --parsable $R/n4.sbatch); echo "n4: Submitted batch job $a"
        b=$(sbatch --parsable --dependency=afterany:$a $R/n8.sbatch); echo "n8: Submitted batch job $b"
        c=$(sbatch --parsable --dependency=afterany:$b $R/n16.sbatch); echo "n16: Submitted batch job $c"
        d=$(sbatch --parsable --dependency=afterany:$c $R/mack_gpu.sbatch); echo "mack_gpu: Submitted batch job $d" ;;
    10) bash $R/arm.sh line_b40_n4 ;;
    11) bash $R/arm.sh line_b40_n8 ;;
    12) bash $R/arm.sh line_b40_n16 ;;
    13) bash $R/arm.sh line_b40_gpu ;;
    14) mkdir -p $R/martini $R/song $R/sawtooth $R/ex5_ordering $R/modes_figs
        for f in martini song sawtooth ex5_ordering modes_figs; do
          echo "$f: Submitted batch job $(sbatch --parsable $R/$f.sbatch)"
        done ;;
    *)  echo "no step $1"; return 2 ;;
  esac
}
echo "=== cascade step $idx $(date -Is)" >> "$log"
out=$(run_step "$idx" 2>&1) || { echo "$out" >> "$log"; echo "step $idx FAILED" >> "$log"; exit 1; }
echo "$out" >> "$log"
ids=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+" | paste -sd: -)
echo "step $idx job ids: $ids" >> "$log"
next=$((idx+1))
if [ "$next" -le 14 ] && [ -n "$ids" ]; then
  sbatch --dependency=afterany:$ids --partition=LocalQ --time=00:20:00 --job-name="rc10_cascade_$next" \
    --output=$R/cascade_step%j.log --wrap "bash $R/cascade.sh $next" >> "$log" 2>&1
fi
echo "DONE step $idx $(date -Is)" >> "$log"
