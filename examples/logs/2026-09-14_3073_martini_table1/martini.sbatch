#!/bin/bash
#SBATCH --job-name=rc9_martini
#SBATCH --output=$HOME/rc9runs/martini/slurm-%j.out
#SBATCH --error=$HOME/rc9runs/martini/slurm-%j.err
#SBATCH --partition=LocalQ
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Table 1: the six Martini examples and their parity reports on main 06256e4.
set -euo pipefail
. $HOME/rc9runs/common.sh
cd $RC9
./run_all_examples.sh || echo "run_all_examples rc=$?"
echo "=== parity table"; cat examples/martini/martini_parity_table.csv
echo "DONE $(date -Is)"
