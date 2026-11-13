#!/bin/bash
#SBATCH --job-name=rc9_n16
#SBATCH --output=$HOME/rc9runs/mpiscal/n16/slurm-%j.out
#SBATCH --error=$HOME/rc9runs/mpiscal/n16/slurm-%j.err
#SBATCH --partition=LocalQ
#SBATCH --time=04:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
# Savarino Mack point (f=2, beta=45), shipped configs/mack_f2_b45.yaml, 16 ranks, main 06256e4 (Table B, item 4).
set -euo pipefail
. $HOME/rc9runs/common.sh
cd $RC9/examples/savarino
if [ ! -d output/baseflow ] || [ -z "$(ls output/baseflow 2>/dev/null)" ]; then
  echo "=== savarino_to_npy $(date -Is)"
  $PY savarino_to_npy.py $HOME/savarino_zenodo/state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat output/baseflow
fi
echo "BETA=45.0 F=2 resolution=full-res 300x300 stride=1 epsilon=0 leak=rectilinear ranks=16"
LINSTABPY_CASE1_CONFIG=$RC9/examples/savarino/configs/mack_f2_b45.yaml \
LINSTABPY_CASE1_OUTPUT_NPZ=$HOME/rc9runs/mpiscal/n16/mack_n16.npz \
mpiexec -n 16 "$PY" -u -c "from linstabpy.driver.run import run; run()"
echo "DONE $(date -Is)"
