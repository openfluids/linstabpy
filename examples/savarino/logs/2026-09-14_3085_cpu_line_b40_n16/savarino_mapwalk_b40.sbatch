#!/bin/bash
#SBATCH --job-name=rc9_cpuline_n16
#SBATCH --output=$HOME/rc9runs/cpu_line/b40_n16/savarino_mapwalk.slurm-%j.log
#SBATCH --error=$HOME/rc9runs/cpu_line/b40_n16/savarino_mapwalk.slurm-%j.log
#SBATCH --partition=LocalQ
#SBATCH --time=02:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
set -euo pipefail
set +u
# setvars.sh --force returns non-zero when oneAPI is already in the submitting shell;
# under set -e that killed the job before mpiexec (job 1860).
if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; fi
set -u
export LINSTABPY_MPI_STACK=intelmpi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Rank placement decides the factor time (1.6x on c9, job 1861): print the pin map into
# this log, and take the job's own I_MPI_* / LINSTABPY_PYTHON from the generator.
export I_MPI_DEBUG=4
export LINSTABPY_PYTHON=<venv>/bin/python
export I_MPI_PIN=off
export I_MPI_DEBUG=4
# Leave two minutes so LineWalker can stop before Slurm kills the job.
export LINSTABPY_JOB_DEADLINE=$(( $(date +%s) + 7200 - 120 ))
REPO="<repo>"
PY="${LINSTABPY_PYTHON:-$HOME/linstabpy/.venv/bin/python}"
export PYTHONPATH="${REPO}/src:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "host=$(hostname -s) sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo dirty)"
echo "resolution=full-res 300x300 stride=1 epsilon=0 skip_band_marking=true ranks=16"
echo "engine=LineWalker stem=savarino_mapwalk beta=40.0 driver_args=[ --n-anchor 99 --ncv 5]"
mpiexec -n 16 "$PY" -u examples/savarino/savarino_mapwalk.py \
  --config <repo>/examples/savarino/configs/mack_f2_b45.yaml \
  --out $HOME/rc9runs/cpu_line/b40_n16 \
  --farm-csv examples/savarino/output/savarino_map2_200pt.csv \
  --beta 40.0 --n-anchor 99 --ncv 5
echo "DONE savarino_mapwalk_b40"
