#!/bin/bash
#SBATCH --job-name=sawtooth
#SBATCH --output=sawtooth.slurm-%j.out
#SBATCH --partition=LocalQ
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
set -euo pipefail
set +u
[ -f /opt/intel/oneapi/setvars.sh ] && . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR}"
PY="${LINSTABPY_PYTHON:-$HOME/linstabpy/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
echo "resolution=packed stride-2 300x340  dissipation=on epsilon=0.10"
mpiexec -n 4 "$PY" -m examples.sawtooth.sawtooth
