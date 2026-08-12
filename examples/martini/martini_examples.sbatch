#!/bin/bash
#SBATCH --job-name=martini-ex
#SBATCH --output=martini_examples.slurm-%j.out
#SBATCH --partition=LocalQ
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail
set +u
[ -f /opt/intel/oneapi/setvars.sh ] && . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR}"
export LINSTABPY_PYTHON="${LINSTABPY_PYTHON:-$HOME/linstabpy/.venv/bin/python}"
export LINSTABPY_BASEFLOW_DATA_DIR="${LINSTABPY_BASEFLOW_DATA_DIR:-$HOME/linstab2d_reference}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
./run_all_examples.sh --fresh --skip-parity
