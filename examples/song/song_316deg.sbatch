#!/bin/bash
#SBATCH --job-name=song316
#SBATCH --output=song_316deg.slurm-%j.out
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
LINSTABPY_CASE1_CONFIG="$ROOT/examples/song/song_316deg.yaml" \
LINSTABPY_CASE1_OUTPUT_NPZ="$ROOT/examples/song/song_316deg_paper.npz" \
mpiexec -n 4 "$PY" -c "from linstabpy.driver.run import run; run()"
