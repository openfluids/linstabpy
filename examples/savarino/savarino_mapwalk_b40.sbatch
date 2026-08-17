#!/bin/bash
#SBATCH --job-name=savarino_mapwalk_b40
#SBATCH --output=/home/rfrantz/openfluids/linstabpy-rc5/examples/savarino/savarino_mapwalk.slurm.log
#SBATCH --error=/home/rfrantz/openfluids/linstabpy-rc5/examples/savarino/savarino_mapwalk.slurm.log
#SBATCH --partition=LocalQ
#SBATCH --time=08:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
set -euo pipefail
set +u
[ -f /opt/intel/oneapi/setvars.sh ] && . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
export LINSTABPY_MPI_STACK=intelmpi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Leave two minutes so LineWalker can stop before Slurm kills the job.
export LINSTABPY_JOB_DEADLINE=$(( $(date +%s) + 8 * 3600 - 120 ))
REPO="${LINSTABPY_MAPWALK_ROOT:-/home/rfrantz/openfluids/linstabpy-rc5}"
PY="${LINSTABPY_PYTHON:-$HOME/linstabpy/.venv/bin/python}"
export PYTHONPATH="${REPO}/src:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "host=$(hostname -s) sha=$(git -C "$REPO" describe --tags --always)"
echo "resolution=full-res 300x300 stride=1 epsilon=0 skip_band_marking=true ranks=4"
echo "engine=LineWalker stem=savarino_mapwalk beta=40"
mpiexec -n "$SLURM_NTASKS" "$PY" -u examples/savarino/savarino_mapwalk.py \
  --config examples/savarino/savarino_point_f2_b45.yaml \
  --out examples/savarino \
  --farm-csv examples/savarino/savarino_map2_200pt.csv \
  --beta 40
echo DONE
