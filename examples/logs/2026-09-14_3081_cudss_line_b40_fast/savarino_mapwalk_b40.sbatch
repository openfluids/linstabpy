#!/bin/bash
#SBATCH --job-name=savarino_mapwalk_b40
#SBATCH --output=$HOME/rc9runs/cudss_line/b40_fp64_fast/savarino_mapwalk.slurm-%j.log
#SBATCH --error=$HOME/rc9runs/cudss_line/b40_fp64_fast/savarino_mapwalk.slurm-%j.log
#SBATCH --partition=LocalQ
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
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
export LINSTABPY_PYTHON=$HOME/phase5/venv-cudss/bin/python
export I_MPI_PIN=off
export PETSC_OPTIONS='-use_gpu_aware_mpi 0'
export OMP_NUM_THREADS=10
export LINSTABPY_FAST_REFACTOR=1
# One GPU per job for the job's lifetime (no GRES on this Slurm): first free
# device by lock; the lock is fd 9 of this shell and dies with the job.
for _g in $(seq 0 1); do
  exec 9>"/tmp/linstabpy_gpu${_g}.lock"
  if flock -n 9; then export CUDA_VISIBLE_DEVICES="$_g"; break; fi
done
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no free GPU among 2"; exit 3; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
( exec 9>&-; while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES"; sleep 5; done > "$HOME/rc9runs/cudss_line/b40_fp64_fast/nvsmi.${SLURM_JOB_ID:-local}.log" 2>&1 ) &
_NVSMI_PID=$!
trap 'kill $_NVSMI_PID 2>/dev/null || true' EXIT
# Leave two minutes so LineWalker can stop before Slurm kills the job.
export LINSTABPY_JOB_DEADLINE=$(( $(date +%s) + 7200 - 120 ))
REPO="<repo>"
PY="${LINSTABPY_PYTHON:-$HOME/linstabpy/.venv/bin/python}"
export PYTHONPATH="${REPO}/src:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "host=$(hostname -s) sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo dirty)"
echo "resolution=full-res 300x300 stride=1 epsilon=0 skip_band_marking=true ranks=1"
echo "engine=LineWalker stem=savarino_mapwalk beta=40.0 driver_args=[ --n-anchor 99 --ncv 5]"
mpiexec -n 1 "$PY" -u examples/savarino/savarino_mapwalk.py \
  --config <repo>/examples/savarino/savarino_point_cudss.yaml \
  --out $HOME/rc9runs/cudss_line/b40_fp64_fast \
  --farm-csv examples/savarino/output/savarino_map2_200pt.csv \
  --beta 40.0 --n-anchor 99 --ncv 5
echo "DONE savarino_mapwalk_b40"
