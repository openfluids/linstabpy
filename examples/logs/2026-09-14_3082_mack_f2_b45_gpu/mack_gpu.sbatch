#!/bin/bash
#SBATCH --job-name=rc9_mack_gpu
#SBATCH --output=$HOME/rc9runs/mpiscal/gpu/slurm-%j.out
#SBATCH --error=$HOME/rc9runs/mpiscal/gpu/slurm-%j.err
#SBATCH --partition=LocalQ
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# Mack point (f=2, beta=45) on one A6000 with cuDSS, same driver and point as the 4/8/16-rank CPU runs
# (jobs 3076-3078), release tree 06256e4; run twice for repeatability. Fourth row of tab:cost_mpi.
set -euo pipefail
set +u
if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; fi
set -u
export LINSTABPY_MPI_STACK=intelmpi
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export I_MPI_DEBUG=4 I_MPI_PIN=off
export PETSC_OPTIONS='-use_gpu_aware_mpi 0'
RC9=<repo>
PY=$HOME/phase5/venv-cudss/bin/python
export LINSTABPY_PYTHON=$PY
export PYTHONPATH="$RC9/src:$RC9"
OUT=$HOME/rc9runs/mpiscal/gpu
for _g in $(seq 0 1); do
  exec 9>"/tmp/linstabpy_gpu${_g}.lock"
  if flock -n 9; then export CUDA_VISIBLE_DEVICES="$_g"; break; fi
done
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no free GPU among 2"; exit 3; }
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
( exec 9>&-; while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES"; sleep 2; done > "$OUT/nvsmi.$SLURM_JOB_ID.log" ) &
_NVSMI_PID=$!
trap 'kill $_NVSMI_PID 2>/dev/null || true' EXIT
echo "SHA=$(git -C $RC9 rev-parse HEAD) $(git -C $RC9 describe --tags --always) host=$(hostname -s) date=$(date -Is)"
$PY -c "import linstabpy, nvmath; p=linstabpy.__file__; assert 'linstabpy-rc9' in p, p; print('linstabpy from', p, 'nvmath', nvmath.__version__)"
cd $RC9/examples/savarino
for rep in 1 2; do
  echo "=== rep $rep $(date -Is)  BETA=45.0 F=2 resolution=full-res 300x300 stride=1 epsilon=0 backend=cudss ranks=1 gpu=1"
  LINSTABPY_CASE1_CONFIG=$RC9/examples/savarino/savarino_point_cudss.yaml \
  LINSTABPY_CASE1_OUTPUT_NPZ=$OUT/mack_gpu_rep$rep.npz \
  mpiexec -n 1 "$PY" -u -c "from linstabpy.driver.run import run; run()"
  echo "=== rep $rep done $(date -Is)"
done
echo "DONE $(date -Is)"
