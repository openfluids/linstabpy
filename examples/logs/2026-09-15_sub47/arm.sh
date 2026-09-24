#!/bin/bash
# 47-point subset (beta = 40 and 45 lines) on the release tree 06256e4, four arms, one arm at a time:
# cpu_farm (4 MUMPS ranks per point, Slurm packs 5 jobs on 20 cores), cpu_lines (2 lines x 4 ranks),
# gpu_farm (1 rank + 10 threads + one A6000 per job, 2 jobs by GPU lock), gpu_lines (same, fast refactor path).
set -euo pipefail
ARM="${1:?arm: cpu_farm | cpu_lines | gpu_farm | gpu_lines}"
REPO=$HOME/openfluids/linstabpy-rc9
PYCPU=$HOME/openfluids/linstabpy-rc5/.venv/bin/python
PYGPU=$HOME/phase5/venv-cudss/bin/python
OUT=$HOME/rc9runs/sub47/$ARM
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=$REPO/src:$REPO
echo "arm=$ARM sha=$(git rev-parse --short HEAD) out=$OUT $(date -Is)"
CPU=(--nranks 4 --env LINSTABPY_PYTHON="$PYCPU" --env I_MPI_PIN=off --env I_MPI_DEBUG=4)
GPU=(--nranks 1 --cpus-per-task 10 --gpus 2 --env LINSTABPY_PYTHON="$PYGPU" --env I_MPI_PIN=off
     --env "PETSC_OPTIONS=-use_gpu_aware_mpi 0" --env OMP_NUM_THREADS=10)
CFGCPU=examples/savarino/configs/map_base.yaml
CFGGPU=examples/savarino/savarino_point_cudss.yaml
case "$ARM" in
  cpu_farm)  $PYCPU examples/savarino/gen_campaign.py --out "$OUT" --beta 40 45 --config "$CFGCPU" "${CPU[@]}" ;;
  cpu_lines) for b in 40 45; do $PYCPU examples/savarino/gen_mapwalk_campaign.py --out "$OUT" --beta $b --hours 4 --n-anchor 99 --ncv 5 --driver-out "$OUT/b$b" --config "$CFGCPU" "${CPU[@]}"; done ;;
  gpu_farm)  $PYGPU examples/savarino/gen_campaign.py --out "$OUT" --beta 40 45 --config "$CFGGPU" "${GPU[@]}" ;;
  gpu_lines) for b in 40 45; do $PYGPU examples/savarino/gen_mapwalk_campaign.py --out "$OUT" --beta $b --hours 4 --n-anchor 99 --ncv 5 --driver-out "$OUT/b$b" --config "$CFGGPU" "${GPU[@]}" --env LINSTABPY_FAST_REFACTOR=1; done ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
