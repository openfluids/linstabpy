#!/bin/bash
# Full 200-point Savarino grid (14 beta lines) on the release tree 06256e4, four arms, one arm at a time:
# cpu_farm (4 MUMPS ranks per point, Slurm packs 5 jobs on 20 cores), cpu_lines (one job per beta line, 4 ranks,
# two lines at a time by a dependency window), gpu_farm (1 rank + 10 threads + one A6000 per job, 2 jobs by
# GPU lock and core count), gpu_lines (same, fast refactor path, two lines at a time).
# Copy of ~/rc9runs/sub47/arm.sh without the --beta 40 45 filter; lines submitted longest first.
set -euo pipefail
ARM="${1:?arm: cpu_farm | cpu_lines | gpu_farm | gpu_lines}"
REPO=$HOME/openfluids/linstabpy-rc9
PYCPU=$HOME/openfluids/linstabpy-rc5/.venv/bin/python
PYGPU=$HOME/phase5/venv-cudss/bin/python
OUT=$HOME/rc9runs/map200/$ARM
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=$REPO/src:$REPO
echo "arm=$ARM sha=$(git rev-parse --short HEAD) out=$OUT $(date -Is)"
CPU=(--nranks 4 --env LINSTABPY_PYTHON="$PYCPU" --env I_MPI_PIN=off --env I_MPI_DEBUG=4)
GPU=(--nranks 1 --cpus-per-task 10 --gpus 2 --env LINSTABPY_PYTHON="$PYGPU" --env I_MPI_PIN=off
     --env "PETSC_OPTIONS=-use_gpu_aware_mpi 0" --env OMP_NUM_THREADS=10)
CFGCPU=examples/savarino/configs/map_base.yaml
CFGGPU=examples/savarino/savarino_point_cudss.yaml
# beta lines of the author200 grid, longest first (38,38,38,15,15,15,9,6,6,6,6,6,1,1 points)
BETAS=(10 163 45 30 50 70 40 0 100 140 180 200 15 7.5)
# Generate one line job per beta (dry run), then submit with a sliding window of two running lines:
# job k waits (afterany) on job k-2.
submit_lines() {
  local py=$1 cfg=$2; shift 2
  local -a ids=()
  for b in "${BETAS[@]}"; do
    $py examples/savarino/gen_mapwalk_campaign.py --out "$OUT" --beta $b --hours 4 --n-anchor 99 --ncv 5 \
      --driver-out "$OUT/b$b" --config "$cfg" "$@" --dry-run >/dev/null
    local f=$OUT/savarino_mapwalk_b${b/./p}.sbatch
    local dep=()
    if [ ${#ids[@]} -ge 2 ]; then dep=(--dependency=afterany:${ids[$((${#ids[@]}-2))]}); fi
    local id
    id=$(sbatch --parsable "${dep[@]}" "$f")
    ids+=("$id")
    echo "savarino_mapwalk_b$b: Submitted batch job $id ${dep[*]:-}"
  done
}
case "$ARM" in
  cpu_farm)  $PYCPU examples/savarino/gen_campaign.py --out "$OUT" --config "$CFGCPU" "${CPU[@]}" ;;
  cpu_lines) submit_lines "$PYCPU" "$CFGCPU" "${CPU[@]}" ;;
  gpu_farm)  $PYGPU examples/savarino/gen_campaign.py --out "$OUT" --config "$CFGGPU" "${GPU[@]}" ;;
  gpu_lines) submit_lines "$PYGPU" "$CFGGPU" "${GPU[@]}" --env LINSTABPY_FAST_REFACTOR=1 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
