#!/bin/bash
# Savarino campaign arms on the released tree v1.0.0rc10. One arm at a time, node otherwise idle.
# cpu_farm: 4 MUMPS ranks per point, Slurm packs 5 jobs on 20 cores.
# cpu_lines: one job per beta line, 4 ranks, five lines at a time (the August setting).
# gpu_farm / gpu_lines: 1 rank + 10 threads + one A6000 per job, two at a time by GPU lock.
set -euo pipefail
ARM="${1:?arm}"
REPO=$HOME/rc10runs/tree
PYCPU=$HOME/openfluids/linstabpy-rc5/.venv/bin/python
PYGPU=$HOME/phase5/venv-cudss/bin/python
OUT=$HOME/rc10runs/$ARM
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=$REPO/src:$REPO
echo "arm=$ARM tag=$(git describe --tags --always) out=$OUT $(date -Is)"
CPU=(--nranks 4 --env LINSTABPY_PYTHON="$PYCPU" --env I_MPI_PIN=off --env I_MPI_DEBUG=4)
GPU=(--nranks 1 --cpus-per-task 10 --gpus 2 --env LINSTABPY_PYTHON="$PYGPU" --env I_MPI_PIN=off
     --env "PETSC_OPTIONS=-use_gpu_aware_mpi 0" --env OMP_NUM_THREADS=10)
CFGCPU=examples/savarino/configs/map_base.yaml
CFGGPU=examples/savarino/savarino_point_cudss.yaml
BETAS=(10 163 45 30 50 70 40 0 100 140 180 200 15 7.5)
submit_lines() {  # $1 python, $2 config, $3 window, $4.. betas then -- then extra args
  local py=$1 cfg=$2 win=$3; shift 3
  local -a betas=() ; while [ "$1" != "--" ]; do betas+=("$1"); shift; done; shift
  local -a ids=()
  for b in "${betas[@]}"; do
    $py examples/savarino/gen_mapwalk_campaign.py --out "$OUT" --beta $b --hours 4 --n-anchor 99 --ncv 5 \
      --driver-out "$OUT/b$b" --config "$cfg" "$@" --dry-run >/dev/null
    local f=$OUT/savarino_mapwalk_b${b/./p}.sbatch
    local dep=()
    if [ ${#ids[@]} -ge $win ]; then dep=(--dependency=afterany:${ids[$((${#ids[@]}-win))]}); fi
    local id; id=$(sbatch --parsable "${dep[@]}" "$f"); ids+=("$id")
    echo "savarino_mapwalk_b$b: Submitted batch job $id ${dep[*]:-}"
  done
}
case "$ARM" in
  map200_cpu_farm)  $PYCPU examples/savarino/gen_campaign.py --out "$OUT" --config "$CFGCPU" "${CPU[@]}" ;;
  map200_cpu_lines) submit_lines "$PYCPU" "$CFGCPU" 5 "${BETAS[@]}" -- "${CPU[@]}" ;;
  map200_gpu_farm)  $PYGPU examples/savarino/gen_campaign.py --out "$OUT" --config "$CFGGPU" "${GPU[@]}" ;;
  map200_gpu_lines) submit_lines "$PYGPU" "$CFGGPU" 2 "${BETAS[@]}" -- "${GPU[@]}" ;;
  map317_cpu_farm)  $PYCPU examples/savarino/gen_campaign.py --out "$OUT" --grid dense317 --config "$CFGCPU" "${CPU[@]}" ;;
  sub47_cpu_farm)   $PYCPU examples/savarino/gen_campaign.py --out "$OUT" --beta 40 45 --config "$CFGCPU" "${CPU[@]}" ;;
  sub47_cpu_lines)  submit_lines "$PYCPU" "$CFGCPU" 2 40 45 -- "${CPU[@]}" ;;
  sub47_gpu_farm)   $PYGPU examples/savarino/gen_campaign.py --out "$OUT" --beta 40 45 --config "$CFGGPU" "${GPU[@]}" ;;
  sub47_gpu_lines)  submit_lines "$PYGPU" "$CFGGPU" 2 40 45 -- "${GPU[@]}" ;;
  line_b40_n4)      submit_lines "$PYCPU" "$CFGCPU" 1 40 -- "${CPU[@]}" ;;
  line_b40_n8)      submit_lines "$PYCPU" "$CFGCPU" 1 40 -- --nranks 8 --env LINSTABPY_PYTHON="$PYCPU" --env I_MPI_PIN=off --env I_MPI_DEBUG=4 ;;
  line_b40_n16)     submit_lines "$PYCPU" "$CFGCPU" 1 40 -- --nranks 16 --env LINSTABPY_PYTHON="$PYCPU" --env I_MPI_PIN=off --env I_MPI_DEBUG=4 ;;
  line_b40_gpu)     submit_lines "$PYGPU" "$CFGGPU" 1 40 -- "${GPU[@]}" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
