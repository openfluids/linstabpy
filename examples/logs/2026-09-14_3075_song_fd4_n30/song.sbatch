#!/bin/bash
#SBATCH --job-name=rc9_song
#SBATCH --output=$HOME/rc9runs/song/slurm-%j.out
#SBATCH --error=$HOME/rc9runs/song/slurm-%j.err
#SBATCH --partition=LocalQ
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
# Song 32 deg EVP with the shipped pipeline on main 06256e4 (item 17), then a c_inf-scaled control.
set -euo pipefail
. $HOME/rc9runs/common.sh
cd $RC9
SONG=$RC9/examples/song
echo "=== prepare_baseflow (shipped: velocity / U_inf) $(date -Is)"
$PY -m examples.song.prepare_baseflow --no-plot
$PY - <<'PYEOF'
import numpy as np
d = "<repo>/examples/song/inputs/baseflow/"
u, T, rho = (np.load(d + n + ".npy") for n in ("u", "T", "rho"))
print(f"shape {u.shape} u [{u.min():.4f}, {u.max():.4f}] T [{T.min():.4f}, {T.max():.4f}] rho [{rho.min():.4f}, {rho.max():.4f}]")
assert u.max() < 1.1 and abs(T.min() - 1.0) < 0.05, "freestream-1 gate failed"
print("freestream-1 gate OK")
PYEOF
echo "=== song.py run_fd4_n30 (4 ranks) $(date -Is)"
mpiexec -n 4 "$PY" -u -m examples.song.song --config $SONG/configs/run_fd4_n30.yaml
echo "=== report_matched20 $(date -Is)"
$PY -m examples.song.report_matched20 || echo "report_matched20 rc=$?"
echo "=== control: same field with velocity / c_inf (--mach 1.0) $(date -Is)"
$PY -m examples.song.prepare_baseflow --no-plot --mach 1.0 --output-dir $SONG/inputs_cinf/baseflow
$PY -c "
import numpy as np
d='<repo>/examples/song/inputs_cinf/baseflow/'
u,T=(np.load(d+n+'.npy') for n in ('u','T'))
print(f'control u [{u.min():.4f}, {u.max():.4f}] T [{T.min():.4f}, {T.max():.4f}]')"
sed -e 's#fields: ../inputs/baseflow#fields: ../inputs_cinf/baseflow#' \
    -e 's#output_npz: ../output/fd4_n30/song_2023.npz#output_npz: ../output/fd4_n30_cinf/song_2023.npz#' \
    $SONG/configs/run_fd4_n30.yaml > $SONG/configs/run_fd4_n30_cinf.yaml
grep -n "fields:\|output_npz:" $SONG/configs/run_fd4_n30_cinf.yaml
mpiexec -n 4 "$PY" -u -m examples.song.song --config $SONG/configs/run_fd4_n30_cinf.yaml || echo "control rc=$?"
echo "DONE $(date -Is)"
