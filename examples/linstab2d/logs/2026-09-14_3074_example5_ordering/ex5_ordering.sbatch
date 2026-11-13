#!/bin/bash
#SBATCH --job-name=rc9_ex5ord
#SBATCH --output=$HOME/rc9runs/ex5_ordering/slurm-%j.out
#SBATCH --error=$HOME/rc9runs/ex5_ordering/slurm-%j.err
#SBATCH --partition=LocalQ
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Example 5 under five MUMPS orderings on main 06256e4 (referee 7, item 5b).
set -euo pipefail
. $HOME/rc9runs/common.sh
OUT=$HOME/rc9runs/ex5_ordering
MART=$RC9/examples/martini
cd $RC9
for ord in default metis pord amd qamd; do
  echo "=== ordering=$ord $(date -Is)"
  if [ "$ord" = default ]; then unset LINSTABPY_MUMPS_ORDERING; else export LINSTABPY_MUMPS_ORDERING=$ord; fi
  /usr/bin/time -f "wall=%e s maxrss=%M kB" "$PY" -m examples.martini.example_5_paraboloid > "$OUT/$ord.log" 2>&1 || echo "ordering=$ord FAILED rc=$?"
  tail -3 "$OUT/$ord.log"
  [ -f "$MART/example_5_paraboloid.npz" ] && mv "$MART/example_5_paraboloid.npz" "$OUT/ex5_$ord.npz"
done
"$PY" - <<'PYEOF'
import numpy as np, glob
ref = np.load("<repo>/examples/martini/example_5_Paraboloid_results_values.npz", allow_pickle=True)
gref = np.asarray(ref["results_Paraboloid__gains"], dtype=float).ravel()
print("MATLAB gains", gref)
for f in sorted(glob.glob("$HOME/rc9runs/ex5_ordering/ex5_*.npz")):
    d = np.load(f, allow_pickle=True)
    g = np.asarray(d["gains"], dtype=float).ravel()
    print(f.split("/")[-1], "gains", g, "rel_diff_vs_matlab", (g - gref[:g.size]) / gref[:g.size],
          "omega_used_imag", float(d["omega_used_imag"]))
PYEOF
echo "DONE $(date -Is)"
