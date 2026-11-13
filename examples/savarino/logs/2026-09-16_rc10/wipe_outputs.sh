#!/bin/bash
# Delete every regenerable output in the released tree before the rerun. References stay.
set -euo pipefail
T=$HOME/rc10runs/tree
cd $T
git ls-files -z examples wiki | tr '\0' '\n' | grep -E '\.(png|csv|json)$' \
  | grep -v -E 'LinStab2DResults/|baseflows/|examples/song/reference/|examples/savarino/reference/|examples/savarino/configs/|examples/sawtooth/sawtooth\.npz|martini_manifest\.json' > $HOME/rc10runs/deleted_outputs.txt
wc -l < $HOME/rc10runs/deleted_outputs.txt
xargs -a $HOME/rc10runs/deleted_outputs.txt rm -f
rm -f $T/examples/savarino/output/savarino_map2_200pt.csv $T/examples/savarino/output/savarino_map317pt.csv
rm -f $T/examples/martini/*.npz $T/examples/song/output -r 2>/dev/null || true
git status --short | head -5; git status --short | wc -l
