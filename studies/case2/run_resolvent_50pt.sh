#!/usr/bin/env bash
# Phase 3 batch wrapper: Case 2 50-point (beta, St) resolvent sweep
# at fd6 stride-4 (paper section 6).
#
# Bead: resolvent-25o
# Config: studies/case2/configs/resolvent/resolvent_grid_50pt_fd6_stride4.jsonc
# Generator: studies/case2/scripts/gen_resolvent_grid_50pt.py
#
# DEVIATION FROM plan_c8_campaign.md: the plan called for 50 individual
# .jsonc files run through run_case_once.sh in a bash loop. The actual
# linstabpy schema is incompatible -- run_case_once.sh expects a full
# single-case parity_config-style file, and the single-case JSONC loader
# does not support $include, so 50 individual configs would have to
# duplicate parity_config wholesale. The multi-case perf_memory.py
# orchestrator does the right thing: it inherits _defaults.jsonc +
# parity_config.jsonc via $include, spawns one MPI child per case, and
# already handles per-case failure isolation (status=failed -> continue,
# never abort). Same scientific output, far less file churn. The 50
# per-point definitions live as 50 cases inside one jsonc.
#
# Per-case isolation lives at the perf_memory layer; this shell wrapper
# only manages the outer process lifecycle (nohup, pid, heartbeat, log).
# set -u (NOT -e): a stray failure in the heartbeat loop must not abort
# the campaign.
#
# Launch (background nohup pattern, mirrors Phase 1b):
#   ssh c8 'cat /tmp/launch_resolvent_50pt.sh | bash'
# where the launcher does:
#   cd ~/linstabpy
#   nohup ./studies/case2/run_resolvent_50pt.sh > /tmp/case2_resolvent_50pt.nohup 2>&1 &
#   echo $! > /tmp/case2_resolvent_50pt.pid

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_EXE="$ROOT/.venv/bin/python"

CONFIG_REL="studies/case2/configs/resolvent/resolvent_grid_50pt_fd6_stride4.jsonc"
CONFIG="$ROOT/$CONFIG_REL"

LOG="${LINSTAB2D_PHASE3_LOG:-/tmp/case2_resolvent_50pt.log}"
HEARTBEAT="${LINSTAB2D_PHASE3_HEARTBEAT:-/tmp/case2_resolvent_50pt.heartbeat}"

# Pre-flight
if [ ! -x "$PYTHON_EXE" ]; then
  echo "[case2-50pt] FATAL python not found at $PYTHON_EXE" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "[case2-50pt] FATAL config not found at $CONFIG" >&2
  echo "[case2-50pt] regenerate via $ROOT/studies/case2/scripts/gen_resolvent_grid_50pt.py" >&2
  exit 2
fi

cd "$ROOT/studies/case2"

start=$(date +%s)
sha=$(git -C "$ROOT" rev-parse --short HEAD)
host=$(hostname)
n_cases=$(grep -c '"case_id"' "$CONFIG")

{
  echo "[case2-50pt] START $(date -Iseconds) host=$host sha=$sha n_cases=$n_cases"
  echo "[case2-50pt] config=$CONFIG_REL"
  echo "[case2-50pt] heartbeat=$HEARTBEAT log=$LOG"
  echo "[case2-50pt] perf_memory.py settings inherited from _defaults.jsonc"
  echo "[case2-50pt] expected wall ~47 h serial; ~15 GiB peak RSS per point"
} >> "$LOG"

# Heartbeat writer (background); cleaned up in the trap
(
  while true; do
    elapsed=$(( $(date +%s) - start ))
    # Count completed cases by scanning the output dir for written NPZs
    out_dir="$ROOT/studies/case2/output/resolvent_grid_50pt_fd6_stride4/runs"
    if [ -d "$out_dir" ]; then
      ok_npz=$(find "$out_dir" -maxdepth 2 -name "*.npz" -type f 2>/dev/null | wc -l | tr -d ' ')
    else
      ok_npz=0
    fi
    printf '%s elapsed=%ds done=%s/%s sha=%s\n' \
      "$(date -Iseconds)" "$elapsed" "$ok_npz" "$n_cases" "$sha" > "$HEARTBEAT"
    sleep 60
  done
) &
HBPID=$!
trap 'kill $HBPID 2>/dev/null; rm -f "$HEARTBEAT"' EXIT

# Environment: thread caps (perf_memory.py also sets these via _defaults.env,
# but be explicit so the nohup invocation is reproducible without env leakage).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Point perf_memory.py at the generated 50-case jsonc.
export LINSTAB2D_CASE2_PERF_CONFIG="$CONFIG"

# Run.
"$PYTHON_EXE" perf_memory.py >> "$LOG" 2>&1
rc=$?

# Final.
{
  echo "[case2-50pt] DONE rc=$rc elapsed=$(( $(date +%s) - start ))s date=$(date -Iseconds)"
  # Final accounting from the report json if present
  report_json="$ROOT/studies/case2/output/resolvent_grid_50pt_fd6_stride4/report_latest.json"
  if [ -f "$report_json" ]; then
    echo "[case2-50pt] report=$report_json"
    "$PYTHON_EXE" - <<PY 2>&1 || true
import json
from pathlib import Path
p = Path("$report_json")
d = json.loads(p.read_text())
ok = sum(1 for c in d.get("cases", []) if c.get("status") == "ok")
fail = len(d.get("cases", [])) - ok
print(f"[case2-50pt] ok={ok} fail={fail} of {len(d.get('cases', []))}")
PY
  fi
} >> "$LOG"

exit "$rc"
