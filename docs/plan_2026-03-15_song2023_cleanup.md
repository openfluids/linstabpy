# Song2023 Cleanup Plan

## Goal
Reduce `validation/song_2023/` clutter without deleting source/reference material, then regenerate the canonical figures from the current corrected Song run.

## Keep
- source and config:
  - `song_2023.py`, `song_2023.jsonc`, `song_2023.sh`
  - `baseflow_loader.py`, `bc_setup.py`, `plotter.py`
  - `compare_*.py`
  - `run_song_2023_matrix.py`, `song_2023_matrix.jsonc`
  - `run_shift_scan.py`, `shift_scan.jsonc`
  - `run_shifted_ksp_matrix.py`, `shifted_ksp_matrix.jsonc`
- case/reference inputs:
  - `baseflow/`
  - `matlab_reference/`
  - `matlab_code/`
  - `ref_results/`
- current canonical outputs:
  - `song_2023.npz`
  - `song_2023.log`
  - `song_2023.png`
  - `song_2023_baseflow.png`
  - `song_2023_support_direct_modes.png`

## Remove
- `__pycache__/`
- generated diagnostic run directories:
  - `matrix_runs/`
  - `shift_scan_runs/`
  - `shifted_ksp_runs/`
- generated reduced scan artifacts:
  - `song_2023_scan_*.npz`
  - `song_2023_scan_*.log`

## Regenerate
1. canonical Song figures from the current `song_2023.npz`
2. confirm the regenerated figures correspond to the corrected default config

## Verify
- `find validation/song_2023 -maxdepth 2`
- canonical Song figure files exist
- canonical `song_2023.log` still reports:
  - `baseflow_input.transpose_input=True`
  - `n_modes_stability_requested=24`
