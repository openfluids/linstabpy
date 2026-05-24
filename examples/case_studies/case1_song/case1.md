# Case 1 (SWBLI): Production Hand-off Status

## Why we are doing this
- MATLAB is the reference implementation.
- MATLAB does not scale to production-size matrices.
- This Case 1 is the parity harness used to prove Python/PETSc/SLEPc is mathematically equivalent before scaling to the large production case.

If parity fails, we fix source/operator logic. We do not hardcode MATLAB outputs.

## Current status (2026-02-28)
- Stability operator-stage parity: **PASS** against `MATLAB_exports_resolvent` stages.
- Resolvent parity (gains + modes): **PASS** against `MATLAB_exports_resolvent/res_svd_final.mat`.
- Key fixes that made parity robust:
  - preserve mesh filters during deformation (MATLAB behavior),
  - load `sponge_aux` from MATLAB `case_inputs_used.mat`,
  - use MATLAB-equivalent resolvent extraction (`H^H W H` EVP route),
  - keep canonical parity parameters in `case1/parity_config.jsonc`.

## Canonical run
- Script: `./case1.sh`
- Outputs:
  - `case1.npz`
  - `output/case1_operator_stage_parity.json`
  - `output/case1_spectrum_parity.json`
  - `output/case1_resolvent_parity.json`
  - parity figures (`case1_*.png`)

Note:
- in `sponge_aux` parity mode, the script auto-skips spectrum parity unless forced with `LINSTAB2D_CASE1_CHECK_SPECTRUM=1`, because the available stability spectrum export is from a different legacy setup.

## How Case 1 should switch to production files
1. Copy `case1/parity_config.jsonc` to a new config file (for example `production_case.jsonc`).
2. Update only paths and physical/numerical parameters in that config.
3. Run with:
   - `LINSTAB2D_CASE1_CONFIG=/absolute/path/to/production_case.jsonc ./case1.sh`

## Solver strategy for production
1. Strategy A (robust baseline):
   - `LINSTAB2D_EIGEN_ST_SOLVER=mumps`
   - `LINSTAB2D_RESOLVENT_TYPE=mumps`
2. Strategy B (scalable memory):
   - `LINSTAB2D_EIGEN_ST_SOLVER=iterative`
   - `LINSTAB2D_RESOLVENT_TYPE=iterative`
   - keep strong preconditioning (`bjacobi/asm` + local `lu`) and strict convergence checks.

Use Strategy A to validate correctness first on reduced production grids, then move to Strategy B for large runs that exceed direct memory budget.

Current practical note:
- Strategy B is implemented and configurable, but on this local Mac test node it was much slower for resolvent and required tuning. Production validation/tuning should be done on the target Linux/Xeon machine.
