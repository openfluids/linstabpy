# Hamada Case: MATLAB/Python Parity Status

## Objective
Use one small Hamada reference case to prove operator and eigenspectrum equivalence between:
- MATLAB workflow: `example_6_hamada/workflow/main.m`
- Python PETSc/SLEPc workflow: `example_6_hamada.py` + `example_6_hamada.sh`

## Key Finding (Major)
The MATLAB exported inputs are internally inconsistent:
- `run_metadata.mat/kappa = 1.31`
- `case_inputs_used.mat/baseFlow.kappa = 1.4`

Operator parity follows the **actual baseFlow value** (`1.4`), not `run_metadata`.

## Current Measured Status
With the MATLAB-effective settings:
- `kappa = 1.4`
- viscosity-temperature derivatives zeroed in the 2D coefficient path (`dmu/dT = 0`, `d2mu/dT2 = 0`)

Measured parity:
- `L0_preBC max_abs ≈ 2.62e-7`
- full pipeline spectrum parity (`./example_6_hamada.sh`, temporal convention):
- `n_compare = 100`
- `max_abs_error = 1.618039e-07`
- `max_rel_error = 1.864684e-08`
- `pass = true`

## Interpretation
1. The large mismatch was caused by export inconsistencies, not PETSc/SLEPc numerics.
2. Matching the effective MATLAB operator settings resolves matrix and spectrum parity for this case.
3. The pipeline is now aligned for the Hamada reference calibration case.

## What Is Already Fixed
- Curvilinear Jacobian index bug (`J12/J21`) in `linstab2d/mesh/mesh.py`.
- Curvilinear mixed-derivative Jacobian-derivative bug (`Dy @ (Dx @ x/y)` path).
- Mixed-second-derivative coefficient wiring fix for transformed `Dxy/Dyx`.
- 2D `inv(B/1j)` sign path fixed in operator assembly.
- Hamada default `kappa` aligned to MATLAB effective baseFlow (`1.4`) in:
  - `example_6_hamada.py`
  - `example_6_hamada.sh`
- Hamada default now enforces MATLAB-export viscosity-derivative behavior for parity:
  - `LINSTAB2D_HAMADA_ZERO_MU_T_DERIVS=1`

## Remaining Work (Targeted)
1. Keep this parity mode frozen as the regression baseline.
2. Document and decide whether non-parity “physical mode” should use nonzero `dmu/dT` and metadata `kappa`.
3. Add a regression test that checks Hamada spectrum parity stays within tolerance.

## Acceptance Gate
1. `L0_preBC` residual remains near numerical precision in parity mode.
2. Spectrum parity remains pass for `n_compare=100`.
3. Generated parity figures remain consistent with MATLAB reference.
