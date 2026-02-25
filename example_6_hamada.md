# Hamada Case Adaptation Plan

## Goal
Integrate Hamada's case into `linstab2d` as a first-class Python workflow, with reproducible parity checks against the provided MATLAB outputs.

## Source Material in `install_hamada`
- `workflow_hamada/`: preprocessing + MATLAB execution workflow.
- `hamada_test/`: initial Python port attempt and case data (`x.npy`, `y.npy`, `rho.npy`, `u.npy`, `v.npy`, `w.npy`, `T.npy`, `sponge.npy`).
- `example_6_hamada/MATLAB_results/`: reference outputs (`LO.mat`, `spectrum.mat`, sample eigenvector file).

## Gaps to Fix Before Integration
1. Mesh/baseflow orientation consistency.
- `Mesh` expects `(ny, nx)` arrays (`linstab2d/mesh/mesh.py`), while `hamada_test/test_hamada.py` builds/transposes inconsistently for coordinates vs fields.

2. Neumann penalty mismatch.
- `apply_neumann` takes `penalty` as last argument, but `mesh.ngp` is passed there in `hamada_test/test_hamada.py`, so BC strength is wrong versus MATLAB (`1e8`).

3. Temporal eigenvalue convention mismatch.
- `temporal_eigenvalues` already applies the `i*L0` convention; additional `L0 /= -1.0j` in `hamada_test/test_hamada.py` double-converts.

4. Spanwise baseflow component dropped.
- `w.npy` is loaded but replaced by zeros in `hamada_test/test_hamada.py`.

5. Integration packaging gap.
- Current script is standalone in `hamada_test/`; not integrated into `linstab2d/examples/` and not covered by focused tests/parity checks.

## Implementation Plan
1. Add a dedicated Hamada case loader.
- Implement loader for `.npy` case files with explicit shape/orientation validation.
- Keep Fortran-order conventions explicit where vectorization is used.

2. Convert `hamada_test/test_hamada.py` into a proper example.
- Move to `linstab2d/examples/` with the same structure used by existing examples.
- Fix mesh creation/deformation orientation.
- Fix Neumann penalty usage and match MATLAB BC setup.
- Remove extra eigensolver conversion and preserve API conventions.
- Use loaded `w.npy` unless user explicitly wants `W=0`.

3. Add MATLAB parity checker for Hamada case.
- Compare Python temporal spectrum against `example_6_hamada/MATLAB_results/spectrum.mat`.
- Report sorting policy, tolerances, matched/unmatched modes, and max residuals.

4. Add focused regression/smoke tests.
- Add one cheap test that builds the operator and runs a reduced eigen solve for Hamada input.
- Keep full heavy solves out of default CI; gate with reduced mode count.

5. Add minimal docs and run instructions.
- Document required files, run command, parity command, and expected artifact paths.

## Validation Criteria
1. Example run succeeds with project interpreter:
- `.venv/bin/python -m linstab2d.examples.<hamada_example_module>`

2. Parity checker succeeds against MATLAB reference:
- Spectrum mismatch within agreed tolerance.

3. No regressions in existing tests:
- `.venv/bin/python -m pytest tests/ -x -q`

## Deliverables
- Integrated Hamada loader and example under `linstab2d/`.
- Parity checker against MATLAB reference data.
- Focused tests and short runbook updates.

## Diagnostics Executed (Step 1-4)
Artifacts:
- `artifacts/hamada_diagnostics_step1_4.json`

Baseline parity failures:
- Spectrum (`n=100`): `max_abs=1.214e+01`, `rms_abs=8.596e+00`
- Mode-7 `u` field: `|Δω|=2.236e+00`, `rel_complex=1.468`, `rel_real=1.169`, `corr_real=-0.040`

Convention check (spectrum, `n=100`):
- `py`: `rms_abs=8.596e+00`
- `conj(py)`: `rms_abs=3.390e+00` (best among simple transforms, still far)
- `-py`: `rms_abs=1.334e+01`
- `-conj(py)`: `rms_abs=1.521e+01`

Operator baseline diagnostics vs MATLAB `LO.mat`:
- `nnz` matches exactly (`5678509`)
- action residual still high (`~3.25`), interior residual (`~1.92`) is large
- diagonal residual high (`~2.90`)

One-factor sweep summary (`n_modes=12` for fast spectral diagnostics):
- `transpose_input_false`: best spectral RMS (`3.91`) but catastrophic operator mismatch (action `~27.1`, `nnz` mismatch)
- `bc_pressure_side`: spectral RMS improves (`4.18`) but operator mismatch worse than baseline
- `penalty_1e4`: operator action drops (`~1.00`) but spectral mismatch unchanged
- `zero_spanwise_true`: no effect vs baseline

## Interpretation
1. Mismatch is not explained by a simple sign/conjugate convention.
2. BC penalty and spanwise-baseflow choices are not primary spectrum drivers here.
3. The dominant mismatch is in the assembled operator physics/interior terms (not only boundary rows).
4. Direct matrix comparison must account for MATLAB script conventions (`LO.mat` saved after `L0 = L0 / -1i`), but even after testing scalar/sign variants, large residual remains.

## Next Root-Cause Tasks (Highest Priority)
1. Validate exact MATLAB `readbaseflow` mapping for Hamada case (U/V/W assignment, transpose policy).
2. Verify variable ordering/index mapping parity in assembled state vector (`rho,u,v,w,T`) against Hamada MATLAB solver path.
3. Instrument and compare coefficient blocks (`A0, Ax, Ay, Axx, Ayy, Axy`) at sampled grid points against MATLAB for the same case.
4. Re-check full operator parity with proper MATLAB convention alignment (`i*L0` storage path) after mapping fixes.
5. Only then tighten spectrum/mode parity tolerances in `tests/test_hamada_parity.py`.
