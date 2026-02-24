# Hamada Case Adaptation Plan

## Goal
Integrate Hamada's case into `linstab2d` as a first-class Python workflow, with reproducible parity checks against the provided MATLAB outputs.

## Source Material in `install_hamada`
- `workflow_hamada/`: preprocessing + MATLAB execution workflow.
- `hamada_test/`: initial Python port attempt and case data (`x.npy`, `y.npy`, `rho.npy`, `u.npy`, `v.npy`, `w.npy`, `T.npy`, `sponge.npy`).
- `hamada_test/MATLAB_results/`: reference outputs (`LO.mat`, `spectrum.mat`, sample eigenvector file).

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
- Compare Python temporal spectrum against `hamada_test/MATLAB_results/spectrum.mat`.
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
