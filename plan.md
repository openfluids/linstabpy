# Martini Tutorial Expansion Plan

Date: 2026-04-24
Branch: `main`
Status: implemented on `main`

Second-pass status: implemented on `main`

## Goal

Expand `tutorial/tutorial.tex` so the Martini example family is useful as a
guided reader path, not only as a command checklist. The new section should
cover every shipped example one by one, explain what each example demonstrates,
how to run it, what files it generates, what figures to inspect, and how to
interpret the physical or numerical result.

Primary source text:

- `/Users/rfrantz/Documents/projects/resolvent/archive/notes/martini2024.md`

Repository sources to keep aligned:

- `tutorial/tutorial.tex`
- `tutorial/tutorial.pdf`
- `examples/example_0_curvilinear_validation.py`
- `examples/example_1_blasius.py`
- `examples/example_1_blasius_smoke.py`
- `examples/example_2_jet.py`
- `examples/example_3_streaky_bl.py`
- `examples/example_4_cavity.py`
- `examples/example_5_paraboloid.py`
- matching `examples/*.jsonc`
- `run_all_examples.sh`
- `validation/martini2024/tools/compare_martini_examples.py`

## Constraints

- Do not change solver behavior, example parameters, or generated outputs as
  part of this pass.
- Do not claim fresh numerical execution unless it is actually run and passes.
- The local interpreter currently lacks `petsc4py`; if that remains true, limit
  verification to documentation build, source/path checks, and cache-aware
  command correctness.
- Use `make -C tutorial` to rebuild the PDF.
- Keep the text didactic and restrained. Paraphrase the Martini note instead of
  copying long passages.
- Preserve the existing one-runner message: `run_all_examples.sh` is the
  canonical acceptance/MPI runner; direct module commands are for learning and
  debugging.

## Tutorial Structure

Replace the compressed Martini section with a sequence like this:

1. Section intro: what the examples cover as a progression.
2. Common run pattern:
   - list examples with `bash run_all_examples.sh --list`
   - run all examples with `bash run_all_examples.sh`
   - rerun one case with `--fresh --example N --skip-parity`
   - clean generated artifacts with `--clean --example N --skip-parity`
   - run MPI through `run_all_examples.sh --mpi ...`
3. Cache and figure pattern:
   - module run uses existing `.npz` to plot when possible
   - delete the `.npz` or use runner cleanup to force recomputation
   - generated `.npz` and `.png` files are ignored artifacts
4. One subsection per example:
   - Purpose
   - Run
   - Main configuration knobs
   - Generated files
   - What to inspect
   - Interpretation
   - Common caution or validation note
5. Parity section:
   - how the examples relate to the Martini MATLAB source
   - why parity compares saved spectra/gains/artifacts
   - known paper-vs-source corrections from the note

## Example Subsections

### Example 0: Curvilinear Derivative Validation

Role:

- Numerical preflight, not a flow-physics case.
- Demonstrates the curvilinear mapping and derivative matrix behavior before
  the flow examples.

Include:

- Command: `.venv/bin/python -m examples.example_0_curvilinear_validation`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 0 --skip-parity`
- Outputs:
  - `examples/example_0_curvilinear_validation.npz`
  - `examples/example_0_curvilinear_validation.png`
  - `examples/example_0_curvilinear_validation_baseflow.png`
  - `examples/example_0_curvilinear_validation_support_error.png`
  - `artifacts/curvilinear_derivative_parity.json`
- Interpretation:
  - inspect the mapped grid and derivative error;
  - this validates geometry/differentiation plumbing before physical modes.

### Example 1: Blasius Boundary Layer Resolvent

Role:

- First physical resolvent case.
- Cartesian boundary-layer example for Tollmien-Schlichting wave amplification.

Include:

- Smoke command first: `.venv/bin/python -m examples.example_1_blasius_smoke`
- Full command: `.venv/bin/python -m examples.example_1_blasius`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 1 --skip-parity`
- Outputs:
  - `examples/example_1_blasius.npz`
  - `examples/example_1_blasius.png`
  - `examples/example_1_blasius_baseflow.png`
  - `examples/example_1_blasius_support_components.png`
- Interpretation:
  - leading forcing is upstream; strongest response is downstream;
  - this is the amplifier-flow/non-normality signature;
  - suboptimal modes use similar physics but must remain orthogonal, so their
    response patterns modulate and their gains are lower.
- Validation note:
  - the note records `Re = 6e5` in MATLAB/Python; the paper text appears to
    understate it as `60,000`.

### Example 2: Axisymmetric Jet Resolvent

Role:

- Cylindrical/axisymmetric resolvent case.
- Demonstrates the `axy` model and azimuthal mode handling.

Include:

- Command: `.venv/bin/python -m examples.example_2_jet`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 2 --skip-parity`
- Outputs:
  - `examples/example_2_jet.npz`
  - `examples/example_2_jet.png`
  - `examples/example_2_jet_baseflow.png`
  - `examples/example_2_jet_support_components.png`
- Interpretation:
  - leading forcing sits near the nozzle lip / mixing layer;
  - response forms a downstream Kelvin-Helmholtz wavepacket;
  - large leading gain reflects convective amplification of an inflectional
    mean/base flow.
- Validation note:
  - base flow and sponge come from the same `.mat` source used by MATLAB.

### Example 3: Streaky Boundary Layer Temporal And Spatial Stability

Role:

- Stability example rather than resolvent.
- Demonstrates temporal EVP, spatial EVP, periodicity, and Floquet logic.

Include:

- Command: `.venv/bin/python -m examples.example_3_streaky_bl`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 3 --skip-parity`
- Outputs:
  - `examples/example_3_streaky_bl.npz`
  - `examples/example_3_streaky_bl.png`
  - `examples/example_3_streaky_bl_baseflow.png`
  - `examples/example_3_streaky_bl_spatial.png`
  - `examples/example_3_streaky_bl_spectra.png`
- Interpretation:
  - temporal spectrum shows unstable streak modes;
  - Floquet treatment separates harmonic and subharmonic content;
  - spatial analysis follows the corresponding spatial branch and supports the
    convective/absolute interpretation.
- Validation note:
  - the note records MATLAB/Python using `125 x 250`; the paper text appears
    to report a lower grid count.

### Example 4: Cavity Direct And Adjoint Eigenvalues

Role:

- Complex-domain / quasi-structured mesh example.
- Demonstrates masking, deformed grid, direct EVP, and adjoint EVP.

Include:

- Command: `.venv/bin/python -m examples.example_4_cavity`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 4 --skip-parity`
- Outputs:
  - `examples/example_4_cavity.npz`
  - `examples/example_4_cavity.png`
  - `examples/example_4_cavity_baseflow.png`
  - `examples/example_4_cavity_direct.png`
  - `examples/example_4_cavity_adjoint.png`
- Interpretation:
  - direct and adjoint spectra should overlay after the documented conjugation;
  - unstable modes are Rossiter-type feedback between shear-layer instability
    and acoustic feedback inside the cavity;
  - adjoint support near the upstream cavity region marks receptivity.
- Caution:
  - this example currently stays on the SciPy assembly path because the adjoint
    workflow remains SciPy-based.

### Example 5: Paraboloid Resolvent

Role:

- Curvilinear plus axisymmetric resolvent example.
- Demonstrates mapping around a body and azimuthal mode `m = 5`.

Include:

- Command: `.venv/bin/python -m examples.example_5_paraboloid`
- Runner equivalent: `bash run_all_examples.sh --fresh --example 5 --skip-parity`
- Outputs:
  - `examples/example_5_paraboloid.npz`
  - `examples/example_5_paraboloid.png`
  - `examples/example_5_paraboloid_baseflow.png`
  - `examples/example_5_paraboloid_support_components.png`
- Interpretation:
  - forcing and response concentrate around the body/stagnation-region
    boundary-layer system;
  - mode structure is tied to streamwise vortices, downwash/upwash, and streak
    formation through lift-up;
  - suboptimal modes show additional wall-normal/radial nodes and lower gains.
- Validation note:
  - the note records a previous Python mapping-scale issue as fixed; do not
    present that as a current bug.

## Parity Notes To Include

Add a compact table or short callout after the per-example sections:

| Example | What parity checks | Known note |
| --- | --- | --- |
| 1 Blasius | gains and mode artifacts | paper `Re` text appears inconsistent with MATLAB/Python |
| 2 Jet | gains, baseflow, sponge-loaded data | `.mat` source matches MATLAB path |
| 3 Streaky BL | temporal and spatial spectra | MATLAB/Python grid is `125 x 250` |
| 4 Cavity | direct/adjoint spectra | adjoint convention is post-processed to overlay direct spectrum |
| 5 Paraboloid | gains and mapped-domain artifacts | mapping scaling has been fixed in the Python port |

## Implementation Phases

1. Audit current tutorial and examples: completed.
   - confirm commands, JSONC paths, generated files, and runner options;
   - check `examples/fix_examples_figures.md` for figure-specific inspection
     notes.
2. Draft the expanded Section 5 in `tutorial/tutorial.tex`: completed.
   - keep the existing earlier installation and smoke-test sections;
   - replace the compressed example descriptions with the subsection template
     above.
3. Add parity/correction notes: completed.
   - use concise paraphrase from `martini2024.md`;
   - avoid long verbatim quotations.
4. Rebuild and inspect: completed.
   - `make -C tutorial`
   - `pdftotext tutorial/tutorial.pdf - | rg "Example 1|Example 5|parity|Tollmien|Kelvin|Rossiter|lift-up"`
5. Verify repository state: completed where local dependencies allow.
   - `git diff --check`
   - `python -m compileall -q src studies validation tests`
   - `bash -n run_all_examples.sh`
   - do not run full examples unless the PETSc/SLEPc environment is available.
6. Commit the documentation update:
   - stage named files only;
   - likely files: `plan.md`, `tutorial/tutorial.tex`, `tutorial/tutorial.pdf`;
   - include README only if the tutorial changes require cross-reference fixes.

## Implementation Notes

- The expanded guide stays in `tutorial/tutorial.tex`, Section 5, and the
  README points users to `tutorial/tutorial.pdf`.
- Example 0 remains inside the Martini example family as the numerical
  preflight before the five physical paper examples.
- Generated example figures are included conditionally with `\IfFileExists`;
  the PDF embeds them when present, while fresh checkouts get a clear note to
  regenerate the figures instead of a LaTeX failure.
- Full example execution was not rerun in this local environment because the
  interpreter lacks `petsc4py`; documentation build and static checks were run
  locally.

## Second-Pass Review Plan

Goal: keep the existing structure, but add the small reader-facing details that
make the figure walkthrough easier to use without changing solver behavior or
plot-generation code.

Add:

- a short guide for reading the generated figures: baseflow panels, spectrum or
  gain panels, mode/support panels, shared-limit comparisons, and residual or
  parity outputs;
- one or two extra inspection details per example, especially where the figure
  can be misread without context;
- a clearer distinction between numerical health checks, figure aesthetics, and
  physical interpretation;
- a more concrete next-pass figure checklist, still framed as proposed plotting
  work rather than completed figure changes.

Do not add:

- new numerical claims without rerunning the examples;
- new generated artifacts;
- plotting-code edits in the same pass.
