# linstabpy — Agent Instructions

# Motto

Honesty in numerics.
Honesty in physical interpretation.
Precision in mathematics.
Preference for organic solutions over patched fixes.
No absolute claims without proof.

## Rules

- State uncertainty directly.
- Separate proof, numerical evidence, and intuition.
- Do not hide non-convergence, sensitivity, or numerical fragility.
- Do not assign physical meaning to numerical artifacts.
- Do not force agreement with ad hoc corrections.
- Match the strength of every claim to the strength of the evidence.
- `nconv > 0` is not success. Always check postcheck residual, target distance,
  and the returned family together.

## Generality

The code must work for **all cases**. No artisanal, case-specific fixes; no
hardcoded settings tuned to a single configuration. Fixes belong in the general
solver, not in per-case branches.

`results/case2/` is the hardest configuration we run and serves as a stress
test — if something breaks there, treat it as a signal about the general code,
not a reason to special-case Case 2.

`numerics.alpha_filter` is an operator-level mesh filter, not a mean-flow
cleanup mechanism. Filtered mean flows are separate baseflow datasets
(`Qraw` vs `Qfilt`), selected via `paths.case_dir`.

## Repo layout

- `src/`            — library code
- `examples/`       — runnable `example_*.py` + `.jsonc` configs; `.npz` caches results
- `tests/`          — pytest suite (incl. MPI helpers in `mpi_helpers.py`)
- `tools/parity/`   — parity checks vs Martini reference
- `results/case2/`  — active Case 2 EVP sweeps (`.jsonc` configs, `PLAN.md`, `tutorial.md`)
- `docs/`, `paper/`, `validation/` — write-ups and validation artifacts
- `install.sh`      — builds and patches `.venv` (run once, or after PETSc/SLEPc upgrade)
- Sibling docs: `README.md`, `LINUX_QUICKSTART.md`, `PETSC.md`

## How to run Python

**Always use `.venv/bin/python`. Never `uv run`.**

```bash
.venv/bin/python -m examples.example_1_blasius_smoke   # run example
.venv/bin/python -m pytest tests/ -v                   # full test suite
.venv/bin/python -m pytest tests/test_solvers.py -v    # single file
.venv/bin/python -m pytest -k blasius                  # by keyword
mpirun -n 4 .venv/bin/python -m examples.example_2_jet # MPI run
```

`uv run` rebuilds `petsc4py`/`slepc4py` from source on every call and (a) contends
for the same cache lock across concurrent invocations, (b) on macOS additionally
fails because of rpath patching not handled without `--no-build-isolation`. The
venv produced by `install.sh` already has these wheels patched.

## Examples

Convention: if `<example>.npz` exists, the script plots from cache; delete it to
recompute.

```bash
rm examples/example_1_blasius_smoke.npz
.venv/bin/python -m examples.example_1_blasius_smoke
```

## Parity comparison

Compares current assembly against the Martini reference implementation — run
after any change to operators, mesh, or boundary conditions.

```bash
.venv/bin/python tools/parity/compare_martini_examples.py
```

## Case 2

Case 2 is the hardest configuration we run and drives much of the active
work. Context is split across three files in `results/case2/`:

- `PLAN.md` — active campaign state, phase plan with status, acceptance
  criteria, open TODOs, "do not" list. **Load this first.**
- `REFERENCE.md` — stable technical background: code lineage, governing
  equations, Chu norm, parameter map, flow configuration, Physical and
  Numerical Framing, references. Load when a question touches background.
- `README.md` — operational (Portuguese): how to run a case or campaign.

Do not duplicate their contents here — this file is general guidelines only.
Do not create new files unless discussed with the user.
Remember our primary interest is to find the issues and find a set of parameters that work as expected, logging and writing are secondary untill the numerics sweet spot is found.
 

## Troubleshooting

Stale `uv` locks (rare):

```bash
rm -f ~/.cache/uv/sdists-v9/pypi/{petsc,petsc4py,slepc,slepc4py}/*/.lock \
      ~/.cache/uv/.lock
```

On Linux HPC nodes, `petsc4py`/`slepc4py` come from the module environment;
`install.sh` handles paths automatically. The macOS rpath issue above does not
apply.
