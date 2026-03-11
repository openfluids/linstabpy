# linstab2d — Claude Code Instructions

## CRITICAL: How to Run Python

**DO NOT use `uv run` to execute examples or tests in this project.**

`uv run` attempts to resolve and build all dependencies (including `petsc4py` and
`slepc4py`) from source on every invocation. This fails on macOS because:

1. `petsc4py` requires special macOS rpath patching and `--no-build-isolation` build
   flags not handled by a plain `uv run`.
2. Compiling `petsc4py` from source takes many minutes and requires PETSc headers
   that `uv` does not locate automatically.
3. Concurrent `uv run` calls compete for the same distribution cache lock
   (`~/.cache/uv/sdists-v9/pypi/petsc/.../.lock`) and time out after 300 s.

**Always use the pre-built venv:**

```bash
# Correct — uses the patched venv created by install.sh
.venv/bin/python -m examples.example_1_blasius

# Correct — for tests
.venv/bin/python -m pytest tests/

# WRONG — will try to rebuild petsc4py and fail
uv run python -m examples.example_1_blasius
```

The venv is created and patched once by `install.sh`. On Linux HPC nodes a
different environment is used; adjust the python path accordingly (e.g.
`mpirun -n 1 /path/to/venv/bin/python`).

## Killing stale uv locks

If previous `uv run` calls left lock files behind:

```bash
pkill -9 -f uv
rm -f ~/.cache/uv/sdists-v9/pypi/petsc/3.24.5/.lock \
      ~/.cache/uv/sdists-v9/pypi/petsc4py/3.24.5/.lock \
      ~/.cache/uv/sdists-v9/pypi/slepc4py/3.24.2/.lock \
      ~/.cache/uv/.lock
```

## Running examples

Examples follow the pattern: `.npz` exists → plot only; delete `.npz` → recompute.

```bash
# Run (uses cache if present)
.venv/bin/python -m examples.example_1_blasius

# Force recompute
rm examples/example_1_blasius.npz
.venv/bin/python -m examples.example_1_blasius
```

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Parity comparison

```bash
.venv/bin/python tools/parity/compare_martini_examples.py
```

## Linux / HPC notes

On Linux (Pop!_OS, RHEL, cluster nodes) `petsc4py` and `slepc4py` are installed
via the system package manager or a module environment. The `install.sh` script
handles Linux-specific paths automatically. The `uv run` restriction above is
macOS-specific; on Linux the environment is already set up without rpath patching.
