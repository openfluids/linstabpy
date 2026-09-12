# Sparse factor backends

Owner: `linstabpy.linalg`.

| File | Role |
|---|---|
| `factorize.py` | Sparse LU / shift-invert factor |
| `backends.py` | Backend names, `mumps` default, probe |
| `shifted.py` | Shifted operators |
| `cudss_factor.py` | Optional NVIDIA cuDSS path |

Requested backends are checked with a tiny complex PETSc LU setup probe.
Unavailable backends fail loudly.

Install isolation: one virtual environment per PETSc/SLEPc stack. The CPU
MUMPS reference stays `.venv`. Other stacks use `.venv-<stack>` and must
not delete `.venv` on `--force`.
