# Python API: the 9 lazy names

`import linstabpy` starts neither PETSc nor MPI. Each public name loads
from its module on first use and is cached on the module. Full generated
table (name → module → attribute) in `generated/python-api.md`.

| Name | Use it for |
|---|---|
| `build_lnse_system` | Assemble L0 + index map + BC indices (the `LNSESystem` handoff). |
| `build_lnse_operator` | Assemble the raw sparse LNSE operator only. |
| `compute_chu_norm` | Chu compressible energy inner product (diagonal weights). |
| `temporal_eigenvalues` | SLEPc temporal EVP: `omega = eig(i*L0)`. |
| `adjoint_eigenvalues` | Adjoint temporal EVP for sensitivity studies. |
| `wavemaker` | Structural sensitivity from direct + adjoint modes. |
| `resolvent_analysis` | Leading gain, forcing, response at one `(beta, omega)`. |
| `resolvent_map` | Frequency walk with anchor factorization reuse. |
| `BaseFlow` | Mean flow, sponge damping, viscosity. |

```python
# Config without PETSc: typed loader, light import.
from linstabpy.config.typed import load_case_config

config = load_case_config("cases/my_case.yaml")
print(config.meta.case_name, config.solver.mode)
```

```python
# Solver call: PETSc/MPI initialize here, on first use.
import linstabpy

system = linstabpy.build_lnse_system(...)  # assembled operator + BC info
```

Rules: never `from linstabpy.stability._eigenvalues import ...` in new
code — use the public lazy names. Never import `petsc4py`/`slepc4py` at
module top level in library code; keep solver imports inside the
function that solves so config-only paths stay light. Run solver code
under `.venv/bin/python` (and `mpiexec -n N` for distributed runs).
