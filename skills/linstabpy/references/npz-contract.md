# NPZ results contract

Every run writes one `.npz` archive. Read it with plain NumPy — no
linstabpy import needed:

```python
import numpy as np

data = np.load("results/my_case.npz")
omega = data["omega"]   # complex eigenvalues; unstable iff imag > 0
modes = data["modes"]   # eigenvectors
```

Key names are a compatibility surface defined by
`RUNTIME_NPZ_CONTRACT_KEYS` in `src/linstabpy/driver/payload.py` — full
generated list by group in `generated/npz-keys.md`. Groups:

- **core**: `omega`, `direct_modes`, `gains`, `responses`, `forcings`,
  mesh/fields (`X Y U sponge`), Chu weights (`W_diag invW_diag`),
  run scalars (`St beta fd_order alpha_filter omega_target`, mode counts).
- **`stability_`**: EPS solver stats, residuals, postchecks, acceptance
  gates (`..._accepted_count`, `..._accepted_indices`,
  `..._rejected_indices`), conjugate-partner bookkeeping.
- **`resolvent_`**: forward/adjoint density weights.
- **`numerics_`**: operator-dissipation enable/epsilon/order/applies-to.
- **`solver_`**: eigen-ST backend used, resolvent routing.
- **`provenance_` + `config_md5 git_sha hostname compute_mode
  mpi_stack`**: config hash, git SHA, where/how it ran.
- **`cost_`**: wall times (total/assemble/factor/solve), peak RSS/GPU,
  ranks, PETSc options, hardware tag.
- **`bc_`**: effective/column-zeroed/neumann/overlap index sets.

Contract rule: additive keys are fine. Renames or removals require a
migration note, test updates, and reader updates
(`docs/runtime_npz_contract.md`).
