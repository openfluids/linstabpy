---
name: linstabpy
description: Run compressible-flow linear stability and resolvent analysis with linstabpy (PETSc/SLEPc + MPI + MUMPS): install the solver stack, validate YAML cases, launch env-var CLI runs, use the 9-name lazy Python API, and read contract-stable NPZ results. Use when the user mentions linstabpy, eigenvalues, resolvent gain, Chu norm, temporal/spatial stability, SBP operators, or YAML case configs.
license: GPL-3.0-only
compatibility: Linux reference, macOS supported; needs one MPI stack plus complex-scalar PETSc/SLEPc with MUMPS (via install.sh) for any solve; pure import and YAML validation need only Python + NumPy + PyYAML.
metadata:
  repo: "https://github.com/openfluids/linstabpy"
  docs: "https://linstabpy.vercel.app/"
---

# linstabpy

Sparse linear stability and resolvent analysis of compressible viscous
flows. You supply a steady base flow; linstabpy builds the linearized
compressible Navier-Stokes operator and solves eigenvalue / resolvent
problems at HPC scale.

## Read first: the install gate

Nothing that solves works until the solver stack is verified. `import
linstabpy` succeeding proves nothing (it never touches PETSc by design).

```bash
bash install.sh --diagnose   # read-only preflight; fix host issues first
bash install.sh              # builds .venv (PETSc complex + MUMPS, SLEPc, editable repo)
bash install.sh --check      # import + complex-scalar + tiny NHEP probe; must be all green
```

Rules: one MPI stack for launcher, mpi4py, PETSc, SLEPc, and mpiexec.
Always run `.venv/bin/python` directly, never `uv run` (breaks macOS
BLAS layout). Every MPI run exports one thread per rank
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`).
Full matrix in `references/troubleshooting.md`.

## The CLI contract (where agents fail)

`linstabpy` takes NO case arguments. Only `--version` exists. Config and
output travel by env vars; unknown bare words and `--flags` warn on stderr, then are
ignored (exit code unchanged).

```bash
export LINSTABPY_CASE_CONFIG=cases/my_case.yaml
export LINSTABPY_CASE_OUTPUT_NPZ=results/my_case.npz
mpiexec -n 8 linstabpy            # full run; prints "Cache valid" + skips if output is current
linstabpy --version              # light: never initializes PETSc or MPI
```

- Never invent flags: there is no `--config`. `parse_known_args` warns
  and drops unknown bare words and `--flags`, so a guessed command still runs the
  wrong case.
- One output path per concurrent run, or set
  `LINSTABPY_FORCE_RECOMPUTE=1` deliberately.
- All env vars + YAML schema: `references/cli-and-yaml.md`.

## Workflows

**1. Smoke test (no data, <1 min, run first).** Pure NumPy, proves the
import chain: `.venv/bin/python -m
examples.linstab2d.example_0_curvilinear_validation`. If this fails, the
problem is the Python env, not MPI or data.

**2. YAML case via CLI (packed-field research route).** Validate, then run:

```bash
.venv/bin/python -c "
from linstabpy.config.schema import validate_config
from linstabpy.utils.yaml_io import load_config
validate_config(load_config('cases/my_case.yaml'), source='cases/my_case.yaml')"
export LINSTABPY_CASE_CONFIG=cases/my_case.yaml LINSTABPY_CASE_OUTPUT_NPZ=results/my_case.npz
mpiexec -n 4 linstabpy
```

Start from `assets/minimal-case.yaml`. Example 1 (Blasius resolvent,
reads the shipped `BL_BF.mat`) is the first real
science run; family map + runner in `references/examples.md`.

**3. Library use (9 lazy names, PETSc starts on first solver call).**
`build_lnse_system`, `build_lnse_operator`, `compute_chu_norm`,
`temporal_eigenvalues`, `adjoint_eigenvalues`, `wavemaker`,
`resolvent_analysis`, `resolvent_map`, `BaseFlow`. Signatures, lazy
semantics, and snippets: `references/python-api.md`.

## Reading results

Results are contract-stable `.npz`: `data = np.load(path)`,
`omega = data["omega"]` (complex eigenvalues),
`modes = data["modes"]`. Full key table with stability/resolvent/
provenance/cost groups: `references/npz-contract.md`.

## Load-bearing conventions

1. `omega = eig(i*L0)`, `exp(-i*omega*t)`; unstable iff `Im(omega) > 0`.
2. SBP orders 2/4/6/8 only. 3. Default shifted solve `KSPPREONLY`+`PCLU`
   via MUMPS. 4. YAML: `.yaml`/`.yml` suffix, top-level mapping, complex
   as `[real, imag]`. 5. NPZ keys additive-only (renames need migration
   note + tests + readers).
6. Resolvent `H = [(gamma - i*omega) I - L0]^-1` (physical sign);
   `solver.resolvent_sign_convention: matlab_parity` restores the LinStab2D
   sign (`wiki/resolvent-sign-convention.md`).

## References (progressive disclosure)

- `references/cli-and-yaml.md` — env vars, TOP_LEVEL_ALLOWED keys, validation
- `references/python-api.md` — the 9 lazy names with snippets
- `references/npz-contract.md` — RUNTIME_NPZ_CONTRACT_KEYS by group
- `references/examples.md` — example family, data needs, runner, costs
- `references/troubleshooting.md` — MPI mixing, macOS BLAS, cache-valid,
  silent args, thread policy
- `references/generated/` — key tables generated from source; never edit
- `assets/minimal-case.yaml` — smallest valid case template
