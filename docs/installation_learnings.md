# Installation learnings (multi-machine)

Notes from exercising the `install_tester` workflow (diagnose, install,
benchmarks) on different hosts. They are **observations and troubleshooting
patterns**, not a substitute for upstream `install.sh` headers or MPI vendor
documentation.

## Related paths in this repo

- [`scripts/ci/README.md`](../scripts/ci/README.md) — how to run diagnose, install, benchmark
- [`PROGRESS.md`](../PROGRESS.md) — workflow steps and checklist
- [`installations_problems.md`](../installations_problems.md) — known install issues and workarounds maintained in-repo

## One MPI stack per environment

PETSc, mpi4py, and the `mpiexec` you use for verification must all agree on **one** MPI implementation.

- Mixing stacks (e.g. system Open MPI plus MPICH plus Intel oneAPI on `PATH`) often produces:
  - import failures such as `libmpi.so.*: cannot open shared object file` (wrong soname for the actually loaded `petsc4py`)
  - verification crashes or confusing `mpiexec` behavior

**Practical approach:** choose MPICH or Open MPI (or Intel MPI only if you standardize on it everywhere), remove or hide the others from `PATH` for the install session, then reinstall from a clean venv if you already built against a different stack.

## Stale `uv` / pip caches after changing MPI

If the PETSc line completes in **seconds** after you changed system MPI, you may be reusing a **cached** sdist/wheel built against the **previous** stack.

**Remediation:** remove the project `.venv`, run `uv cache clean`, and run the installer again so PETSc and petsc4py link against the MPI that is on the machine **now**.

## `mpiexec` probes and hung diagnostics

On some systems, `mpiexec` for one vendor **hangs** or times out on `mpiexec -n 1 /bin/true` while another launcher works (this showed up with older Open MPI on one host while MPICH was fine).

The `scripts/ci/diagnose.sh` script uses short timeouts and force-kill so a broken stack does not wedge an interactive shell. Prefer the launcher that passes the probe when setting `LINSTABPY_PETSC_MPIEXEC` / `LINSTABPY_VERIFY_MPIEXEC` if the installer doc suggests it.

## GNU toolchain vs Intel MPI on `PATH`

If Intel oneAPI MPI appears first on `PATH`, `mpicc` / `mpiexec` may be Intel’s wrappers even when you intend a **distro MPICH/Open MPI + gcc/gfortran** stack. For parity with typical project assumptions, either:

- remove Intel from `PATH` for the install, or
- install and select distro MPICH (or Open MPI) and verify `which mpicc` / `which mpiexec` before installing.

`install.sh` already biases Linux PETSc configuration toward GNU compilers when its Linux helper runs; the remaining pitfall is **which MPI** the wrappers wrap.

## Source-of-truth checkout

`install_tester` now runs against the real repository checkout instead of a
copied `src/` and `examples/` bundle. The harness defaults to the parent
directory as the checkout root; set `LINSTABPY_REPO_ROOT=/path/to/linstabpy`
when the harness is copied elsewhere.

This keeps `src/linstabpy/` as the only importable library source tree. The
repository-root `install.sh` still performs the editable install from the real
checkout.

## Running examples

Example scripts use **relative imports** (`from ._helpers import ...`). Run them as modules from the tree that contains `examples/` as a package:

```bash
.venv/bin/python -m examples.example_0_curvilinear_validation
.venv/bin/python -m examples.example_1_blasius
```

Running `python examples/example_0_...py` directly triggers `ImportError: attempted relative import with no known parent package`.

## `ldconfig` vs `libmpi`

`diagnose.sh` may print that `libmpi.so` does not appear in the `ldconfig` cache while MPICH-based `mpiexec` still works. Treat that as a **warning** to verify with real imports, not an automatic failure.

## Benign MPICH finalize noise

After checks under `mpiexec -n 1`, lines such as `write_line error` / `Bad file descriptor` on finalize have been seen **even when** Python imports and installation checks succeed. If verification passes, this is usually harmless I/O teardown noise from the process manager.

## Benchmarks: what is being measured

The `scripts/ci/benchmark.sh` wrapper delegates to the repository-root
`benchmark.sh` harness:

- **Example 0** (curvilinear validation): mostly **NumPy**; it is run **single-process**. MPI rank sweeps do not accelerate that path.
- **Example 1** (Blasius resolvent): uses **PETSc/SLEPc**; it is run with **`mpiexec -n N`** for `N` equal to all physical cores and again for half, with `OMP_NUM_THREADS=1` (and related BLAS thread env vars pinned to 1).

**Cold runs:** the harness deletes matching `examples/*.npz` caches before
each timed run so timings are not dominated by "plot only" reuse.

### Example: one 20-core Ubuntu 18.04 host

On one lab workstation, after aligning on MPICH and a clean rebuild, benchmark JSON showed roughly:

- Example 0: on the order of **8 s** wall time (single process).
- Example 1: about **70 s** with 20 ranks vs **72 s** with 10 ranks for the **same** JSON-fixed case — **little speed-up** when doubling ranks at this problem size, consistent with communication and serial phases dominating or ordering differences in the direct sparse solver.

Peak RSS **did not** decrease monotonically with more ranks on that run (fewer ranks sometimes implied larger per-rank local work and factorizations). Use this as a reminder that **memory vs rank count** is case- and solver-dependent, not intuitive.

Always treat machine-specific numbers as **samples**; compare using your own `benchmark_results.json` outputs.

## CSV vs JSON in benchmark output

`benchmark_summary.csv` may be affected by **locale** (e.g. comma as decimal separator). For automation, prefer **`benchmark_results.json`** in the same `benchmark_runs/<id>/` directory.

## Artifact locations

- Install logs and snapshots: `scripts/ci/install_runs/<run_id>/` (`install_tester.log`, `machine_versions.txt`)
- Benchmark outputs: `benchmark_runs/<bench_id>/` at the repository root (`benchmark_results.json`, per-case `.log` and `/usr/bin/time -v` files where available)

## Multi-machine workflow (higher level)

For syncing many machines and indexing runs, see [`docs/plan_2026-03-26_distributed_hub_workflow.md`](plan_2026-03-26_distributed_hub_workflow.md) (manifest-oriented workflow in-repo; adapt as needed).
