# macOS SLEPc NHEP investigation

Date: 2026-04-22

## Goal

Understand why the Martini examples fail on this macOS machine while the Linux
machine reportedly worked. Treat this as an installation/runtime issue unless
evidence points back to example code.

## Current facts

- `bash diagnose.sh` passes on macOS after using `/usr/bin/true` for the MPI
  launcher probe.
- `bash install.sh --check` is intentionally strengthened to fail on a tiny
  Python SLEPc non-Hermitian EPS solve.
- Native C SLEPc `EPS_NHEP` with Krylov-Schur works.
- Python `slepc4py` `EPS_HEP` works.
- Python `slepc4py` `EPS_NHEP` with LAPACK works.
- Python `slepc4py` `EPS_NHEP` with Krylov-Schur / Arnoldi aborts with PETSc
  signal 10 / exit 59.

## Questions

1. Is the abort inside SLEPc algorithm code, Python callback glue, or an ABI
   mismatch?
2. Does the Python path pass the same matrix type, scalar type, and communicator
   as native C?
3. Is the binding using stale build metadata or the wrong dynamic library?
4. Can the installer detect and reject this state reliably?

## Evidence Log

- Python reproducer: `tools/debug/macos_slepc_nhep_repro.py`.
- Native C reproducer: `tools/debug/macos_slepc_nhep_probe.c`.
- Before the runtime fix, Python `EPS_NHEP` with Krylov-Schur aborted with
  PETSc exit `59`.
- Native C `EPS_NHEP` with Krylov-Schur returned
  `native_c_nhep_nconv=2`.
- lldb backtrace for the Python crash stopped at Apple Accelerate LAPACK:
  `libLAPACK.dylib` `ZLADIV`, called from PETSc's downloaded
  `libscalapack.2.2.dylib` `zlarfg_`, then `libpetsc.dylib` `zlahqr_`,
  then SLEPc `DSSolve_NHEP_Private`, then `EPSSolve_KrylovSchur_Default`.
- `DYLD_PRINT_LIBRARIES=1` showed the Python process loading Apple
  Accelerate `libBLAS.dylib` and `libLAPACK.dylib` before PETSc/SLEPc.
- Setting
  `DYLD_INSERT_LIBRARIES=/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib`
  made the same Python NHEP reproducer pass.
- The current installer now adds a macOS runtime wrapper for `.venv/bin/python`
  that inserts Homebrew OpenBLAS before Python starts. This is macOS-only.
- The installer also patches stale PETSc/SLEPc pkg-config and make metadata
  paths that pointed into uv's transient build cache.
- `bash install.sh --check` now includes the Python non-Hermitian EPS solve.
- `./run_all_examples.sh --continue-on-error` passed all six examples on this
  Mac after the runtime fix.

## Root cause

The failing path was not the Martini examples. It was a macOS runtime-library
ordering issue. PETSc's downloaded ScaLAPACK library contained LAPACK symbols
but, when loaded inside Python, its LAPACK calls resolved to Apple Accelerate's
LAPACK first. The iterative non-Hermitian SLEPc path then crashed in
Accelerate `ZLADIV`. Loading Homebrew OpenBLAS before Python imports NumPy /
PETSc prevents that symbol interposition.

## Robust path

- Linux keeps the existing downloaded PETSc dependency path.
- macOS should prefer Homebrew OpenBLAS and ScaLAPACK at PETSc configure time.
- The macOS Python wrapper remains as a runtime guard for existing venvs and
  for any fallback build that still uses PETSc's downloaded ScaLAPACK.
- The installer must not report success unless a tiny Python `EPS_NHEP`
  Krylov-Schur solve passes.
