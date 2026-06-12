# Changelog

All notable changes to linstabpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work toward **0.0.6**. Development happens on the `dev` branch (versioned
`0.0.6.dev0`) and is promoted to `main` through a reviewed pull request once a
coherent slice is green. Entries accumulate here until the next patch tag is cut.

### Added
- Solver-stack reproducibility manifest: runtime provenance now records python/numpy/scipy/petsc4py/slepc4py versions, PETSc arch, MUMPS version, and the MPI implementation (failed probes record `unavailable`, never crash a run); the block is echoed at INFO at save time and `scripts/tools/freeze_env.sh` emits a pip-constraints file for release freezes. (bead r3uu.21)
- Per-run metrics schema gate (`metrics_schema`): a single source-of-truth schema for the cost+quality sidecar with `validate_metrics` reporting every missing or ill-typed key at once; sidecars failing validation are written with a `schema_errors` field and a loud warning so the run is visibly quarantined from the figure pipeline. (bead r3uu.7)
- RSVD certification gate (`solvers.router.certification`): `certified_rsvd` runs five algebraic checks before trusting an inexact resolvent action — σ-accuracy-derived solve tolerance, forward/adjoint consistency with realized-residual enforcement, adjoint-of-approximate-factor verification, deterministic linearity probed on both forward and adjoint actions, and multi-seed RSVD reproducibility — and fails loud naming the tripped check. (bead r3uu.14.4)
- Phase-zero EVP-guided band marking (`solvers.router.band_marking`): labels requested forcing frequencies stable / near-pole / smooth by distance to the resolvent poles (`-omega_eig`) before any sweep. (bead r3uu.14.3)
- Config-selectable FGMRES shift-invert path with direct-LU preconditioning, including STRUMPACK backend selection when available.
- Runtime result archives and metrics sidecars now include git, config, host, compute-mode, and MPI-stack provenance.
- CPU/MUMPS default-path parity fixture on a small Blasius case and standalone regression guard for ≤1e-12 baseline drift. (bead r3uu.1.13)

### Changed
- Runtime cost metrics now use `unknown` as the default hardware tag when no hardware tag is configured.

## [0.0.5] - 2026-06-11

Stages 2–6 of the clean-code revamp, completing the refactor program under the
frozen-numerics invariant: the `linalg/`, `resolvent/`, `stability/`, `driver/`,
and `operators/metrics` seams, each shipped with deprecation shims and certified
operator-identical. Final gate on the verification workstation: full suite
1058 passed / 5 skipped, examples 6/6, MATLAB parity unchanged; the assembled
operator is bit-identical to the pre-refactor anchor (v0.0.2) by SHA-256 digest.

### Changed
- **Stage 2 (refactor):** new `linstabpy.linalg` package — `factorize()` /
  `Factorization` (was `make_linear_solver` / `LinearSolver`, with
  `.solve_transpose` renamed `.solve_adjoint`), a typed `Backend` registry
  (`select`/`ensure`/`available`), and `shifted_operator` (was private).
  `solvers.linsolve` / `solvers.backends` remain as deprecation shims for one
  minor version.
- **Stage 3 (refactor):** new `linstabpy.resolvent` package —
  `ResolventOperator` in `resolvent/operator.py` and `WeightedNormalOperator`
  plus the gain driver in `resolvent/gain.py`;
  `resolvent_analysis_with_residuals` folded into
  `resolvent_analysis(residuals=True)`. Numerics frozen: gains and modes
  identical to baseline.
- **Stage 6 (test):** metrics extraction certified operator-identical by the
  assembly-block oracle, MATLAB source parity, metamorphic tests, full suite,
  and Martini example parity. (bead r3uu.6.7.2)
- **Stage 6 (refactor):** assembly metrics, stage timers, and CSR derivative
  cache helpers moved into `linstabpy.operators.metrics`; `assembly.py` now
  imports the plumbing while preserving compatibility re-exports. (bead r3uu.6.7.1)
- **Stage 4 (refactor):** new `linstabpy.stability` package unifies the three
  eigenvalue entry points — `eigenvalues` (was `temporal_eigenvalues`),
  `adjoint_eigenvalues`, `spatial_eigenvalues` — on one shared private
  shift-invert core (`_solve_near_target`, moved verbatim from
  `solvers.eigenvalue._eigs_near_target`). Old names in `solvers.eigenvalue` /
  `solvers.spatial_eig` remain as deprecation shims for one minor version;
  top-level `linstabpy.temporal_eigenvalues` / `adjoint_eigenvalues` are
  unchanged and warning-free. Numerics frozen: full suite and spectra
  identical to baseline. (bead r3uu.6.5.1)
- **Stage 4 (refactor):** `utils.stability_acceptance` moved verbatim to
  `linstabpy.stability.acceptance`, colocating the acceptance gate
  (postcheck residual, target distance, conjugate-aware verdict,
  `reject_reason_code` bitmask — all unchanged) with the eigenvalue core.
  Old path remains a deprecation shim for one minor version. (bead r3uu.6.5.2)
- **Stage 5 (refactor):** new `linstabpy.driver` package; the case-pipeline
  orchestration moved verbatim from `results_runtime.compute` to
  `linstabpy.driver.run` (`from linstabpy.driver import run`).
  `results_runtime.compute` now delegates and remains the supported
  entrypoint; PETSc command-line options are initialised before `petsc4py`
  import on the new path as well. Numerics frozen: full suite and spectra
  identical to baseline. (bead r3uu.6.6.1)
- **Stage 5 (refactor):** runtime `.npz` archive I/O isolated in
  `linstabpy.driver.io` (`save_results` — verbatim move of the writer —
  and `load_results`). The provenance/cost key set is unchanged and now
  guarded by a round-trip test; `results_runtime._stage_write_npz` remains
  as a back-compatible wrapper. (bead r3uu.6.6.2)
- **Stage 5 (refactor):** new CLI front door `linstabpy.driver.cli` with a
  `--version` flag; the cache-aware `main()` moved there verbatim.
  `results_runtime` is now a one-minor-version deprecation shim re-exporting
  its previous public surface (`compute` → `driver.run`, `main` →
  `driver.cli.main`, helpers → their new homes) with `DeprecationWarning`;
  all in-repo importers migrated. (bead r3uu.6.6.3)
- **Stage 5 (test):** driver split certified numerics-inert by the full suite,
  driver provenance round-trip, `--version` smoke, and examples parity gate.
  (bead r3uu.6.6.4)

### Verified
- Full test suite on the verification workstation (Intel-MPI): 1058 passed, 5 skipped.
- Examples: 6/6 PASS; MATLAB (Martini) parity: 5 PASS + 1 known SOFT_FAIL (jet), unchanged.
- No-MPI pure-math lane (dissipation gate): 91 passed.
- Cumulative-drift gate: assembled-operator SHA-256 digests bit-identical
  v0.0.2 → 0.0.5 (32/32 arrays, equivalence-harness cases 2D + axy).
- `ruff check` and `ruff format --check`: clean.

## [0.0.4] - 2026-06-10

Stage 1 of the clean-code revamp: SolverConfig centralization, complete and
verified under the frozen-numerics invariant, plus the Case 1 MATLAB parity
pipeline brought to green on the verification workstation for the first time.

### Added
- `docs/DEVELOPMENT.md`: the development and release workflow — two-branch model,
  the granular `0.0.x` tick cadence (a few beads → verify → tag), the merge gate,
  and CHANGELOG discipline. README gained a short "Branching and releases"
  pointer section.
- `config.typed`: a typed, frozen `Config` tree (`LinSolveCfg`, `ResolventCfg`,
  `EigenCfg`, `SpatialCfg`, `MeshCfg`, `MpiCfg`) with a single `Config.load`
  reader that parses JSONC + env once at the boundary, precedence `JSONC > ENV >
  default`. Defaults are byte-faithful to the legacy `_scfg*` surface, guarded
  by a differential oracle test.
- `utils.petsc_core.mumps_icntl_petsc_options(mpi)`: pure helper mapping the
  typed `MpiCfg` ICNTL fields to PETSc option strings (unit-covered).

### Changed
- **Stage 1 complete:** all 79 `_scfg*` production call sites migrated onto the
  typed `cfg.*` tree (solvers, runtime, diagnostics). `utils/_solver_cfg.py` is
  retained as the low-level parsing-primitive layer that `config.typed` builds
  on; its dead policy surface (`setting*`, `linear_solver_type`,
  `eigen_st_solver`, `root_scatter_enabled`, `_mumps_petsc_options`) was
  removed. Numerics unchanged by construction and by oracle: all seven
  first-principles oracle groups and the Martini example parity are identical
  to the pre-refactor baseline.

### Fixed
- `case1.sh`: repo-root resolution broke after the tree reorganization
  (`../..` → `../../..`); the script aborted with "missing Python runtime" on
  any fresh checkout.
- Case 1 parity stability solve stagnated (EPS `nconv=0` for 60+ restarts):
  the EPS now converges on the shift-invert transformed residual at tol 1e-8,
  matching the MATLAB `eigs` semantics that produced the reference
  `spectrum.mat` (mode quality remains gated by the full-operator postcheck
  and the acceptance bitmask).
- Case 1 stability parity pass now runs sponge-free like the MATLAB reference
  run (whose exported sponge aux is identically zero); the native
  `case1_resolvent` sponge remains active for the resolvent pass. Spectrum
  parity vs MATLAB: max rel. error 2.8e-08 over the 100 closest-to-target
  modes (was O(1) with the sponge mismatch).

### Verified
- Verification workstation, dev @ `800cd5c`: full suite **1034 passed /
  5 skipped / 0 failed**, including — for the first time on that machine —
  the Case 1 MATLAB spectrum-parity and resolvent-parity tests actually
  running (resolvent SVD postcheck residuals ~1e-10). Martini example parity
  table byte-identical to the v0.0.1 baseline across all six examples.

## [0.0.3] - 2026-06-08

First stable, locally verified baseline. Replaces the withdrawn 0.0.2 (see note).

### Fixed
- Single-rank (no-MPI) execution path: the runtime test stubs now provide an
  `allreduce` collective with `SUM`/`MAX`/`MIN` reduction-op sentinels and a
  PETSc `Options` stub, so the cost-metrics MPI reduction works correctly when
  `mpi4py`/`petsc4py` are absent. This restores the pure-math (no-MPI) test lane.

### Changed
- Repository-wide formatting and lint cleanup (`ruff format` plus selected lint
  fixes). A frozen validated-math file keeps a scoped `F841` ignore via config so
  its body is never edited.
- Build: add `g++` to the container image so the PETSc C++ configure step
  succeeds.
- Paused per-push continuous integration and dependency-bump automation while the
  codebase stabilizes; verification is performed locally (full suite plus the
  no-MPI pure-math lane). A leaner CI will be reintroduced at public release.

### Verified
- Full test suite: 749 passed, 14 skipped.
- No-MPI pure-math lane (dissipation gate): 91 passed.
- `ruff check` and `ruff format --check`: clean.

> **Note on 0.0.2.** A `0.0.2` tag was briefly published on a commit whose CI was
> red, then withdrawn. It was never a real release. `0.0.3` replaces it with a
> clean tag that is cut once at the verified baseline and never moved.

## [0.0.1] - 2026-06-04

Initial tagged baseline. It carried known issues that were resolved in later
work — an MPI vector that was replicated across ranks instead of distributed, an
SVD resolvent-postcheck residual that was not energy-norm consistent, and a
relocated run-kit path. Prefer `0.0.3` or later.

[Unreleased]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/ricardofrantz/linstabpy/releases/tag/v0.0.1
