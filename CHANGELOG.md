# Changelog

All notable changes to linstabpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work toward **0.0.4**. Development happens on the `dev` branch (versioned
`0.0.4.dev0`) and is promoted to `main` through a reviewed pull request once a
coherent slice is green. Entries accumulate here until the next patch tag is cut.

### Added
- `docs/DEVELOPMENT.md`: the development and release workflow — two-branch model,
  the granular `0.0.x` tick cadence (a few beads → verify → tag), the merge gate,
  and CHANGELOG discipline.

### Changed
- Clean-code revamp in progress under frozen-numerics invariants: the validated
  operator/energy-norm/SBP math is hash-frozen and must stay numerically
  identical (independent oracle parity, not a self-referential snapshot), while
  the surrounding solver/driver structure is refactored. The full test suite and
  the no-MPI pure-math lane stay green at every commit.

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

[Unreleased]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/ricardofrantz/linstabpy/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/ricardofrantz/linstabpy/releases/tag/v0.0.1
