# Changelog

All notable changes to linstabpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0rc2] - 2026-06-25

GPU-resolvent dual-factorization adjoint heal + amortized stable-sweep wired
into the turnkey gain-map driver. On an NVIDIA A6000, the healed GPU backend
resolvent gain and modes match the MUMPS CPU reference to ~2e-14. Regime-(b)
discounted resolvent and amortized stable-sweep solver acceleration are
validated and wired into the `resolvent_map` entry point. Turnkey gain-map
scripts (`run_gain_map.py` / `fig_gain_map.py`) are now shipped as public
examples.

### Added
- Turnkey gain-map driver scripts: `examples/case_studies/case1_song/scripts/run_gain_map.py`
  (compute (β,f) gain map end-to-end, resumable) and `fig_gain_map.py` (plot and
  annotate the result). Both read the Song 2023 box base flow and are self-contained
  executables shipped as public-facing Case 1 examples.
- Regime-(b) discounted-resolvent wiring into `resolvent_map`: NEAR_POLE band points
  compute the stabilized resolvent via `L0 - (γ + iω)I` (default γ=0.25, user-configurable);
  every regime-b point records `resolvent_discount_gamma` and is hard-gated to a finite gain.
  The regime-a direct path (γ=0) is unchanged and bit-identical to existing runs.
- Amortized stable-sweep solver acceleration into `resolvent_map`: factorizations are
  refactorized across anchor blocks (sparsity reuse), and RSVD probe blocks route through
  `solve_block` (PETSc `MatMatSolve` multi-RHS). Time savings reported in per-run metrics.

### Fixed
- **GPU resolvent adjoint heal via dual factorization:** STRUMPACK's complex transpose-solve
  is not a true adjoint (defect ~3.2e-2). The sweep now engages a dual-factorization heal:
  the operator is factored twice (once forward, once transpose) and the transpose solve
  routes through the transposed factor, recovering the true adjoint to machine precision
  (~2e-14 on an A6000; matching MUMPS). Dual-factor mode is transparent and automatic;
  its engagement is logged. Validated on an A6000 with Song 2023 base flow: leading gain
  and structural-sensitivity modes match the MUMPS reference. This closes the GPU-resolvent
  correctness blocker (bead r3uu.14.17). CPU/MUMPS path unchanged (still single-factor).
- `wire factor_backend` into iterative LU preconditioner: the shift-invert solver now passes
  the selected `factor_backend` (e.g. STRUMPACK) through to the PETSc LU preconditioner
  (`KSP` path), instead of always falling back to MUMPS, so GPU-accelerated FGMRES
  preconditioners route to the GPU backend. Validated on c9 A6000 with complex resolvent:
  residuals match the direct path to ~2e-14.

### Changed
- Case 1 Song base flow is now genericised in the config comment: `song_box_evp.jsonc`
  reads "Song SWBLI box" (removed author name).

## [1.0.0-rc1] - 2026-06-18

Physics-campaign freeze release candidate: the turnkey `resolvent_map(base_flow,
beta_range, f_range, cfg)` entry point (base-flow in, (β,f) gain map + cost metrics out)
with region masks on the CPU/MUMPS-validated frozen-numerics path. Seven Appendix-A
oracles verified to <1e-10; full suite green; single frozen SHA for the validation campaign.

### Added
- Turnkey region masks for `resolvent_map`: `input_region`/`output_region` cfg specs (`{"x": [x0,x1], "y": [y0,y1], "vars": [...]}`, any key omittable → full extent / all vars) build the diagonal 0/1 forcing/response operators (`B_mat`/`C_mat`) internally, so a windowed or component-restricted resolvent gain needs no hand-built ndof×ndof matrices and no knowledge of the LNSE state layout. The mask builder honors the layout exactly — active points from `mesh.X/Y.ravel(order="F")[mesh.used_ind]`, variable blocks from `idx['{rho,u,v,w,T}_j']` — and is a sibling of the existing BC-forcing mask helper; an explicit `B_mat`/`C_mat` still takes precedence, then region specs, then the default BC-forcing mask. A full-domain all-vars region is the identity on every dof and reproduces the unmasked resolvent gain and modes to machine precision (~1e-12, phase-aligned); empty/out-of-domain/zero-active-point/unknown-variable specs loud-fail with a diagnostic `ValueError`; per-variable mask coverage (`n_active/n_total` + bounding box) is logged at INFO. Sharp 0/1 masks only (the spec is a dict so smoothed edges can extend it later). A mask changes the *question* to a windowed/component-restricted gain — it is not an approximation of the global gain. (bead resolvent-xc-program-v0-1-0-r3uu.14.15)
- Turnkey resolvent gain-map entry point `resolvent_map(base_flow, beta_range, f_range, cfg) -> RouterResult` (`solvers/router/resolvent_map.py`, re-exported as the public `linstabpy.resolvent_map`): the single user-facing thesis symbol — base-flow in, `(len(beta), len(f))` leading-gain map + intended band labels + CP3 cost-metrics sidecar out. Per β it assembles `L0(β)` once via the existing assembly seam (`build_lnse_system`), asks `mark_unstable_band` for the per-frequency band labels, and computes every `(β,f)` point through the certified/direct `resolvent_analysis` path. The discounted (regime b) and ROM (regime c) speed engines do not exist yet, so the router is honest about it: every point currently runs on regime (a), the band label is recorded as the *intended* route, and the regime-b/c paths are gated off by default (`enable_regime_b`/`enable_regime_c`) — when left off a one-line fallback note is logged, when switched on they raise `NotImplementedError` naming the missing bead rather than silently faking a result. Each point logs index/total, ω, β, leading gain, residual, wall, peak RSS via the existing CP3 cost block + stage-heartbeat env (no new logging system); the sweep is resumable (points already present in the output `.npz` are skipped, same cache-on-existence convention as the examples) and per-point failure-tolerant (a failed solve is caught, recorded with its exception class + last state, marked `NaN`+reason-code, and retried on a resumed run rather than aborting the map). Numerics are reused, not re-implemented; an e2e test asserts a regime-(a) map point matches a standalone `resolvent_analysis` oracle within tolerance. (bead resolvent-xc-program-v0-1-0-r3uu.14.6)
- Frequency-count break-even router `break_even_decision(L0_pattern, n_frequencies, cfg) -> RoutingDecision` (`solvers/router/break_even.py`): decides direct per-frequency factorization vs the anchor-FGMRES sweep from a **measured** cost model, the frequency-count complement to the memory-driven `ram_autoselect` (r3uu.14.13). It times one real SciPy `splu` factorization (`T_fact`) and one representative solve (`T_apply`) on the supplied nonzero pattern (fabricated well-conditioned values; timing/fill depend on pattern not values), then computes the crossover `N_f* = K_anchors / (1 − iters·T_apply/T_fact)`: below it routes `"direct"`, at/above it routes `"fgmres-sweep"`. When `iters·T_apply ≥ T_fact` the sweep can never amortize its anchor factorizations, so the decision stays `"direct"` for every frequency count (crossover `inf`). `RoutingDecision` carries the chosen path, the measured crossover and timings, and a provenance note, so a cost figure can report which path each map used. The crossover is a measured number, not a guessed constant (the test varies only a probe's apply timing and asserts the crossover follows it). Illustrative small-pattern measurement — no production-scale speedup claim — and exposed but not auto-wired into any solver path (`resolvent_map`, r3uu.14.6, does not exist yet). (bead resolvent-xc-program-v0-1-0-r3uu.14.14)
- Stable-sweep solver acceleration (cost-only; numerically exact): the `linalg` factorization seam gained an additive `Factorization.solve_block` (PETSc `MatMatSolve` block multi-RHS, with a single-RHS-loop fallback for iterative/unsupported backends) and `Factorization.refactorize` (reuse the symbolic analysis/fill ordering for a new operator of identical nonzero pattern). `sweep_stable_regime` now factors the first anchor once and refactorizes in place across the remaining anchors — all anchors share K's sparsity (only the diagonal shift differs) — and routes the RSVD probe block through `solve_block`; it falls back to per-anchor factorization if reuse is unavailable or raises. The assembled/solved results are unchanged (block columns equal the single-RHS loop to round-off; symbolic-reuse gains equal recompute within the S7 σ-tolerance). `refactorize` rejects a differing nonzero pattern (collectively across MPI ranks) so reuse can never produce a silent wrong factor. Reductions in analysis time and block-vs-loop solve time are reported in `SweepResult.metrics`; these were demonstrated on a small representative case and carry no production-scale speedup claim. (bead resolvent-xc-program-v0-1-0-r3uu.14.8)
- Opt-in size-based factor-backend auto-routing: `factor_backend="auto"` resolves to STRUMPACK at or above a DOF threshold and to MUMPS below it, chosen at solve time once the operator size is known. The threshold is configurable (`auto_route_dof_threshold` JSONC field / `LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD` env, precedence env > JSONC > default) with a default of 224k DOF — the c9 A6000 measured lower bound below which MUMPS wins; the true crossover up to ~900k is unmeasured, so the default is a tunable approximation, not a universal constant. When STRUMPACK is not available in the build the route falls back to MUMPS (logged), so `"auto"` is safe on CPU-only machines and never applies an absent backend. The default factor backend is unchanged (MUMPS); auto-routing is strictly opt-in, and resolving `"auto"` to MUMPS produces results numerically identical to an explicit MUMPS run. The GPU speedup itself is measured separately and is not exercised by this routing logic. (bead resolvent-xc-program-v0-1-0-r3uu.1.42)
- Documented the NVPL (NVIDIA Performance Libraries) CPU BLAS/LAPACK build recipe for Grace aarch64 in `docs/COMPUTE_BACKENDS.md`: the PETSc `--with-blaslapack-lib` linkage against `libnvpl_blas_*`/`libnvpl_lapack_*` (LP64-sequential pair for the default 32-bit-index complex build, with the ILP64/`_gomp` variants and the MUMPS-rejects-64-bit-BLAS-indices caveat noted), that SLEPc inherits the linkage, the NumPy/SciPy Meson option, and a complex128 (Z) `ldd`/`show_config` verification step. `install.sh` ships no NVPL stack, so it builds manually into an isolated `.venv-nvpl`; on-hardware activation and verification are deferred until Grace aarch64 hardware is available (no speed claim is made). (bead resolvent-xc-program-v0-1-0-r3uu.1.15)
- Adaptive pre-sweep anchor placement `choose_anchors_adaptive(omega_range, band_marks, cfg) -> (anchors, records)` (`solvers/router/anchors.py`): starts from a coarse anchor set and inserts anchors where the S6 EVP distance to the imaginary axis (`BandMarks.distances`) falls below a radius or where measured FGMRES iterations/residual at a sample frequency exceed a threshold, replacing fixed uniform spacing. It is a pure pre-sweep helper — it returns a *fixed* anchor set (plus a per-anchor `AnchorTriggerRecord` provenance trail naming each insertion's trigger) that the caller passes into `sweep_stable_regime`, so the chosen set is fixed per RSVD pass and the S7 determinism/linearity guarantees are unaffected. The FGMRES-iteration signal is supplied by an injectable probe (or measured arrays in cfg), defaulting to a cheap probe, rather than self-measured inside the chooser. On a small case the scheme places more anchors near a hard sub-band and fewer in the easy bulk, and the resulting sweep matches a direct MUMPS oracle within the S7 σ-tolerance; the adaptive-vs-fixed factorization-count comparison is logged as a measurement (no superiority gate — "adaptive beats fixed" stays a hypothesis to measure). Exposed but not auto-wired into a default sweep path. (bead resolvent-xc-program-v0-1-0-r3uu.14.11)
- RAM-calibrated operating-point selector `ram_autoselect(L0_pattern, cfg) -> OperatingPoint` (`solvers/router/ram_autoselect.py`): decides the resolvent solve operating point from the machine, not the user. It reads available RAM portably (injectable for tests; else `/proc/meminfo` `MemAvailable`, else `os.sysconf`), runs a cheap **empirical** sparse-LU analysis probe on the supplied nonzero pattern to measure the real factor fill count and bytes-per-nonzero on *this* build (the §7c formula is only the prior; the probe is the correction, R5), estimates the memory footprint, and picks `direct` when the factor fits the budget or `compressed-sweep` (raising compression / dropping precision / adding ranks) when it does not. `OperatingPoint` returns only resource knobs (backend, reorder, compression level, restart, ranks, precision); user physics choices (grid, fd order, modes, β/f ranges) are never read or altered — selection changes COST, not the numerical answer (the direct end matches a direct MUMPS oracle within the S7 σ-tolerance). The compressed-sweep end's memory ratios are an uncalibrated prior, and its end-to-end oracle check loud-skips where STRUMPACK is unavailable or its complex adjoint is non-certifiable (see r3uu.14.9). The selector is exposed but not auto-wired into `resolvent_map` defaults. (bead resolvent-xc-program-v0-1-0-r3uu.14.13)
- Opt-in GCRO-DR-style Krylov subspace recycling for the stable resolvent sweep (`recycling` JSONC block): carries a *fixed* deflation/recycle subspace from one shifted system to the next within an anchor block and feeds it to the right-preconditioned GMRES, so consecutive shifts reuse spectral structure. The recycle subspace is prepared once per shift solve and never adapted mid-solve, so the solve remains a linear map and the S7 `deterministic_tol_linearity` certification check still passes on the recycled path (measured defects ~1e-12 against a 1e-8 tolerance). `SweepResult.metrics` reports total matvec and GMRES-iteration counts with and without recycling plus the delta. The default path (recycling off) is unchanged. Demonstrated on a small representative case (6×6 operator, 4 frequencies, recycle dim 3): matvecs 1837 → 1513 (−17.6%, including 18 setup matvecs), gains matching an independent direct MUMPS oracle to ~1.5e-13; this is a small-case illustrative measurement and carries no production-scale speedup claim. Recycling and compressed anchors are not combined in this evaluation path. (bead resolvent-xc-program-v0-1-0-r3uu.14.10)
- Opt-in compressed-anchor option for the stable resolvent sweep (`compressed_anchor` JSONC block): routes the anchor factorization through the STRUMPACK backend with BLR/HSS rank compression via the existing `linalg` seam, with the S7 adjoint-of-approximate-factor certification (check #3) wired to the *compressed* factor's own `solve_adjoint` (not the exact-LU assumption), and the measured factor-compression ratio + effective rank reported on `SweepResult.metrics`. RECORDED FINDING (measured on real STRUMPACK): the compressed-anchor path is **not currently usable in the certified sweep** — STRUMPACK's transpose-solve is a non-conjugate transpose for complex operators, so the adjoint-identity defect is ~3.2e-2 (vs ~3e-16 for MUMPS), five orders above the certification tolerance, and it is *identical with BLR compression on and off* — a STRUMPACK complex-adjoint property, not a compression artifact. The sweep therefore correctly refuses (raises) rather than silently using an uncertified factor; the compression-vs-memory hypothesis cannot be evaluated in the certified sweep until this prior backend-adjoint blocker is resolved. The default path (MUMPS-exact) is unchanged. (bead resolvent-xc-program-v0-1-0-r3uu.14.9)

### Changed
- 2-D LNSE block assembly (`_build_block_csr`) now builds each ngp×ngp block CSR-natively instead of materializing a COO `(rows, cols, data)` triplet and letting SciPy `coo_tocsr` sort it: blocks with ≤2 contributions are summed as row-scaled CSR terms, and denser blocks are filled directly into precomputed `indptr`/`indices` (one `cumsum` + per-term position scatter, no sort), eliminating the ~200 MB-per-dense-block COO intermediate. The assembled operator is byte-identical — `‖L0_new − L0_old‖ = 0` with identical sparsity on the Blasius smoke and a Case 2 stride-2 case (510k DOF, 23.2M nnz), and the seven Appendix-A oracles (<1e-10) plus the full suite stay green with no re-baseline — so this is a cost-only change. Block assembly is ~10–20% faster on the Case 2 stride-2 case (single-run, timing-noise-dependent; no production-scale speedup claim). The layout is recomputed per call and deliberately *not* memoized across calls — a module cache keyed on object ids / buffer addresses would be unsafe under allocator address reuse (a freed derivative cache's address reused by a different sparsity pattern could silently return a wrong operator). (bead resolvent-rbi6)
- 2-D LNSE boundary-condition application now runs in a single CSC→CSR→CSC format context across the whole recipe instead of converting the operator per BC rule. `apply_bc_recipe`'s SciPy path zeroes the original Dirichlet columns once in CSC, converts to CSR once, then applies all row-zeroing / Neumann row-insertion / penalty-diagonal writes in CSR before a single final CSC conversion — collapsing the redundant `csr_tocsc` round-trips (9 → 3 on the profiled case). The assembled operator is numerically unchanged — guarded ≤1e-10 by the seven Appendix-A oracles and the equivalence/parity harness (full suite green) — and assembly is ~15–20% faster on a Case 2 stride-2 representative case (510k DOF, 23.2M nnz, single rank: ~3.6–3.9 s → ~3.1 s; no speedup claim at production scale). `apply_dirichlet`/`apply_neumann` keep their public CSC-in/CSC-out contracts for direct callers, and `apply_bc_recipe` no longer mutates its input operator in place. (bead resolvent-rbi6)
- Documented the SuperLU_DIST PETSc build recipe (`--download-superlu_dist` plus the METIS/ParMETIS graph-partitioning downloads; the GPU variant adds `--with-cuda`) in `docs/COMPUTE_BACKENDS.md`, noting that `install.sh` does not yet ship a `superlu_dist` build stack so it must be built manually in an isolated `.venv-superlu_dist`. Clarified that SuperLU_DIST has two distinct roles: a live CPU mixed-precision iterative-refinement path (FP32 factor + FP64 refine via `pc_precision single`) and a complex sparse-direct GPU solve that was rejected on the tested CUDA build (order-one residuals). The previous backend status read as a wholesale "rejected"; the rejection applies only to the complex+CUDA sparse-direct path, not to the live CPU mixed-precision role. (bead resolvent-xc-program-v0-1-0-r3uu.1.16)

### Fixed
- Build requirement corrected to `setuptools >= 77.0` (was `>= 68.0`): the PEP 639 SPDX `license = "GPL-3.0-only"` expression in `pyproject.toml` needs setuptools 77+, so a fresh install on setuptools 68–76 previously failed metadata generation (`invalid pyproject.toml config: project.license`).
- STRUMPACK `EPS.setUp` factorization failures traced to static-pivoting `ZERO_PIVOT` now raise `StrumpackFactorizationFailed` and trigger a one-shot MUMPS fallback, with fallback provenance recorded in solver info; the default MUMPS path and unrelated setup failures remain unchanged. (bead resolvent-xc-program-v0-1-0-r3uu.1.43)
- Factor-backend accuracy probe is now role-aware: a backend used as the preconditioner inside an iterative-outer shift-invert solve (`st_solver` in `iterative`/`krylov`/`slepc`/`fgmres_lu`/`strumpack_fgmres`) is no longer rejected on its bare direct-solve residual. The strict `residual_rel_max` (~1e-8) gate is the right acceptance test for a standalone direct solve, but for a preconditioner the outer Krylov/SLEPc convergence is the real gate and a moderate (~1e-6) single-apply residual is healthy — previously this forced users to disable the probe by hand to run a STRUMPACK-preconditioned route. The genuine direct path (`mumps`/`direct`/`lu`, default-MUMPS exemption) keeps the unchanged strict gate; the preconditioner role applies a relaxed sanity ceiling so a catastrophically ineffective preconditioner still fails fast. The preconditioner-role detection reuses the shared `ITERATIVE_EIGEN_SOLVER_TYPES` set so it cannot drift from the solver-config vocabulary. (bead resolvent-xc-program-v0-1-0-r3uu.1.40)
- `EPS.setUp` failures now surface a useful error message. When petsc4py's `CHKERR` collapses a PETSc/STRUMPACK error into a bare `SystemError: ... returned a result with an exception set`, the raised `RuntimeError` now explains that the real PETSc error was emitted to stderr (and how to capture it via stderr redirection or `-error_output_stderr`) instead of reporting the opaque `SystemError`; a `PETSc.Error` now names its integer error code (`ierr`). All existing diagnostic context (ndof, backend, `st_solver`, MUMPS infog) is preserved. (bead resolvent-xc-program-v0-1-0-r3uu.1.41)

## [0.0.7] - 2026-06-13

GPU and alternative-backend groundwork plus the inexact-resolvent sweep
machinery, all on the CPU/MUMPS-validated default path under the frozen-numerics
invariant — no GPU backend is wired or benchmarked. Adds the MUMPS ordering and
BLR knobs, the mixed-precision factorization config surface, per-stack build
isolation, the parity-vs-MUMPS protocol, and the GPU decision docs (FP64 gate,
cuDSS C-API note); the anchor-preconditioned stable-regime sweep with RSVD
certification and EVP-guided band marking; the wavemaker structural-sensitivity
field; run provenance and a per-run metrics schema. Reconciles the package
version to a single `pyproject.toml` source and gives the band-marking guard a
physical omega-plane default.

### Fixed
- Package version source-of-truth reconciled to a single canonical source. `pyproject.toml` is now the only place the version is written, and `linstabpy.__version__` is derived from `importlib.metadata` instead of a hardcoded string that had drifted to `0.1.0` — the reserved release-candidate tag — while the build metadata said `0.0.6.dev0`. `__version__`, `importlib.metadata.version("linstabpy")`, and `linstabpy --version` now always agree with the installed distribution, guarded by a consistency test that checks every surface against `pyproject.toml`. (bead resolvent-sr3l)

### Changed
- 2-D LNSE assembly now applies the analytical `inv(B)` as a direct CSR row scaling (`data *= per-row factors`) instead of materializing a sparse diagonal and multiplying, and the SciPy raw-assembly path builds CSR natively with a single final `tocsc()` (one fewer format conversion). The assembled operator is numerically unchanged — guarded at ≤1e-10 against the validated block reference, with the seven Appendix-A oracles green — and ~26% faster on a small fd6 Blasius representative case (measured median on a shared box; no speedup claim at production scale). (bead r3uu.14.2)
- Solver module docstrings (`solvers/eigenvalue.py`, `spatial_eig.py`, `resolvent.py`, `linsolve.py`) now document the factor-backend seam: path-accurate resolution order (including that `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE` is not consulted on the reference direct eigen path), probe-based availability with loud failure, conservative backend statuses, and the env-only enforcement status of the experimental precision/BLR/ordering knobs; doc-only, behavior proven unchanged by docstring-stripped AST comparison. (bead r3uu.1.9)
- Phase-zero band-marking default `danger_radius` is now a physical omega-plane scale instead of an unrelated linear-solve residual scale. When no `danger_radius` is configured, the guard now uses `0.5 * median` of the successive spacing of the requested (finite, unique) forcing frequencies — flagging a frequency `near-pole` when a resolvent pole lies within about half a sweep cell — replacing the old fallback to `solver.eigen.backend_probe.residual_rel_max` (~1e-8), which was a residual tolerance that effectively never fired in the omega plane. With fewer than two distinct finite frequencies no sweep scale exists and the default is `0.0` (no `near-pole` labels); NaN/inf frequencies are excluded and non-uniform grids use the median successive spacing. Explicit `danger_radius` config keys still override in the same precedence and are still validated as finite non-negative distances; the least-damped `|Im omega_eig|` basis was rejected because it collapses near a stability threshold, exactly when a near-real pole most needs a band. (bead resolvent-awwc)

### Added
- Hybrid MPI + threaded-BLAS placement guidance (`docs/COMPUTE_BACKENDS.md`): documents the deliberate `OMP_NUM_THREADS`/`OMP_PROC_BIND`/`OMP_PLACES` (+ `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`) recipe as `threads_per_rank = physical_cores // ranks`, the explicit non-oversubscription invariant `ranks * OMP_NUM_THREADS <= physical cores` (SMT/HT siblings excluded), when to pick pure-MPI (the validated MUMPS sparse-direct default) vs hybrid (dense-heavy / threaded vendor BLAS), and a copy-paste `mpirun` sanity check with a clear pass rule. Placement guidance only — no measured speedup claim. (bead r3uu.1.3)
- Line-3 GPU FP64 decision gate (`docs/GPU_FP64_GATE.md`): documents why FP64 peak FLOPS is the wrong axis for the sparse-direct shift-invert workload class, a measurable go/no-go criteria table (parity quality via the acceptance-anchored `PARITY_RECORD` tolerances; factor-memory capacity; sustained bandwidth; wall time at equal mode quality; factorization-reuse economics; transfer overhead — every non-measured threshold tagged `[assumed]`, hardware numbers tagged `[vendor spec]`), and the recorded recommendation: Line 3 commits only as a capacity/bandwidth/mixed-precision play, FP64-FLOPS framing rejected, with explicit reopen conditions. (bead r3uu.1.8)
- Parity-vs-MUMPS validation protocol + CPU harness skeleton (`tests/test_parity_vs_mumps.py`): candidate factor backends are gated against the MUMPS reference on a small Blasius temporal EVP with acceptance-anchored tolerances (postcheck residual <= 1e-6, target distance <= 0.11, conjugate-aware `-conj(omega)` matching, matched-eigenvalue distance bounded by the postcheck gate — no backend-specific epsilon), emitting a structured `PARITY_RECORD` with matrix size, nnz, factor memory, wall times, ranks, exact solver options, and subspace/convergence state. Ships a mumps-vs-mumps self-consistency floor plus the CPU hard-path SuperLU_DIST `-pc_precision single` E2E test, which skips loudly on PETSc builds without superlu_dist; single-precision is verified from PETSc option consumption, never from request state. GPU execution deferred. (bead r3uu.1.6)
- MUMPS fill-reducing ordering knob (off/unset by default): `LINSTABPY_MUMPS_ORDERING` selects `amd`/`pord`/`metis`/`qamd`/`auto` via `-mat_mumps_icntl_7` (JSONC `mumps_ordering` carried, env-only enforced for now); unset expresses no opinion — it neither sets nor clears `ICNTL(7)`, preserving values installed by the legacy `mpi.mumps_icntl_7` gate or user CLI options (pinned by a real-wiring preservation test); when set it overwrites external values for that process. Unknown names fail listing allowed values; any non-MUMPS backend (including the PETSc-default path) fails loud. Real-wiring tests read `ICNTL(7)` back from the factored matrix after `ksp.setUp()`. No performance claim. (bead r3uu.1.2)
- MUMPS BLR low-rank compression knob (experimental, off by default): `LINSTABPY_MUMPS_BLR` / `LINSTABPY_MUMPS_BLR_TOL` env knobs (JSONC `mumps_blr` / `mumps_blr_tol` fields validated and carried, env-only enforced for now, like `factor_precision`) set `-mat_mumps_icntl_35 1` and `-mat_mumps_cntl_7 <tol>` for MUMPS factorizations; options persist in the options DB so MUMPS actually consumes them at factorization setup (verified by real-wiring tests against the factored matrix: ICNTL(35)/CNTL(7) read back after `ksp.setUp()`, including a non-stickiness check that an off-request clears leftovers); enabling BLR with any non-MUMPS backend (including the PETSc-default path) fails loud naming the backend and knob. No performance claim. (bead r3uu.1.1)
- Per-stack build isolation in `install.sh`: `--stack <name>` / `LINSTABPY_BUILD_STACK` select side-by-side virtual environments (`.venv-<stack>`), keeping `.venv` exclusively for the validated CPU/MUMPS reference build; non-default stacks hard-refuse to remove, overwrite, or alias (symlink) the CPU venv, `--force` is scoped to the stack's own directory, and `--dry-run` prints the resolved plan without touching the filesystem. The GPU configure branch (`--with-cuda --download-strumpack`) is documented as design-only in `docs/COMPUTE_BACKENDS.md` — no GPU build is claimed. (bead r3uu.1.7)
- Wavemaker / structural-sensitivity utility (`linstabpy.wavemaker`, `stability/sensitivity.py`): Giannetti–Luchini field `zeta = |q_dir| |q_adj| / |<q_adj, q_dir>_W|` from direct and Chu-weighted adjoint modes — the first user-facing consumer of the free matrix adjoint. Pairs modes by `|lambda_dir - conj(lambda_adj)|` with a tolerance gate that fails loud listing the offending eigenvalues, normalizes with an explicit retained inner-product divisor, reshapes Fortran-order per state component, and logs a per-mode pairing table at INFO. The adjoint convention (`adjoint_eigenvalues` returns `W^-1 L0^H W` modes, no extra transform) is pinned by a discriminating test. (bead r3uu.20)
- Public-boundary input validation: `Mesh.create`, `resolvent_analysis`, `eigenvalues`, `adjoint_eigenvalues`, and `spatial_eigenvalues` now reject malformed arguments at the call boundary with typed errors naming the bad argument (non-square or non-sparse operators, mismatched weight-vector lengths, `n_modes < 1`, non-finite frequencies/targets, grids below the SBP stencil minimum for the requested `fd_order`) instead of failing deep inside assembly; error messages never leak filesystem paths. (bead r3uu.3.2)
- Mixed-precision factorization config surface (experimental, inert by default): `LINSTABPY_FACTOR_PRECISION` env knob and JSONC `factor_precision` field accept `double` (default, zero effect on the options DB) or `single`, which is honored only with the `superlu_dist` factor backend (inserts `pc_precision single`) and fails loud with any other backend; the JSONC field is carried in the typed config and will be consumed by the Phase B.4 iterative-refinement work. (bead r3uu.1.4)
- Anchor-preconditioned stable-regime frequency sweep (`solvers.router.stable_sweep`): `sweep_stable_regime` factors the shifted operator only at caller-given anchor frequencies and sweeps neighboring frequencies with right-preconditioned GMRES reusing the nearest anchor factor; forward and adjoint actions share one factorization per anchor, every leading singular triplet passes the `certified_rsvd` gate, and per-frequency iteration counts and residuals are recorded as measured quantities, never gated. (bead r3uu.14.5)
- Solver-stack reproducibility manifest: runtime provenance now records python/numpy/scipy/petsc4py/slepc4py versions, PETSc arch, MUMPS version, and the MPI implementation (failed probes record `unavailable`, never crash a run); the block is echoed at INFO at save time and `scripts/tools/freeze_env.sh` emits a pip-constraints file for release freezes. (bead r3uu.21)
- Per-run metrics schema gate (`metrics_schema`): a single source-of-truth schema for the cost+quality sidecar with `validate_metrics` reporting every missing or ill-typed key at once; sidecars failing validation are written with a `schema_errors` field and a loud warning so the run is visibly quarantined from the figure pipeline. (bead r3uu.7)
- RSVD certification gate (`solvers.router.certification`): `certified_rsvd` runs five algebraic checks before trusting an inexact resolvent action — σ-accuracy-derived solve tolerance, forward/adjoint consistency with realized-residual enforcement, adjoint-of-approximate-factor verification, deterministic linearity probed on both forward and adjoint actions, and multi-seed RSVD reproducibility — and fails loud naming the tripped check. (bead r3uu.14.4)
- Phase-zero EVP-guided band marking (`solvers.router.band_marking`): labels requested forcing frequencies stable / near-pole / smooth by distance to the resolvent poles (`-omega_eig`) before any sweep. (bead r3uu.14.3)
- Config-selectable FGMRES shift-invert path with direct-LU preconditioning, including STRUMPACK backend selection when available.
- Runtime result archives and metrics sidecars now include git, config, host, compute-mode, and MPI-stack provenance.
- CPU/MUMPS default-path parity fixture on a small Blasius case and standalone regression guard for ≤1e-12 baseline drift. (bead r3uu.1.13)

### Changed
- Runtime cost metrics now use `unknown` as the default hardware tag when no hardware tag is configured.

### Fixed
- `install.sh --dry-run` `planned_remove` now mirrors the real venv lifecycle exactly: it reports a removal only when the venv actually exists and would be removed — under `--force` on an existing venv, or a Python-version mismatch that triggers recreate — and reports `none` for an absent venv or in `--check` mode (which never removes). Previously it keyed off `--force` alone, over-reporting a removal for an absent venv or under `--check`, and under-reporting the mismatch-recreate case. Plan-reporting fidelity only, via a single shared `venv_will_be_removed` predicate; real removals were already correctly guarded by `remove_venv_dir`. (bead resolvent-e4s8)
- `linstabpy --version` and `--help` (`python -m linstabpy.driver.cli`) now work on a base install without the optional PETSc/MPI stack: the `linstabpy.driver` package initializer no longer eagerly re-exports `run` from the PETSc/MPI-backed `linstabpy.driver.run`, so the CLI's existing lazy import is no longer defeated at package-import time. Verified by a test that blocks `petsc4py`/`mpi4py` via `sys.meta_path` and asserts neither is imported on the `--version`/`--help` paths. Import the runtime explicitly (`from linstabpy.driver.run import run`) where it is needed; no consumer change required. (bead resolvent-xs4h)

### Deprecated
- The legacy `mpi.mumps_icntl_7` / `LINSTABPY_PETSC_SET_MUMPS_ICNTL_7` boolean is now a deprecated compatibility alias for `mumps_ordering=metis`; it is still accepted (and still installs `ICNTL(7)=5`) but new configs should use the canonical `mumps_ordering` surface (`mumps_ordering: "metis"` / `LINSTABPY_MUMPS_ORDERING=metis`). The `metis`→`ICNTL(7)=5` value is now drawn from a single shared `MUMPS_ORDERING_TO_ICNTL_7` table (new PETSc-free `linstabpy.mumps_options` module) consumed by both the legacy gate and the ordering knob, so the two surfaces can no longer drift. Numerically identical to the prior behavior for every existing config (real-wiring `ICNTL(7)` read-back: legacy alias → 5, explicit `mumps_ordering` overrides the alias, unset leaves the key absent → MUMPS default 7). (bead resolvent-38ww)

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
