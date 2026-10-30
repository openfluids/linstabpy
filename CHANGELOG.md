# Changelog

All notable changes to linstabpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)

- The resolvent uses the physical frequency sign: `H(omega) = [(gamma - i*omega) I - L0]^-1`,
  the response to a forcing `exp(-i*omega*t)`. This is the time convention of the
  stability solver. The code inverted `L0 - (gamma + i*omega) I` before, the
  LinStab2D matrix, which is `-H(-omega)`. Exact time integration of
  `dq/dt = L q + f exp(-i*omega*t)` agrees with the new form to 1e-15 and differs
  from the old form by 120 % to 150 % (test operator: stable, non-normal, complex,
  6 x 6). Effect on results you already have:
  - Gains for a flow with `W = 0` (no spanwise or azimuthal base velocity), at any
    `beta`, are unchanged.
  - For `beta != 0` and `W != 0` (also Floquet and swirl cases) the gain map is
    mirrored in `omega`: `gain_new(omega) = gain_old(-omega)`.
  - Forcing and response modes change in every case. For `W = 0` they are the
    complex conjugates of the old modes, the response also changes sign relative
    to the forcing, and for `beta != 0` the spanwise velocity component also
    changes sign. For `beta != 0` and `W != 0` the modes at `omega` are the old
    modes at `-omega`, with the response sign reversed.
  - Waves in the old modes travelled upstream when animated with
    `Re[q exp(-i*omega*t)]`. The new modes travel in the correct direction.

  To reproduce an old result set `solver.resolvent_sign_convention: matlab_parity`
  (`shifted_operator(..., matlab_parity=True)` in code). The key sits beside
  `solver.resolvent`, not inside it. `validate_config` rejects other values.
  The LinStab2D parity examples 1, 2 and 5 set it. Band marking puts the
  resolvent pole at `+omega_eig` (it was `-omega_eig`) and centres its EVP target
  on the band, not on its negative; `matlab_parity` restores the old placement.
  Saved resolvent results and `resolvent_map` stores carry the key
  `resolvent_sign_convention`. `resolvent_map` refuses to resume a store from the
  other convention, and a store without the key counts as `matlab_parity`.
  How to convert old maps and modes, with the derivation and the evidence, is in
  `wiki/resolvent-sign-convention.md`.

- The `linstabpy` command refuses a case file that sets a top-level key it does
  not read (`Re`, `Ma`, `Pr`, `kappa`, `T_0`, `sponge`, `fd_order`, `beta`,
  `penalty`, `St`, `omega_target`, `run_stability`, `run_resolvent`, `n_modes`,
  `ncv` and others). Before, the key was accepted and the default value was used
  without a warning. The error names the nested key to use (for example
  `physics.Re`). Example scripts that read these keys from their own YAML files
  are not affected. The `logging` block (`stage_heartbeat_seconds`) is now
  accepted.
- `resolvent_map` refuses to resume a store written for other problem inputs:
  configuration, mesh, base flow, energy weight, forcing norm, mode count,
  discount or sign convention. The error names the inputs that differ and the
  store is not changed. Before, the store was ignored and then overwritten.
  Solver settings (backend, factorization, tolerances, router settings) may
  differ: the run resumes with one warning that names them, and the user is
  responsible for the result. A field that exists in only one of the two
  fingerprints (for example a new attribute after an upgrade) also resumes with a
  warning. The `beta` and `f` arrays may be extended: when every stored value is
  in the new arrays, the stored points are copied and only the new points are
  computed. A grid that drops a stored value is refused. A store without a
  fingerprint resumes with a warning. A store that cannot be read is moved to
  `<name>.unreadable-<time>`.
- Band marking searches for poles at up to 9 targets across the requested
  frequency range, not only at its centre. Poles near the ends of the range were
  missed before. Each target costs one shift-invert factorization;
  `skip_band_marking` still removes the stage.

- Temporal `adjoint_eigenvalues` returns `conj(omega)` of the direct eigenvalues.
  It returned `-omega` before, which broke the documented contract
  `adjoint_omegas == conj(temporal_omegas)` whenever `Re(omega) != 0`. The
  adjoint eigenvalue is `i*conj(omega)`, so `conj(omega) = -i*lambda_adj`.
  Code that used the old values must change sign of the real part.

- The `linstabpy` command has no default case file. Without `LINSTABPY_CASE_CONFIG` it stops
  and names the variable. It looked for `./configs/parity_config.yaml` before.

- `physics.Re` has no default in the case file. A case file without it raises
  `ConfigValidationError`. The default was 395000.

- `paths.case_inputs_mat` has no default. It is read only when
  `physics.use_matlab_sponge_aux` is true. The default was
  `MATLAB_exports_resolvent/case_inputs_used.mat`.

### Fixed

- `resolvent_map` stores the discount `gamma` used at each point
  (`discount_gamma`, 0 for ordinary points). Points near a pole are solved with
  a discount and were stored like ordinary gains. Stores written before this
  change give NaN at their near-pole points on resume.
- Dissipation on a masked mesh is built on the used points, one operator per
  line segment, with the same segment rule as the masked derivatives. Before,
  the assembly failed with a shape error.
- `deform` composes the metric of earlier deforms: a chain of deforms has the
  metric of the final map. Before, a one-direction deform reset the metric of
  the other direction to the identity, and a full-2D deform of a deformed mesh
  dropped the earlier metric. A deform of an undeformed mesh is unchanged bit
  for bit.
- The raw-eigenvalue diagnostic in the driver uses the `omega = eig(i*L0)`
  convention for its target and its results, and accepts a PETSc `L0`.
- The frequency walk starts each inner solve from zero. The previous solution
  belonged to another right-hand side and saved no iterations.
- The block solve lets PETSc split the columns of the dense right-hand sides
  across ranks (results were correct before at 2 and 3 ranks).

- `limit_strumpack_host_threads()` sets the OpenMP thread count to one
  before the first STRUMPACK factorization, because the library is not thread
  safe. It runs one time for each process, writes one line in the log, and
  warns instead of failing where `libgomp` does not load. MUMPS, cuDSS and the
  BLAS are untouched, so only the STRUMPACK host work becomes serial: 4.10 s
  against 3.64 s on the small test problem, unmeasured on a production
  operator.
- Neumann rows on the `l` and `r` borders use the inward normal, as `b` and `t` did.
  The outward normal added a spurious mode with `Im(omega) ~ penalty/h`: 6.35e4 at
  penalty 1e4 on a 10 x 9 fd4 mesh, against 8.2 without the condition. Results of
  every case with a Neumann rule on `l` or `r` change. The condition `dq/dn = 0` does
  not depend on the sign. LinStab2D's `BC_Neumann.m` uses `+d/dx` and `+d/dy` on all
  edges, so it has the same spurious mode on `r` and `t`.
- Boundary sponge edges combine by maximum, so corner values no longer depend on the
  edge order. The parabolic ramp no longer writes the row or column at index `width`,
  which erased the ramp of the perpendicular edge. Sponge values change at corners and
  next to the ramps.
- `apply_preprocess` filters copies and leaves the caller's arrays unchanged.
- The Song run wrote its 430 MB result over the shipped 6.6 MB trimmed file
  `examples/song/output/fd4_n30/song_2023.npz`. The clone then showed a
  modified tracked file, and the tracked-file-size test failed. The run now
  writes to `output/fd4_n30_full/`, which git ignores. The report and plot
  scripts read that file when it exists, and the shipped copy otherwise.

- Stability acceptance pairs `omega` with `-conj(omega)`, not `conj(omega)`. With
  `omega = eig(i*L0)` and a real `L0`, the spectrum is closed under
  `omega -> -conj(omega)`. The old conjugate-aware target distance admitted modes
  near `conj(target)`, which have the opposite growth sign, and the pair
  completeness check looked for the wrong partner. The driver default for
  conjugate-aware acceptance is now off only when `beta != 0` and the spanwise
  base flow `W != 0` (pairing mismatch 6.9e-2 there, against 1e-9 in the other
  cases). A value set in the YAML still wins.
- SBP operators need `n >= 2r` points (`r` rows in each boundary closure). The two
  closures overlap below that size, and the identity `H D1 + (H D1)^T = B` failed
  (defect 0.16 at fd4 with `n = 6`). `min_points` is now `max(m, 2r)` (3, 8, 12, 16
  for orders 2 to 8). All SBP builders refuse a smaller `n`. A short masked
  segment takes the highest order that fits, not order 2. Overlapping closure
  rows are cleared, so `D1` of a constant is zero. Operators on meshes without a
  hole and without a segment shorter than `2r` are bitwise identical to before.
- A masked mesh with a one-point hole splits its derivative lines at the hole.
  The stencil spanned the hole before (`max|Dx x - 1| = 0.5`).
- The dissipation layout follows the size of `Q_diss_x`, not the point count. A
  mask that keeps every point failed with a shape error before.
- `backend: auto` assembles with SciPy when operator dissipation is on. It chose
  the direct PETSc builder, which has no dissipation term, and the run failed.
- The result NPZ is written to a temporary file and renamed, under the exact name
  in `LINSTABPY_CASE_OUTPUT_NPZ`. A name without `.npz` got `.npz` appended, and the
  cache never hit. A killed run no longer leaves a truncated file. The metrics
  file writes `null` for infinite values (`lambda_over_L` at `beta = 0`).
- A truncated cached NPZ is recomputed. Before, every later call stopped with
  `zipfile.BadZipFile` on rank 0.
- The recorded git SHA comes from the package directory. It was `unknown`, or the
  SHA of another repository, when you ran from a case directory.
- The top-level `preprocess` block applies in the case driver. The filter and the
  shock cleanup were skipped without a message before.
- The NPZ provenance records the settings the run used: the configured factor
  precision (it was always `double`), the resolvent factor precision from
  `resolvent.linear`, and the `conjugate_aware` flag. Root-assembly mode now
  applies `root_scatter`. The dense eigenvalue fallback no longer raises
  `TypeError` after the solve.
- The iterative shift-and-invert branch applies `sub_ksp_type` and `sub_pc_type`.
  Before, PETSc error 58 was swallowed and the two options were dropped. The
  spatial solver raises an error for an unknown `st_solver`. It used the SLEPc
  default without a message before.
- The line walk keeps the last forward solve as the next warm start. It stored an
  adjoint solve. `sweep_stable_regime` reads `reuse_symbolic` and
  `use_block_solve` as booleans, so the string `false` is false. `savarino_line`
  uses `invW = 1` for `forcing_norm: l2`. It computed the Chu gain before.
- The adjoint self-test fails on NaN. A NaN solve passed the identity and oracle
  checks before. `Factorization.destroy()` frees the dual `K^H` factor.
- `numerics.floquet_shift` overrides `LINSTABPY_FLOQUET_SHIFT`. The environment
  value won over the YAML before, and a value from an earlier `run()` in the same
  process persisted.
- Cost metrics read `LINSTABPY_GPUS` and `LINSTABPY_HARDWARE_TAG`. Only the old
  `LINSTAB2D_*` names were read, so the new names gave an unknown-variable
  warning and `cost_gpus` stayed 0. The old names are now ignored (see Removed).
- With root-only assembly, the target-distance gate works. It rejected every mode
  with `postcheck failed` before. Asking for the residual gate in that mode,
  which cannot run there, is now a configuration error.
- The `resolvent_map` store is written by MPI rank 0 only. All ranks wrote the
  same file and all but one failed with `FileNotFoundError`, which stopped a
  4-rank run with `output_npz`. Rank 0 also moves an unreadable store aside.

### Changed

- The axisymmetric (`model: axy`) operator keeps the mu'(T) T' terms by
  default: mu'(T) T' div(S_bar) in the three momentum rows and the matching
  mu' and k' terms in the energy row. LinStab2D `getCoeffsLaminar.m` omits
  them. Results of axy runs with temperature-dependent viscosity change by
  O(1/Re): the operator by about 1e-4/Re, the leading eigenvalue by 1e-4 at
  Re 500 down to 1e-6 at Re 5e4, and resolvent gains by up to 1.2e-2 at
  Re 500. Set `matlab_parity_dmudt: true` (keyword `matlab_parity_dmudt=True`
  in the assembly functions) to restore the LinStab2D operator. The jet and
  paraboloid examples set it.

- The `linstabpy` command refuses a case that sets `beta` or `kx` at the
  top level. It reads the spanwise wavenumber only from `numerics.beta`, so
  such a case ran at beta = 0 without a message. The example scripts still
  read the top-level keys from their own YAML files, and `validate_config`
  still accepts them.

- `build_lnse_operator` and `build_lnse_system` refuse a positive
  `dissipation_epsilon` when the mesh has no dissipation matrices
  (`Q_diss_x`, `Q_diss_y`). They used to return the undamped operator
  without a message. A mesh gets the matrices from
  `Mesh.create(..., numerics_operator_dissipation=...)`. An epsilon of zero
  gives the same operator as before. The map router's own check for this
  case is removed, because the assembly now makes it.

- The STRUMPACK known issue of 1.0.0rc11 is measured again and restated
  below. It is not a rare fault of one build: it follows the thread count.

- The documentation has one host, https://linstabpy.vercel.app/, the site
  the paper cites. The GitHub Pages workflow is now `docs.yml`, which only
  checks that the wiki builds with `mkdocs build --strict`.

- Dissipation has one config surface. `resolvent_map` now reads its strength
  from `numerics.operator_dissipation` (`enabled`, `epsilon`,
  `apply_to_resolvent`), as the driver does, through one function,
  `config.defaults.stage_dissipation_epsilon`. It used to read a top-level
  `dissipation_epsilon` (alias `eps_resolv`) and build the damping operator
  from the block, so setting only one of the two ran the map undamped without
  a warning. The top-level keys now stop the run with an error that names the
  block. Migration: move the value to `numerics.operator_dissipation.epsilon`
  and set `enabled: true`. A nonzero strength on a caller-supplied mesh
  without `Q_diss` is refused for the same reason. The shipped Savarino
  configs set `enabled: false`, so their numbers do not change.

- `examples/martini/` is now `examples/linstab2d/`, the name the paper uses
  for the six LinStab2D validation cases. Import paths change from
  `examples.martini` to `examples.linstab2d`. The file contents are the same
  apart from the path.

- The wiki build command in `docs/DEVELOPMENT.md` uses its own environment
  (`UV_PROJECT_ENVIRONMENT=.venv-wiki`). `uv run` on `.venv` replaced the PETSc
  environment that `install.sh` builds.

### Added

- The `linstabpy` command is installed with the package (console script
  `linstabpy = linstabpy.driver.cli:main`). Before, only
  `python -m linstabpy.driver.cli` worked.

- `examples/odd_even/`: the script that reproduces the odd-even decoupling
  figure of the paper (wavy-wall mesh, 81 x 41 and 161 x 81, six values of
  epsilon), its run record `odd_even.csv`, and the figure script. It uses the
  SciPy operator path, so it runs without PETSc.

- `python -m examples.song.prepare_baseflow --from-reference` writes the Song
  solver input from the shipped `reference/baseflow/song_32deg.npz`, so the
  Song case runs from a clone without Song's raw Plot3D files. The npz stores
  velocity / c_inf; the command divides u, v and w by Ma, which gives the
  velocity / U_inf input of the paper run to 1e-16.

### Removed

- `LINSTABPY_FAST_REFACTOR`. Serial cuDSS now always refactorizes in place,
  which has been the default since 1.0.0rc10. A run that still sets the
  variable gets the unknown-variable warning with the reason.

- `examples/savarino/reference/savarino_response_{mack,liftup}.npz`. They
  are repackaged from the Savarino Zenodo modes, whose record states no
  redistribution license. `pack_reference_modes.py` builds them from the
  fetched modes, and `compare_modes.py` prints the commands when they are
  missing.

- All backward-compatibility names, before 1.0:
  - `LINSTABPY_CASE1_CONFIG`, `LINSTABPY_CASE2_CONFIG`, `LINSTABPY_CASE1_OUTPUT_NPZ`,
    `LINSTABPY_CASE2_OUTPUT_NPZ`, `LINSTABPY_CASE1_ROOT_ASSEMBLY` and
    `LINSTABPY_CASE2_ROOT_ASSEMBLY`. Use `LINSTABPY_CASE_CONFIG`,
    `LINSTABPY_CASE_OUTPUT_NPZ` and `LINSTABPY_CASE_ROOT_ASSEMBLY`. The driver
    ignores the old names and warns that they are unknown variables.
  - `LINSTAB2D_GPUS`, `LINSTAB2D_HARDWARE_TAG` and `LINSTAB2D_STAGE_HEARTBEAT_SECONDS`.
    Use `LINSTABPY_GPUS`, `LINSTABPY_HARDWARE_TAG` and
    `LINSTABPY_STAGE_HEARTBEAT_SECONDS`.
  - `linstabpy.solvers.eigenvalue.temporal_eigenvalues`,
    `linstabpy.solvers.eigenvalue.adjoint_eigenvalues` and
    `linstabpy.solvers.spatial_eig.spatial_eigenvalues`, with the deprecation
    warning that forwarded them. Use `linstabpy.stability.eigenvalues`,
    `linstabpy.stability.adjoint_eigenvalues` and
    `linstabpy.stability.spatial_eigenvalues`. The same three names are gone from
    `linstabpy.solvers`.
  - `solve_transpose` and `solve_transpose_vec` on the factorization object. Use
    `solve_adjoint` and `solve_adjoint_vec`.
  - `AnchorTriggerRecord.trigger`. Use `"+".join(record.triggers)`.
  - `linstabpy.driver.preflight.SAWTOOTH_BETA_BAND`. Nothing used it.

### Known issues

- **The STRUMPACK LU returns a wrong solve when it runs multithreaded.** The
  GPU is not needed for it and a newer STRUMPACK does not fix it. Reproducer:
  factor `K = L0 - sigma*I` for the Blasius temporal problem (750 unknowns,
  sigma = -0.1j) and measure the probe residual on a seeded right-hand side;
  sixteen fresh processes per arm, one A6000, one script across three builds.
  A run is bad when its residual is above 1e-8. The first two builds ran on
  17 September, the third on 25 September.

                              PETSc 3.24.2            PETSc 3.25.5            PETSc 3.25.5
                              STRUMPACK v8.0.0        STRUMPACK master        STRUMPACK 273d9068
      one thread, GPU on      2.102e-10 x16           2.102e-10 x16           2.102e-10 x16
      ten threads, GPU on     8e-11..2.5e-3, 2 bad    8e-11..7.0e-7, 1 bad    7e-11..8.7e-9, 0 bad,
                                                                              1 crash
      ten threads, GPU off    1e-11..1.1e-5, 1 bad,   9e-12..1.0e+01, 2 bad   1e-11..1.1e-8, 1 bad
                              1 crash

  One thread gives the same value sixteen times, and the same value on all
  three builds. Ten threads scatter on all three builds whether the card is
  used or not. PETSc 3.25.5 pins STRUMPACK master of May 2026, eighteen months
  and 24 commits after the v8.0.0 tag, including one that sets the CUDA device
  per OpenMP thread; the fault survives it. The third build takes STRUMPACK
  master at commit 273d9068 of 23 September 2026, the newest at the time of
  the test; none of the commits since May names threads or OpenMP. The fault
  survives it too, and one ten-thread process ended without a result. On
  both PETSc 3.25.5 builds PETSc reports `-mat_strumpack_gpu 0` as unused, so
  their GPU-off row may have used the card. Upstream's own TODO says the
  library "is not completely thread safe at the moment".

  The error reaches the answer. Through FGMRES with a STRUMPACK
  preconditioner, the five leading eigenvalues of the same example missed the
  MUMPS reference by 1.1e-11 to 6.9e-9 on ten threads, two runs of eight above
  1e-9, against 2.209e-11 eight times out of eight on one thread. MUMPS on ten
  threads stayed at 1e-11.

  LinStabPy now sets the STRUMPACK OpenMP thread count to one itself; see
  Fixed above. The accuracy probe stays behind it, and so does the FGMRES residual
  check on the answer, which is what keeps a bad factorization out of a
  published number.

## [1.0.0rc11] - 2026-09-17

### Fixed

- The dissipation advisory warned on every run without artificial
  dissipation, whatever the mesh. On a rectilinear mesh the bare operator
  cannot hold an odd-even (sawtooth) mode, so the warning fired against the
  correct setting for both released cases. The verdict now comes from
  `mesh_advisories()`, which measures the mesh that the run uses and reports
  the number in every case: the released Savarino and Song meshes measure
  4.8e-15 and 2.1e-12 against a gate of 1e-8 and get a note, and a bent mesh
  without dissipation still gets the warning. A config that does not state
  its dissipation choice at all is reported by `config_advisories()` as
  `DISS-UNDECLARED`.
- The base-flow check compared the largest value in the field with 1, so
  every compressible flow with a shock or a hot wall looked mis-scaled. It
  reported density 1.39 and 1.51 and temperature 1.81 and 1.83 for the two
  released base flows, whose freestream is exactly 1. The check now looks
  for the freestream itself, the part of the field where density, velocity
  and temperature are all 1, and records the share of the field it covers
  (1.9 % and 3.7 %). A field scaled with the speed of sound has no such
  region and is still reported, now with the values of the largest uniform
  region so the actual convention is visible.
- The MUMPS ICNTL entries went into the PETSc option database at startup on
  every run. A run with no MUMPS stage left them unused and PETSc ended with
  "There are options you set that were not used". They now go in only when a
  stage can use MUMPS, and the two cuDSS example configs name the card for
  every stage instead of naming MUMPS for stages they never run.
- `examples/savarino/savarino_mapwalk.py` wrote the real host name into the
  `machine` column of every line record, which put a private host name in
  three shipped run records. It follows the package policy now: the label is
  `unknown` unless `LINSTABPY_HOSTNAME` is set.
- `examples/savarino/verify_norm_convention.py` ended with a traceback when
  the Zenodo base flow and modes were not in the tree, which is the state of
  a fresh clone because their licence does not allow redistribution. It
  prints the two fetch commands and stops, like `run_compare_points.py`.

### Changed

- The test that pins the committed beta = 40 farm record checks its gains,
  which a rerun reproduces, instead of the sum of its wall times, which is a
  property of the machine on the day.

### Known issues

- **The STRUMPACK LU returns a wrong solve about one run in three on this
  build.** Measured at the shift-invert probe of the Blasius temporal
  problem, sixteen fresh processes on byte-identical input (c9, one A6000,
  PETSc 3.24.2): eleven runs gave a relative residual of 6.3e-13 to 8.7e-12
  and five gave 1.9e-5, 2.9e-5, 3.0e-5, 6.7e-2 and 1.1e-1. Nothing in the
  input varies: the probe right-hand side is seeded and the mesh is fixed.
  The accuracy probe rejects every bad solve, which is what it exists for,
  and the eigenvalue and resolvent paths refine with FGMRES and check the
  residual afterwards, so a bad factorization cannot reach a published
  number. The cause is not established. Prefer MUMPS or cuDSS for
  production work on this build until it is.


## [1.0.0rc10] - 2026-09-15

### Changed

- A line of frequencies on one card is faster. For a serial cuDSS
  factorization, a refactorize now takes the CSR arrays of the new operator
  once, checks the nonzero pattern against the first operator, writes the
  values into the same PETSc matrix, and gives them to cuDSS. The adjoint
  factor gets K^H from a permutation of those values that is computed once
  per factorization, not from a sparse transpose at every frequency. On the
  450,000-unknown Savarino operator, one A6000, the median line point went
  from 32.1 s to 25.8 s over the eight frequencies after the first of the
  beta = 40 line, with gains identical to ten digits. The path is the default; set
  `LINSTABPY_FAST_REFACTOR=0` to use the general path. A different nonzero
  pattern raises `ValueError`.
- The run summary reports the peak memory of the largest rank next to the
  sum over ranks.
- `run_all_examples.sh` prints `SOFT_FAIL` with the case names when the
  Martini parity comparison has soft failures. Before, it printed `PASS`.

### Fixed

- `examples/savarino/savarino_mapwalk.py` compared each line gain with
  `output/savarino_map2_200pt.csv`, a farm computed before the filter-order
  fix. The `farm_gain` and `rel_err` columns of a line record then showed
  differences up to 4.4e-3 that are not in the line result. The default
  reference is now `savarino_map3_200pt.csv`, the committed map of record,
  in the script, in `savarino_mapwalk.sbatch.template`, which passed the old
  file explicitly, and in the test and docstrings that named it.

### Added

- `wiki/` architecture pages and `.github/workflows/pages.yml`. MkDocs
  builds the site. Pages are hand-written and hand-maintained.

### Fixed

- The `solvers/router/resolvent_map` module docstring said every gain map
  needs SBP sawtooth dissipation. That claim is wrong on rectilinear
  meshes. The odd-even mode exists only on a mesh that is curvilinear in
  both directions, where the metric cross-terms mix grid-scale content.
  The docstring now states the mesh condition, points to
  `driver.preflight.metric_cross_coupling()` as the test, and scopes the
  tuned `epsilon ~ 0.10` to the curved-wall mesh it came from. This aligns
  the last remaining text with the measured story already in
  `config/defaults.py` and `driver/preflight.py`.
- The Martini parity manifest still pointed Example 3 at a combined
  `example_3_streaky_bl.npz` that nothing writes since the example was split
  into the four paper campaigns, so every full parity run reported a hard
  failure on a missing artifact. The compare now walks the per-campaign
  NPZs, each against its own MATLAB bundle, with the same residual-first
  gates as before, and aligns the component-sliced mode subsets the
  repository bundles store (idx rows, saved-mode columns) before pairing.
  With the wiring repaired, the full parity run reports four passes, two
  documented structural soft failures, and no hard failures.

## [1.0.0rc9] - 2026-08-27

Release-hardening pass before 1.0.0.

### Changed

- Consolidated duplicated private helpers into shared modules: config lookup
  (`cfg_get`, `as_bool`) in `utils/_solver_cfg`, `peak_rss_bytes()` in
  `utils/cost_metrics`, the spatial Dirichlet helpers in
  `boundary_conditions/dirichlet`, the backend normalizer in
  `linalg/backends`, and one `_matvec(operator, x, adjoint=...)` in the
  solver router in place of a forward/adjoint pair. Net -142 lines with no
  behavior change; every module now imports the one canonical copy instead
  of carrying a private variant that could drift.

### Fixed

- Four call sites in `solvers/router/stable_sweep.py` still called the
  removed `_adjoint_matvec` after the mat-vec merge, so PETSc-backend sweeps
  raised `NameError`. These paths only execute where petsc4py is installed,
  which is why a PETSc-less checkout did not catch it.
- Environment-absence tests (missing nvidia-smi, missing PETSc stack,
  missing repository `.venv`, empty PETSc options database) now simulate the
  absent component instead of asserting properties of the host machine. The
  full suite passes unmodified on CPU-only machines and on GPU nodes.
- The Martini export tools now locate the upstream MATLAB code through
  `LINSTAB2D_ROOT` only, and the one parity test that reads a raw MATLAB
  fixture bundle takes its directory from `LINSTABPY_MATLAB_FIXTURES_NPZ`;
  the machine-specific fallback paths are gone, and the public-provenance
  scanner now rejects those path forms.

### Verification

The tagged commit passes the full test suite with zero failures on both
supported stacks: 1574 tests pass on a CPU-only environment (MKL BLAS,
MUMPS) and 1584 on a GPU environment (CUDA-enabled PETSc with STRUMPACK
plus the cuDSS backend, on an NVIDIA RTX A6000). Every remaining skip
names either an optional backend absent from that environment or a
reference file that is not distributed with the repository, and between
the two environments every backend-gated test runs at least once.

The six examples (the five Martini flows plus the Example 0 derivative
protocol) were recomputed from scratch for this release — every cached
NPZ deleted first — through `./run_all_examples.sh`, and all six pass.
Against the shipped MATLAB reference bundles, the Blasius and jet
resolvent gains agree to 1.7e-11 and 3.3e-12 relative, the cavity spectra
agree to 6e-6 with eigenvalue residuals below 2e-8, and the Example 3
temporal and spatial eigenpairs satisfy their discrete problems to
residuals of 5e-9 or better. (The Example 3 parity check required the
manifest repair recorded under Unreleased; the artifacts it verifies are
the ones recomputed here.)

## [1.0.0rc8] - 2026-08-21

### Added --- cuDSS GPU sparse-direct backend

The largest performance change since the package was created. A new GPU
backend factors the linearized operator with NVIDIA cuDSS (through
`nvmath-python`) and wraps it in the same right-preconditioned FGMRES that
already owns the accuracy of the iterative path. On the 450,000-unknown
SWBLI resolvent operator (2.0--2.3e7 nonzeros, complex double) one
factorization drops from 41--55 s (STRUMPACK on an NVIDIA RTX A6000) to
5.7--9.0 s (cuDSS, double precision) on the same card. A complete
frequency-line resolvent point --- two refactorizations plus ten forward and
ten adjoint solves converged to 1e-10 --- costs 23 s, against 60--82 s for
the four-rank MUMPS CPU line mode and 130--180 s for the STRUMPACK GPU
path. The GPU changes from the slower arm into the fastest solver on the
node, in full double precision, with no accuracy loss.

- Added `linalg/cudss_factor.py`: a cuDSS solver wrapper with the four
  operations the resolvent map needs: `plan` (analysis, once per sparsity
  pattern), `factorize`, `refactorize` (new values on the kept pattern,
  through `reset_operands(a=...)` --- nvmath copies a host operand to the
  device when it is set, not on each call), and per-right-hand-side `solve`
  (`reset_operands(b=...)` for the same reason). The cuDSS threading layer
  (`libcudss_mtlayer_gomp`) is discovered inside the installed NVIDIA wheel
  and enabled with the job's host threads; it halves the analysis time.
- Added `factor_backend: cudss` to the solver configuration: FGMRES in
  complex double, right-preconditioned by the cuDSS LU, tolerance owned by
  the iteration (`ksp_rtol`, default 1e-10). The direct MUMPS and STRUMPACK
  paths are unchanged.
- Adjoint solves build a second cuDSS factorization of the conjugate
  transpose. cuDSS 0.8 has no transposed or adjoint solve
  (`CUDSS_CONFIG_SOLVE_MODE` accepts only its default), so the dual factor
  is the design, not a heal: both factors share one analysis cost profile
  and refactorize together along a frequency line.
- Added `factor_precision: single` (opt-in, cudss only): the factorization
  runs in complex64 --- 0.7--1.2 s per factor, half the device memory ---
  while FGMRES stays in complex double and restores the solution to 1e-12.
  The option force-enables the two settings measured to be necessary:
  MC64-style matching (`MAX_DIAG_PRODUCT`) and double-precision row/column
  equilibration before the cast. Without matching the single-precision
  factor fails near zero frequency; with both, it converges at every tested
  point of the gain map.
- The adjoint self-test for this backend draws its probe vectors from the
  forcing-masked subspace instead of using dense random vectors. Random
  probes excite the boundary-penalty rows (row norms up to 5.5e12), where a
  single-precision factor cannot precondition; they would fail a
  configuration that is healthy on every right-hand side the resolvent map
  actually produces.
- Added the optional dependency group `cudss` (`pip install
  "linstabpy[cudss]"`): `nvmath-python[cu12] >= 1.0, < 2`, which brings the
  cuDSS 0.8 wheels. Requires a CUDA 12 runtime and an NVIDIA GPU supported
  by cuDSS. The core install is unchanged.

### Performance --- measured (dual-socket Xeon Silver 4210R, MKL, one RTX A6000 48 GB)

All numbers from the 450,000-unknown SWBLI operator of the validation
study, on an otherwise idle node, with the solve tolerance at 1e-10.

- cuDSS double: analysis 3.3--3.4 s (threading layer on; 6.7--8.1 s
  without), factorization 5.7--9.0 s, 4.1--5.8e8 nonzeros in L+U, one solve
  26--30 ms, two resident factors (K and its adjoint) 15--20 GB device
  memory.
- Refactorization on the kept analysis costs a full numeric factorization
  (9.0--9.2 s); what a frequency line saves is the analysis. Timings are
  reproducible across independent runs to about 1 percent.
- Line-mode point (two refactorizations + 20 solves): 23 s double, 12--14 s
  single at a benign point --- the production sweep below shows single is
  slower overall. Reference points on the same operator and node: MUMPS CPU line
  mode 60--82 s (4 ranks), MUMPS factorization alone 24--33 s (10 threads),
  STRUMPACK GPU point 130--180 s.
- FGMRES iteration counts: 1--2 (double), 5--20 (single with matching and
  equilibration, at points where it converges; the 47-point sweep below found
  21 points where it exceeds the 400-iteration cap).
- Production campaign (two A6000s, one job per card, two concurrent): the
  47-point beta=40/45 subset runs at 57.1 s/pt as independent jobs (CPU record
  118.5 s/pt) and 34.0 s/pt in line mode (CPU 76.6 s/pt); the full 200-point
  map runs at 57.1 s/pt, 103.9 min wall against 83.6 min for five concurrent
  CPU jobs on twenty cores --- wall-clock is bound by device count, per-point
  cost halves. Run records committed as
  `examples/savarino/savarino_cudss_*.csv`.
- Single precision, measured end-to-end on the same 47 points: converged
  points average 207 s/pt against 58 s/pt double (the weaker factor needs the
  extra iterations), and 21 of 47 points --- the resonant band
  2 <= f <= 7 at beta=45, the response peak at beta=40, and the
  zero-frequency corner --- exceed 400 FGMRES iterations. Evaluated and
  rejected for production; the opt-in stays for the masked-RHS niche and as
  the memory route (halved factor storage).

### Accuracy --- measured

- Solutions through the cuDSS-preconditioned FGMRES match the MUMPS double
  reference to 1.2e-12 relative; the discrete adjoint identity holds to
  2.5e-14--5.9e-12 with the dual factor.
- The full production campaign reproduces the committed map-of-record gains
  to every printed digit: all 200 map points and both 47-point arms (jobs and
  line mode) show zero difference at the CSV's ten-digit precision.
- The single-precision option reaches the same 1e-12 through the double
  iteration. Its scope is the production right-hand-side family (forcing-
  and response-masked vectors); on dense unmasked vectors the iteration can
  stall, which is why the option stays opt-in and the self-test uses masked
  probes.
- For a unit-norm right-hand side the attainable true-residual floor is
  about 1e-5, set by the boundary-penalty rows; MUMPS shows the same floor
  on the same operator. Solution error stays at 1e-12.

### Fixed --- adjoint documentation

- Corrected the explanation of why the GPU path needs a second
  factorization. PETSc's STRUMPACK interface registers no transposed solve,
  and `MatSolveTranspose` then silently substitutes the forward solve (the
  factor's symmetry flag is unknown, which the fallback treats as true).
  The measured "transpose defect" therefore comes from the PETSc interface,
  not from STRUMPACK's arithmetic. Earlier comments and the paper draft
  attributed it to STRUMPACK's transpose solve; code comments and
  documentation now state the fallback. The dual-factor construction was
  and remains the correct route on both GPU backends.

### Fixed --- reproduction documentation

- A reproduction audit of the shipped tree closed four seams between the new
  backend and its documentation: the "Current Status" backend table in
  `docs/COMPUTE_BACKENDS.md` still said cuDSS was not wired (contradicting
  the new section in the same file); the install line named only the `cudss`
  extra when the iterative path also needs the `petsc` extra, and did not say
  that a standard CPU PETSc/SLEPc build is enough (the GPU work happens
  inside cuDSS); the Savarino example README did not mention the two shipped
  cuDSS configs (`savarino_point_cudss.yaml`, `savarino_point_cudss32.yaml`)
  or that the campaign generators drive them via `--config`; and both
  campaign generators' `--config` help text named only the STRUMPACK GPU
  route. All four are corrected.

### Changed --- Savarino base flow fetched, no longer tracked

- The published Savarino base flow (`reference/baseflow/*.dat`, 10.9 MB) is no
  longer tracked: its Zenodo record (10.5281/zenodo.18854407) is open access
  but states no redistribution license, so the repository ships no file from
  it. `fetch_zenodo.py baseflow` downloads it (md5 checked against the Zenodo
  files API) and `savarino_to_npy.py` re-verifies the md5 before converting.
  The shipped `reference/savarino_response_*.npz` (repackaged from the same
  record) remain tracked pending the same license decision.

### Added --- every published number now has its run record and recipe in the tree

A second reproduction audit, this time claim-by-claim against the manuscript,
brought in everything a reader needs to check or rerun the measurements:

- The STRUMPACK GPU campaign records (`savarino_strumpack_farm_fp64.csv`,
  `savarino_strumpack_lines_fp64.csv`) and the dedicated CPU 47-point
  references (`savarino_cpu_farm_47pt.csv`, `savarino_cpu_lines_47pt.csv`) —
  the previously unshipped halves of the backend comparison.
- The backend benchmarks (`examples/savarino/benchmarks/`): the five
  measurement CSVs behind the per-backend factorization times and the
  single-precision verdict, together with the harness scripts that produced
  them and the operator-dump tool they consume.
- `savarino_campaign_walltimes.csv`: wall-clock of every campaign arm from
  Slurm accounting, first job start to last job end, with job-id ranges. This
  corrected one published number: the cuDSS 200-point map wall-clock is
  103.9 min, not 100.2 (the earlier figure was measured from the submission
  loop's completion, while the first jobs were already running).
- `savarino_mpi_scaling.csv`: MPI strong scaling at the Mack point measured
  on the same node as every other row (190.5 / 154.6 / 148.2 s at 4/8/16
  ranks, gain identical to ten significant digits). This replaces an earlier
  scaling table whose hardware could not be traced to any machine or record.
- `compare_backend_modes.py`: prints the Chu-weighted correlation between the
  leading response modes of two runs of the same point, so the "GPU modes
  match the CPU modes" statement is checkable with one command.
- The example README now lists, for every shipped CSV, the exact generator
  command and config that produced it.

### Evaluated and rejected --- measured dead ends, recorded so they are not re-walked

- Block low-rank (BLR) compression as a loose FGMRES preconditioner, the
  standard advice for cheaper factors: STRUMPACK GPU BLR at tolerances 1e-2
  and 1e-4 produces factors whose plain solve residuals are 1e16--1e31 on
  this operator (the CPU BLR arm reached 4e39) and stalls FGMRES outright,
  while factoring 9--21x slower than the exact GPU factorization (469--1118
  s against 41--55 s). MUMPS BLR at 1e-3 stalls the iteration; at 1e-5 it
  converges in 36--39 iterations and the extra solves cancel the 20 percent
  factor saving. Shock-dominated operators with 11 decades of row-norm
  spread do not compress.
- cuDSS explicit batching (factoring K and its adjoint, or several
  frequencies, in one call): requires device-resident operands and is moot
  at 1-second factors.
- Hybrid CPU+GPU execution and pivot-type overrides: no measurable fit for
  this operator size.
- A single-precision factorization without matching: fails (stalls) near
  zero frequency even with equilibration --- the reason
  `factor_precision: single` force-enables matching.

Martini compute scripts wrote NPZ and a log only. Plot scripts loaded those
NPZ files and wrote the PNG figures. A conda file installed the CPU stack
from `pyproject.toml`.

PETSc-free omit-list coverage rose from 75% (4969/6618) to 83% (5467/6618).

### Added
- `environment.yml` installed the package from `pyproject.toml` on a conda-forge complex PETSc stack.

### Changed
- PETSc-only tests skipped collection when mpi4py or petsc4py was missing.
- Auto PETSc matrix layout stayed local when petsc4py was absent.
- `driver.run` re-exported `RUNTIME_NPZ_CONTRACT_KEYS` from the payload module.
- Dissipation tests dropped PETSc stubs after import. Later tests saw a missing stack.
- README installation named the CPU stack. It stated that no GPU conda file ships.
- Martini `example_*.py` drivers wrote NPZ and a log only.
- `example_*_plots.py` loaded NPZ files and wrote PNG figures.
- `run_all_examples.sh` computed each Martini case, then ran the plot module.
- `benchmark.sh` ran Example 0 as `examples.martini.example_0_curvilinear_validation`.
- README, tutorial, and runner help used `examples.martini.*` module names.

### Removed
- `.pre-commit-config.yaml`. The file-size guard stayed in pytest and CI.
- Unused `tests/fixtures/curvilinear_derivative_parity.json`.
- Campaign leftover `examples/martini/physics_audit_campaign_vs_repo.json`.
- Example 0 write to `artifacts/curvilinear_derivative_parity.json`.

### Verified
- Tests locked `environment.yml` against `pyproject.toml`.
- Tests locked the Martini compute/plot split and the leftover-file ban.
- PETSc-free omit-list coverage rose from 75% (4969/6618) to 83% (5467/6618).
- Full-package coverage without PETSc was 50% (5467/11008).
- `.coveragerc` left out 4390 solver, linalg, and PETSc-helper statements at 0%.
- 31 omit-list files reached 100%. 22 omitted files stayed at 0%.
- The PETSc-free suite passed 1125 tests. It skipped 113 tests. 29 MATLAB-parity tests stayed deselected.
- `ruff check` and `ruff format --check` passed on every Python file.
- Tests covered driver CLI, sponge, shock cleanup, Floquet shift, and wavemaker inputs.
- Tests covered grid-field load, filters, conservative-momentum forcing, and typed solver config.
- Tests covered mesh create limits, SBP dissipation errors, and empty Dirichlet recipes.

Omit-list files (`.coveragerc` source, solvers and linalg omitted). Hit 5467, miss 1151, stmts 6618.

| File | Hit | Miss | Stmts | Cover |
|---|---:|---:|---:|---:|
| `resolvent/gain.py` | 19 | 100 | 119 | 16% |
| `operators/spatial.py` | 39 | 101 | 140 | 28% |
| `stability/_eigenvalues.py` | 102 | 232 | 334 | 31% |
| `resolvent/operator.py` | 18 | 33 | 51 | 35% |
| `resolvent/analysis.py` | 47 | 78 | 125 | 38% |
| `driver/run.py` | 447 | 213 | 660 | 68% |
| `utils/provenance.py` | 78 | 26 | 104 | 75% |
| `operators/assembly.py` | 519 | 165 | 684 | 76% |
| `boundary_conditions/recipe.py` | 172 | 44 | 216 | 80% |
| `boundary_conditions/dirichlet.py` | 150 | 32 | 182 | 82% |
| `utils/cost_metrics.py` | 42 | 8 | 50 | 84% |
| `boundary_conditions/neumann.py` | 127 | 24 | 151 | 84% |
| `mesh/sbp_generator.py` | 147 | 23 | 170 | 86% |
| `operators/metrics.py` | 178 | 15 | 193 | 92% |
| `mesh/mesh.py` | 615 | 47 | 662 | 93% |
| `driver/io.py` | 55 | 2 | 57 | 96% |
| `config/typed.py` | 296 | 4 | 300 | 99% |
| `baseflow/sponge.py` | 205 | 2 | 207 | 99% |
| `stability/sensitivity.py` | 104 | 1 | 105 | 99% |
| `baseflow/preprocess.py` | 139 | 1 | 140 | 99% |
| `__init__.py` | 28 | 0 | 28 | 100% |
| `baseflow/__init__.py` | 5 | 0 | 5 | 100% |
| `baseflow/baseflow.py` | 21 | 0 | 21 | 100% |
| `baseflow/grid_fields.py` | 63 | 0 | 63 | 100% |
| `baseflow/shock_cleanup.py` | 215 | 0 | 215 | 100% |
| `baseflow/viscosity.py` | 23 | 0 | 23 | 100% |
| `boundary_conditions/__init__.py` | 11 | 0 | 11 | 100% |
| `config/__init__.py` | 4 | 0 | 4 | 100% |
| `config/defaults.py` | 3 | 0 | 3 | 100% |
| `config/schema.py` | 53 | 0 | 53 | 100% |
| `driver/__init__.py` | 1 | 0 | 1 | 100% |
| `driver/cli.py` | 63 | 0 | 63 | 100% |
| `driver/payload.py` | 12 | 0 | 12 | 100% |
| `driver/preflight.py` | 165 | 0 | 165 | 100% |
| `mesh/__init__.py` | 11 | 0 | 11 | 100% |
| `mesh/fdweights.py` | 25 | 0 | 25 | 100% |
| `mesh/filters.py` | 189 | 0 | 189 | 100% |
| `mesh/sbp.py` | 211 | 0 | 211 | 100% |
| `metrics_schema.py` | 38 | 0 | 38 | 100% |
| `mumps_options.py` | 3 | 0 | 3 | 100% |
| `operators/__init__.py` | 4 | 0 | 4 | 100% |
| `operators/coefficients.py` | 453 | 0 | 453 | 100% |
| `operators/energy_norm.py` | 12 | 0 | 12 | 100% |
| `resolvent/__init__.py` | 12 | 0 | 12 | 100% |
| `resolvent/conservative_forcing.py` | 99 | 0 | 99 | 100% |
| `stability/__init__.py` | 20 | 0 | 20 | 100% |
| `stability/acceptance.py` | 93 | 0 | 93 | 100% |
| `utils/__init__.py` | 2 | 0 | 2 | 100% |
| `utils/_solver_cfg.py` | 67 | 0 | 67 | 100% |
| `utils/floquet.py` | 32 | 0 | 32 | 100% |
| `utils/yaml_io.py` | 30 | 0 | 30 | 100% |
| **Total** | **5467** | **1151** | **6618** | **83%** |

Files omitted by `.coveragerc`. Statement count from the coverage parser. Hit 0 without PETSc.

| File | Hit | Miss | Stmts | Cover |
|---|---:|---:|---:|---:|
| `solvers/router/stable_sweep.py` | 0 | 675 | 675 | 0% |
| `solvers/router/resolvent_map.py` | 0 | 573 | 573 | 0% |
| `linalg/factorize.py` | 0 | 452 | 452 | 0% |
| `utils/petsc_core.py` | 0 | 387 | 387 | 0% |
| `solvers/eigenvalue.py` | 0 | 320 | 320 | 0% |
| `solvers/router/map_walk.py` | 0 | 265 | 265 | 0% |
| `linalg/backends.py` | 0 | 262 | 262 | 0% |
| `solvers/router/ram_autoselect.py` | 0 | 252 | 252 | 0% |
| `solvers/router/anchors.py` | 0 | 204 | 204 | 0% |
| `solvers/router/recycling.py` | 0 | 199 | 199 | 0% |
| `solvers/router/certification.py` | 0 | 172 | 172 | 0% |
| `solvers/router/break_even.py` | 0 | 159 | 159 | 0% |
| `solvers/spatial_eig.py` | 0 | 102 | 102 | 0% |
| `solvers/router/band_marking.py` | 0 | 96 | 96 | 0% |
| `solvers/router/map_record.py` | 0 | 78 | 78 | 0% |
| `utils/petsc_init.py` | 0 | 66 | 66 | 0% |
| `solvers/_eps_helpers.py` | 0 | 63 | 63 | 0% |
| `linalg/shifted.py` | 0 | 21 | 21 | 0% |
| `solvers/router/__init__.py` | 0 | 19 | 19 | 0% |
| `linalg/__init__.py` | 0 | 13 | 13 | 0% |
| `solvers/__init__.py` | 0 | 11 | 11 | 0% |
| `solvers/petsc_utils.py` | 0 | 1 | 1 | 0% |
| **Total** | **0** | **4390** | **4390** | **0%** |

## [1.0.0rc7] - 2026-08-19

Comments in the example scripts and tests were shortened. Ruff formatted
the test tree. The Martini audit-plan notes were removed.

### Changed
- Example and test comments.

### Removed
- Martini audit-plan and parity-summary notes.

## [1.0.0rc6] - 2026-08-19

The supplement examples ship complete: Martini 0-5, Song 32 deg, and
Savarino, with their parity audits, slim reference fixtures, and the
paper-style figures. Two operator corrections came out of that audit.
On a deformed y-symmetric mesh the metric terms now come from the full
eta operators, so the family second derivatives converge at the axis.
The Floquet shift defaults to the continuum chain rule; the LinStab2D
discrete form stays behind `matlab_parity`.

### Added
- Each Savarino map-walk row records its commit, its Slurm job, and its machine.
- `resolvent_analysis` returns the factor, SVD, and residual stage clocks.
- The Savarino line campaign times a recycled sweep against a factor at each frequency.
- **Song (2023) example complete** (`examples/song/`, θ = 32° only): self-contained
  supplement pipeline from Song FV export through LinStabPy validation figures.
  - `prepare_baseflow.py` — Plot3D → nodal `.npy` (601→599 trim, interpolation, nondim).
  - `plot_baseflow.py` — mesh + mean-flow panels (`examples.figure_style` LaTeX fonts).
  - `configs/run_fd4_n30.yaml` — FD4, *n* = 30, β = 19.2, no operator dissipation (ε = 0).
  - `song.py` — EVP driver; default config → `configs/run_fd4_n30.yaml`.
  - `report_matched20.py`, `plot_spectrum_matched20.py`, `plot_eigenmode_lead.py` —
    Hungarian matched-20 spectrum and lead eigenmode vs Song reference.
  - `song_plot_utils.py`, `reference_paths.py`; shipped reference under `reference/`.
  - Shipped validation figures under `figures/` (baseflow mesh/fields, matched-20
    spectrum, lead eigenmode; FD4 *n* = 30, ε = 0).
  - Local-only outputs: `inputs/baseflow/`, `output/fd4_n30/` (gitignored).
  - Validated (FD4 *n* = 30, ε = 0): 20/20 Song modes within |Δω| ≤ 0.01; lead |Δω| ≈ 0.002.
- **Martini (2024) examples 0–5 — presentation + parity closure** (`examples/martini/`):
  audit **CLOSED 2026-08-18** (`AUDIT_PLAN.md`; steps A→E, user sign-off all six cases).
  - Per-case plotters: `example_{0..5}_plots.py` plus shared
    `plot_compare_utils.py`, `split_martini_fixtures.py`,
    `run_all_martini_examples_and_plots.py` — thesis-style mosaics (LinStab2D |
    linstabpy | peak-normalized % error), spectra, and baseflow panels.
  - Slim MATLAB reference fixtures under `examples/martini/LinStab2DResults/`
    (compact u+v / u+v+w, **complex128**); bundled `linstab2D_scripts/` and
    `regenerate_martini_fixtures.sh` for MATLAB → NPZ regen.
  - Bundled baseflow `.mat` files under `examples/martini/baseflows/`
    (`BL_BF.mat`, `Step_BF.mat`, `M09jet.mat`, `Paraboloid_BF.mat`).
  - Shipped presentation PNGs next to each driver (baseflow, spectrum, mode
    mosaics for Ex. 1–5; validation triptych for Ex. 0).
- **Savarino (2026) example — reorganized supplement reproduction** (`examples/savarino/`,
  θ = 30.8° SWBLI resolvent validation, Figs. S4–S6):
  - **Layout:** `configs/` (YAMLs + grids), `reference/` (author baseflow + packed
    response modes), `output/` (run records, figures, local intermediates).
  - **Configs** (`configs/`): `map_base.yaml`, `mack_f2_b45.yaml`,
    `liftup_f1em4_b163.yaml`, `point.sbatch.template`, `savarino_map317_points.csv`
    (317-pt β–f grid for dense-map campaigns).
  - **Shipped reference:** Tecplot base flow under `reference/baseflow/`; packed
    `reference/savarino_response_{mack,liftup}.npz` (primitive ρ′, u′, v′, w′, T′).
  - **Shipped results** (`output/`): `savarino_map2_200pt.csv` (200-pt gain-map run
    record), `savarino_map317pt.csv` (317-pt minimal dense subset from PhD
    `map_minimal_keep_author200`), validation figures under `output/figures/`
    (mode comparisons + 200-/317-pt gain maps).
  - `collect_campaign.py --from-npz` — aggregate NPZ + `.metrics.json` trees into
    the run-record CSV layout (for PhD dense-map campaigns without Slurm logs).
  - `gen_campaign.py --grid {author200,dense317}` — Slurm/local campaign generator
    for 200- or 317-point grids.
  - Plotters aligned with PhD Savarino2026 style:
    - `plot_gain_map.py` — tensor/PCHIP interpolation, σ₁² colorbar with ×10⁴
      on top, Mack / lift-up peak markers (`plot_resolvent_map_savarino_style.py`
      parity); optional `--show-points` / `--baseline-summary` for diagnostics.
    - `compare_modes.py` — Song eigenmode layout, peak-normalised ±1 colourbar,
      ymax 0.06 (Mack) / 0.1 (lift-up) (`plot_savarino_modes_linstabpy_template.py`
      parity).

### Changed
- Recycled GMRES restarts. The restart is a Krylov length. It is never the system size.
- `certified_rsvd` logs every check when one check fails. One seed skips the multi-seed check.
- **Floquet shift default is now `continuum`** (`LINSTABPY_FLOQUET_SHIFT`,
  `numerics.floquet_shift`): `DZ += iα I`, `D2RZ += iα DR`, `D2Z += 2iα DZ − α² I`,
  the chain rule of `(∂z + iα)`. The LinStab2D algebra (`D2RZ += iα I` and `D2Z`
  built from the already-shifted `DZ`, i.e. `−3α² I`) stays available as
  `matlab_parity`; the Martini Ex. 3 YAMLs and `examples/sawtooth` pin it.
  Plane Poiseuille at Re = 10⁴, α = 1 (Orszag 1971): `continuum` converges to
  ω = 0.23752649 + 0.00373967i (|Δω| 3.8e-4 → 1.8e-6 for ny 128 → 512);
  `matlab_parity` stalls at |Δω| ≈ 3.2e-4, a growth rate 8.5 % low, at every
  resolution. Any unpinned Floquet run changes with this default.
- Replaced the legacy Song figure/pack/matrix scripts with the pipeline above;
  see `examples/song/README.md` for run order.
- `song.py` creates the output NPZ parent directory before save and reads
  `numerics.operator_dissipation` from YAML (disabled, ε = 0 for this test).
- `song_plot_utils.load_song_leading_mode_npz` transposes the shipped reference
  eigenmode NPZ to LinStabPy `(ny, nx)` layout for plotting.
- `.gitignore` re-ignores regenerated example outputs (`examples/**/figures/`,
  `examples/**/output/`, Song `inputs/baseflow/`, logs) after the example PNG
  whitelist; Song `figures/` and reference NPZ are shipped; raw Plot3D exports
  under `inputs/raw/` stay local.
- Song plot scripts require LaTeX via `figure_style.check_plot_environment`;
  see `examples/song/README.md` for system packages (`texlive-latex-extra`, …).
- `examples.figure_style`: `check_plot_environment(require_latex=…)` with install
  hint, optional `dpi` / `pad_inches` on `save()`; tests in `tests/test_figure_style.py`.
- Grouped Song (2023) **reference** assets under `examples/song/reference/`:
  `baseflow/song_32deg.npz`, `eigenmode/song_32deg_leading_mode.npz`,
  `spectrum/eigenvalues.dat`, `spectrum/omega_i.csv`. Added `reference_paths.py`
  and updated configs, figure scripts, and tests.
- **Martini fixture layout and git policy:**
  - Shipped NPZ: **`examples/martini/LinStab2DResults/*.npz` only** (slim MATLAB
    reference). Root-level `example_*.npz` from linstabpy drivers are
    **gitignored** (regenerated locally; several exceed the 60 MiB tracked-file
    guard).
  - Raw MATLAB `.mat` exports stay outside Git in `martini_examples/` (gitignored
    scratch; recreated by `regenerate_martini_fixtures.sh`).
  - Run logs under `examples/martini/_run_logs/` are gitignored.
  - Renamed `examples/martini/data/` → `baseflows/`.
- **Martini plot/compare layer** (repo mosaics ≤ **0.49%** peak-norm vs campaign;
  Ex. 4 cavity audit closed — residual tail is MATLAB `eigs` convergence, not
  operator/BC):
  - `plot_compare_utils.py`: `used_ind` 0-based gridding, `matlab_plot_grid_shape`,
    `compact_resolvent_modes_uvw`, peak-normalized mosaic triptychs.
  - `example_4_plots.py` / `split_martini_fixtures.py`: slim MATLAB u+v+w compare
    with joint complex scaling; Example 4 YAML pins `Pr: 0.7` and
    `zero_dmudT: true` for LinStab2D parity.
  - `compare_martini_examples.py` and `tests/test_martini_parity_compare.py`
    updated for `LinStab2DResults/` fixture paths and compact mode layout.
- Moved library validation CLIs from `examples/martini/` to `tests/tools/`
  (`assembly_parity.py`, `assembly_metrics_diff.py`, `lnse_equivalence_harness.py`,
  `matlab_source_parity.py`, `symbolic_compressible_nse.py`,
  `coefficients_reference_base.py`, `NSE_DERIVATION.md`). See
  `tests/tools/README.md` for rationale and owner review checklist.
- Removed unused maintainer-only scripts from `examples/martini/`:
  `check_parity_matrices.py`, `compare_operator_construction.py`,
  `perf_memory_operator_parity.py`, `operator_matrix.json`, `branch_matrix.json`,
  `martini_c64_max_rel.txt`, `martini_examples.sbatch`.
- **Savarino example layout and git policy:**
  - Moved point YAMLs, campaign template, and 317-pt grid into `configs/`; run
    records into `output/` (`savarino_map2_200pt.csv` was at example root).
  - `reference_paths.py`: `GAIN_MAP_CSV`, `GAIN_MAP_317_CSV`, figure paths under
    `output/figures/`; compare outputs under `output/compare/`.
  - `.gitignore` ships **only** curated Savarino outputs: both map CSVs and
    `output/figures/*.png`. **Gitignored:** `output/baseflow/*.npy` (from
    `savarino_to_npy.py`), `output/compare/**/*.npz` (from `run_compare_points.py`),
    `reference/zenodo/`. Whitelist repeated after root `output/` rule.
  - `run_compare_points.py`, `gen_campaign.py`, `collect_campaign.py`, `mode_oracle.py`
    updated for `configs/` paths.
  - See `examples/savarino/README.md` for minimal workflow and maintainer scripts
    (`verify_norm_convention.py`, `mode_oracle.py`, `fetch_zenodo.py`,
    `pack_reference_modes.py`).

### Fixed
- `Mesh.deform` on a y-symmetric mesh with a non-separable map: the `dJ`/`Q`
  metric terms of the symm/asymm family operators are now differentiated with
  the full `Dy`/`D2y` (the coordinate map has no η parity), as LinStab2D
  `DeformMesh.m` does. Before, the family stencils were applied to the
  coordinates themselves and `D2y_asymm`/`D2y_symm` did not converge at the axis
  rows (relative error growing like 1/h under refinement: 7.7 → 61 for
  n = 16 → 128 on a polynomial map; 3.5 → 29 on the Z² map). After: 2.2e-4 → 3.9e-7
  (order 3, asymm) and 2.1e-3 → 3.1e-5 (order 2, symm). Only `y_symmetry=True`
  meshes on the full-2D deform path are affected (Martini Ex. 5 paraboloid;
  the SWBLI cases are untouched). Regression: `test_deform_full_2d_family_metric_terms_use_full_operators`.
- The walk probe uses the last response, not a ones vector.
- A walk GMRES has a four-iteration cap. A dead walk raises and factors this omega.
- `append_row` refuses a row that does not match the header.

### Removed
- Dropped `song_fig{1,2,3}_*.py`, `pack_run.py`, `run_song_2023_*.py`, `song.sh`,
  `song_32deg_spectrum_quoted.csv`, and the old `song_32deg*.yaml` / sbatch configs
  from `examples/song/`.
- Dropped committed LinStabPy **outputs** from the old flat `examples/song/` layout
  (`song_32deg_modes.npz`, `song_32deg_spectrum.csv`, `song_matlab_timing.json`,
  root-level `song_fig*.png`). Large EVP dumps and raw Plot3D inputs stay local;
  curated reference NPZ and validation PNGs now live under `reference/` and
  `figures/` respectively.
- Dropped the 31.6 deg Song example from `examples/song/`: baseflow
  (`song_316deg.npz`, `song_316deg_leading_mode.npz`), reference spectrum
  (`song_316_eigenvalues.dat`), configs (`song_316deg.yaml`, `song_316deg.sbatch`),
  and comparison helpers (`compare_evp_to_song316.py`, `parse_song316_eigenvalues.py`).
  The shipped example is **32 deg only**; 31.6 deg validation stays in the PhD case tree.
- Dropped committed **linstabpy driver NPZ** from `examples/martini/` root
  (`example_*.npz`, legacy `*_values.npz` / `*_modes.npz` / `*_logs.npz` splits).
  Regenerate locally via `./run_all_examples.sh` or per-case drivers; parity
  against MATLAB uses `LinStab2DResults/` only.
- Dropped old debug/support PNGs from `examples/martini/` (per-case
  `*_support_components*.png`, legacy single-panel `example_*_{blasius,jet,...}.png`
  layouts superseded by mosaic figures).
- Dropped `savarino_corner_probe_uv.yaml` and `savarino_corner_probe_wonly.yaml`
  (side experiments; not part of the S4–S6 reproduction path).
- Dropped example-root Savarino YAMLs and Slurm template (now under `configs/`):
  `savarino_point_f2_b45.yaml`, `savarino_point.sbatch.template`; renamed
  `compare_liftup_f1em4_b163.yaml` → `liftup_f1em4_b163.yaml`.
- Dropped committed PNGs from `examples/savarino/reference/` (figures now under
  `output/figures/`).


## [1.0.0rc5] - 2026-08-15

The gain map walks each frequency line. The first two frequencies
pay a MUMPS factor. Later frequencies walk only when a probe beats
that measured cost. Otherwise the line factors again.

### Added
- `LineWalker` walks one beta line. The first two frequencies factor with MUMPS.
- A cold forward and adjoint probe decides whether a walk can beat a new factor.
- `map_record` writes each finished point and resumes unfinished frequencies.

### Changed
- `resolvent_map` walks each beta line with `LineWalker` and SLEPc.
- A walk needs a measured apply count and a cheap probe. Otherwise the line factors.
- GMRES stops when a deadline has passed.
- The map store writes a temp file and then replaces the old store.

### Fixed
- The walk reads GMRES lengths from the walk module, not from the recycle module.

## [1.0.0rc4] - 2026-08-13

The Example 2 jet resolvent matched the MATLAB LinStab2D gains. The axis
coefficient used the true radius. The spatial filter sat inside the
forcing and response masks.

A forcing region composed with the boundary-condition mask.
`input_region.space: conservative_momentum` mapped conservative momentum
forcing onto the primitive residual.

Case files became YAML. One loader read a packed field. The tree dropped
JSONC, import shims, `scripts/`, and the unused Dockerfile.

### Fixed
- `compute_cylindrical_coefficients` no longer floored R at the smallest
  positive spacing. A grid that contains r = 0 raised ValueError.
- `ResolventOperator` applied the spatial filter inside the input and output
  masks, matching MATLAB `resolvent.m`.
- The Example 2 leading gain moved from 411.72 to the MATLAB 413.72 reference.
- A forcing or response region composed with the boundary-condition mask.
- `mask_bc_forcing` stayed true by default. A region no longer forced
  constraint rows.
- A region that selected only constrained rows raised.
- `mask_bc_forcing: false` still forced those rows on purpose.
- The Floquet shift also updated D2RZ. The default mode matched LinStab2D.
- The spatial EVP finite-differenced L in kx, not in the Floquet exponent.
- Example 3 applied spatial Dirichlet after Lc + omega Lw.

### Added
- `input_region.space: conservative_momentum` mapped conservative momentum
  forcing onto the primitive residual with a mass-weighted L2 norm.
- Momentum forcing induced a consistent temperature forcing
  `T' = -kappa(kappa-1) Ma^2 (ubar · u')`.
- `load_config` loaded YAML case files.
- `load_case_fields` read a packed npy directory or one npz.
- `examples/song` and `examples/savarino` shipped the reproduction chains.
- `examples/sawtooth` added the Lui 2022 field on a packed stride-2 300 x 340 mesh.
- Four paper Example 3 YAML files reused `example_3_streaky_bl.py`.
- `examples/martini/matlab_baseflow.py` held the Martini MATLAB field constructors.
- `build_runtime_payload` grouped the runtime NPZ keys before `save_results`.
- `install.sh --diagnose` printed the host preflight and installed nothing.
- `tests/tracked_file_size.py` and `tests/public_provenance.py` replaced `scripts/`.

### Changed
- Example figures probed TeX by writing a PNG. They used usetex when that worked.
- A failed usetex save retried with Computer Modern mathtext.
- A missing TeX stack no longer crashed the figure.
- Case files under `examples/` became YAML. The loader rejected a leftover `.jsonc` suffix.
- Shipped YAML files now comment each key. `validation_cases.yaml` is YAML.
- Comments and the tutorial now say YAML, not JSONC.
- Sponge presets dropped the `case1_` prefix.
- Moved the Martini, Song, and Savarino cases to `examples/martini`,
  `examples/song`, and `examples/savarino`.
- Packed the Song reference fields as float64 npz. The 32 deg field uses
  velocity / c_inf and temperature / T_inf.
- Packed Martini mode arrays as complex64. The largest file is
  `example_3_Streaky_BL_results_temporal_modes.npz`.
- Raised the tracked-file size limit from 50 MiB to 60 MiB for that file.

### Removed
- `load_jsonc`, `load_hamada_case`, `results_runtime`, and the solver import shims.
- The `run/` campaign kit and the broken Docker PETSc CI job.
- The `mumps_icntl_7` boolean alias. Use `mumps_ordering: metis`.
- The unused `Dockerfile` that did not configure PETSc.
- The `scripts/` folder and `diagnose.sh`.

## [1.0.0rc3] - 2026-08-10

### Changed
- **BREAKING (default):** `numerics.operator_dissipation.apply_to_resolvent` now
  defaults to `true` instead of `false`, matching `apply_to_stability`. Previously,
  enabling dissipation gave a damped eigenvalue problem and a **bare** resolvent
  operator — a combination that runs to completion and returns grid-scale sawtooth
  (spurious high-wavenumber streaks, gains orders of magnitude too large) with nothing
  in the log to indicate it. Configs that deliberately want an undamped resolvent must
  now set `apply_to_resolvent: false` explicitly. No config shipped with the project
  changes behaviour: every one that runs a resolvent either disables dissipation or sets
  the switch explicitly. Both defaults now live in one place,
  `config.defaults.DISSIPATION_APPLY_DEFAULTS`, so the driver and the preflight checks
  cannot disagree about them.
- `solver.factor_backend` is now honoured as a fallback for all solver stages
  (`linsolve`, `resolvent`, and the shift-invert eigen path). It was previously read by
  nothing: every resolver receives a *sub*-config, so a key at solver level silently did
  nothing — one symptom being a run intended for GPU that executed entirely on CPU. It
  sits at the bottom of the precedence chain (per-stage key > environment >
  `solver.factor_backend` > built-in default), so no configuration that resolved before
  resolves differently now. Route-specific defaults (for example `strumpack_fgmres`
  requiring `strumpack`) are unaffected.

### Added
- Preflight and post-run advisories (`driver.preflight`), reported in the run log. Each
  advisory names a concrete remedy rather than only a symptom. Preflight covers:
  dissipation enabled but not reaching a stage that is about to run; no dissipation at
  all; dissipation enabled with `epsilon = 0`; a spanwise wavenumber inside the band
  where grid-scale sawtooth dominates while undamped; a base flow that is not normalised
  to the canonical freestream (`rho = u = T = 1`); a shadowed `factor_backend`; and a
  strided (reduced-resolution) mesh. Post-run checks flag a leading gain far above the
  physical scale for these cases and a postcheck residual above tolerance.
- A run **banner** and closing **summary** in the log, reporting *effective* values —
  including the per-stage dissipation epsilon actually applied — so a run's real
  configuration and the trustworthiness of its result are visible without loading the
  output archive.
- Translation of opaque factorization failures into actionable messages. A MUMPS
  workspace exhaustion (`INFOG(1) = -9`) previously surfaced only as
  `SLEPc EPS.setUp failed`; it now names the cause and the remedy (`ICNTL(14)`).

- A **band-marking backend** advisory. `resolvent_map` runs a mandatory band-marking
  eigenvalue problem, and that eigenproblem reads `solver.eigen` — not
  `solver.resolvent`. A configuration that moved only the resolvent onto a scalable
  backend therefore left the eigenproblem on the default direct solver, where at full
  resolution it does not fail but simply never finishes; the run looks like a slow
  resolvent sweep for hours. The advisory names the three ways out (`solver.factor_backend`,
  an explicit `solver.eigen.factor_backend`, or `analysis.skip_band_marking`).
- A source-wide lint (`tests/test_mpi_collective_lint.py`) rejecting MPI collectives
  inside rank-0-only regions, in both the `if rank == 0:` and `if rank != 0: return`
  forms. Such a call deadlocks every multi-rank run while remaining invisible to the
  serial test suite, which runs at size 1. The lint ships with positive controls, since
  a lint never shown to fire cannot be distinguished from one that cannot fire.

### Fixed
- The single-point resolvent path now honours `input_region` and `output_region`, and no
  longer forces the response mask to equal the forcing mask. `driver.run._stage_run_resolvent`
  built its own boundary-condition-only diagonal and then set `C_mat = B_mat.copy()`, so two
  things went wrong at once and neither said anything: a config naming a forcing region ran
  with no region at all, because only `solvers.router.resolvent_map` ever read those keys; and
  even a correct region was copied onto the response, which turns an asymmetric problem
  (restricted forcing, free response) into a different one. Both paths now call the same
  `_forcing_response_mats` helper, so one config key means one thing everywhere, and `C`
  defaults to the boundary-condition mask rather than to `B`. Runs that set no region are
  unchanged: `B` and `C` are still the boundary-condition mask, asserted byte-for-byte by a
  regression test. `scripts/validation/song_2023/song_2023.py` builds its masks separately and
  cannot honour a region, so it now raises instead of ignoring the key.
- MPI is now initialised at `MPI_THREAD_MULTIPLE` on every import path, restoring the
  STRUMPACK factorization backend. Whichever library calls `MPI_Init` first fixes the
  thread level for the process, and it can only be *requested* at init — never raised
  afterwards. `mpi4py` requests `MPI_THREAD_MULTIPLE`, PETSc only
  `MPI_THREAD_FUNNELED`. `utils.petsc_init.init_petsc_from_argv` already imported
  `mpi4py` first, but seven modules imported `petsc4py`/`slepc4py` at *module scope*
  and bypassed it, so whichever of them was imported first decided the level. When
  PETSc won, `MatGetFactor_aij_strumpack()` failed with PETSc error 57 and `SLATE
  requires MPI_THREAD_MULTIPLE`, taking the whole STRUMPACK/GPU route out of service —
  8 of 14 failures in the GPU test venv had this single cause, and none of them
  reported anything about MPI. The guard is a plain `import mpi4py.MPI`, chosen over a
  first-party helper module because isort orders third-party imports above first-party
  ones: a `linstabpy.*` guard could not stay above `petsc4py`, and `ruff --fix` would
  have silently reordered the fix away. `mpi4py` sorts above `petsc4py` on its own.
  `tests/test_mpi_thread_level.py` enforces the ordering as a source-wide AST lint plus
  a runtime thread-level assertion that skips loudly without petsc4py.
- The freestream-normalisation check and the MUMPS error translation were defined but
  never called: both shipped with full unit-test coverage and zero call sites, so
  neither ran in production. They are now invoked from the driver and the eigensolver
  respectively, and a test asserts that every advisory producer is reachable from the
  runtime — unit tests establish that a guard works, not that anything calls it.
- ASM/BJacobi sub-preconditioner settings on the `inexact` shift-invert path are now
  actually applied. They were configured before `eps.setUp()`, where PETSc cannot yet
  materialise sub-solvers (`ierr=58`), and the error was swallowed — so `sub_pc_type`
  and `pc_factor_levels` were discarded. PETSc's default sub-PC is also `ilu`, which
  masked the type being dropped, but the fill level was not masked: a config asking for
  ILU(2) ran with ILU(0). The settings are now applied immediately after `eps.setUp()`,
  failures raise instead of downgrading, and the realized type and fill level are logged
  rather than assumed. This mirrors what `linalg.factorize` already did for the linsolve
  path. Only `st_solver: "inexact"` reached the defect — no shipped configuration selects
  it — so no previously reported result changes.
- MATLAB-parity tests are **opt-in**. This repository ships no MATLAB reference data,
  so the seven tests that compare against it called `pytest.skip` when it was absent —
  which is every machine but the two the data lives on. A run reporting
  "1361 passed, 13 skipped" read as near-total coverage without disclosing that seven
  of those skips could never be satisfied by anything in the repository. They now carry
  a `matlab_parity` marker, deselected by default, and are requested explicitly with
  `LINSTABPY_MATLAB_PARITY=1 pytest -m matlab_parity`. When requested, a missing
  reference is a **failure**, not a skip: a comparison you asked for that quietly does
  not run is indistinguishable from one that passed. A default run now reports
  6 skips, every one an absent optional backend — a true statement about the machine.
- Subprocess timeouts in the test suite are now load-aware everywhere, via a single
  shared helper. One test file already scaled its timeout with load average; its two
  siblings used hard limits of 120 s and 60 s, leaving as little as 1.4x headroom over
  the test's own runtime. On a shared or busy machine those expired and surfaced as
  test *failures*, indistinguishable from a numerical disagreement — a contributor
  running the suite with a browser open would see phantom errors. These timeouts guard
  against a hang; they are not performance assertions, so the bases are deliberately
  generous. The helper documents a limitation found in practice: load average counts
  runnable processes and does not detect memory-bandwidth starvation, so a co-resident
  memory-bound job slowed a test run ~3x while the computed multiplier was 1.17x.
- `production_case_template.jsonc`, the file the documentation tells a new user to copy
  for a production run, shipped with no `operator_dissipation` block at all. A first
  production run made from it was sawtooth-contaminated with nothing in the log to say
  so. It now enables dissipation and documents that `epsilon` must be re-tuned at the
  target resolution rather than inherited.
- The Streaky-BL MATLAB parity test guarded only one of the two references it needs, so
  restoring the parity bundle while the base-flow data directory was still absent turned
  a clean skip into a hard error that resembled a numerical disagreement. The guard now
  checks every reference and announces which one is missing.
- PETSc is now imported lazily by `linalg.backends`, `linalg.factorize`,
  `linalg.shifted`, and `solvers.router.stable_sweep`, matching the existing
  `utils.petsc_core` pattern. Importing these modules — or the
  `solvers.router` package, which re-exports `resolvent_map` — no longer pulls in
  petsc4py or initializes PETSc/SLEPc; initialization now happens on first
  actual solver use. This makes lightweight helpers such as
  `solvers.router.band_marking` usable without starting PETSc. Public APIs and
  numerical behavior are unchanged.

### Fixed
- Restored the guarantee that importing `solvers.router.band_marking` does not
  initialize PETSc (regressed when `resolvent_map` was added to the
  `solvers.router` package exports).
- The CPU/MUMPS parity regression guard no longer fails on freshly built
  environments. Its tolerance was tighter (1e-12) than both the eigensolver's own
  convergence tolerance (`eps_tol=1e-10`) and the measured build-to-build spread
  of the converged eigenvalues (~1e-8 between independently configured
  PETSc/MUMPS/BLAS builds), so it could not be stably satisfied. The guard now
  uses 1e-6 — far above that floor, far below any physically meaningful drift —
  and the frozen reference was regenerated. Repeat runs within one build remain
  bit-identical.

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
  correctness blocker. CPU/MUMPS path unchanged (still single-factor).
- `wire factor_backend` into iterative LU preconditioner: the shift-invert solver now passes
  the selected `factor_backend` (e.g. STRUMPACK) through to the PETSc LU preconditioner
  (`KSP` path), instead of always falling back to MUMPS, so GPU-accelerated FGMRES
  preconditioners route to the GPU backend. Validated on c9 A6000 with complex resolvent:
  residuals match the direct path to ~2e-14.

### Changed
- Case 1 Song base flow is now genericised in the config comment: `song_box_evp.jsonc`
  reads "Song SWBLI box" (removed author name).

## [1.0.0rc1] - 2026-06-18

Physics-campaign freeze release candidate: the turnkey `resolvent_map(base_flow,
beta_range, f_range, cfg)` entry point (base-flow in, (β,f) gain map + cost metrics out)
with region masks on the CPU/MUMPS-validated frozen-numerics path. Seven Appendix-A
oracles verified to <1e-10; full suite green; single frozen SHA for the validation campaign.

### Added
- Turnkey region masks for `resolvent_map`: `input_region`/`output_region` cfg specs (`{"x": [x0,x1], "y": [y0,y1], "vars": [...]}`, any key omittable → full extent / all vars) build the diagonal 0/1 forcing/response operators (`B_mat`/`C_mat`) internally, so a windowed or component-restricted resolvent gain needs no hand-built ndof×ndof matrices and no knowledge of the LNSE state layout. The mask builder honors the layout exactly — active points from `mesh.X/Y.ravel(order="F")[mesh.used_ind]`, variable blocks from `idx['{rho,u,v,w,T}_j']` — and is a sibling of the existing BC-forcing mask helper; an explicit `B_mat`/`C_mat` still takes precedence, then region specs, then the default BC-forcing mask. A full-domain all-vars region is the identity on every dof and reproduces the unmasked resolvent gain and modes to machine precision (~1e-12, phase-aligned); empty/out-of-domain/zero-active-point/unknown-variable specs loud-fail with a diagnostic `ValueError`; per-variable mask coverage (`n_active/n_total` + bounding box) is logged at INFO. Sharp 0/1 masks only (the spec is a dict so smoothed edges can extend it later). A mask changes the *question* to a windowed/component-restricted gain — it is not an approximation of the global gain.
- Turnkey resolvent gain-map entry point `resolvent_map(base_flow, beta_range, f_range, cfg) -> RouterResult` (`solvers/router/resolvent_map.py`, re-exported as the public `linstabpy.resolvent_map`): the single user-facing thesis symbol — base-flow in, `(len(beta), len(f))` leading-gain map + intended band labels + CP3 cost-metrics sidecar out. Per β it assembles `L0(β)` once via the existing assembly seam (`build_lnse_system`), asks `mark_unstable_band` for the per-frequency band labels, and computes every `(β,f)` point through the certified/direct `resolvent_analysis` path. The discounted (regime b) and ROM (regime c) speed engines do not exist yet, so the router is honest about it: every point currently runs on regime (a), the band label is recorded as the *intended* route, and the regime-b/c paths are gated off by default (`enable_regime_b`/`enable_regime_c`) — when left off a one-line fallback note is logged, when switched on they raise `NotImplementedError` naming the missing engine rather than silently faking a result. Each point logs index/total, ω, β, leading gain, residual, wall, peak RSS via the existing CP3 cost block + stage-heartbeat env (no new logging system); the sweep is resumable (points already present in the output `.npz` are skipped, same cache-on-existence convention as the examples) and per-point failure-tolerant (a failed solve is caught, recorded with its exception class + last state, marked `NaN`+reason-code, and retried on a resumed run rather than aborting the map). Numerics are reused, not re-implemented; an e2e test asserts a regime-(a) map point matches a standalone `resolvent_analysis` oracle within tolerance.
- Frequency-count break-even router `break_even_decision(L0_pattern, n_frequencies, cfg) -> RoutingDecision` (`solvers/router/break_even.py`): decides direct per-frequency factorization vs the anchor-FGMRES sweep from a **measured** cost model, the frequency-count complement to the memory-driven `ram_autoselect`. It times one real SciPy `splu` factorization (`T_fact`) and one representative solve (`T_apply`) on the supplied nonzero pattern (fabricated well-conditioned values; timing/fill depend on pattern not values), then computes the crossover `N_f* = K_anchors / (1 − iters·T_apply/T_fact)`: below it routes `"direct"`, at/above it routes `"fgmres-sweep"`. When `iters·T_apply ≥ T_fact` the sweep can never amortize its anchor factorizations, so the decision stays `"direct"` for every frequency count (crossover `inf`). `RoutingDecision` carries the chosen path, the measured crossover and timings, and a provenance note, so a cost figure can report which path each map used. The crossover is a measured number, not a guessed constant (the test varies only a probe's apply timing and asserts the crossover follows it). Illustrative small-pattern measurement — no production-scale speedup claim — and exposed but not auto-wired into any solver path (`resolvent_map` does not exist yet).
- Stable-sweep solver acceleration (cost-only; numerically exact): the `linalg` factorization seam gained an additive `Factorization.solve_block` (PETSc `MatMatSolve` block multi-RHS, with a single-RHS-loop fallback for iterative/unsupported backends) and `Factorization.refactorize` (reuse the symbolic analysis/fill ordering for a new operator of identical nonzero pattern). `sweep_stable_regime` now factors the first anchor once and refactorizes in place across the remaining anchors — all anchors share K's sparsity (only the diagonal shift differs) — and routes the RSVD probe block through `solve_block`; it falls back to per-anchor factorization if reuse is unavailable or raises. The assembled/solved results are unchanged (block columns equal the single-RHS loop to round-off; symbolic-reuse gains equal recompute within the S7 σ-tolerance). `refactorize` rejects a differing nonzero pattern (collectively across MPI ranks) so reuse can never produce a silent wrong factor. Reductions in analysis time and block-vs-loop solve time are reported in `SweepResult.metrics`; these were demonstrated on a small representative case and carry no production-scale speedup claim.
- Opt-in size-based factor-backend auto-routing: `factor_backend="auto"` resolves to STRUMPACK at or above a DOF threshold and to MUMPS below it, chosen at solve time once the operator size is known. The threshold is configurable (`auto_route_dof_threshold` JSONC field / `LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD` env, precedence env > JSONC > default) with a default of 224k DOF — the c9 A6000 measured lower bound below which MUMPS wins; the true crossover up to ~900k is unmeasured, so the default is a tunable approximation, not a universal constant. When STRUMPACK is not available in the build the route falls back to MUMPS (logged), so `"auto"` is safe on CPU-only machines and never applies an absent backend. The default factor backend is unchanged (MUMPS); auto-routing is strictly opt-in, and resolving `"auto"` to MUMPS produces results numerically identical to an explicit MUMPS run. The GPU speedup itself is measured separately and is not exercised by this routing logic.
- Documented the NVPL (NVIDIA Performance Libraries) CPU BLAS/LAPACK build recipe for Grace aarch64 in `docs/COMPUTE_BACKENDS.md`: the PETSc `--with-blaslapack-lib` linkage against `libnvpl_blas_*`/`libnvpl_lapack_*` (LP64-sequential pair for the default 32-bit-index complex build, with the ILP64/`_gomp` variants and the MUMPS-rejects-64-bit-BLAS-indices caveat noted), that SLEPc inherits the linkage, the NumPy/SciPy Meson option, and a complex128 (Z) `ldd`/`show_config` verification step. `install.sh` ships no NVPL stack, so it builds manually into an isolated `.venv-nvpl`; on-hardware activation and verification are deferred until Grace aarch64 hardware is available (no speed claim is made).
- Adaptive pre-sweep anchor placement `choose_anchors_adaptive(omega_range, band_marks, cfg) -> (anchors, records)` (`solvers/router/anchors.py`): starts from a coarse anchor set and inserts anchors where the S6 EVP distance to the imaginary axis (`BandMarks.distances`) falls below a radius or where measured FGMRES iterations/residual at a sample frequency exceed a threshold, replacing fixed uniform spacing. It is a pure pre-sweep helper — it returns a *fixed* anchor set (plus a per-anchor `AnchorTriggerRecord` provenance trail naming each insertion's trigger) that the caller passes into `sweep_stable_regime`, so the chosen set is fixed per RSVD pass and the S7 determinism/linearity guarantees are unaffected. The FGMRES-iteration signal is supplied by an injectable probe (or measured arrays in cfg), defaulting to a cheap probe, rather than self-measured inside the chooser. On a small case the scheme places more anchors near a hard sub-band and fewer in the easy bulk, and the resulting sweep matches a direct MUMPS oracle within the S7 σ-tolerance; the adaptive-vs-fixed factorization-count comparison is logged as a measurement (no superiority gate — "adaptive beats fixed" stays a hypothesis to measure). Exposed but not auto-wired into a default sweep path.
- RAM-calibrated operating-point selector `ram_autoselect(L0_pattern, cfg) -> OperatingPoint` (`solvers/router/ram_autoselect.py`): decides the resolvent solve operating point from the machine, not the user. It reads available RAM portably (injectable for tests; else `/proc/meminfo` `MemAvailable`, else `os.sysconf`), runs a cheap **empirical** sparse-LU analysis probe on the supplied nonzero pattern to measure the real factor fill count and bytes-per-nonzero on *this* build (the §7c formula is only the prior; the probe is the correction, R5), estimates the memory footprint, and picks `direct` when the factor fits the budget or `compressed-sweep` (raising compression / dropping precision / adding ranks) when it does not. `OperatingPoint` returns only resource knobs (backend, reorder, compression level, restart, ranks, precision); user physics choices (grid, fd order, modes, β/f ranges) are never read or altered — selection changes COST, not the numerical answer (the direct end matches a direct MUMPS oracle within the S7 σ-tolerance). The compressed-sweep end's memory ratios are an uncalibrated prior, and its end-to-end oracle check loud-skips where STRUMPACK is unavailable or its complex adjoint is non-certifiable. The selector is exposed but not auto-wired into `resolvent_map` defaults.
- Opt-in GCRO-DR-style Krylov subspace recycling for the stable resolvent sweep (`recycling` JSONC block): carries a *fixed* deflation/recycle subspace from one shifted system to the next within an anchor block and feeds it to the right-preconditioned GMRES, so consecutive shifts reuse spectral structure. The recycle subspace is prepared once per shift solve and never adapted mid-solve, so the solve remains a linear map and the S7 `deterministic_tol_linearity` certification check still passes on the recycled path (measured defects ~1e-12 against a 1e-8 tolerance). `SweepResult.metrics` reports total matvec and GMRES-iteration counts with and without recycling plus the delta. The default path (recycling off) is unchanged. Demonstrated on a small representative case (6×6 operator, 4 frequencies, recycle dim 3): matvecs 1837 → 1513 (−17.6%, including 18 setup matvecs), gains matching an independent direct MUMPS oracle to ~1.5e-13; this is a small-case illustrative measurement and carries no production-scale speedup claim. Recycling and compressed anchors are not combined in this evaluation path.
- Opt-in compressed-anchor option for the stable resolvent sweep (`compressed_anchor` JSONC block): routes the anchor factorization through the STRUMPACK backend with BLR/HSS rank compression via the existing `linalg` seam, with the S7 adjoint-of-approximate-factor certification (check #3) wired to the *compressed* factor's own `solve_adjoint` (not the exact-LU assumption), and the measured factor-compression ratio + effective rank reported on `SweepResult.metrics`. RECORDED FINDING (measured on real STRUMPACK): the compressed-anchor path is **not currently usable in the certified sweep** — STRUMPACK's transpose-solve is a non-conjugate transpose for complex operators, so the adjoint-identity defect is ~3.2e-2 (vs ~3e-16 for MUMPS), five orders above the certification tolerance, and it is *identical with BLR compression on and off* — a STRUMPACK complex-adjoint property, not a compression artifact. The sweep therefore correctly refuses (raises) rather than silently using an uncertified factor; the compression-vs-memory hypothesis cannot be evaluated in the certified sweep until this prior backend-adjoint blocker is resolved. The default path (MUMPS-exact) is unchanged.

### Changed
- 2-D LNSE block assembly (`_build_block_csr`) now builds each ngp×ngp block CSR-natively instead of materializing a COO `(rows, cols, data)` triplet and letting SciPy `coo_tocsr` sort it: blocks with ≤2 contributions are summed as row-scaled CSR terms, and denser blocks are filled directly into precomputed `indptr`/`indices` (one `cumsum` + per-term position scatter, no sort), eliminating the ~200 MB-per-dense-block COO intermediate. The assembled operator is byte-identical — `‖L0_new − L0_old‖ = 0` with identical sparsity on the Blasius smoke and a Case 2 stride-2 case (510k DOF, 23.2M nnz), and the seven Appendix-A oracles (<1e-10) plus the full suite stay green with no re-baseline — so this is a cost-only change. Block assembly is ~10–20% faster on the Case 2 stride-2 case (single-run, timing-noise-dependent; no production-scale speedup claim). The layout is recomputed per call and deliberately *not* memoized across calls — a module cache keyed on object ids / buffer addresses would be unsafe under allocator address reuse (a freed derivative cache's address reused by a different sparsity pattern could silently return a wrong operator).
- 2-D LNSE boundary-condition application now runs in a single CSC→CSR→CSC format context across the whole recipe instead of converting the operator per BC rule. `apply_bc_recipe`'s SciPy path zeroes the original Dirichlet columns once in CSC, converts to CSR once, then applies all row-zeroing / Neumann row-insertion / penalty-diagonal writes in CSR before a single final CSC conversion — collapsing the redundant `csr_tocsc` round-trips (9 → 3 on the profiled case). The assembled operator is numerically unchanged — guarded ≤1e-10 by the seven Appendix-A oracles and the equivalence/parity harness (full suite green) — and assembly is ~15–20% faster on a Case 2 stride-2 representative case (510k DOF, 23.2M nnz, single rank: ~3.6–3.9 s → ~3.1 s; no speedup claim at production scale). `apply_dirichlet`/`apply_neumann` keep their public CSC-in/CSC-out contracts for direct callers, and `apply_bc_recipe` no longer mutates its input operator in place.
- Documented the SuperLU_DIST PETSc build recipe (`--download-superlu_dist` plus the METIS/ParMETIS graph-partitioning downloads; the GPU variant adds `--with-cuda`) in `docs/COMPUTE_BACKENDS.md`, noting that `install.sh` does not yet ship a `superlu_dist` build stack so it must be built manually in an isolated `.venv-superlu_dist`. Clarified that SuperLU_DIST has two distinct roles: a live CPU mixed-precision iterative-refinement path (FP32 factor + FP64 refine via `pc_precision single`) and a complex sparse-direct GPU solve that was rejected on the tested CUDA build (order-one residuals). The previous backend status read as a wholesale "rejected"; the rejection applies only to the complex+CUDA sparse-direct path, not to the live CPU mixed-precision role.

### Fixed
- Build requirement corrected to `setuptools >= 77.0` (was `>= 68.0`): the PEP 639 SPDX `license = "GPL-3.0-only"` expression in `pyproject.toml` needs setuptools 77+, so a fresh install on setuptools 68–76 previously failed metadata generation (`invalid pyproject.toml config: project.license`).
- STRUMPACK `EPS.setUp` factorization failures traced to static-pivoting `ZERO_PIVOT` now raise `StrumpackFactorizationFailed` and trigger a one-shot MUMPS fallback, with fallback provenance recorded in solver info; the default MUMPS path and unrelated setup failures remain unchanged.
- Factor-backend accuracy probe is now role-aware: a backend used as the preconditioner inside an iterative-outer shift-invert solve (`st_solver` in `iterative`/`krylov`/`slepc`/`fgmres_lu`/`strumpack_fgmres`) is no longer rejected on its bare direct-solve residual. The strict `residual_rel_max` (~1e-8) gate is the right acceptance test for a standalone direct solve, but for a preconditioner the outer Krylov/SLEPc convergence is the real gate and a moderate (~1e-6) single-apply residual is healthy — previously this forced users to disable the probe by hand to run a STRUMPACK-preconditioned route. The genuine direct path (`mumps`/`direct`/`lu`, default-MUMPS exemption) keeps the unchanged strict gate; the preconditioner role applies a relaxed sanity ceiling so a catastrophically ineffective preconditioner still fails fast. The preconditioner-role detection reuses the shared `ITERATIVE_EIGEN_SOLVER_TYPES` set so it cannot drift from the solver-config vocabulary.
- `EPS.setUp` failures now surface a useful error message. When petsc4py's `CHKERR` collapses a PETSc/STRUMPACK error into a bare `SystemError: ... returned a result with an exception set`, the raised `RuntimeError` now explains that the real PETSc error was emitted to stderr (and how to capture it via stderr redirection or `-error_output_stderr`) instead of reporting the opaque `SystemError`; a `PETSc.Error` now names its integer error code (`ierr`). All existing diagnostic context (ndof, backend, `st_solver`, MUMPS infog) is preserved.

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
- Package version source-of-truth reconciled to a single canonical source. `pyproject.toml` is now the only place the version is written, and `linstabpy.__version__` is derived from `importlib.metadata` instead of a hardcoded string that had drifted to `0.1.0` — the reserved release-candidate tag — while the build metadata said `0.0.6.dev0`. `__version__`, `importlib.metadata.version("linstabpy")`, and `linstabpy --version` now always agree with the installed distribution, guarded by a consistency test that checks every surface against `pyproject.toml`.

### Changed
- 2-D LNSE assembly now applies the analytical `inv(B)` as a direct CSR row scaling (`data *= per-row factors`) instead of materializing a sparse diagonal and multiplying, and the SciPy raw-assembly path builds CSR natively with a single final `tocsc()` (one fewer format conversion). The assembled operator is numerically unchanged — guarded at ≤1e-10 against the validated block reference, with the seven Appendix-A oracles green — and ~26% faster on a small fd6 Blasius representative case (measured median on a shared box; no speedup claim at production scale).
- Solver module docstrings (`solvers/eigenvalue.py`, `spatial_eig.py`, `resolvent.py`, `linsolve.py`) now document the factor-backend seam: path-accurate resolution order (including that `LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE` is not consulted on the reference direct eigen path), probe-based availability with loud failure, conservative backend statuses, and the env-only enforcement status of the experimental precision/BLR/ordering knobs; doc-only, behavior proven unchanged by docstring-stripped AST comparison.
- Phase-zero band-marking default `danger_radius` is now a physical omega-plane scale instead of an unrelated linear-solve residual scale. When no `danger_radius` is configured, the guard now uses `0.5 * median` of the successive spacing of the requested (finite, unique) forcing frequencies — flagging a frequency `near-pole` when a resolvent pole lies within about half a sweep cell — replacing the old fallback to `solver.eigen.backend_probe.residual_rel_max` (~1e-8), which was a residual tolerance that effectively never fired in the omega plane. With fewer than two distinct finite frequencies no sweep scale exists and the default is `0.0` (no `near-pole` labels); NaN/inf frequencies are excluded and non-uniform grids use the median successive spacing. Explicit `danger_radius` config keys still override in the same precedence and are still validated as finite non-negative distances; the least-damped `|Im omega_eig|` basis was rejected because it collapses near a stability threshold, exactly when a near-real pole most needs a band.

### Added
- Hybrid MPI + threaded-BLAS placement guidance (`docs/COMPUTE_BACKENDS.md`): documents the deliberate `OMP_NUM_THREADS`/`OMP_PROC_BIND`/`OMP_PLACES` (+ `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`) recipe as `threads_per_rank = physical_cores // ranks`, the explicit non-oversubscription invariant `ranks * OMP_NUM_THREADS <= physical cores` (SMT/HT siblings excluded), when to pick pure-MPI (the validated MUMPS sparse-direct default) vs hybrid (dense-heavy / threaded vendor BLAS), and a copy-paste `mpirun` sanity check with a clear pass rule. Placement guidance only — no measured speedup claim.
- Line-3 GPU FP64 decision gate (`docs/GPU_FP64_GATE.md`): documents why FP64 peak FLOPS is the wrong axis for the sparse-direct shift-invert workload class, a measurable go/no-go criteria table (parity quality via the acceptance-anchored `PARITY_RECORD` tolerances; factor-memory capacity; sustained bandwidth; wall time at equal mode quality; factorization-reuse economics; transfer overhead — every non-measured threshold tagged `[assumed]`, hardware numbers tagged `[vendor spec]`), and the recorded recommendation: Line 3 commits only as a capacity/bandwidth/mixed-precision play, FP64-FLOPS framing rejected, with explicit reopen conditions.
- Parity-vs-MUMPS validation protocol + CPU harness skeleton (`tests/test_parity_vs_mumps.py`): candidate factor backends are gated against the MUMPS reference on a small Blasius temporal EVP with acceptance-anchored tolerances (postcheck residual <= 1e-6, target distance <= 0.11, conjugate-aware `-conj(omega)` matching, matched-eigenvalue distance bounded by the postcheck gate — no backend-specific epsilon), emitting a structured `PARITY_RECORD` with matrix size, nnz, factor memory, wall times, ranks, exact solver options, and subspace/convergence state. Ships a mumps-vs-mumps self-consistency floor plus the CPU hard-path SuperLU_DIST `-pc_precision single` E2E test, which skips loudly on PETSc builds without superlu_dist; single-precision is verified from PETSc option consumption, never from request state. GPU execution deferred.
- MUMPS fill-reducing ordering knob (off/unset by default): `LINSTABPY_MUMPS_ORDERING` selects `amd`/`pord`/`metis`/`qamd`/`auto` via `-mat_mumps_icntl_7` (JSONC `mumps_ordering` carried, env-only enforced for now); unset expresses no opinion — it neither sets nor clears `ICNTL(7)`, preserving values installed by the legacy `mpi.mumps_icntl_7` gate or user CLI options (pinned by a real-wiring preservation test); when set it overwrites external values for that process. Unknown names fail listing allowed values; any non-MUMPS backend (including the PETSc-default path) fails loud. Real-wiring tests read `ICNTL(7)` back from the factored matrix after `ksp.setUp()`. No performance claim.
- MUMPS BLR low-rank compression knob (experimental, off by default): `LINSTABPY_MUMPS_BLR` / `LINSTABPY_MUMPS_BLR_TOL` env knobs (JSONC `mumps_blr` / `mumps_blr_tol` fields validated and carried, env-only enforced for now, like `factor_precision`) set `-mat_mumps_icntl_35 1` and `-mat_mumps_cntl_7 <tol>` for MUMPS factorizations; options persist in the options DB so MUMPS actually consumes them at factorization setup (verified by real-wiring tests against the factored matrix: ICNTL(35)/CNTL(7) read back after `ksp.setUp()`, including a non-stickiness check that an off-request clears leftovers); enabling BLR with any non-MUMPS backend (including the PETSc-default path) fails loud naming the backend and knob. No performance claim.
- Per-stack build isolation in `install.sh`: `--stack <name>` / `LINSTABPY_BUILD_STACK` select side-by-side virtual environments (`.venv-<stack>`), keeping `.venv` exclusively for the validated CPU/MUMPS reference build; non-default stacks hard-refuse to remove, overwrite, or alias (symlink) the CPU venv, `--force` is scoped to the stack's own directory, and `--dry-run` prints the resolved plan without touching the filesystem. The GPU configure branch (`--with-cuda --download-strumpack`) is documented as design-only in `docs/COMPUTE_BACKENDS.md` — no GPU build is claimed.
- Wavemaker / structural-sensitivity utility (`linstabpy.wavemaker`, `stability/sensitivity.py`): Giannetti–Luchini field `zeta = |q_dir| |q_adj| / |<q_adj, q_dir>_W|` from direct and Chu-weighted adjoint modes — the first user-facing consumer of the free matrix adjoint. Pairs modes by `|lambda_dir - conj(lambda_adj)|` with a tolerance gate that fails loud listing the offending eigenvalues, normalizes with an explicit retained inner-product divisor, reshapes Fortran-order per state component, and logs a per-mode pairing table at INFO. The adjoint convention (`adjoint_eigenvalues` returns `W^-1 L0^H W` modes, no extra transform) is pinned by a discriminating test.
- Public-boundary input validation: `Mesh.create`, `resolvent_analysis`, `eigenvalues`, `adjoint_eigenvalues`, and `spatial_eigenvalues` now reject malformed arguments at the call boundary with typed errors naming the bad argument (non-square or non-sparse operators, mismatched weight-vector lengths, `n_modes < 1`, non-finite frequencies/targets, grids below the SBP stencil minimum for the requested `fd_order`) instead of failing deep inside assembly; error messages never leak filesystem paths.
- Mixed-precision factorization config surface (experimental, inert by default): `LINSTABPY_FACTOR_PRECISION` env knob and JSONC `factor_precision` field accept `double` (default, zero effect on the options DB) or `single`, which is honored only with the `superlu_dist` factor backend (inserts `pc_precision single`) and fails loud with any other backend; the JSONC field is carried in the typed config and will be consumed by the Phase B.4 iterative-refinement work.
- Anchor-preconditioned stable-regime frequency sweep (`solvers.router.stable_sweep`): `sweep_stable_regime` factors the shifted operator only at caller-given anchor frequencies and sweeps neighboring frequencies with right-preconditioned GMRES reusing the nearest anchor factor; forward and adjoint actions share one factorization per anchor, every leading singular triplet passes the `certified_rsvd` gate, and per-frequency iteration counts and residuals are recorded as measured quantities, never gated.
- Solver-stack reproducibility manifest: runtime provenance now records python/numpy/scipy/petsc4py/slepc4py versions, PETSc arch, MUMPS version, and the MPI implementation (failed probes record `unavailable`, never crash a run); the block is echoed at INFO at save time and `scripts/tools/freeze_env.sh` emits a pip-constraints file for release freezes.
- Per-run metrics schema gate (`metrics_schema`): a single source-of-truth schema for the cost+quality sidecar with `validate_metrics` reporting every missing or ill-typed key at once; sidecars failing validation are written with a `schema_errors` field and a loud warning so the run is visibly quarantined from the figure pipeline.
- RSVD certification gate (`solvers.router.certification`): `certified_rsvd` runs five algebraic checks before trusting an inexact resolvent action — σ-accuracy-derived solve tolerance, forward/adjoint consistency with realized-residual enforcement, adjoint-of-approximate-factor verification, deterministic linearity probed on both forward and adjoint actions, and multi-seed RSVD reproducibility — and fails loud naming the tripped check.
- Phase-zero EVP-guided band marking (`solvers.router.band_marking`): labels requested forcing frequencies stable / near-pole / smooth by distance to the resolvent poles (`-omega_eig`) before any sweep.
- Config-selectable FGMRES shift-invert path with direct-LU preconditioning, including STRUMPACK backend selection when available.
- Runtime result archives and metrics sidecars now include git, config, host, compute-mode, and MPI-stack provenance.
- CPU/MUMPS default-path parity fixture on a small Blasius case and standalone regression guard for ≤1e-12 baseline drift.

### Changed
- Runtime cost metrics now use `unknown` as the default hardware tag when no hardware tag is configured.

### Fixed
- `install.sh --dry-run` `planned_remove` now mirrors the real venv lifecycle exactly: it reports a removal only when the venv actually exists and would be removed — under `--force` on an existing venv, or a Python-version mismatch that triggers recreate — and reports `none` for an absent venv or in `--check` mode (which never removes). Previously it keyed off `--force` alone, over-reporting a removal for an absent venv or under `--check`, and under-reporting the mismatch-recreate case. Plan-reporting fidelity only, via a single shared `venv_will_be_removed` predicate; real removals were already correctly guarded by `remove_venv_dir`.
- `linstabpy --version` and `--help` (`python -m linstabpy.driver.cli`) now work on a base install without the optional PETSc/MPI stack: the `linstabpy.driver` package initializer no longer eagerly re-exports `run` from the PETSc/MPI-backed `linstabpy.driver.run`, so the CLI's existing lazy import is no longer defeated at package-import time. Verified by a test that blocks `petsc4py`/`mpi4py` via `sys.meta_path` and asserts neither is imported on the `--version`/`--help` paths. Import the runtime explicitly (`from linstabpy.driver.run import run`) where it is needed; no consumer change required.

### Deprecated
- The legacy `mpi.mumps_icntl_7` / `LINSTABPY_PETSC_SET_MUMPS_ICNTL_7` boolean is now a deprecated compatibility alias for `mumps_ordering=metis`; it is still accepted (and still installs `ICNTL(7)=5`) but new configs should use the canonical `mumps_ordering` surface (`mumps_ordering: "metis"` / `LINSTABPY_MUMPS_ORDERING=metis`). The `metis`→`ICNTL(7)=5` value is now drawn from a single shared `MUMPS_ORDERING_TO_ICNTL_7` table (new PETSc-free `linstabpy.mumps_options` module) consumed by both the legacy gate and the ordering knob, so the two surfaces can no longer drift. Numerically identical to the prior behavior for every existing config (real-wiring `ICNTL(7)` read-back: legacy alias → 5, explicit `mumps_ordering` overrides the alias, unset leaves the key absent → MUMPS default 7).

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
  and Martini example parity.
- **Stage 6 (refactor):** assembly metrics, stage timers, and CSR derivative
  cache helpers moved into `linstabpy.operators.metrics`; `assembly.py` now
  imports the plumbing while preserving compatibility re-exports.
- **Stage 4 (refactor):** new `linstabpy.stability` package unifies the three
  eigenvalue entry points — `eigenvalues` (was `temporal_eigenvalues`),
  `adjoint_eigenvalues`, `spatial_eigenvalues` — on one shared private
  shift-invert core (`_solve_near_target`, moved verbatim from
  `solvers.eigenvalue._eigs_near_target`). Old names in `solvers.eigenvalue` /
  `solvers.spatial_eig` remain as deprecation shims for one minor version;
  top-level `linstabpy.temporal_eigenvalues` / `adjoint_eigenvalues` are
  unchanged and warning-free. Numerics frozen: full suite and spectra
  identical to baseline.
- **Stage 4 (refactor):** `utils.stability_acceptance` moved verbatim to
  `linstabpy.stability.acceptance`, colocating the acceptance gate
  (postcheck residual, target distance, conjugate-aware verdict,
  `reject_reason_code` bitmask — all unchanged) with the eigenvalue core.
  Old path remains a deprecation shim for one minor version.
- **Stage 5 (refactor):** new `linstabpy.driver` package; the case-pipeline
  orchestration moved verbatim from `results_runtime.compute` to
  `linstabpy.driver.run` (`from linstabpy.driver import run`).
  `results_runtime.compute` now delegates and remains the supported
  entrypoint; PETSc command-line options are initialised before `petsc4py`
  import on the new path as well. Numerics frozen: full suite and spectra
  identical to baseline.
- **Stage 5 (refactor):** runtime `.npz` archive I/O isolated in
  `linstabpy.driver.io` (`save_results` — verbatim move of the writer —
  and `load_results`). The provenance/cost key set is unchanged and now
  guarded by a round-trip test; `results_runtime._stage_write_npz` remains
  as a back-compatible wrapper.
- **Stage 5 (refactor):** new CLI front door `linstabpy.driver.cli` with a
  `--version` flag; the cache-aware `main()` moved there verbatim.
  `results_runtime` is now a one-minor-version deprecation shim re-exporting
  its previous public surface (`compute` → `driver.run`, `main` →
  `driver.cli.main`, helpers → their new homes) with `DeprecationWarning`;
  all in-repo importers migrated.
- **Stage 5 (test):** driver split certified numerics-inert by the full suite,
  driver provenance round-trip, `--version` smoke, and examples parity gate.

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
  the granular `0.0.x` tick cadence (a few changes → verify → tag), the merge gate,
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

[Unreleased]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc11...HEAD
[1.0.0rc11]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc10...v1.0.0rc11
[1.0.0rc10]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc9...v1.0.0rc10
[1.0.0rc9]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc8...v1.0.0rc9
[1.0.0rc8]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc7...v1.0.0rc8
[1.0.0rc7]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc6...v1.0.0rc7
[1.0.0rc6]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc5...v1.0.0rc6
[1.0.0rc5]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc4...v1.0.0rc5
[1.0.0rc4]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc3...v1.0.0rc4
[1.0.0rc3]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc2...v1.0.0rc3
[1.0.0rc2]: https://github.com/openfluids/linstabpy/compare/v1.0.0rc1...v1.0.0rc2
[1.0.0rc1]: https://github.com/openfluids/linstabpy/compare/v0.0.5...v1.0.0rc1
[0.0.5]: https://github.com/openfluids/linstabpy/releases/tag/v0.0.5
