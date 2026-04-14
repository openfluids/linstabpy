# Numerical Setup

Status: provisional draft

This document records the parts of the numerical setup that are stable enough
to keep for future users even while the Case 2 calibration study is still in
progress.

It intentionally separates:

- facts we know from the implementation
- convergence rules that remain valid across machines and cases
- defaults that are still under calibration and therefore are not fixed here

## Solver Facts

These points are properties of the current implementation, not tentative
recommendations.

### Temporal stability

The current temporal stability path uses:

- SLEPc `EPS`
- `KRYLOVSCHUR` as the outer eigensolver
- shift-invert spectral transformation (`ST=SINVERT`)
- direct LU factorization with MUMPS for the shifted linear systems

In code, this path is configured in
[src/linstab2d/solvers/eigenvalue.py]($HOME/linstabpy/src/linstab2d/solvers/eigenvalue.py).

The temporal eigenproblem is formed on `i * L0`, so the reported temporal
eigenvalues are `omega = eig(i * L0)`.

### Residual bookkeeping

The code records more than one residual-like quantity:

- SLEPc EPS residuals from `EPS.computeError(...)`
- the configured EPS convergence test (`abs`, `rel`, or `norm`)
- the selected reported residual type (`absolute`, `relative`, or `backward`)
- an explicit postcheck residual computed outside SLEPc on the returned modes

These quantities are related, but they are not interchangeable.

More precisely:

- `EPSSetTolerances()` sets the tolerance and maximum iteration count used by
  the EPS convergence test
- `EPSSetConvergenceTest()` selects how that EPS convergence test is measured:
  absolute, relative to the eigenvalue, or relative to matrix norms
- `EPSComputeError()` reports an error bound for a returned eigenpair, but this
  is not guaranteed to be the same quantity used internally during the solve
- `EPSSetTrueResidual(true)` forces explicit true-residual computation for the
  EPS convergence test; this is more expensive, but is specifically important
  when a spectral transform is used because cheap Krylov residual estimates can
  be too optimistic for the original problem

For this codebase, that distinction matters because the temporal EVP is solved
with shift-invert.

### PETSc KSP tolerances

`KSPSetTolerances()` controls the stopping criteria for a linear Krylov solve,
that is, the relative/absolute/divergence tolerances and maximum iteration
count of the `KSP` object.

This is not an eigenpair tolerance.

In the current production temporal-stability path, the shifted linear systems
use:

- `KSPPREONLY`
- `PCLU`
- MUMPS

That means the current linear solve is a direct factorization path, not an
iterative Krylov path. In this configuration, `KSP` tolerances are not the main
control knob for eigenpair accuracy; the dominant knobs are the EPS settings,
the Krylov subspace size `ncv`, the target, and the conditioning of the shifted
operator.

### Boundary-condition acceptance

Returned eigenmodes are postchecked against:

- full-operator residual
- boundary-condition leakage
- constrained-Dirichlet leakage

This is essential because an interior eigenvalue solve can report apparently
good solver residuals while still returning modes that are not acceptable under
the full operator and boundary treatment.

## Convergence Principles

These points are worth remembering even if the recommended default values later
change.

### `eps_tol` alone does not define mode quality

For interior non-Hermitian problems, tightening the requested EPS tolerance does
not guarantee that the accepted modes will improve in the same proportion.

Mode quality can be limited by:

- Krylov-subspace size `ncv`
- target placement
- conditioning of the shifted operator
- non-normality and clustering near the target

Also, the meaning of `eps_tol` depends on the selected EPS convergence test.
The SLEPc default is convergence relative to the eigenvalue. Official SLEPc
documentation warns that this criterion can give poor accuracy for eigenvalues
near the origin, in which case an absolute criterion is often more appropriate.

### `ncv` is a quality-control parameter

`ncv` is not only a runtime parameter.

If the target neighborhood is crowded or strongly non-normal, a too-small `ncv`
can cause accepted-mode quality to plateau even though the solve is active and
the reported minimum EPS error appears small.

### Eigenvalues can stabilize before eigenvectors

In non-normal problems, the spectrum may stop moving while the eigenvectors are
still insufficiently separated.

That means:

- "the eigenvalues look stable" is not enough
- the explicit postcheck and BC-leakage checks still matter

### Shift-invert quality depends on the shifted operator

The outer eigensolver only sees the quality of the shifted solves.

If `iL0 - sigma I` is poorly conditioned, residuals may plateau at a finite
precision floor even when the linear systems are solved with direct LU.

When shift-invert is active, SLEPc also warns that cheap Krylov residual
estimates may be optimistic for the original problem. This is why enabling true
residuals and doing an explicit postcheck are both reasonable for this
application, even though they increase cost.

### Target choice matters

A target on the real axis can sit in a crowded neighborhood. In that case,
convergence can improve when the target is moved slightly into the complex plane
to isolate the branch more cleanly.

This should be treated as a numerical search choice, not as a change in the
underlying physics being sought.

### Branch-localize before scaling up `nev`

For temporal Case 2 studies, the reliable workflow is:

- start with a small solve, such as `nev=10`
- try a few complex targets that sit on the branch of interest
- judge success using the explicit postcheck residual together with distance to
  the requested target
- only after that branch is localized should `nev` and `ncv` be increased

In other words, raw `nconv` is not enough to say that the requested physics has
been captured. A run can return many EPS-converged Ritz pairs while still
locking onto the wrong spectral pocket.

## What To Check On Every Run

Before trusting a spectrum, check all of the following:

- requested `nev`
- actual `nconv`
- `ncv`
- EPS tolerance and EPS convergence test
- whether true residuals were enabled
- whether the linear solve was direct (`KSPPREONLY+LU`) or iterative
- explicit postcheck residual counts
- BC-leakage diagnostics
- wall time
- peak RSS
- MPI rank count and rank placement

For this project, a run should not be accepted based only on SLEPc's internal
convergence report.

## What Not To Assume

Future users should not assume any of the following:

- that passing the EPS convergence test implies a small full-operator residual
- that lowering `eps_tol` is always the cheapest way to improve results
- that a larger `nev` automatically gives more trustworthy leading modes
- that the best setup on one workstation will also be best on another

## Defaults Still Under Calibration

The following are still experiment-backed questions, not frozen guidance:

- best `ncv` for Case 2 temporal stability
- best `eps_tol` for accepted-mode quality versus cost
- best `nev` for production studies
- best rank count and socket placement
- when to move from `fd_order = 4` to `6` or `8`

The active calibration plan is tracked in
[plan_2026-04-14_case2_numerical_setup.md]($HOME/linstabpy/docs/plan_2026-04-14_case2_numerical_setup.md).

Once the calibration runs are complete, this document should gain one more
section only:

- measured defaults supported by the run matrix

Until then, this file should remain conservative and methodological.
