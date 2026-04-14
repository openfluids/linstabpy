# Case 2 Numerical-Setup Plan

Date: 2026-04-14

## Purpose

Build an evidence-backed numerical setup guide for Case 2.

This plan does not assume we already know the best settings. It treats the
current configuration as a baseline to test, not as a recommendation to
publish.

## Fixed Baseline

Keep these choices fixed until the solver-calibration phases are complete:

- Case: `results/case2/parity_config.jsonc`
- Baseflow: Case 2 full-resolution baseflow
- MPI layout: 10 ranks pinned to one physical socket
- Threads: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Operator path: distributed PETSc matrix assembly
- Temporal EVP: SLEPc EPS Krylov-Schur + shift-invert
- Shifted linear solves: direct LU with MUMPS
- BC recipe: no-corner Neumann recipe already validated against the earlier bug
- Physics knobs held fixed for solver calibration:
  - `fd_order = 4`
  - `alpha_filter = 0.0`
  - current Case 2 sponge path from the baseline config

## Questions To Answer

1. How large does `ncv` need to be before the accepted spectrum saturates?
2. How much does tightening `eps_tol` improve explicit postchecked residuals?
3. For a fixed accepted-quality target, what is the cheapest working setup?
4. Once the small-`nev` problem is stable, how far can we increase `nev`
   before cost becomes unacceptable?
5. After the solver is calibrated on one socket, how much do rank placement
   and rank count change runtime and memory?
6. Only after the solver setup is stable: how sensitive are the spectra to
   `fd_order`, filtering, and sponge choices?

## Acceptance Metrics

Every run must record the same metrics:

- wall time
- peak RSS
- MPI rank count and launch layout
- requested `nev`
- actual `nconv`
- EPS relative residual counts
- explicit postcheck residual counts
- median and max explicit postcheck residual
- BC leakage diagnostics

The setup guide should quote only metrics measured from these outputs.

## Phase Order

### Phase 1: `ncv` sweep at fixed `nev`

Goal: find the useful Krylov-basis range without changing any other knob.

Status:
- already completed for `nev = 10`, `eps_tol = 1e-4`
- output:
  - `results/case2/output/eigs_calibration_ncv_socket0/case2_eigs_calibration_ncv_table.csv`

Current evidence:
- `ncv = 200` is much cheaper than `700` and already gives median explicit
  postcheck residual around `8.5e-6`
- `ncv = 700` is slightly better on median residual, but much more expensive
- `ncv = 1000` is not currently justified by quality gain

Working conclusion:
- the next phase should focus on `ncv = 200` and `ncv = 700`

### Phase 2: tolerance sweep at fixed `nev = 10`

Goal: determine whether tighter `eps_tol` actually improves explicit
postchecked residuals or only increases cost.

Run matrix:
- `ncv = 200`, `eps_tol = 1e-4, 3e-5, 1e-5, 3e-6, 1e-6`
- `ncv = 700`, `eps_tol = 1e-4, 3e-5, 1e-5, 3e-6, 1e-6`

Decision rule:
- if tighter tolerance does not materially improve explicit postcheck
  residuals, do not keep paying for it

### Phase 3: `nev` growth at the best solver point

Goal: increase requested eigenpairs only after the per-mode setup is stable.

Planned ladder:
- `nev = 10, 20, 50, 100`

Rule:
- choose `ncv` from Phase 2, then scale `ncv` with `nev` conservatively
- stop increasing `nev` when accepted-mode quality degrades or cost explodes

### Phase 4: MPI placement and rank-count study

Goal: separate numerical convergence from hardware-launch effects.

Planned comparisons:
- 1 socket, 10 ranks
- 2 sockets, 20 ranks
- optional 1 rank reference only if cost is reasonable

Measure:
- wall time
- peak memory
- accepted-mode counts at the same numerical setup

### Phase 5: physics/discretization sensitivity

Goal: once the solver is stable, test the physical discretization knobs.

Planned comparisons:
- `fd_order = 4, 6, 8`
- filter choices as needed
- sponge sensitivity only after the solver setup is no longer the dominant
  uncertainty

## Deliverables

The final `docs/numerical_setup.md` should contain only:

- solver facts we know from the code
- defaults supported by measured runs
- open questions still under test

It should not present guesses as recommendations.

Current note:
- a provisional `docs/numerical_setup.md` draft has already been created to
  capture the durable, atemporal guidance before it is forgotten
