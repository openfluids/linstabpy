# Case 2 — Active Campaign Plan

Last updated: 2026-04-17.

This file holds the **active state** of the Case 2 EVP campaign: where we
are, what's next, what's blocked. Stable background (governing equations,
flow configuration, parameter meanings, physical framing, references) lives
in `REFERENCE.md`. Operational how-to-run lives in `README.md` and
`tutorial.md`.

Inherits motto and rules from the repo-root `CLAUDE.md`.

---

## Context files (read in this order)

An agent joining this campaign should load context from:

1. **`/CLAUDE.md`** — project-wide motto, rules, generality principle,
   `nconv > 0 ≠ success`, `alpha_filter` vs mean-flow-filter distinction.
2. **This file (`PLAN.md`)** — current state, acceptance criteria, phase
   plan with status, open TODOs, "do not" list. Always load.
3. **`REFERENCE.md`** (same directory) — code lineage (Martini & Schmidt
   2024 port), governing equations, Chu energy norm, parameter map, flow
   configuration, Physical and Numerical Framing (resolvent / mean-flow /
   LES–DNS gap), how residuals should be read, references. Load when a
   question touches background.
4. **`README.md`** (same directory, Portuguese) — operational: how to run
   one case, how to run a campaign, MPI presets, artifact layout.
5. **`tutorial.md`** and **`tutorial_ducros.md`** — step-by-step walk-throughs.

Supporting files in this directory:

- `parity_config.jsonc` — canonical full-resolution configuration;
  `Re=3.95e5`, `Ma=2.0`, `Pr=0.747`, `T₀=1540 K`, `γ=1.4`, `β=0`.
- `case2.py` — single-case entry point.
- `perf_memory.py` — campaign orchestrator.
- `run_case_once.sh`, `run_perf_queue_once.sh` — wrappers.
- `evp_stride*.jsonc` — per-experiment configs (mesh × target × solver).
- `output/<campaign>/runs/*.npz`, `logs/*.log` — artifacts.
- `baseflow/` — active mean-flow dataset (`Qraw`).
- `legacy/` — retired configs. Naming `eigs_branch_localize_ziming_*`
  references the Song 2023 validation case, not Case 2.

External references in `/docs/`:

- `martini2024.md` — the code paper (method-level anchor).
- `robinet2007.md` — original SWBLI global-stability study.
- `song2023.md` — reference SWBLI at M=2.15, Re=1e5 (Ziming Song).
- `plan_2026-04-17_case2_meanflow_solver_coupling.md` — sibling planning
  note; check for overlap before diverging from it.

Related validation data:

- `/validation/song_2023/` — Ziming Song reference spectra / eigenvalues.

---

## At a glance

- **Flow.** Suction-side oblique SWBLI of the Lui 2022 supersonic turbine
  cascade. `Ū` = spanwise/time average of Lui's LES. `Re=3.95e5`,
  `Ma=2.0`, `Pr=0.747`, `γ=1.31` (real turbine gas; Lui ran LES with
  `γ=1.4`, see REFERENCE "Gas-model caveat"). Frequencies: `St = Re(ω)/(2π)`.
- **Objective.** Stabilise one EVP setup at `M1` that is numerically
  convergent on the filtered mean flow. Resolvent is the publication
  target; EVP is the validation gate.
- **Active mesh.** `M21 = (2, 1)` — screening only. `ksi` streamwise is
  coarsened; this is a cost hack, not physical. Final answer must be on
  `M1`.
- **Active mean flows.**
  - `Qraw` = Lui LES mean, unfiltered. **Known to be contaminated** —
    contaminated modes have been observed.
  - `Qfilt` = Lui LES mean after **Lele (1992) 2nd-order compact filter**,
    applied upstream of `linstabpy`. 4th-order Lele was tested and did
    not suppress the oscillations.
- **Screening defaults.** `ncv=500`, `eps_tol=5e-3`, `nev=20`, `beta=0`,
  `alpha_filter="none"`, `run_stability=true`, `run_resolvent=false`.
- **Targets being pursued** (empirical `M2` fits, re-localisation expected):
  - stationary `~ 0 − 0.83i` (`St` undefined; Re(ω)=0). Anchor recovered
    on `M2`.
  - oscillatory `~ 0.56 − 1.11i` → `St ≈ 0.089`. Not portable to finer
    meshes.
- **Current phase.** Phase 1 (import `Qfilt` dataset) → Phase 1.5
  (baseflow spectrum, optional sanity check) → Phase 2 (M21 screening on
  `Qfilt`, comparison to `Qraw`).
- **Resolvent gate (softened).** As soon as the EVP converges numerically
  on one mean flow at one mesh, resolvent can start in parallel. Full
  cross-mesh tracking (Phase 8) is desirable, not a hard prerequisite.
- **Open blockers:** all prior blockers are now resolved or downgraded to
  TODOs below. Previously outstanding items (geometry, M21 asymmetry,
  Qfilt spec, γ value) are now documented or decided.
- **Action note:** `parity_config.jsonc` has been updated to `γ=1.31`. All
  results produced before 2026-04-17 used `γ=1.4` and are not directly
  comparable to subsequent runs without recomputation.
- **Next deliverable.** One `M21 + Qfilt` localization campaign covering
  the stationary and oscillatory pockets under the provisional
  Acceptance Criteria below.

---

## Acceptance criteria (provisional)

We are in uncharted territory; these thresholds are **initial working
gates**, not calibrated ground truth. Revisit once Phase 2 data arrives.

For a run to count as a **successful localization** of an intended family:

- `stability_postcheck_residual_rel` median on accepted modes `≤ 1e-4`;
- target-distance median `≤ 0.1 · |omega_target|`;
- for the oscillatory branch: at least one accepted mode with
  `|Re(omega)| > 0.2`;
- `accepted_count ≥ 1`.

For a branch to count as **preserved** across a knob change (mesh, target
perturbation, `eps_tol`, `ncv`, solver config, `fd_order`):

- returned `omega` shifts by `≤ 5 %` relative to the reference run;
- returned eigenvector stays in the same spatial pocket (diagnostic to be
  made precise).

All numbers are provisional. The point of writing them down is so later
tuning is a visible, dated update, not silent drift.

---

## What we know so far

| # | Setup | Result | Verdict |
|---|---|---|---|
| 1 | `M2`, target `0 − 0.85i`, nev=6 | accepted `0 − 0.8306i`, postcheck `3.4e-6`, target dist `1.9e-2` | stationary pocket recoverable on `M2` |
| 2 | `M2`, target real `~6.28`, nev=20/50 | many modes, postcheck `~2e-5`, target dist `~6–7`, accepted=0 | converged but wrong physics |
| 3 | `M2`, oscillatory target exact vs offset | exact `0.561 − 1.112i` localises best | OK on `M2` |
| 4 | `M1`, inherited osc target, ncv=? | stalled ~22 ks; `nconv=0`; min monitor error `2.6e-4` | `M2→M1` transfer of target fails |
| 5 | `M21`, inherited osc target, ncv=500, tol=1e-4 | converged 1 iter, returned `0 − 0.82i`, `0 − 1.59i`; postcheck `7.9e-6`; target dist `0.69`; accepted=0 | `M21` numerically stable, but target no longer localises the osc family |

Full artifact paths are in `output/<campaign>/runs/*.npz` and
`output/<campaign>/logs/*.log`.

**Headline conclusions.**

- `M2` recovers the stationary family.
- The `M2` oscillatory target is not portable to finer meshes.
- `M21` with `ncv=500` is numerically tractable; target portability is
  the open question.
- Mean-flow quality must be treated as a primary campaign variable (see
  Framing §3 in `REFERENCE.md`).

---

## Phase plan

Status legend: ✅ done · 🔄 in progress · ⏸ not started · 🚧 blocked.

### Phase 1 — Import `Qfilt` dataset ⏸

`Qfilt` = Lui LES mean after **Lele (1992) 2nd-order compact filter**,
produced upstream of `linstabpy`. Import as a separate baseflow dataset
alongside `Qraw`; switch via `paths.case_dir`. Record filter provenance
(source, filter order, cutoff parameter) in the dataset metadata. No
filter machinery is added to `linstabpy` itself.

### Phase 1.5 — Characterise the baseflow before filtering ⏸

Cheapest diagnostic before committing to `Qfilt`:

- wall-normal spectrum of `u` at a few streamwise stations;
- check for high-k content above the `M2` Nyquist.

If `Qraw` is smooth at the relevant scales, `Qfilt` may not be the right
axis of attack at all.

### Phase 2 — M21 screening of `Qraw` vs `Qfilt` ⏸

For both mean-flow variants, localise stationary and oscillatory pockets
on `M21` with the screening defaults at the top of this file.

`eps_tol = 5e-3` is deliberate: Gabriel's evidence says forcing `1e-4` too
early on fine meshes may block progress. Locate first, tighten later.

### Phase 3 — Re-localize targets on M21 ⏸

Do not assume the `M2` oscillatory target is still correct. Separate
sweeps for:

- stationary family near `0 − 0.83i`;
- oscillatory family near `0.56 − 1.11i`.

For the oscillatory branch, check `|Re(omega)| > 0.2` explicitly.

### Phase 4 — Tolerance sweep on the winning mean flow ⏸

Once the right pocket is found on `M21`, vary `eps_tol ∈ {5e-3, 1e-3,
5e-4, 1e-4}`. Smallest practical tolerance that still preserves the
correct family (per Acceptance Criteria).

### Phase 5 — `ncv` sweep ⏸

With mean flow and target fixed, vary `ncv ∈ {200, 300, 500, 700}`.
Smallest subspace that preserves the correct family.

### Phase 6 — PETSc / SLEPc configuration sweep ⏸

Freeze mean flow, target, `eps_tol`, `ncv`. Sweep `st_solver`,
`petsc_matrix_mode`, `conv_test`, `true_residual` one at a time. Does the
family stay in the same pocket? Is what we see a physical branch or a
solver-selection artifact?

### Phase 7 — Stencil-order sweep ⏸

With `M21` setup frozen, test `fd_order ∈ {4, 6, 8}`. A family that only
appears for one stencil order is not publication-quality evidence.

### Phase 8 — Promote to M1 (cross-mesh tracking desirable, not gating) ⏸

Move the winning setup to `M1`. A demonstration that the same branch
appears at `M2`, `M21`, and `M1` within Acceptance Criteria tolerances is
the ideal deliverable for the paper; absence of cross-mesh consistency
would have to be acknowledged honestly in the write-up.

Softened: per current campaign decision, this phase does not block
resolvent. Once the EVP converges numerically on one mean flow at one
mesh (e.g. `Qfilt` on `M21` or `M1`), resolvent can start in parallel.

### Phase 9 — Activate mean-flow cleanup ⏸

Only after a stable `M1` baseline exists. Two routes, in order of
preference (see `REFERENCE.md` Framing §3):

1. mean-flow low-pass filter (Visbal–Gaitonde-type) of `Ū`;
2. eddy-viscosity augmentation of `L_Ū` (frozen `ν_t`).

**Do not** call either "Ducros". Ducros is an LES-time shock-capturing
sensor, not a post-hoc mean-flow smoother.

### Resolvent — as soon as EVP converges numerically ⏸

The novelty target. Gate softened (see at-a-glance): start once any
`Qfilt`-based EVP converges and passes Acceptance Criteria. Cross-mesh
tracking (Phase 8) runs in parallel rather than in series.

---

## Open TODOs

| Owner | TODO | Blocks |
|---|---|---|
| you | **Deliver `Qfilt` dataset** (Lele 1992 2nd-order compact filter applied to Lui mean flow). Place under `results/case2/` with metadata recording filter provenance. | Phase 1, Phase 2 |
| you (optional) | **Physics-based Strouhal seed.** If Lui 2022 or separation-bubble breathing literature gives an expected `St` for the oscillatory family, record it in `REFERENCE.md` "Expected qualitative mode structure". Would replace the M2-empirical target seed. | Phase 3 seeding |
| me | **Baseflow spectrum diagnostic** (Phase 1.5). Wall-normal `u` spectra at 3–4 streamwise stations on `Qraw` and `Qfilt`. Mostly a sanity check now that we know `Qraw` is contaminated — confirm `Qfilt` looks clean. | Confidence in Phase 2 |
| me | **Branch-identity diagnostic.** A clustering or mode-overlap metric on eigenvectors to operationalise "same spatial pocket" in Acceptance Criteria. | Phases 4–8 |
| me | **Recompute / archive `γ=1.4` artefacts.** Past `output/` runs used the wrong `γ`. Mark them in a `legacy/` note and do not overlay on new `γ=1.31` results. | Comparability hygiene |

---

## What we should not do next

- do not keep pushing `M1` with the old oscillatory target without
  re-localizing;
- do not use `alpha_filter` as a proxy for mean-flow filtering;
- do not judge success by `nconv` alone;
- do not mix target, PETSc, and stencil-order changes in the same
  comparison;
- do not claim branch robustness if it exists only for one arbitrary
  `fd_order` or one arbitrary PETSc configuration;
- do not activate mean-flow filtering or eddy-viscosity augmentation
  before one unfiltered baseline is stable enough to be understood;
- do not call a post-hoc mean-flow smoother "Ducros" — Ducros is an
  LES-time shock-capturing sensor.
