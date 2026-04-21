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
5. **`tutorial.md`** — step-by-step walk-through.

Supporting files in this directory:

- `parity_config.jsonc` — canonical full-resolution configuration;
  `Re=3.95e5`, `Ma=2.0`, `Pr=0.747`, `T₀=1540 K`, `γ=1.4`, `β=0`.
- `case2.py` — single-case entry point.
- `perf_memory.py` — campaign orchestrator.
- `run_case_once.sh`, `run_perf_queue_once.sh` — wrappers.
- `evp_stride*.jsonc` — per-experiment configs (mesh × target × solver).
- `output/<campaign>/runs/*.npz`, `logs/*.log` — artifacts.
- `baseflow/` — active mean-flow dataset (`Qraw`).

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
- **Active mesh.** `M2 = (stride_ksi=2, stride_eta=2)` for screening,
  `M1 = (stride_ksi=1, stride_eta=1)` for final. **Asymmetric strides
  are retired from the campaign** as of 2026-04-18 — in a body-fitted
  curvilinear mesh, decimating one computational axis but not the
  other produces cells whose physical aspect ratio depends on
  location (because the wall curves; the Jacobian varies along the
  wall). That is not a natural coarsening of the physical problem;
  it is an artefact. Therefore only symmetric strides are used:
  `stride_ksi == stride_eta` always. `M21 = (2, 1)` and `M12 = (1, 2)`
  are no longer valid. This also moots the stride-name swap
  (F2 / I1) for campaign purposes — when both strides are equal,
  which axis each name decimates is immaterial — but the naming
  inconsistency in the loader remains documented (see F2) so that
  future readers who try asymmetric strides do not get surprised.
- **Active mean flows.**
  - `Qraw` = Lui LES mean, unfiltered. **Known to be contaminated** —
    contaminated modes have been observed.
  - `Qfilt` = Lui LES mean after **Lele (1992) 2nd-order compact filter**,
    applied upstream of `linstabpy`. 4th-order Lele was tested and did
    not suppress the oscillations.
- **Screening defaults.** `ncv=500`, `eps_tol=5e-3`, `nev=20`, `beta=0`,
  `alpha_filter="none"`, `run_stability=true`, `run_resolvent=false`.
- **Targets being pursued** (empirical `M2` fits, re-localisation
  **required** — see F6 below):
  - stationary `~ 0 − 0.83i` (`St` undefined; Re(ω)=0). On M21/γ=1.31
    the closest eigenvalue is at (0, -0.683i), i.e. shifted 17% from
    the γ=1.4 anchor and still failing the target-distance gate.
  - oscillatory `~ 0.56 − 1.11i`. On M21/γ=1.31 **there is no
    eigenvalue in a neighbourhood of (0.56, -1.11i)**. The nearest
    returned modes sit at (0, -0.683i) and (0, -1.580i), at complex
    distance ~0.7. The operator's actual oscillatory conjugate pairs
    live at much higher frequency (`|Re ω|` = 0.79 up to 4.9) with
    much stronger damping (`|Im ω|` = 5.4 to 9.8). Phase 3
    re-localisation from a spectral topology map (I6) is mandatory
    before any target seed can be claimed.
- **Current phase.** Phase 1.5 baseflow diagnostic (done on Qraw
  2026-04-18, see F3 below) → Phase 1 (import Qfilt, awaiting
  upstream delivery + provenance documentation I5) → Phase 2 (M21
  screening Qraw vs Qfilt, Qraw side already run as reference in
  `worklog.md` Entry 8 — see F6).
- **Resolvent gate (softened).** As soon as the EVP converges numerically
  on one mean flow at one mesh, resolvent can start in parallel. Full
  cross-mesh tracking (Phase 8) is desirable, not a hard prerequisite.
- **Open blockers (new, 2026-04-18):** I1 (stride-name swap, doc or
  code decision), I5 (Qfilt pipeline provenance), I6 (spectral topology
  map for target re-localisation), I7 (apparently-unstable Mode 3 —
  blocks any stability claim). Detailed in the Findings section below.
  Previously outstanding items (geometry, M21 asymmetry interpretation,
  γ value) are now documented — the "M21 asymmetry" item turned out
  to be the stride-name swap itself.
- **Action note:** `parity_config.jsonc` has been updated to `γ=1.31`. All
  results produced before 2026-04-17 used `γ=1.4` and are not directly
  comparable to subsequent runs without recomputation.
- **Next deliverable.**
  1. Rerun the Phase-2 screening on **M2 (symmetric, stride 2,2)**.
     All asymmetric-stride artefacts from the 2026-04-18 Qraw sweep
     are superseded; they live in `worklog.md` for historical record
     but do not enter the paper.
  2. Receive Qfilt with provenance (I5).
  3. Then: side-by-side Qraw-vs-Qfilt on M2 using the already-built
     screening infrastructure (`phase1p5_*.py`,
     `QFILT_COMPARISON_PROTOCOL.md`).
  4. In parallel, run the spectral topology map (I6) on M2 Qraw to
     re-localise targets honestly for Phase 3, and run the Mode-3
     investigation (I7) to resolve the stability question on M2.

---

## Research strategy — how this becomes a paper

Added 2026-04-18. The previous phase plan is waterfall-shaped
(Phase 1 → 2 → ... → 9); that creates a single failure point at every
stage. This section reframes the work as **parallel tracks** with
**tier-ed commitments** so that the paper succeeds even if some
tracks stall.

### What the paper actually needs

The LinStabPy framework paper (`paper/brief.tex` + `paper/supp.tex`,
now in draft) already has Case 1 parity and Robinet/Song SWBLI
validation. Case 2 is the **application showcase**. The minimum
publishable Case 2 contribution is:

1. One spectrum panel on Case 2 M2 (Qraw **or** Qfilt) with honest
   classification of the returned modes (wall-local vs global,
   contamination vs physical);
2. One resolvent gain curve `σ₁(ω)` over a frequency range, with the
   SVD extraction implemented through the same PETSc/SLEPc stack;
3. One pair of optimal forcing / optimal response mode shapes at the
   gain peak;
4. Honest discussion of the DNS-on-LES-mean consistency gap
   (REFERENCE §3) and the filter-based cleanup strategy.

Anything beyond this is stretch. The paper does **not** require
finding the "correct" oscillatory branch at a pre-chosen target; it
requires the method to produce *defensible* results on a non-trivial
compressible SWBLI mean flow.

### Tiered goals

| Tier | Commitment | Content |
|------|------------|---------|
| T1 (floor) | **COMMIT** | 1 spectrum + 1 resolvent gain + 1 F/R pair on Case 2 M2 Qfilt (or Qraw fallback) |
| T2 (stretch) | **PURSUE** | M1 cross-mesh check + Mode-3 resolution (I7) + multiple branch families |
| T3 (moonshot) | **OPPORTUNISTIC** | Full spectral topology map + physical branch attribution (shock-breathing vs BL-pulsation) + unstable mode if real |

### Why resolvent is the robust path

Shift-invert EVP picks the **densest cluster of eigenvalues** near
the target — that is why the M2/γ=1.4 seed failed on M21/γ=1.31
(F6). Resolvent analysis picks **`ω` itself** and computes
`σ(ω) = ‖(iωI − L)^{−1}‖`. There is no attractor problem; any
chosen `ω` gives a gain and a forcing/response. The paper's
novelty is resolvent-side — run it on a grid of ω values and report
the gain curve.

**Corollary:** resolvent is not gated on EVP re-localisation success.
As soon as the MUMPS shifted-solve converges at one `ω` on
Qfilt M2, the paper's tier-1 resolvent deliverable is reachable.

### Physics priors not yet used (free signal)

- **Lui 2022** (the source LES): the suction-side SWBLI has a
  dominant unsteady frequency. That Strouhal number should seed the
  resolvent `ω` scan. Gabriel: please extract it from the Lui 2022
  paper and put it in `REFERENCE.md` under "Expected qualitative
  mode structure" (blocked on that today, flagged as you-TODO).
- **SWBLI literature** (Dussauge 2006, Touber & Sandham 2009, Humble
  et al. 2009): typical St of the low-frequency shock breathing is
  `O(0.02–0.05)`, corresponding to `|Re ω| ≈ 0.13–0.3` in code
  units. The M21 screening returned oscillatory pairs at
  `|Re ω| = 0.79, 1.07, 1.98, 2.82, 3.50, 4.84+`. Only the lowest
  is close to the Dussauge range; **the rest are at BL-acoustic or
  shock-resonance frequencies, not low-frequency breathing**. This
  is a real physical observation to lean on.

### Kill criteria / pivot triggers

- **If Qfilt delivery is delayed > 1 week past today (2026-04-18)**:
  pursue T1 on **Qraw M2** with explicit mean-flow-contamination
  framing. The paper still has a story ("here is what unfiltered
  Case 2 reveals; filtering is route-3 cleanup per Mettot 2013").
- **If resolvent fails to converge on Qfilt M2 after ~5 days of
  tuning**: fall back to (a) Qraw M2 resolvent, (b) eddy-viscosity
  augmentation (REFERENCE §3 route 1, previously declared
  out-of-scope — reconsider as tier-3 if needed for convergence).
- **If Mode 3 (0, +3.17i) turns out to be a genuine instability**:
  that is a **paper strength, not failure** — reframe Case 2 as
  demonstrating the tool's ability to surface a previously-unresolved
  unstable mode at γ=1.31 absent at γ=1.4.

### Strategy for Phase 2 onward (revised)

- **EVP track and resolvent track run in parallel** as soon as Qfilt
  lands. Resolvent starts immediately without waiting for "correct"
  EVP target localisation (F6 consequence).
- **EVP re-localisation uses the empirical spectral topology map
  (I6)** plus physics priors above, **not** the M2/γ=1.4 seed.
- **Diagnostics track runs continuously** — classifier (I3), 2D-FFT
  metric (I2), Mode-3 tests (I7) — so that every EVP and resolvent
  run comes with a classification of what family the mode belongs to.
- **Paper figures are generated from the M2 runs as they arrive**, not
  held back for "final results" — this lets us know on day 3 whether
  the figures are compelling enough to support T1.

### Consolidated task dependency map (tracks)

```
Track A — EVP validation (sequential within track)
  I0  rerun Phase 2 screening on M2 Qraw ────────┐
  I5  Qfilt provenance [you]                     │
  A1  Import Qfilt as dataset (paths.case_dir)  ─┤
  A2  M2 screening Qfilt                         │
  A3  re-localize from empirical topology (I6)   ├──→ T1 spectrum panel
  A4  tolerance / ncv / fd_order sweep            │
  A5  M1 promotion (T2)                          │
                                                 │
Track B — Resolvent (parallel to A, gated only on Qfilt)
  B1  resolvent ω-scan on Qfilt M2 ──────────────┼──→ T1 gain curve
  B2  F/R modes at gain peak                     ├──→ T1 F/R pair
  B3  resolvent at Dussauge-range frequencies     │
                                                 │
Track C — Diagnostics (continuous, feeds A and B)
  I2  2D-FFT metric                              │
  I3  Chu-energy branch classifier               │
  I6  Spectral topology map (needs I0)           │
  I7  Mode-3 resolution (needs I0 + I3)          ┘

Track D — Paper (continuous)
  D1  T1 figure set (start as soon as any M2 result lands)
  D2  T2 supplementary material
```

### Commit order if compute is limited

If we can only run 3 things next, run them in this order:

1. **I0** (M2 symmetric screening Qraw) — already running as
   task #19. 20-min wall time. Baseline for everything else.
2. **B1/T1-start** (Qraw M2 resolvent at 1 physically-seeded ω) —
   confirms the tool works end-to-end on Case 2 and gives us the
   first T1 artefact.
3. **I6** (topology map on M2 Qraw) — overnight. Defines where
   EVP re-localisation should land.

Everything else is downstream of these three.

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
| 6 | **γ=1.31 reproduction of row 5**: asymmetric `M21` Qraw, nev=2, ncv=500, tol=1e-4, osc target | 2 modes returned: `0 − 0.683i` and `0 − 1.580i`; postcheck `~6e-5`; target dist `0.71`; accepted=0. Both modes wall-localised at mean-flow gradient peaks | γ shift moved the stationary-axis pocket by 17% (0.82→0.68); qualitative contamination picture unchanged. **Superseded by 2026-04-18 asymmetric-stride retirement; historical only — rerun on symmetric M2 required** |
| 7 | **γ=1.31 Phase 2 screening on asymmetric M21**: Qraw, nev=20, ncv=500, tol=5e-3, both stat + osc targets | 64 and 74 Ritz values converged. Returned 20 per case — the same 20. Stationary axis: (0, -0.68), (0, -1.58), (0, -4.59), (0, -8.92), **(0, +3.17) unstable**. Conjugate pairs at `|Re ω|` = 0.79–4.9 with `|Im ω|` = 5.4–9.8. Nothing near (0.56, -1.11i) | **Superseded — asymmetric-stride mesh retired 2026-04-18.** Rerun on symmetric M2 before any of these numbers are used. Historical findings retained in `worklog.md` Entries 5–8 and reviewed in F4–F7 below, with the caveat that they were obtained on a mesh we no longer use |

Full artifact paths are in `output/<campaign>/runs/*.npz` and
`output/<campaign>/logs/*.log`.

**Headline conclusions.**

- `M2` at γ=1.4 recovered a stationary eigenvalue at (0, -0.83i);
  the γ=1.31 screening on M21 shows this pocket is really at
  (0, -0.683i), shifted 17%.
- Contaminated modes have concrete physics: they peak at mean-flow
  gradient maxima (shock foot `|∂Ū/∂x|`, recovery wall `|∂Ū/∂y|`)
  and are wall-localised. Shift-invert is attracted to the dense
  `Re(ω) = 0` cluster they form.
- The M2/γ=1.4 oscillatory target (0.56, -1.11i) has **no
  eigenvalue** in its neighbourhood on M21/γ=1.31. The real
  oscillatory families of the operator sit at much higher frequency
  (`|Re ω|` ≥ 0.79). Re-localisation from a topology map (I6) is
  mandatory.
- On the (now-retired) asymmetric M21 mesh, `ncv=500` at
  `eps_tol=5e-3` was numerically rich — converged 64+ modes per run;
  tighter `eps_tol=1e-4` gave only 2–4 modes per run. The same
  numerical behaviour should be confirmed on M2 (symmetric)
  before use.
- The screening spectrum contains an apparently **unstable** mode
  at (0, +3.17i). Residuals at machine precision and Dirichlet leak
  nil rule out obvious numerical artefact. Resolving this (I7) is a
  prerequisite to any paper claim of global stability.
- Mean-flow quality is confirmed as a primary campaign variable, but
  the Lele F2 filter the upstream pipeline is expected to use only
  attenuates shock-region `|∂Ū/∂x|` (7–20% reduction) — it does
  not touch the near-wall `|∂Ū/∂y|_max` because the Dirichlet wall
  value is held fixed. I5 must clarify the upstream pipeline before
  a null Qfilt result is interpreted.

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

### Phase 2 — M2 (symmetric) screening of `Qraw` vs `Qfilt` ⏸

For both mean-flow variants, localise stationary and oscillatory pockets
on `M2 = (stride_ksi=2, stride_eta=2)` with the screening defaults at
the top of this file. **M2 is symmetric**; asymmetric screening meshes
are not used (see at-a-glance note on curved-domain stride policy).

`eps_tol = 5e-3` is deliberate: Gabriel's evidence says forcing `1e-4` too
early on fine meshes may block progress. Locate first, tighten later.

### Phase 3 — Re-localize targets on M2 ⏸

Do not assume the `M2` oscillatory target is still correct. Separate
sweeps for:

- stationary family near `0 − 0.83i`;
- oscillatory family near `0.56 − 1.11i`.

For the oscillatory branch, check `|Re(omega)| > 0.2` explicitly.

### Phase 4 — Tolerance sweep on the winning mean flow ⏸

Once the right pocket is found on `M2`, vary `eps_tol ∈ {5e-3, 1e-3,
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

With the `M2` setup frozen, test `fd_order ∈ {4, 6, 8}`. A family that
only appears for one stencil order is not publication-quality evidence.

### Phase 8 — Promote to M1 (cross-mesh tracking desirable, not gating) ⏸

Move the winning setup to `M1`. A demonstration that the same branch
appears at `M2` and `M1` within Acceptance Criteria tolerances is
the ideal deliverable for the paper; absence of cross-mesh consistency
would have to be acknowledged honestly in the write-up. Only two
meshes are used (M2, M1); asymmetric strides such as `(2, 1)` are
**not** an additional mesh — they are not a natural coarsening of a
body-fitted domain (see at-a-glance).

Softened: per current campaign decision, this phase does not block
resolvent. Once the EVP converges numerically on one mean flow at one
mesh (`Qfilt` on `M2` or `M1`), resolvent can start in parallel.

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
| you | **Deliver `Qfilt` dataset** (Lele 1992 2nd-order compact filter applied to Lui mean flow). Place under `results/case2/` with metadata recording filter provenance, including: filter order, α, number of passes, BC treatment, and whether any resampling/interpolation is applied in the upstream pipeline. See Investigation I5 below. | Phase 1, Phase 2 |
| you (optional) | **Physics-based Strouhal seed.** If Lui 2022 or separation-bubble breathing literature gives an expected `St` for the oscillatory family, record it in `REFERENCE.md` "Expected qualitative mode structure". Would replace the M2-empirical target seed. | Phase 3 seeding |
| me | **Baseflow spectrum diagnostic** (Phase 1.5). Wall-normal `u` spectra at 3–4 streamwise stations on `Qraw` and `Qfilt`. *Partially done on `Qraw` 2026-04-18; see `worklog.md` Entries 1–4. Lessons learned: 1D plateau metric is compromised by DFT non-periodicity. 2D FFT is the informative view.* | Confidence in Phase 2 |
| me | **Branch-identity diagnostic.** A clustering or mode-overlap metric on eigenvectors to operationalise "same spatial pocket" in Acceptance Criteria. | Phases 4–8 |
| — | **Recompute `γ=1.4` artefacts.** All prior `output/` runs (wrong `γ`) have been deleted as part of the 2026-04-18 cleanup. New runs start from `γ=1.31`. | — (resolved) |

---

## Findings from 2026-04-18 (Phase 1.5 campaign on Qraw)

Detailed trail in `worklog.md` (Entries 1–8). The key findings below are
now baked into the campaign plan. Each one has an associated
investigation task (I1–I5) that needs more work before we act on it.

### F1 — Post-load array axis convention (verified three ways)

After `load_hamada_case(case_dir, transpose_input=True)` returns:

- array shape `(ny, nx) = (679, 599)` at M1 native,
- **axis 0** = computational `η` = **physical wall-normal**,
- **axis 1** = computational `ξ` = **physical streamwise along the
  (curved, body-fitted) wall**.

Verified (`worklog.md` Entry 8 addendum):

- `U[0, :]` has mean `|U| = 0.0000` — the Dirichlet wall (no-slip) lies
  exactly at axis-0 index 0.
- axis-0 mesh spacing ratio (max/min) = **470×** (wall clustering is
  along axis 0).
- axis-1 mesh spacing ratio = 4.6× (quasi-uniform, streamwise).
- axis-1 path length at the wall = 0.666 units (streamwise extent of
  the curved surface).
- axis-0 path length at the inlet = 0.476 units (wall-to-freestream
  extent).

The body-fitted mesh follows the curving wall; "wall-normal" rotates
along the wall, but the computational axis 0 always points across the
wall, and the finite-difference stencils Dy operate in computational
(index) space.

### F2 — Loader parameter names `stride_ksi` / `stride_eta` are swapped (documentation only, no campaign impact)

`src/linstab2d/baseflow/case1.py:97–98` applies
`arr[::stride_ksi, ::stride_eta]`. Post-transpose:

- `stride_ksi` **decimates axis 0 = η = wall-normal** (despite being
  named for ξ = streamwise);
- `stride_eta` **decimates axis 1 = ξ = streamwise** (despite being
  named for η = wall-normal).

In `Mesh.create`, the convention is the opposite of what the loader's
parameter names suggest: the mesh treats axis 0 as η and axis 1 as ξ
(verified on a toy 9×11 mesh where `Dx` differentiates along axis 1
and `Dy` along axis 0).

**Campaign impact: none, because only symmetric strides are allowed.**
With the 2026-04-18 decision to use only symmetric strides on the
curved body-fitted mesh (at-a-glance), `stride_ksi == stride_eta` in
every allowed config, and the naming swap becomes observationally
invisible. The documentation is kept so that future readers who
might want to try asymmetric strides are warned immediately.

**Investigation I1 — downgraded to a code-hygiene task.**

Previously proposed doc-only / code-fix / both routes. Because
asymmetric strides are no longer used, the urgency is gone. The
recommended minimal action:

1. Add a prominent comment in `case1.py:97` stating
   "stride_ksi decimates axis 0 (wall-normal post-transpose),
   stride_eta decimates axis 1 (streamwise) — the names are swapped
   relative to the Mesh convention. Use symmetric strides only."
2. Add an assertion in the loader that
   `stride_ksi == stride_eta` for this case family, raising a clear
   error otherwise. This is a cheap guardrail against the exact
   mistake this PLAN's previous M21 interpretation made.
3. Leave the public parameter names alone (historical stability).

**Status of this investigation changed from HIGH to LOW priority.**

### F3 — `Qraw` has shock-wideband content outside the M2 resolvable box (2D FFT)

Wall-normal 1D plateau is partially a DFT non-periodicity artefact
(the wall-to-freestream step in `u(η)` generates a 1/k² tail that
looks like a noise floor on log scale). `worklog.md` Entry 4 details
this mistake and its resolution.

The 2D FFT of `u(ξ, η)` and `T(ξ, η)` tells a cleaner story: the
high-wavenumber content is organised as **two oblique diagonal
ridges** in `(k_ξ, k_η)` space, extending from the origin past the
M2 Nyquist box to the M1 boundary. These ridges are the Fourier
signature of the two oblique shocks visible in the field.

**Investigation I2: quantify shock-ridge content that escapes the
M2 box.** Needed diagnostic:

- integrate `|FFT|^2` inside and outside the M2 Nyquist box;
- report the fraction per variable (ρ, u, v, T);
- do the same on the eventual `Qfilt`;
- compare.

This is the *correct* replacement for the flawed 1D plateau metric.
Artefact `phase1p5/qraw_u_2d_fft.png` already visualises this; needs
a quantitative number.

*Owner: me. Low priority until `Qfilt` arrives.*

### F4 — Contaminated modes peak at mean-flow gradient maxima

*Obtained on the now-retired asymmetric M21 mesh; needs confirmation
on symmetric M2. The qualitative coincidence between mode peaks and
mean-flow gradient peaks is mesh-independent physics and likely
survives, but exact eigenvalues will shift.*

On M21 Qraw (Entry 5):

- Mode 0 at `ω = 0 − 0.683i`, peak at (x, y) = (0.818, -0.099).
- Mode 1 at `ω = 0 − 1.580i`, peak at (x, y) = (0.962, -0.150).

Both wall-attached, 2.3% / 7.1% of gridpoints have `|u| > 0.5·peak`.
Direct coincidence with mean-flow gradient peaks:

- `|∂Ū/∂x|_max` on M21 is at **(0.815, -0.092)** — the shock foot on
  the wall. **Matches Mode 0 peak** within one cell.
- `|∂Ū/∂y|_max` on M21 is at `(0.969, -0.154)`, j=0 row (wall). Mode
  1 sits nearby at `(0.962, -0.150)`.

Physical reading: the LNSE operator assembles coefficients dominated
by local mean-flow gradients. Where those gradients are extreme
(shock foot, near-wall recovery), the operator has locally-trapped
damping modes on the `Re(ω) = 0` axis. Shift-invert Krylov-Schur is
attracted to those dense clusters and returns them instead of
whatever globally-propagating mode the target was intended to find.

**Investigation I3: branch-identity classifier.** Need a systematic
way to distinguish "wall-local shock-driven mode" from "propagating
global mode". Candidate metrics:

- `frac_|u|>threshold` over domain area (wall modes ≪ 10%);
- centroid location relative to shock foot / wall (wall modes within
  δ of wall-strip);
- autocorrelation length of `|u|(x, y)` (wall modes short, global
  modes long);
- phase winding along streamwise direction (propagating modes have
  non-zero phase winding; wall-trapped modes have ~zero).

Artefact `phase1p5/qraw_gradient_vs_mode_peaks.png` shows this
coincidence visually. A quantitative classifier blocks Acceptance
Criteria's "same spatial pocket" test for Phases 4–8.

*Owner: me. Priority once Phase 2 Qfilt comparison reveals multiple
mode families.*

### F5 — A simple Lele F2 compact filter does NOT reduce `|∂Ū/∂y|_max`

*Measured on the asymmetric M21 mesh. The BC-preservation argument
that drives this result is mesh-independent; numbers will shift
slightly on symmetric M2 but qualitative finding holds.*

Built a Python Lele 1992 F2 / F4 filter as an independent reference
(diagnostic script, not in linstabpy core —
`results/case2/phase1p5_lele_filter_reference.py`). Applied to
Qraw on M21, then compared gradient peaks:

| filter | `|∂Ū/∂x|_max` | `|∂Ū/∂y|_max` |
|---|---|---|
| Qraw | 49.2 | 1460 |
| F2 α=0 one pass | 45.5 (−7.7%) | 1456 (−0.2%) |
| F2 α=0 five passes | 38.9 (−21%) | 1448 (−0.8%) |

The `|∂Ū/∂y|_max` is untouched because it sits at the first interior
cell adjacent to the Dirichlet wall; my local filter passes through
the wall value to preserve `U(wall) = 0`. The upstream Qfilt pipeline
must do one of: (a) filter before Dirichlet enforcement, (b) use
non-pass-through BC, (c) resample / interpolate in the pipeline.

**Investigation I5: upstream Qfilt pipeline specification.** Before
drawing conclusions from a future Qfilt comparison, document:

- exact filter formula (F2, F4, other) with α value;
- order of operations: filter → apply BC, or apply BC → filter;
- whether resampling happens (if yes, from what grid to what grid);
- whether the filter is applied per-variable or per-component;
- number of passes.

Without this, a null result (Qfilt unchanged from Qraw in EVP
outcome) can be misdiagnosed as "filtering doesn't help" when it
might be "the filter didn't touch the feature that drives the
contamination".

*Owner: you (upstream pipeline provenance). Blocks meaningful Phase 2
comparison.*

### F6 — PLAN's seed oscillatory target (0.56 − 1.11i) has no eigenvalue there on M21 / γ=1.31

*Screening run was on the asymmetric M21 mesh. The finding that the
PLAN seed is not where an eigenvalue lives is likely robust to mesh
choice, but the exact spectrum reported here must be re-measured on
M2 before it enters the paper.*

Screening run (nev=20, ncv=500, eps_tol=5e-3) on Qraw M21 converged
64–74 Ritz values in one iteration. The 20 returned for each target
are essentially the same 20 eigenvalues of the operator (modulo
conjugate-pair sign):

- stationary axis discrete modes: (0, -0.68), (0, -1.58), (0, -4.59),
  (0, -8.92);
- stationary axis "unstable" mode: **(0, +3.17)** — see F7;
- oscillatory conjugate pairs at `|Re ω|` in {0.79, 1.07, 1.98, 2.82,
  3.50, 4.84, 4.86, 4.90}, with `|Im ω|` in 5.4–9.8;
- nothing near (0.56, -1.11i).

The M2/γ=1.4 target position is **empty** on M21/γ=1.31.

Two implications for the campaign:

- **PLAN Phase 3 (re-localisation) is mandatory**, not an optional
  refinement. The M2/γ=1.4 oscillatory seed is not portable as-is.
- **Both targets attract the same mode cluster** at screening tol.
  Target-directed exploration requires (i) sufficient spatial
  separation between targets OR (ii) tighter `eps_tol` + smaller
  `nev`. The low-target-density plot in
  `phase1p5/screening_*_spectrum.png` shows this directly.

**Investigation I6: spectral topology map.** Before picking any new
target, sweep a coarse grid of targets over the relevant complex
region:

- Re(ω) ∈ {0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0};
- Im(ω) ∈ {-0.5, -1.0, -2.0, -4.0, -6.0, -8.0}.

Each run nev=2-4, tol=1e-3 so they complete in ~10 min each. Union
the returned eigenvalues: that is the empirical spectrum topology
of the operator. Choose the re-localised oscillatory seed from that
map, with a physics argument for why it's the "right" mode (shape
localisation, frequency matching, etc.).

*Owner: me. Cost: ~(9·6)·10 min = 9 hours on one socket. Run
overnight. Blocks Phase 3.*

### F7 — Apparently-unstable mode at `ω = 0 + 3.17i` on Qraw M21 γ=1.31

*Observed on the asymmetric M21 mesh. The very first action of
investigation I7 is therefore to check whether this mode survives on
symmetric M2 — if it does not, it was an asymmetric-stride artefact
and the stability question is unchanged from PLAN's original
assertion.*

Screening run also returned a mode with **Im(ω) > 0**. Under the
code's MATLAB convention (`ω = eig(i·L0)`), `Im(ω) > 0` corresponds
to a growing perturbation. This is the only non-wall-localised mode
in the 20 returned — it shows global diagonal-band structure across
the domain, reminiscent of a trapped acoustic mode in the
shock-cavity region.

Diagnostics:

- `postcheck_residual_rel` = 8.1e-5 (passes loose gate);
- `stability_residual_relative` = 2.6e-5 (solver residual);
- `dirichlet_leak_max` = 3e-15 (machine precision);
- `leading_weighted_norm_error` = 3e-15 (Chu-norm consistent).

These exclude "obvious numerical artefact". But with a contaminated
mean flow, a finite residual does NOT guarantee physical validity.

**Three possibilities, all requiring investigation:**

- (a) Case 2 actually has a globally unstable mode at γ=1.31 that
  was missed under γ=1.4;
- (b) Mode 3 is an **operator-level contamination** artefact: the
  DNS-linearised operator on the LES-mean has a spurious unstable
  mode that does not correspond to any physical instability of the
  underlying flow;
- (c) The mode was already present at γ=1.4 but in a region of the
  spectrum not probed by past runs.

**Investigation I7: resolve the Mode-3 question.** Concrete tests,
in order of information / cost:

1. Run same screening on M2 at γ=1.31 (both axes halved). If
   Mode 3 disappears → mesh-dependent artefact. If Mode 3 persists
   at similar ω → real mode.
2. Run same screening on M1 at γ=1.31 if tractable. Same logic.
3. Compute the mode's Chu energy localisation: spatial distribution
   of energy norm, not just |u|. A physical acoustic mode shows
   characteristic pressure-velocity coupling; a spurious mode
   doesn't.
4. Run on `Qfilt` when delivered. If Mode 3 disappears on Qfilt →
   mean-flow-gradient contamination (route (b)); if it survives →
   probable real mode.
5. Check β-sweep: does Mode 3 appear at β ≠ 0? If it's spanwise-mode
   sensitive, it's likely a cavity resonance (real physics); if it's
   β-insensitive, could be numerical.

*Owner: me. Priority HIGH — an unstable mode reframes the entire
paper narrative about Case 2. Must be resolved before any publication
claim of stability.*

### Summary of investigations required

| ID | Topic | Owner | Blocks | Priority (2026-04-18) |
|----|-------|-------|--------|-----------------------|
| I0 | **Rerun the Phase 2 screening on symmetric M2** (stride 2,2) to replace the retired asymmetric-M21 artefacts | me | everything below | HIGH — prerequisite |
| I1 | Stride-name swap: add guard-rail comment + `stride_ksi == stride_eta` assertion in loader | me (cheap) | documentation only | LOW (symmetric-only policy already prevents the mistake) |
| I2 | 2D-FFT quantitative metric | me | Qfilt comparison quality | LOW until Qfilt lands |
| I3 | Branch-identity classifier | me | Phases 4–8 acceptance | MEDIUM |
| I5 | Qfilt pipeline provenance | you | Phase 2 interpretation | HIGH (blocks Phase 2) |
| I6 | Spectral topology map on symmetric M2 | me | Phase 3 re-localisation | HIGH |
| I7 | Mode-3 resolution — first step is "does it survive on M2?" | me | paper claim of stability | HIGH |

All findings (F1–F7) and investigations (I0–I7, excluding the
vacated I4) are logged in `worklog.md` with raw numbers,
artefact paths, and commit-ready scripts. This PLAN.md change is
the summary; `worklog.md` is the record.

The 2026-04-18 campaign pivot is: **M2 symmetric is the only valid
screening mesh**. All M21 artefacts are superseded. Resume Phase 2
as soon as I0 (M2 screening rerun) and I5 (Qfilt provenance) are in
hand.

---

## Wall-trapped modes — mitigation roadmap (added 2026-04-20)

**Empirical finding** (consolidated 2026-04-20). 20–35 % of every M2
screening batch classifies as `WALL_LOCAL` (Chu-energy localisation,
`phase1p5_classify_modes.py`). Only one genuine `GLOBAL` oscillatory
mode survives on M2 Qraw: `ω = (±1.7416, −6.317i)`, `St = 0.277`. The
loose-tolerance "stationary mode" at `ω = (0, −0.6948i)` is *not*
converging under tight EVP (`eps_tol = 1e-8`, `postcheck_rel_max = 1e-6`):
the iteration plateaus around `err_range ~ 5–10` from SLEPc iter 100
onward (tight-certify run 2026-04-20). This is exactly the signature
of a wall-local cluster: loose tolerance rounds a non-isolated near-axis
eigenvalue into the "accepted" bucket; tight tolerance unmasks it.

The resolvent σ₁(ω) curve is **not** affected — resolvent SVD integrates
over the whole domain in the Chu norm, and the certificate at ω = 0.40
gives `ρ_adj = 2.4e-10` at `eps_tol = 1e-8`. So the pollution is
**EVP-only**.

### Mitigations — tiered by effort × impact

| # | Approach | Status | Effort | Expected impact | Priority |
|---|---|---|---|---|---|
| M1 | Baseflow filtering (real Qfilt, upstream delivery) | blocked on I5 | external | **Root cause.** Smoothing `∂Ū/∂y` peaks removes the coefficient inflation that breeds wall-local clusters. Lele F2/F4 stand-in barely moved the spectrum (F5); real Qfilt is the live experiment. | HIGH |
| M2 | Chu-localisation classifier + reject | ✅ done (I3) | — | Gatekeeper, not a fix — the modes still exist, we just don't publish them. | — |
| M3 | **Contour-integration eigensolver (FEAST / Beyn)** in place of shift-invert Krylov–Schur | not started | MEDIUM–HIGH | Extracts *all* eigenvalues inside a user-drawn contour. Draw a contour that excludes the dense wall-local axis. No "closest-to-target" attractor. SLEPc has a FEAST wrapper (`EPS_FEAST`) — can prototype without rewriting the operator. | HIGH (user priority, 2026-04-20) |
| M4 | Energy-weighted generalized EVP: solve `L q = ω M q` with `M` = Chu-energy weight | not started | MEDIUM (operator refactor) | Residuals and convergence criteria measured in Chu-energy norm → de-emphasises modes whose energy is concentrated in few cells. Cleanest theoretical answer. | MEDIUM |
| M5 | Sponge / absorbing strip adjacent to wall + shock foot | not started | SMALL | Classical global-stability trick. Wall-local modes acquire large `\|Im ω\|` and fall out of any reasonable target window. Risk: introduces non-physical behaviour if the strip is too thick — must be validated against a known-clean case. | MEDIUM |
| M6 | Mesh refinement in wall/shock regions | — | — | **Won't help.** Wall-local modes are a property of the continuous operator, not discretisation error. Finer mesh often produces *more* of them. | avoid |
| M7 | Different physical BC (e.g. stress-free wall) | — | — | **Don't.** Changes the physics. Not an option for SWBLI. | avoid |

### What we recommend for the paper (Case 2 scope)

1. Report the classifier as a mandatory post-filter: EVP results are
   framed as "only `GLOBAL`-class modes enter physics discussion."
2. Frame the wall-local axis as a **physics finding**, not a failure:
   "sharp mean-flow gradients at the shock foot and recovery wall
   generate a dense axis of wall-trapped damping modes that dominate
   shift-invert searches; this motivates (i) Chu-localisation
   classification as post-filter and (ii) baseflow filtering (Qfilt,
   ongoing) as a physical mitigation."
3. List M3 (FEAST) and M4 (energy-weighted EVP) as **future work** —
   beyond the framework-paper scope, target of the Case 2 follow-up.

### Probe experiment (scoped, paper-adjacent)

Before scheduling a full FEAST campaign, run a single proof-of-concept:

- Target: the one genuine `GLOBAL` mode `ω = (1.7416, −6.317i)`.
- Contour: small rectangle around this eigenvalue that **excludes** the
  `Re(ω) = 0` axis where wall-local modes accumulate.
- Deliverable: "FEAST finds the mode cleanly; Krylov–Schur on the same
  target has to fight a wall-local cluster." Single plot, single
  certificate. Paper-appendix material.

*Owner: me. Priority: MEDIUM — runs after tight-EVP certification
results land (task #26) and after the paper correction to `supp.tex`.*

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
  LES-time shock-capturing sensor;
- do not use asymmetric strides (`stride_ksi != stride_eta`). A
  body-fitted curvilinear mesh coarsened in one computational
  direction but not the other has a cell aspect ratio that varies
  along the curving wall — that is not a natural coarsening of the
  physical problem. Only `M2 = (2, 2)` and `M1 = (1, 1)` are valid
  campaign meshes;
- do not claim Case 2 is globally stable until I7 (Mode-3 question)
  is resolved — the screening run showed a mode with `Im(ω) > 0`.
