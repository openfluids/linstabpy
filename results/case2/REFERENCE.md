# Case 2 — Technical Reference

Stable background for the Case 2 campaign. Update this file only when the
governing equations, norms, flow configuration, physical framing, or parameter
meanings change. For current campaign state (what's done, what's next, what's
blocked), see `PLAN.md`. For how to run a case, see `README.md`.

Inherits the project motto and rules from the repo-root `CLAUDE.md`.

---

## Code Lineage and Governing Equations

`linstabpy` is a Python / PETSc / SLEPc reimplementation of the MATLAB
**LinStab2D** tool of Martini & Schmidt (2024), *Theor. Comp. Fluid Dyn.*
(full paper: `docs/martini2024.md`). It inherits the following modelling
choices, which determine what every residual and every eigenvalue means:

- **Compressible Navier–Stokes**, non-dimensionalised by free-stream
  density/velocity/length; pressure by `2 × ½ρ_∞ u_∞²`. Sutherland viscosity.
  Primitive-variable state `q = [ρ, u, T]ᵀ`.
- **Reynolds decomposition** `q(x,t) = q̄(x) + q'(x,t)`, linearization in
  **primitive variables** (Martini & Schmidt 2024, §2). Linearizing in
  conservative variables yields a numerically different operator — we do not
  mix conventions. The parity tool
  `tools/parity/compare_martini_examples.py` checks we match the MATLAB
  reference under the primitive choice.
- **Generalized eigenproblem** `M ∂q'/∂t = L q' + f'`, with `M = M(q̄)`
  accounting for the primitive→conservative time-derivative mapping. Defining
  `A = M⁻¹ L`, temporal stability is `-iω q̂ = A q̂`. `M` invertibility /
  conditioning matters, not just `L`.
- **Compressible energy norm** (Chu 1965; George & Sujith 2011; Martini &
  Schmidt 2024 Eq. 10): weighted L² that includes `ρ', u', T'` with the
  correct compressible fluctuating-kinetic-energy weights. Non-normality,
  pseudospectra, and resolvent gain are defined in this norm — not in a naive
  L².
- **Spanwise / azimuthal wavenumber.** With homogeneous `z`, the ansatz
  `q'(x,y,z,t) = q̂(x,y) exp(iβz − iωt)` yields `A_β` and the EVP
  `A_β q̂ = −iω q̂`. The `numerics.beta` parameter is exactly this `β`. The
  cylindrical variant swaps `βz` for `mθ`.
- **Non-uniqueness caveat** carried over from the source paper. Martini &
  Schmidt (2024, §2): *"if q̄ is not a fixed point of the system, the results
  obtained from linear analysis are not unique"*. This is the Beneddine et al.
  2016 warning applied at the code level — see the Framing section below.

**Consequence.** The parity test validates **operator assembly**, not the
physical correctness of the mean-flow linearization. A run that passes
parity can still be physically meaningless if the mean flow is wrong, the
target is in the wrong pocket, or the LES/DNS consistency gap contaminates
the operator.

---

## Flow Configuration

**Geometry.** Case 2 is the **suction-side oblique shock-wave / laminar
boundary-layer interaction of the supersonic turbine cascade studied by
Lui (2022)** — see `docs/lui2022.md` and `docs/lui2022/`. The mean flow
we linearise around is the spanwise-and-time average of Lui's LES run
(same simulation, not a derivative).

Active physical parameters (from `results/case2/parity_config.jsonc`):

- `Re = 3.95 × 10⁵`
- `Ma = 2.0`
- `Pr = 0.747`
- `T₀ = 1540 K` (reference stagnation temperature; enters Sutherland)
- `γ = 1.31` (real turbine working fluid; see caveat below)
- `β = 0.0` (pinned for this sub-campaign)

**Gas-model caveat.** Lui (2022) ran the LES with `γ = 1.4` and `Pr = 0.72`
as a calorically-perfect-gas approximation. We linearise with `γ = 1.31`
because that is the physically correct value for the real turbine gas.
This introduces a second, deliberate inconsistency on top of the LES-mean
/ DNS-operator gap (Framing §3): the linear operator sees a gas that is
not quite the gas the LES simulated. We accept this in exchange for
physical realism of the linear analysis. All eigenvalues computed before
2026-04-17 were obtained with `γ = 1.4` and are not comparable to
subsequent results without recomputation.

**Length and frequency conventions.**

- Global non-dim length: **chord**. A reported `x/L = 0.1` means 10 % of
  chord.
- Local non-dim length in the operator: `L = mean separation-bubble
  length`, so `u_∞` and `L` set the operator's time scale.
- Since `ω` is non-dim in `L/u_∞`, the physically meaningful frequency is
  `St = Re(ω) / (2π)`. Report frequencies as Strouhal wherever possible;
  `ω` alone is not self-explanatory.

### Expected qualitative mode structure

Regardless of the precise geometry, the campaign is organised around a
structure of the Robinet 2007 / Song & Hao 2023 type (see `docs/song2023.md`):

- a **stationary mode-1 family** (Re(ω) = 0, stable Im(ω) < 0 in our regime)
  — recovered near `0 − 0.83i` on `M2`;
- an **oscillatory mode-2 family** (Re(ω) ≠ 0, stable) — aimed for near
  `0.56 − 1.11i`, non-portable under mesh refinement so far;
- globally stable operator + non-normal LNS ⇒ strong **non-modal
  amplification** in the resolvent, peaking at finite `ω` on the mode-2
  branch. This is the physics the paper will ultimately claim, not the EVP
  itself.

Song & Hao (2023) report, in the *globally unstable* regime, that mode 2 and
its conjugate collide at small spanwise wavelength and bifurcate into two
stationary branches — a warning that the mode-2 line in the complex plane is
not a passive parameter path when `β` is swept.

### Related reference configurations

- `validation/song_2023/` — Ziming Song's oblique SWBLI at `M=2.15`, `Re=1e5`
  (Song & Hao, *Phys. Fluids* 35, 084121, 2023). Used as a validation
  reference, **not** to be confused with Case 2 itself.
- `docs/robinet2007.md` — Robinet's original global-stability SWBLI study.
- `docs/martini2024.md` — the code paper.
- `docs/song2023.md` — full text of the Song & Hao reference study.

---

## Physical and Numerical Framing

Three things are true at once in this campaign. Keeping them separate is
essential — every diagnostic has a different meaning depending on which one
it addresses.

### 1. Why resolvent, not just eigenvalues

Case 2 is **globally stable**: spectrum in the lower half-plane, containing
(a) a stationary mode and (b) at least one stable oscillatory branch. Modal
analysis alone predicts decay.

But the linearized compressible NS operator is strongly **non-normal** — its
eigenvectors are not orthogonal in the energy inner product — so eigenvalues
do not bound short-time energy growth. The relevant objects are the
**pseudospectrum** and the **resolvent norm** `‖(iωI − L)⁻¹‖`, whose SVD
gives **optimal forcing** (right singular vectors), **optimal response**
(left singular vectors) and **gain** `σ_j(ω)` (singular values). Standard
**input–output / amplifier-flow** framework (Trefethen & Embree 2005; Schmid
2007 *ARFM*; McKeon & Sharma 2010 *JFM*; Sipp & Marquet 2013; Jovanović &
Bamieh 2005 *JFM*).

**Stable spectrum + expected non-modal amplification ⇒ resolvent is the
correct tool, not a convenience.**

### 2. Mean flow vs base flow — the closure gap

Linearization point is a **mean flow** `Ū` (time average), not a **base flow**
(fixed point `N(u_b)=0`). `Ū` does not satisfy NS; residual is

```
N(Ū) = −∇·⟨u'u'⟩ ,
```

the **Reynolds-stress divergence**, which would have to appear as steady
forcing to make `Ū` a fixed point. Fluctuation equation:

```
(iωI − L_Ū) q̂ = f̂ ,
```

where `f̂` contains the **nonlinear forcing from second-order fluctuation
statistics** (McKeon & Sharma 2010; Beneddine, Sipp, Arnault, Dandois &
Lesshafft 2016 *JFM* 798).

Mean-flow linearization is an *ansatz*, not a derivation. Well justified when
the operator is a strong **low-rank selective amplifier** (`σ₁ ≫ σ₂`) and
forcing projects onto optimal forcing — Beneddine et al. 2016 validity
criterion. Can fail in weakly nonlinear regimes (Sipp & Lebedev 2007 *JFM*
593).

### 3. LES mean flow, DNS-linearized operator — the consistency gap

Our specific situation and the reason for current numerical fragility.

- `Ū` comes from an **LES**: resolved-scale energy only; sub-filter content
  absorbed by a subgrid-scale (SGS) model during time integration.
- The operator we build linearises the **unfiltered** compressible NS — a
  **DNS operator** around an LES-filtered mean.
- No canonical name; *"inconsistent linearization of a filtered mean flow"*
  or *"model–operator mismatch"* is the phrasing in practice.

Expected failure modes, exactly what we see or worry about:

- **energy pile-up near the grid cutoff** in eigenmodes and resolvent
  response modes;
- **spurious high-wavenumber interior modes** that the shift-invert target
  then attracts;
- **poor Krylov/Arnoldi convergence** for interior eigenvalues (the `M1`
  stall);
- **continuous-spectrum contamination**.

Standard treatments, in order of prevalence:

1. **Eddy-viscosity augmentation** of the linear operator. Add frozen `ν_t`
   to `L_Ū`, typically from RANS k–ε or calibrated to data. Dominant choice
   for jets and swirling flows (Pickering, Rigas, Schmidt, Sipp & Colonius
   2021 *JFM* 917; Tammisola & Juniper 2016 *JFM* 792; Oberleithner, Rukes &
   Soria 2014 *JFM* 755). *Ad hoc* — Symon et al. 2019 *PRF* 4 show choice of
   `ν_t` matters quantitatively.
2. **Linearize the LES/RANS equations**, including the SGS or turbulence
   model. Cleaner in principle; requires a differentiable closure; not what
   this code currently does.
3. **Low-pass filter the mean flow** before linearization (Visbal & Gaitonde
   compact filters, *JCP* 181, 2002; Gaussian / box at the LES cutoff;
   spectral vanishing viscosity, Tadmor 1989). Pragmatic preprocessing;
   acknowledged in Mettot PhD (ONERA, 2013).
4. **Ducros sensor** (Ducros et al. 1999 *JCP* 152): compressibility selector
   `θ = (∇·u)² / ((∇·u)² + ‖∇×u‖² + ε)`, for localising shock-capturing
   dissipation **inside the LES that produced `Ū`**. Not a post-hoc
   mean-flow smoother; should not be used as one in this code.

**Consequence for this campaign.**

- We take **route 3**: low-pass filter of the mean flow. **Filtering is
  performed upstream of `linstabpy`** — `linstabpy` never sees an
  unfiltered mean flow in `Qfilt`. No filter machinery is implemented in
  this code, by design (keeps the code general across cases).
- The filter is the **compact filter family of Lele (1992)**. Empirical
  result on this mean flow (already obtained): the **2nd-order** Lele
  filter suppresses the spurious oscillations; the **4th-order** Lele
  filter does not. `Qfilt` therefore = Lui mean flow after Lele (1992)
  2nd-order compact filter.
- Route 1 (eddy-viscosity augmentation) is **out of scope for this
  campaign**. Kept documented above for record; not pursued here.
- Route 2 (linearise the LES/RANS equations) is not what this code does.
- Route 4 (Ducros) is upstream of `Ū`, inside the LES; not applicable
  here. Do not call a post-hoc mean-flow filter "Ducros".

---

## Parameter Map

Parameters that matter in the active code path. Source-file pointers refer to
`/home/ghamada/linstabpy/src/...`.

### Mean-flow input

- `paths.case_dir` — which baseflow dataset is loaded. Active target:
  `results/case2/baseflow`. Filtered-mean variants live as separate datasets
  (e.g. `baseflow_filt`), switched via this path — **not** via
  `alpha_filter`.

### Mesh resolution

- `numerics.stride_ksi`, `numerics.stride_eta` — coarsening during case
  loading (`results_runtime.py:379,515`). `1` = full; `2` = every other
  point; `4` = coarse legacy.

Active nomenclature:

- `M2`  = `(2, 2)` coarse reference
- `M21` = `(2, 1)` active screening mesh
- `M1`  = `(1, 1)` full resolution

`ksi` is streamwise, `eta` is wall-normal. `M21 = (2, 1)` therefore
coarsens the streamwise direction while keeping the wall-normal
resolution.

**This asymmetry is not physically justified.** It is a cost reduction
used to find *any* numerically converged EVP. The final deliverable has
to be at `M1`. `M21` exists purely as a screening mesh, and any result
obtained on it must be re-confirmed at `M1` before it enters the paper.

### Spatial discretization

- `numerics.fd_order` — SBP finite-difference order, `{2, 4, 6, 8}`
  supported. Campaign baseline is `4`; not proven insensitive.

### Spanwise parameter

- `numerics.beta` — forwarded as `kx=beta` to the LNSE assembly
  (`results_runtime.py:592`, `assembly.py:585`). Pinned `0.0` for this
  sub-campaign. **Caveat:** `β=0` forecloses any 3D branch by construction;
  this is a 2D-subspace study until the baseline is stable.

### Numerical mesh filter in the operator

- `numerics.alpha_filter` — mesh filter strength, builds filter operators in
  `filters.py:293`. Numeric `0.0` ≠ `"none"` in the filter machinery
  (`filters.py:479`). The runtime forces EVP-only runs to `"none"` when
  `run_resolvent = false` (`results_runtime.py:103`). Therefore in the active
  EVP-only campaign `alpha_filter` is effectively disabled. **This parameter
  is not "mean-flow filtering"** — that lives at dataset level.

### Analysis switches

- `analysis.run_stability`, `analysis.run_resolvent` — independent EVP /
  resolvent toggles. Calibration rule: `run_stability = true`,
  `run_resolvent = false`.

### Number of requested eigenpairs

- `analysis.n_modes_stability` — small values (`2`, `6`) for cheap branch
  localization; larger (`20`, `50`) for revealing whether the returned
  pocket is the intended one.

### Spectral target

- `analysis.omega_target` — complex shift-invert target. If omitted, built
  from `St`, `ratio_uinf`, `ratio_l` (`results_runtime.py:389`). In SLEPc:
  `eps.setTarget`, `eps.setWhichEigenpairs(TARGET_MAGNITUDE)`,
  `st.setShift`. **Not a passive hint** — it defines the interior eigensolve
  centre.

### Solver backend

- `solver.petsc_matrix_mode` = `"distributed"` (active)
- `solver.eigen.st_solver` = `"mumps"` (active) — direct shift-invert with
  MUMPS inside EPS. Baseline choices, not proven insensitive.

### EPS tolerance

- `solver.eigen.eps_tol` — SLEPc EPS convergence threshold, **not** the
  postcheck residual (`parity_config.jsonc:57`). **Operational stance:**
  `eps_tol` is a performance/robustness knob. The physics gate is postcheck
  residual + target distance + family identity. A loose `eps_tol` that
  passes the physics gate is acceptable; a tight `eps_tol` that stalls gives
  nothing.

### Arnoldi subspace size

- `solver.eigen.ncv` — Krylov subspace size. Too small → wrong family or
  stall; increasing `ncv` can improve interior-mode localization even at
  fixed target.

### Convergence test and true residual

- `solver.eigen.conv_test = "rel"`, `solver.eigen.true_residual = true` —
  safer settings; currently active.

### Iteration cap

- `solver.eigen.eps_max_it` — EPS iteration cap. The hard `stride=1` test did
  not hit the cap; the run stalled before convergence, so the cap is not the
  bottleneck.

---

## How Residuals Should Be Read

Two residual concepts in saved artifacts:

1. **SLEPc residual** `stability_residual_relative` — from
   `eps.computeError(...)` in `eigenvalue.py:738`.
2. **Explicit postcheck residual** `stability_postcheck_residual_rel` —
   `|| i L0 v − ω v || / ||v||` in `results_runtime.py:191`.

Rules:

- Good SLEPc residuals do **not** guarantee the returned family is the one
  we wanted.
- Target distance still matters.
- Postcheck residuals are the explicit full-operator validation.
- `nconv` alone is not enough.

---

## References

- Trefethen, L.N. & Embree, M. *Spectra and Pseudospectra.* Princeton, 2005.
- Schmid, P.J. 2007. *Nonmodal stability theory.* ARFM 39.
- McKeon, B.J. & Sharma, A.S. 2010. *A critical-layer framework for
  turbulent pipe flow.* JFM 658.
- Sipp, D. & Marquet, O. 2013. *Characterization of noise amplifiers with
  global singular modes: the case of the leading-edge flat-plate boundary
  layer.* Theor. Comp. Fluid Dyn. 27.
- Jovanović, M.R. & Bamieh, B. 2005. *Componentwise energy amplification in
  channel flows.* JFM 534.
- Beneddine, S., Sipp, D., Arnault, A., Dandois, J. & Lesshafft, L. 2016.
  *Conditions for validity of mean flow stability analysis.* JFM 798.
- Sipp, D. & Lebedev, A. 2007. *Global stability of base and mean flows.*
  JFM 593.
- Barkley, D. 2006. *Linear analysis of the cylinder wake mean flow.* EPL 75.
- Mettot, C., Sipp, D. & Bézard, H. 2014. *Quasi-laminar stability and
  sensitivity analyses for turbulent flows.* Phys. Fluids 26.
- Pickering, E., Rigas, G., Schmidt, O.T., Sipp, D. & Colonius, T. 2021.
  *Optimal eddy viscosity for resolvent-based models of coherent structures
  in turbulent jets.* JFM 917.
- Tammisola, O. & Juniper, M.P. 2016. *Coherent structures in a swirling
  jet.* JFM 792.
- Oberleithner, K., Rukes, L. & Soria, J. 2014. *Mean-flow stability
  analysis of oscillating jet experiments.* JFM 755.
- Symon, S., Illingworth, S.J. & Marusic, I. 2019. *Energy transfer in
  turbulent channel flows and implications for resolvent modelling.* PRF 4.
- Ducros, F., Ferrand, V., Nicoud, F., Weber, C., Darracq, D., Gacherieu, C.
  & Poinsot, T. 1999. *Large-eddy simulation of the shock/turbulence
  interaction.* JCP 152.
- Visbal, M.R. & Gaitonde, D.V. 2002. *On the use of higher-order
  finite-difference schemes on curvilinear and deforming meshes.* JCP 181.
- Tadmor, E. 1989. *Convergence of spectral methods for nonlinear
  conservation laws.* SIAM JNA 26.
- Chu, B.-T. 1965. *On the energy transfer to small disturbances in fluid
  flow.* Acta Mech. 1.
- George, K.J. & Sujith, R.I. 2011. *On Chu's disturbance energy.* J. Sound
  Vib. 330.
- Martini, E. & Schmidt, O.T. 2024. *LinStab2D: stability and resolvent
  analysis of compressible viscous flows in MATLAB.* Theor. Comp. Fluid Dyn.
- Robinet, J.-C. 2007. *Bifurcations in shock-wave/laminar-boundary-layer
  interaction: Global instability approach.* JFM 579.
- Song, Z. & Hao, J. 2023. *Global instability of the interaction between an
  oblique shock and a laminar boundary layer.* Phys. Fluids 35, 084121.
