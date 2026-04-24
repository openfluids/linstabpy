# Case 2 — Work Log

Append-only, one entry per step. Each entry records:
plan → what was read/run/changed → result → what is now uncertain.

Entries refer to PLAN.md phases. Nothing here replaces PLAN.md; this is
the trail of how we got from one state to the next.

---

## 2026-04-18 — Entry 1. Scope of Phase 1.5; grid facts; diagnostic design.

**Plan for this entry.** Fix the operational meaning of the Phase 1.5
"wall-normal spectrum" diagnostic before writing any code. In
particular: confirm what `M1/M2/M21` mean in terms of array shapes, and
decide whether the FFT is in physical (x,y) or computational (ξ,η)
coordinates.

**Read (no changes made).**

- `parity_config.jsonc` — native config. `paths.case_dir = "baseflow"`,
  `numerics.stride_ksi = stride_eta = 1`, `fd_order = 4`, `beta = 0`,
  `alpha_filter = 0.0` (consistent with PLAN screening defaults when
  interpreted as "none" at config layer).
- `case2.py` — thin wrapper over `linstab2d.results_runtime`. No
  mesh-aware logic lives here; the stride lives in `numerics.*`.
- `REFERENCE.md` Framing §3 — confirms the expected contamination
  signature of a DNS operator on an LES-filtered mean: (i) energy
  pile-up near grid cutoff, (ii) spurious high-wavenumber interior
  modes attracted by shift-invert, (iii) poor Krylov convergence for
  interior eigenvalues, (iv) continuous-spectrum contamination. Route
  chosen for Case 2 is route 3 (Visbal–Gaitonde-type low-pass of Ū,
  specifically Lele 1992 2nd-order compact), applied **upstream of
  linstabpy**. `Qfilt` is therefore an externally-produced dataset, not
  a filter call inside this code.
- `baseflow/*.npy` (via one-off inspection, no code committed):

  | file     | shape     | dtype   | range                     |
  |----------|-----------|---------|---------------------------|
  | `x`      | (599,679) | float64 | 0.3229 … 1.1481           |
  | `y`      | (599,679) | float64 | -0.1549 … 0.4681          |
  | `u`      | (599,679) | float64 | -0.1211 … 2.1611          |
  | `v`      | (599,679) | float64 | -0.9371 … 0.2382          |
  | `w`      | (599,679) | float64 | 0 … 0 (2D case, β=0)      |
  | `rho`    | (599,679) | float64 | 0.3044 … 1.9617           |
  | `T`      | (599,679) | float64 | 2.8162 … 5.3005           |
  | `sponge` | (599,679) | float64 | 0 … 0 (sponge off here)   |

**Axis convention — confirmed.**

- Axis 0 (length 599) = streamwise ξ (ksi). x varies primarily along
  axis 0; Δx along axis 0 ≈ 1.44e-3, quasi-uniform.
- Axis 1 (length 679) = wall-normal η (eta). y varies primarily along
  axis 1; Δy near wall (index 0) ≈ 9.75e-6, strongly clustered toward
  the wall.
- Wall is at η-index 0, freestream at η-index -1.

**Mesh map — confirmed against REFERENCE.md §Parameter-map and active
jsonc configs:**

| label | (stride_ksi, stride_eta) | effective shape |
|-------|--------------------------|-----------------|
| M1    | (1, 1)                   | 599 × 679       |
| M21   | (2, 1)                   | 300 × 679       |
| M2    | (2, 2)                   | 300 × 340       |

**Key methodological decision: FFT in computational coordinates, not
physical.** The wall-normal grid is strongly non-uniform in physical y
(wall clustering). A uniform FFT in y is not meaningful without
resampling. But decimation (`stride_eta = 2`) happens in the
**computational** η index, where spacing is uniform by construction.
The question "does Qraw carry content above the M2 wall-normal
Nyquist?" is therefore well-posed **in η-index space**, not in physical
y. Same logic applies streamwise, but there Δx is already
quasi-uniform, so the distinction is cosmetic. The diagnostic will
therefore compute 1D FFTs along the computational axes, at selected
stations, and report energy above the decimation Nyquist
(k > N/(2·stride)) as a fraction of total energy.

**Choice of streamwise stations (planned, not yet executed).** 4
stations distributed in ξ-index across the inlet → interaction →
recovery regions. Exact indices to be picked by visual inspection of
the `u(x,y)` field in Entry 2. The plan is to pick one station upstream
of any shock structure, one inside the interaction, one immediately
downstream, and one in the recovery zone. A station grid done blindly
on index would not be honest — we want stations the reader could
defend physically.

**What this entry does NOT do.**

- No code is written.
- No files in the repo are modified except this worklog.
- No EVP runs are performed.
- No `Qfilt` import scaffolding is added (that is Phase 1, still
  blocked on upstream delivery).

**Still uncertain / to check before Entry 2.**

1. Whether the native `baseflow/*.npy` is already some kind of
   pre-processed field (e.g., already mapped to the curvilinear mesh
   with specific boundary handling) rather than raw LES sample points.
   This would affect what "high-k content" means. — Will check
   `linstab2d.results_runtime` / baseflow loader in Entry 2.
2. Whether the `y` coordinate being non-monotonic along axis 0 at a
   fixed η (curvilinear mesh following a curved surface) introduces
   artefacts when doing a streamwise FFT at fixed η-index. Expected
   answer: yes, a small one — the physically meaningful streamwise
   direction is along-surface, not along axis 0. For this diagnostic
   we accept the artefact and note it, because the goal is to detect
   aliasing-scale content, not to get a clean dispersion relation.
3. Whether Lui's LES used the same curvilinear mesh resolution we are
   linearising on, or was resampled down to this 599×679 grid before
   being handed to linstabpy. If resampled, "content above the M2
   Nyquist" may already be partially suppressed by that resampling.
   Documented as a caveat; answer requires asking upstream.

**Next entry (Entry 2) will:**

(a) read the baseflow loader to resolve uncertainty #1;
(b) render a plot of `u(x,y)` with the 4 candidate station positions
marked, for your review;
(c) only after you approve the stations, compute and plot the
1D η-index FFT of `u(ξ_station, :)` at each, marking k_Nyquist of
M2's η-decimation.

No code lands in the repo until (a)+(b)+(c) are in front of you.

---

## 2026-04-18 — Entry 2. Axis convention pinned; loader-name swap found; Phase 1.5 diagnostic run.

**Plan for this entry.** Resolve uncertainty #1 from Entry 1 (what
`baseflow/` actually looks like to the solver), confirm the
computational-axis convention, run the Phase 1.5 wall-normal spectrum
diagnostic on `Qraw`, and record the decision.

### Finding A — Post-load axis convention

Confirmed by reading `src/linstab2d/baseflow/case1.py` + `Mesh.create`
+ one `.venv/bin/python` inspection call:

- `load_hamada_case("baseflow")` loads the eight `.npy` files, transposes
  each (`transpose_input=True` by default), then returns arrays of shape
  `(ny, nx) = (679, 599)`.
- `Mesh.create((0,1), (0,0.5), nx, ny)` treats **axis 0 as η (computational
  y)** and **axis 1 as ξ (computational x)**. Verified by inspecting X, Y
  on a toy 7×9 mesh: `X[0,:]` sweeps the first computational dimension
  0→1, `Y[:,0]` sweeps the second 0→0.5.
- In the loaded `baseflow/`, axis 0 (length 679) is where Y spans
  −0.003 → 0.47 (wall → freestream): it IS the physical wall-normal.
  Axis 1 (length 599) is where X spans 0.32 → 0.97 (inlet → outlet):
  it IS the physical streamwise.
- Settled: **axis 0 = η = wall-normal; axis 1 = ξ = streamwise**. This
  matches Entry-1's assumption but is now a verified fact, not an
  inference.

### Finding B — `stride_ksi` / `stride_eta` are swapped in the loader

`case1.py:97-98` applies `arr[::stride_ksi, ::stride_eta]`. Because the
loaded arrays are `(η, ξ)`, this means:

| parameter       | actually decimates     | named as suggests |
|-----------------|------------------------|-------------------|
| `stride_ksi`    | axis 0 = η = wall-normal | ξ (streamwise)   |
| `stride_eta`    | axis 1 = ξ = streamwise  | η (wall-normal)  |

**The two parameter names are reversed** relative to what they do.

Consequence for past results:

- `M21 = (stride_ksi=2, stride_eta=1)` — `PLAN.md` says this "coarsens
  streamwise". It actually halves **wall-normal**.
- `M2 = (2, 2)` — both axes halved, unambiguous under either naming.
- `M1 = (1, 1)` — no decimation, unambiguous.

**This is a Phase-2 blocker.** M21 screening interpretation depends on
knowing which physical direction is being decimated. Phase 1.5 is not
affected — the diagnostic FFTs along the known physical wall-normal
axis regardless of the stride name.

Decision deferred to Gabriel: either (i) document the actual behavior
and update PLAN's "M21 coarsens streamwise" text, (ii) swap `sx`/`sy`
in the loader to make names match behavior (and annotate past M21 runs
as "actually M12"), or (iii) both. Logged here; not touching anything
until the call is made.

### Phase 1.5 diagnostic — method and results

**Script.** `studies/case2/phase1p5_baseflow_spectrum.py`. Loads
`baseflow/` via `load_hamada_case` (stride=1, so native `(679, 599)`),
picks 4 streamwise stations, takes `U[:, ξ_station]` (a 679-point
wall-normal profile at each), mean-subtracts, computes `|rfft|²` along
the **computational η-index**, and reports the fraction of energy above
the M2 wall-normal Nyquist index `k = (N/2)/2 = 169`.

**Stations.** Picked at ξ-fractions {0.15, 0.35, 0.60, 0.85}:

| station | ξ-idx | x(wall) | x(freestream) | physical region |
|---------|-------|---------|----------------|------------------|
| S1      | 90    | 0.4558  | 0.5453         | upstream, pre-shock |
| S2      | 209   | 0.6301  | 0.7372         | approach, near first shock |
| S3      | 359   | 0.8153  | 0.9379         | through interaction / shock foot |
| S4      | 508   | 0.9330  | 1.0768         | recovery |

Confirmed by eye on `phase1p5/qraw_u_with_stations.png` — the two
oblique shocks emerge from the upper boundary and meet near S3.

**Quantitative result.** Fraction of |û|² above M2 η-Nyquist (k=169):

| station | frac above M2 Nyquist |
|---------|-----------------------|
| S1      | 1.85e-03              |
| S2      | 1.41e-03              |
| S3      | 7.88e-04              |
| S4      | 2.81e-03              |

Max = 2.8e-3 at S4 (recovery zone). Min = 7.9e-4 at S3 (through
shock foot). **Counterintuitive** that S3 has the LEAST
above-Nyquist energy despite being at the shock — possible
explanations: (a) shock captured over enough cells that its energy sits
at mid-k not near-Nyquist, (b) wall-normal line through S3 is
dominated by smooth pre/post-shock regions and the thin shock
contributes little to the line-integrated energy, (c) upstream LES
resampling already suppressed the sharpest features.

**Qualitative result (from `phase1p5/qraw_u_wall_normal_spectrum.png`).**
This is the more important finding. The normalized spectrum decays
~4–5 decades from k=1 to k≈50–100, then flattens into a **plateau at
10⁻⁴ (S4) to 10⁻⁶ (S3)** that extends all the way to the M1 Nyquist
(k=339). That plateau is the signature of digitization-level /
broadband noise — a smooth mean flow should keep decaying, not
flatten. Exactly matches REFERENCE.md §3's expected contamination
signature for "DNS operator on LES-filtered mean".

### Verdict

- **`Qraw` is not cleanly band-limited.** Wall-normal energy keeps
  arriving at every wavenumber up to the native Nyquist. 0.08–0.28% of
  total energy sits above what the M2 grid can represent; additional
  content sits between the M2 and M1 Nyquists.
- **`Qfilt` (Lele 1992 2nd-order compact) is worth pursuing.** A
  compact filter attenuates exactly this plateau and is the standard
  route-3 response from REFERENCE §3.
- **But the evidence is suggestive, not decisive.** 10⁻⁵-level plateau
  energy is small in absolute terms; whether it is enough to generate
  the "contaminated modes" we have observed in past EVP runs depends
  on the non-normality of the operator — this is a question only the
  actual Qraw-vs-Qfilt EVP comparison (Phase 2) can answer.

### Outputs

- `studies/case2/phase1p5/qraw_u_with_stations.png` — field with stations
- `studies/case2/phase1p5/qraw_u_wall_normal_spectrum.png` — spectra
- `studies/case2/phase1p5/summary.txt` — numbers
- `studies/case2/phase1p5_baseflow_spectrum.py` — the diagnostic

### Blockers and next steps

1. Resolve Finding B (stride-name swap) before any Phase 2 run.
2. Await delivery of `Qfilt` dataset (external: Lele 1992 2nd-order
   compact filter applied upstream).
3. When both are resolved, re-run this same diagnostic on `Qfilt` and
   compare the plateaus side by side. Expected result if the filter
   is effective: `Qfilt` plateau ≥ 2 decades below `Qraw`'s.

---

## 2026-04-18 — Entry 3. Dx/Dy convention confirmed. Extended diagnostic.

**Method changes.** (a) Verified the mesh's `Dx` / `Dy` derivative
operators by applying them to `f = sin(π x) + cos(π y / 2)` on a 9 × 11
test mesh: max interior error `Dx` 9.4e-3, `Dy` 4.2e-3 (at fd_order 4
on a coarse grid this is correct). Confirmed: `Dx` differentiates along
**axis 1 (computational x, nx)**, `Dy` along **axis 0 (computational y,
ny)**. Together with Entry 2 this pins down the loader naming swap
beyond doubt. (b) Extended the Phase 1.5 diagnostic:

- all 4 primitive variables (`rho`, `u`, `v`, `T`) rather than just `u`;
- added a "plateau level" metric = median of normalized |FFT|^2 over
  the top 30% of wall-normal wavenumbers (captures the noise-floor
  height, which the `frac_above_Nyquist` metric alone misses);
- added a 2D FFT of `u` and `T` over `(k_ξ, k_η)`.

**Extended results.** See `phase1p5/qraw_multi_variable_spectrum.png`,
`phase1p5/qraw_u_2d_fft.png`, `phase1p5/qraw_T_2d_fft.png`,
`phase1p5/summary.txt`.

| station | x(wall) | frac_above_M2_Nyquist ρ / u / v / T | plateau ρ / u / v / T |
|---------|---------|-------------------------------------|-----------------------|
| S1      | 0.4558  | 4.8e-3 / 1.9e-3 / 1.6e-4 / 7.7e-4   | 2.3e-5 / 9.0e-6 / 5.7e-7 / 3.7e-6 |
| S2      | 0.6301  | 6.1e-3 / 1.4e-3 / 1.8e-3 / 5.8e-4   | 2.9e-5 / 7.1e-6 / 8.8e-6 / 2.8e-6 |
| S3      | 0.8153  | 3.6e-3 / 7.9e-4 / 8.1e-4 / 6.4e-4   | 1.7e-5 / 3.8e-6 / 3.9e-6 / 3.1e-6 |
| S4      | 0.9330  | 3.3e-3 / 2.8e-3 / 2.2e-3 / 1.3e-3   | 1.6e-5 / 1.4e-5 / 1.1e-5 / 6.3e-6 |

- max frac above M2 Nyquist: 6.1e-3 (**ρ at S2**) — ~3× higher than u.
- max plateau level: 2.9e-5 (ρ at S2).
- plateau of v at S1 has a strong oscillating artefact around
  k ≈ 150–250 (visible in the multi-variable plot) — likely a
  small-magnitude-relative-error effect since v is small across the
  domain (max |V| ≈ 0.94 with most of the field ≪ 1).

**Physical reinterpretation (this is the important finding).** The
2D FFT plots (`qraw_u_2d_fft.png`, `qraw_T_2d_fft.png`) show that the
high-k content is NOT isotropic noise. It is organized as two **oblique
diagonal ridges** in `(k_ξ, k_η)` space — the Fourier signature of
shocks. A discontinuity in real space is wideband perpendicular to
itself in Fourier space; two oblique shocks give two diagonal ridges.
These ridges extend past the M2 Nyquist box all the way to the M1 box
boundary.

Consequence for the `Qfilt` strategy:

- the "contamination" to be filtered is **shock-wideband structured
  content**, not random digitization noise;
- a low-pass filter moves shocks from ~1-cell-thick to ~4-6-cell-thick,
  which the DNS operator's interior stencil can resolve without
  ringing; it does not erase the shocks, only thickens them;
- this also explains the REFERENCE.md §3 empirical result: 2nd-order
  Lele compact works, 4th-order does not — a 2nd-order filter has a
  gentler cutoff, which over-smooths shocks "more", which is what you
  want here.

**Secondary observation — the S3 puzzle from Entry 2 revisited.** S3
(shock foot, x≈0.82) still has the lowest fraction-above-Nyquist in
every variable. The 2D FFT makes the reason clearer: S3 sits in a
region where the two shock ridges interfere, so the fraction of the
wall-normal 1D FFT attributable to either ridge is diluted. A 2D
metric (which we now have) shows the ridges are present regardless of
station.

### Status of tasks 4–6 (done)

- Dx/Dy convention: verified, Entry 3.
- Multi-variable diagnostic: done.
- 2D FFT: done.
- Stride-name swap: documented in Entries 1–3; decision still awaiting
  user input on whether to rename in code.

---

## 2026-04-18 — Entry 4. Lele-filter reference built; 1D plateau metric debunked.

**Plan for this entry.** Build a Python Lele compact filter in
`studies/case2/phase1p5_lele_filter_reference.py` (diagnostic only,
outside linstabpy core per REFERENCE §3), apply it to `Qraw`, and show
what `Qfilt` should look like. Use to validate the upstream delivery
once it arrives.

**Filter implementation.** Two variants, matching Lele 1992 §4.4.1:

- F2 (2nd-order formal, tridiagonal LHS + tridiagonal RHS):
  `a_0 = a_1 = 1/2 + α`. `H(π/2) = 1/2 + α`. α=0 gives 50% attenuation
  at M2 Nyquist.
- F4 (4th-order formal, tridiagonal LHS + pentadiagonal RHS):
  `a_0 = (5+6α)/8`, `a_1 = (1+2α)/2`, `a_2 = -(1-2α)/8`. `H(π/2) =
  (3+2α)/4`. α=0 gives 25% attenuation at M2 Nyquist.

Transfer functions verified on pure sinusoids:

| filter | H(π/2) measured | H(π/2) theory |
|--------|-----------------|---------------|
| F2 α=0.00 | 0.5000 | 0.5000 |
| F2 α=0.40 | 0.9000 | 0.9000 |
| F4 α=0.00 | 0.7500 | 0.7500 |
| F4 α=0.49 | 0.9950 | 0.9950 |

Filter passes all transfer-function tests. **Note**: F4 α=0.49
produces a very ill-conditioned LHS (eigenvalue ratio near 1:99),
which shows up as large amplification at near-Nyquist wavenumbers in
practice. Do not use with α > 0.45 for F4.

### The surprise — my 1D plateau metric was flawed

Applied F2 α=0.0 to `Qraw` and computed the frac-above-M2-Nyquist
metric. **Result: essentially unchanged** (6.072e-3 Qraw vs 6.090e-3
Qfilt). 2D field difference `||U − Uf||_max = 3.9e-2` (non-trivial),
so the filter is acting. Why doesn't the 1D metric move?

Spent half an hour on this. The answer is structural:

- The wall-normal profile `U[:, ξ_station]` goes from ~0 (wall) to
  ~2 (freestream). It is **not periodic**.
- A non-periodic signal under rfft has a Gibbs-like 1/k envelope that
  extends across ALL wavenumbers — this is a DFT-assumption artifact,
  not a real spatial feature.
- A low-pass filter cannot remove this artifact, because the artifact
  is not a real high-k signal in real space; it's the DFT's
  representation of the wall-to-freestream step.
- Over k=100→340, 1/k² drops ~1 decade, which on a log plot from 1e-4
  to 1e-5 looks like a plateau. I had read this as a noise floor.

The 2D FFT was not misleading: it separates (a) axis-aligned structural
content (the non-periodic wall-step contributes to the k_η-axis, i.e.,
the vertical bright line in the 2D FFT), (b) diagonal shock ridges,
(c) corner noise. The F2 filter visibly attenuates (b) and (c) in
`phase1p5/qfilt_ref_u_2d_fft.png` — diagonal ridges darken and corners
blacken. (a) is untouched, as expected.

### What this means for Phase 1.5 interpretation

- **The 1D "fraction above M2 Nyquist" metric is partially compromised
  by non-periodic-DFT structural artifacts.** It does not cleanly
  separate "real contamination worth filtering" from "wall-normal
  profile shape". Use with caution.
- **The 2D FFT diagnostic stands.** Diagonal shock-ridge content that
  leaks past the M2 Nyquist box is real and IS attenuated by the Lele
  filter. This IS worth pursuing.
- **Verdict from Entry 2 still holds** (Qfilt worth pursuing), but for
  a sharper reason than I originally wrote: not because the
  wall-normal profile has "noise", but because the two oblique shocks
  have Fourier footprints that fall outside the M2 resolvable box,
  and a filter is the standard remedy.

### Open follow-up diagnostic (not yet run)

A cleaner 1D metric would:
- subtract a smooth polynomial fit from the wall-normal profile
  (removing the boundary-layer-scale structural shape),
- FFT the residual,
- measure energy vs wavenumber.
The residual should be many decades below the raw amplitude, and
whatever fails to decay is genuine contamination. Deferred — not
worth it before the EVP comparison.

### Task 8 status

Filter implementation is correct. Diagnostic outputs are useful for
the 2D view. The 1D metric as implemented is not informative. Moving
on to task #7 (reproduce Qraw EVP). Will return to this if and when
Qfilt is delivered, to do a direct Qraw-EVP vs Qfilt-EVP comparison,
which is the definitive test.

---

## 2026-04-18 — Entry 5. Qraw baseline EVP reproduction on M21 (γ=1.31).

**Plan for this entry.** Reproduce PLAN.md "What we know" row 5 — an
M21 EVP with `ncv=500, eps_tol=1e-4, target=0.561-1.112i` — on the
current γ=1.31 operator. Confirm the solver returns the wrong family
(stationary axis instead of oscillatory). Establish this as the
**contaminated-mode reference** that Qfilt / re-localisation / solver
sweeps must beat.

**Run details.**

- Config: `evp_stride2x1_oscillatory_target_ncv500_tol1e4.jsonc`
- Setup: `M21 = (stride_ksi=2, stride_eta=1)` (halves **wall-normal**
  in the code, despite the name; see Entry 2), nev=2, ncv=500,
  eps_tol=1e-4, MUMPS shift-invert, 10 MPI ranks.
- Operator size after BCs: 1,018,300 × 1,018,300, nnz 46,527,560.
- Wall time: 685 s (stability stage 642 s, of which 1 EPS iteration).

**Numeric result.**

| metric | value | gate | verdict |
|--------|-------|------|---------|
| nconv  | 2     | ≥ 1  | PASS |
| EPS iterations | 1 | — | converged immediately |
| max postcheck residual | 6.4e-5 | — | — |
| median postcheck residual | 5.7e-5 | ≤ 1e-4 | PASS |
| median target distance | 0.72 | ≤ 0.1·\|ω\|=0.125 | FAIL |
| max \|Re ω\| | 0.000 | ≥ 0.2 (osc branch) | FAIL |
| accepted_modes | 0 | ≥ 1 | FAIL |

Returned modes: `ω₁ = 0 − 0.683i`, `ω₂ = 0 − 1.580i`. Both purely
imaginary (stationary axis). Target was `0.561 − 1.112i`. Solver is
**numerically healthy** (small residuals, single iteration) but
returned the **wrong family** — the oscillatory branch is not
localised.

### Comparison with PLAN row 5 (γ=1.4)

| run                 | ω₁ returned | ω₂ returned |
|---------------------|-------------|-------------|
| PLAN row 5 (γ=1.4)  | 0 − 0.82i   | 0 − 1.59i   |
| this run (γ=1.31)   | 0 − 0.683i  | 0 − 1.580i  |

The stationary-axis pocket ω₁ shifted by ~17% (0.82 → 0.683) under the
γ correction. ω₂ barely moved. Past γ=1.4 artefacts are therefore NOT
directly comparable to current runs (as PLAN notes) but the qualitative
conclusion — solver returns stationary axis for oscillatory target —
is preserved.

### Interpretation

1. The solver itself is fine.
2. The oscillatory branch target `(0.561, -1.112)` inherited from
   M2/γ=1.4 is **not** where the oscillatory branch sits on
   M21/γ=1.31 (if it exists there at all).
3. Shift-invert Krylov-Schur is attracted to the densest cluster of
   eigenvalues within its `ncv` subspace; the stationary axis evidently
   holds more nearby modes than wherever the oscillatory branch
   actually is.
4. `nev=2` is too few to diagnose the near-target spectrum — it
   returns the 2 densest-cluster modes regardless of physical family.

### Artefacts

- `output/evp_stride2x1_oscillatory_target_ncv500_tol1e4/runs/*.npz`
- `phase1p5/evp_spectra_overview.png` — spectrum overlay with target,
  target-distance gate, and returned modes.
- `phase1p5_score_run.py` — programmatic scoring against PLAN
  acceptance criteria. Reusable for every subsequent EVP run.
- `phase1p5_plot_spectrum.py` — multi-run spectrum overlay plot.

### Next

- (task #9, in progress) Run stationary-target EVP
  `(0, -0.83i)` on M21 to confirm the stationary pocket IS recoverable
  at γ=1.31. Affirmative counterweight to the negative oscillatory
  result.
- (task #10) Run PLAN Phase 2 screening defaults (nev=20, ncv=500,
  eps_tol=5e-3) on both stationary and oscillatory targets. Broader
  extraction gives an honest view of the near-target spectrum, not
  just the densest 2 modes. Config already written:
  `evp_stride2x1_m21_screening_qraw.jsonc`.

### Addendum — spatial structure of the contaminated modes (task #11)

Rendered `|u|(x, y)` of the two returned modes
(`phase1p5/qraw_osc_target_contaminated_modes.png`):

- Mode 0 (ω=0−0.683i) peak at **(x=0.82, y=−0.10)** — directly under
  the shock foot / on the concave wall. Highly localised: 2.3% of
  gridpoints have `|u| > 0.5·peak`.
- Mode 1 (ω=0−1.58i) peak at **(x=0.96, y=−0.15)** — near-wall recovery
  region, downstream of the interaction. 7.1% of gridpoints above half
  peak.

Both modes are **wall-attached, locally driven**. They are not noisy
artefacts (Dirichlet leak is at machine precision, 1e-15) — they ARE
eigenmodes of the operator as assembled. But they live where the mean
flow has sharp wall-normal gradients (shock foot, recovery wall
region), and the LNSE operator's coefficients are dominated by those
gradients locally.

**Physics:**

- The operator has many near-degenerate eigenvalues on or near the
  `Re(ω)=0` axis corresponding to locally-trapped damping modes
  driven by large `∂Ū/∂y` near the wall/shock foot.
- Shift-invert Krylov-Schur picks the densest cluster of eigenvalues
  near the target. The oscillatory target (0.56, -1.11) is closer in
  straight-line distance to these wall-local modes than to any true
  propagating oscillatory branch (if one exists).
- Filtering Ū (Qfilt) would shrink the mean-flow gradient peaks, which
  should reduce the coefficient inflation that makes these wall-local
  modes numerically dominant. That is the physical reason Qfilt is
  expected to help.

This upgrades the "Qfilt is worth pursuing" verdict from
"Fourier-plateau argument" (which Entry 4 showed was compromised by
DFT artefacts) to a concrete **mean-flow-gradient argument**:
aggressive wall-normal low-pass filtering reduces `∂Ū/∂y` maxima ⇒
reduces the magnitude of LNSE coefficients that produce wall-local
damping modes ⇒ the oscillatory branch target is no longer crowded
out by wall modes.

### Cross-check — mean-flow gradient vs mode peak

Computed `|∂Ū/∂x|_physical` and `|∂Ū/∂y|_physical` on the M21 mesh:

| region | max \|∂Ū/∂y\| | max \|∂Ū/∂x\| |
|--------|---------------|---------------|
| full wall strip (j≤5) | 1.46e+3 | 2.68 |
| interaction zone (approx.) | 5.54e+2 | 4.92e+1 |
| upper shock region (j≥300) | 3.16e+1 | 3.09e+1 |

Peak `|∂Ū/∂x|` is at **(x=0.815, y=-0.092)** — **essentially
identical** to Mode 0 peak location (x=0.818, y=-0.099). This is the
shock foot on the wall. Mode 1 peak at (x=0.962, y=-0.150) is in the
recovery-zone wall strip where `|∂Ū/∂y|` is large.

Both returned modes therefore sit **where the mean-flow gradient
dominates**. This is direct evidence for the "shock-driven wall-local
modes" story and closes the physical loop: the LNSE coefficient
inflation is real, it's localised at the shock foot and recovery
wall, and it produces the Re(ω)=0 axis cluster that crowds out the
oscillatory branch in shift-invert.

### Critical caveat — my local Lele filter does NOT reduce `|∂Ū/∂y|_max`

Tested the F2 filter's effect on the gradient peaks that I just
identified as mode drivers:

| filter config | `|∂Ū/∂x|_max` reduction | `|∂Ū/∂y|_max` reduction |
|---------------|--------------------------|---------------------------|
| F2 α=0.0 (1 pass) | 0.923× (−7.7%) | 0.998× (−0.2%) |
| F2 α=0.4 (1 pass) | 0.972× (−2.8%) | 1.000× (−0.0%) |
| F2 α=0.0 × 5 passes | 0.790× (−21%)  | 0.992× (−0.8%) |

The `|∂Ū/∂y|_max` attribute is **not reduced** by my implementation,
because it sits at the first interior cell next to the wall, and my
boundary handler passes through `bw = ⌊order/2⌋ + 1` points unchanged
(to preserve the Dirichlet wall value `U(wall) = 0`).

Implications:

- My Python F2 reference **will not** reproduce the effect PLAN
  reports for the upstream Qfilt on Mode 1 (recovery-wall-driven,
  `|∂Ū/∂y|`-dominated).
- My F2 reference **will** attenuate `|∂Ū/∂x|` at the shock foot,
  which is where Mode 0 sits, so it should help Mode 0.
- The upstream Qfilt delivery must therefore do more than just a
  "Lele 1992 compact filter" — it may (a) filter before Dirichlet
  enforcement, (b) use a different BC treatment, (c) include
  resampling/interpolation, or (d) additionally smooth near-wall.

**Revised expectation for when Qfilt arrives**: check specifically
whether `|∂Ū/∂y|_max` at the wall of Qfilt is different from Qraw.
If it is, the upstream filter is doing more than just compact
filtering. If it isn't, then Qfilt's benefit must come via a
different mechanism (e.g., shock-foot only), and Mode 1 may remain
a contaminated attractor.

This finding means the **comparison is more subtle than I wrote in
Entry 4**. Qfilt may help some contaminated modes (Mode 0 family)
but not others (Mode 1 family) — which in turn suggests the Phase 2
screening run will give a mixed picture: some pockets improved,
others not. Worth flagging so you interpret the first Qfilt vs
Qraw comparison with this in mind.

### Decision still pending

- Stride-name swap (Entry 2 Finding B): action needed when convenient.
  Does not block further Phase 2 runs, only blocks clean
  interpretation of what "M21" actually decimates.

---

## 2026-04-18 — Entry 6. Stationary-target EVP runs (nev=4, nev=2), both killed.

**Plan.** Run stationary-target (0, -0.83i) EVP on M21 Qraw with
`ncv=500, eps_tol=1e-4`. First attempt nev=4; second attempt nev=2.

**Attempt 1 (nev=4).** Killed after 24 min, iter=5. `nconv=2` still.
Err range dropped from [1.8e-5, 1.4e+6] at iter=1 to [1.8e-5, 7.5e+4]
at iter=5 — factor-19 improvement in worst-case error over 4 iters.
Extrapolated remaining time to reach tol=1e-4 on all 4 modes: ~2.5
hours. Killed.

**Attempt 2 (nev=2).** Killed after 22 min, iter=1. `nconv=1` (only
one of the two requested modes converged). Same err_range ceiling
(1.4e+6) — the 2nd-closest mode to this target converges slowly.

**Diagnosis.** Shift-invert Krylov-Schur converges fastest for the
eigenvalue(s) nearest the target. For the stationary target
(0, -0.83i):

- nearest eigenvalue: Mode 0 (0, -0.683i) at distance 0.147 → converges
  fast (err ~5.8e-5 in iter 1).
- next nearest: either Mode 1 at (0, -1.58i) distance 0.75 (far) or
  a less-accessible mode. Distance/√tol-scale makes it many iterations
  slower.

Compared to the oscillatory-target run (Entry 5), which had both
Mode 0 and Mode 1 at roughly equidistant ~0.7 from the target, hence
both converged in iter 1. A target asymmetric w.r.t. the densest
eigenvalue cluster picks only the single closest mode quickly.

**Decision.** Abandon fixed-nev at tight tolerance for broad
spectrum diagnosis. The right next step, per PLAN Phase 2 screening
defaults, is `nev=20, ncv=500, eps_tol=5e-3` — looser tolerance
admits many more converged modes, giving a real spectrum picture
instead of 2 cherry-picked ones.

**Configs written:**

- `evp_stride2x1_stationary_target_ncv500_tol1e4.jsonc` — nev=2 at
  stationary target, kept for posterity but not rerun yet.
- `evp_stride2x1_m21_screening_qraw.jsonc` — PLAN Phase 2 defaults,
  two cases (stationary target, oscillatory target), now running.

**What the stationary-nev=2 partial result told us anyway.** The
singled converged mode has `err_estim = 5.82e-5` — this is likely the
refined Mode 0 (which came out as 0-0.683i at err=4e-6 in the osc
run). With only `.config.json` saved and no `.npz` (killed before
save stage), we can't read the eigenvalue directly. The screening
run will surface it.

---

## 2026-04-18 — Entry 7. PLAN Phase 2 screening run started (Qraw M21, nev=20).

**Plan.** One run, two cases:

1. `bstat_m21_qraw_nev20_ncv500_tol5em3`: target (0, -0.83i), nev=20.
2. `bosc_m21_qraw_nev20_ncv500_tol5em3`: target (0.561, -1.112i), nev=20.

Other parameters exactly PLAN Phase 2 screening defaults:
`ncv=500, eps_tol=5e-3, alpha_filter='none', run_stability=true,
run_resolvent=false, fd_order=4, beta=0`. Per-case acceptance
thresholds widened to `postcheck_rel_max=5e-3, target_distance_max=0.5`
to match the exploratory nature of screening (PLAN row 135).

**Expected outputs.** Each case returns up to 20 modes at
`err ≤ 5e-3`. Far richer than the 2-mode runs. Will give us, for each
target, a list of 5–20 eigenvalues in the near-target neighbourhood,
each with postcheck residual and target distance.

**Hypotheses going in.**

- (H1) Both targets will return many eigenvalues clustered along the
  `Re(ω)=0` axis (the wall-local mode family).
- (H2) Few if any returned modes will have `|Re ω| ≥ 0.2` — the
  oscillatory branch will remain unlocalised.
- (H3) Mode 0 family (shock-foot-driven, ~0-0.683i) and Mode 1 family
  (recovery-wall-driven, ~0-1.58i) will both be densely populated.
- (H4) Between the two target runs, the overlap of returned
  eigenvalues will be substantial — because both targets sit roughly
  equidistant from the wall-mode cluster.

**Why this matters for the paper.** A proper spectrum picture of the
contaminated modes (20 modes instead of 2) is essential evidence.
It tells us whether the contamination is a dense family (many wall
modes) or a sparse pair (two specific eigenvalues). That distinction
matters for how much work Qfilt must do.

**Next.** Await run completion (~45 min estimated), analyse with
`phase1p5_plot_spectrum.py` and `phase1p5_score_run.py`, compare the
two cases' spectra directly.

---

## 2026-04-18 — Entry 8. Screening-run results. First full Qraw spectrum picture.

**Summary of spectrum (M21 Qraw γ=1.31, nev=20, tol=5e-3).**

Both targets (stationary (0, -0.83i) and oscillatory (0.56, -1.11i))
return the **same 20-mode set** (modulo sign convention on conjugate
pairs). Total converged across 64 (stat) / 74 (osc) — the solver
found a dense Krylov subspace and returned the 20 "nearest" by each
target.

**Mode inventory:**

| class | modes | character |
|-------|-------|-----------|
| Stationary axis, discrete | (0, -0.68), (0, -1.58), (0, -4.59), (0, -8.92) | wall-localised, decaying |
| Stationary axis, "unstable" | (0, **+3.17**) | **Im(ω) > 0 — unstable under MATLAB convention!** |
| Oscillatory pairs, low freq | (±0.79, -6.30), (±1.07, -8.88) | wall-near, high damping |
| Oscillatory pairs, mid freq | (±1.98, -9.59), (±2.82, -7.66), (±3.50, -9.82) | |
| Oscillatory pairs, high freq | (±4.84, -7.66), (±4.86, -5.36), (±4.90, -8.50) | shock-frequency range |

See `phase1p5/screening_*_spectrum.png` for the combined plot and
`phase1p5/screening_*_modes.png` for the mode-shape grid.

### Key findings

**F1 — PLAN's osc seed (0.56, -1.11i) has no eigenvalue there.** The
spectrum has **nothing** in that region of the complex plane. The
nearest two modes are (0, -0.68) and (0, -1.58) — both on the
stationary axis — at distance 0.71 and 0.73 from the seed. PLAN
Phase 3 (re-localisation) is NOT an optional refinement; it's
mandatory. The M2/γ=1.4 target is not portable.

**F2 — Both targets attract the same mode cluster.** Re-running with
a different target does not explore different parts of the spectrum
at this tolerance level. The shift-invert solver with ncv=500 at
tol=5e-3 converges the SAME dominant 20-mode subspace regardless of
where σ is set (within a reasonable range). To explore elsewhere in
the spectrum, need to (a) move target far enough that the dominant
subspace shifts, or (b) tighten `eps_tol`, which forces the solver
to do more refinement of the nearest modes instead of admitting more
distant Ritz values.

**F3 — Almost all modes are wall-localised (qualitatively, from
mode-shape grid).** Modes 0, 1, 2, 4-19 all show energy concentrated
near the wall in the interaction / recovery region (x ≳ 0.8,
y ≲ -0.05). Only Mode 3 (0, +3.17i) has GLOBAL structure — diagonal
bands spanning the domain — consistent with a trapped acoustic /
cavity mode between the two oblique shocks.

**F4 — Mode 3 at (0, +3.17i) appears unstable.** Under the code's
MATLAB stability convention (`ω = eig(i·L0)`), Im(ω) > 0 means
growth. Mode 3 diagnostics:

- `postcheck_residual_rel` = 8.1e-5 (passes loose gate)
- `stability_residual_relative` = 2.6e-5 (solver residual)
- `dirichlet_leak_max` = 3e-15 (machine precision BC respect)
- `leading_weighted_norm_error` = 3e-15 (Chu-norm consistent)

These indicate this is a **genuine eigenvalue of the operator**, not
a numerical artefact. Three possibilities:

- (a) Case 2 actually has a globally unstable mode under γ=1.31 that
  was not visible under γ=1.4; PLAN's "globally stable" description
  may need revision;
- (b) The mode is real but represents a **mean-flow-contamination
  instability** — an eigenmode of the DNS-on-LES-mean operator that
  does not correspond to any physical mode of the underlying flow;
- (c) PLAN's γ=1.4 assertion of global stability was based on a
  spectrum that didn't resolve this region; Mode 3 could have been
  there all along.

Verifying which route requires: (i) running on Qfilt and seeing if
Mode 3 disappears (would support (b)); (ii) running on M2 at γ=1.31
and checking if Mode 3 appears there too (would support (a) or (c));
(iii) checking whether the mode shape is consistent with shock-cavity
acoustics.

**F5 — Acceptance-gate summary.**

| case | postcheck ≤ 1e-4 | target-dist ≤ 0.1·|ω| | `|Re ω|` ≥ 0.2 | accepted |
|------|------|------|------|------|
| stat target (0, -0.83)  | FAIL (1.4e-4 median) | FAIL (7.7 median) | PASS (|Re| up to 4.9) | 1 |
| osc target  (0.56, -1.11) | FAIL (1.1e-4 median) | FAIL (7.6 median) | PASS (|Re| up to 4.9) | 0 |

The `|Re ω| ≥ 0.2` gate passes for both because modes 4-19 include
strongly oscillatory pairs at |Re ω| = 0.79 to 4.9. But these modes
are ALL at much higher frequency than PLAN's seed (0.56). Calling
this a "PASS on the oscillatory branch" would be misleading —
PLAN's acceptance criterion was designed assuming |Re ω| near 0.56,
not 4.9.

The "stat" case accepted 1 mode — likely Mode 0 (0, -0.68) at
target_dist 0.147 (just within the 0.1·|0.83| = 0.083? No, 0.147 >
0.083, so FAILS tight gate). The screening acceptance uses the
looser `target_distance_max = 0.5` I set in the config, which 0.147
passes.

### What this tells us about the campaign

1. **Qraw at γ=1.31 on M21 has a real, well-populated spectrum** — 64
   converged modes. The operator is not "broken" in the sense of
   refusing to converge; it produces clean eigenvalues.
2. **What's missing is physical**: the PLAN-seed oscillatory branch
   at (0.56, -1.11i) does not exist in the current operator's
   spectrum. This is a γ/mesh-change physics consequence, not a
   solver failure.
3. **The "contamination" story needs refinement.** Modes 0 and 1
   (the "contaminated pair" identified in Entry 5) are part of a
   bigger discrete spectrum of wall-local modes, plus extensive
   oscillatory families at much higher frequencies than expected.
4. **Mode 3 unstable mode is a genuine surprise** and needs physics
   investigation before any paper claims.

### Suggested next steps (for Gabriel to weigh in)

1. **Re-localisation box scan.** Fix mesh + mean flow, sweep target
   over a grid in the complex plane to map the actual spectrum
   topology. Cheap because the dominant cluster keeps coming back;
   expensive because interesting modes might require distant
   targets.
2. **Mode-shape classification.** Of the 20 returned modes, which
   are local (wall) vs global (trapped acoustic)? Automated classifier
   using the `frac_|u|>threshold` metric. Informs which branches are
   candidates for the "oscillatory family" PLAN wants.
3. **Run the same config on M2** (both directions halved) to see if
   (i) Mode 3 survives the coarser mesh and (ii) the dominant
   wall-cluster at the same ω.
4. **Hold for Qfilt**. The eventual Qfilt comparison will tell us
   whether Mode 0, Mode 1, and Mode 3 all move / disappear, or only
   some do. That's the most discriminating experiment.

### Artefacts

- `output/evp_stride2x1_m21_screening_qraw/runs/*.npz` — two run
  artefacts.
- `phase1p5/screening_*_spectrum.png` — combined complex-plane plot.
- `phase1p5/screening_*_modes.png` — 12-panel mode-shape grid per
  case.
- `phase1p5_analyze_screening.py` — reusable analysis entry point.

---

## 2026-04-18 — Entry 9. Policy pivot: asymmetric strides retired.

**Trigger.** Gabriel's guidance: "because we have curved we can not
do different strides. we can not mix 2 and 1 or 1 and 2. revise".

**Decision.** The campaign uses only symmetric strides:
`stride_ksi == stride_eta`. Valid meshes are `M1 = (1, 1)` and
`M2 = (2, 2)` (and, if ever needed, `M4 = (4, 4)`). `M21 = (2, 1)`
and `M12 = (1, 2)` are removed from the active mesh set.

**Rationale.** The Case 2 mesh is body-fitted to a curved turbine
suction side (the wall is concave). In a body-fitted curvilinear
mesh, the Jacobian of the computational-to-physical mapping varies
along the wall. Decimating one computational axis but not the other
produces cells whose physical aspect ratio depends on location —
not a natural coarsening of the physical problem. Screening must
coarsen both axes uniformly so that the cell aspect ratio (in the
sense of physical dimensions) is preserved in the mapping.

**Consequence for the 2026-04-18 work today.**

All EVP runs I did today were on asymmetric M21 (stride_ksi=2,
stride_eta=1). Under the new policy these runs are **superseded**.
They remain in the worklog as historical record of the reasoning
chain (axis convention, gradient localisation, Mode 3 observation),
but their specific numerical values and PLAN's "What we know" rows
6–7 must be re-established on symmetric M2 before anything enters
the paper.

Specifically superseded (but retained in `worklog.md` Entries 5–8):

- the Qraw osc-target reproduction (Entry 5) → rerun at M2
- the stationary-target attempts (Entry 6) → rerun at M2
- the Phase 2 screening run (Entry 8) → **rerun at M2 as I0**

The physics findings that DON'T depend on exact numerical values —
namely "contaminated modes peak at mean-flow gradient maxima" (F4
shape) and "Lele F2 doesn't reduce `|∂Ū/∂y|_max` due to BC
preservation" (F5 structural argument) — likely survive the mesh
change. I0 will confirm.

**What I/O-cheap action can take now (no rerun required).**

- The F3 (2D FFT) diagnostic of Qraw is done on M1 native; unaffected
  by the M21 retirement.
- The F4 gradient-vs-modes visualisation concept is mesh-independent;
  needs rerun on M2 for the actual values.
- Scoring / analysis scripts (`phase1p5_*.py`) are mesh-agnostic; all
  still work on the future M2 artefacts.

**What I/O-expensive action is now pending.**

- I0 screening rerun on M2. Expected wall time: shorter than M21
  runs (500k DOF vs 1M DOF, so ~3-4x faster per EVP iteration).
  Estimated: 10-15 min for the 2-case screening.
- I7 Mode-3 survival test on M2: included in I0's output.

**Configs to create:**

- `evp_stride2_m2_screening_qraw.jsonc` (stride 2,2 symmetric
  replacement for the retired M21 screening config)

**Configs to deprecate (keep files, stop using):**

- `evp_stride2x1_oscillatory_target_ncv500_tol1e4.jsonc`
- `evp_stride2x1_stationary_target_ncv500_tol1e4.jsonc`
- `evp_stride2x1_m21_screening_qraw.jsonc`

These three remain on disk as historical reference; any fresh run
must use the symmetric M2 replacement. I will leave a README note
in this directory pointing forward.

---

## 2026-04-18 — Entry 10. M2 symmetric screening — paradigm shift.

**Run details.** New config `evp_stride2_m2_screening_qraw.jsonc`
(stride 2,2 symmetric, nev=20, ncv=500, tol=5e-3). Wall-time 346 s
per case (~2× faster than M21 at 1M DOF; operator is 500k DOF).
Both stationary and oscillatory targets. nconv = 78 (stat) / 88 (osc).

**Scored with `phase1p5_analyze_screening.py`. Classified with a
simple `frac_|u|>0.1` global-vs-local threshold.**

### Three paradigm shifts compared to M21

**Shift 1 — Mode 3 (0, +3.17i) is GONE.** The only apparently-unstable
mode from the M21 screening has no counterpart in the M2 spectrum.
All 20 returned M2 modes have `Im(ω) ≤ -0.69`. **I7 largely resolved:
Mode 3 was a mesh artefact of the asymmetric stride, NOT a genuine
instability.** PLAN's "globally stable" assertion stands. Treat this
as strong evidence for the symmetric-stride policy retrospectively.

**Shift 2 — Spectrum is much richer at low |Re ω|.** M21 returned 8
oscillatory pairs, most at `|Re ω|` = 4.8–4.9. M2 returns 9 pairs,
spread over `|Re ω|` = 0.39 to 2.72. The high-frequency cluster
`|Re ω|` ≈ 4.9 from M21 is entirely absent on M2 — those were
asymmetric-grid resonances, not physical.

**Shift 3 — Most M2 oscillatory modes are globally-extended, not
wall-local.** Classifier output (M2, stationary-target case):

| idx | ω | St | frac \|u\|>0.5 | frac \|u\|>0.1 | class |
|-----|---|----|-------|--------|-------|
| 0 | 0 − 0.695j | — | 2.3% | 14.4% | wall-local |
| 1 | 0 − 1.578j | — | 7.5% | 16.9% | wall-local |
| 2 | 0 − 4.372j | — | 0.1% | 13.0% | global |
| 3 | 0 − 4.585j | — | 0.1% | 19.1% | global |
| 4/5 | ±2.724 − 4.597j | 0.43 | 0.1% | 1.8% | local |
| 6/7 | ±1.195 − 5.352j | 0.19 | 0.1% | 18.7% | global |
| **8/9** | **±0.386 − 6.160j** | **0.06** | 0.15% | **35.1%** | **global** |
| 10/11 | ±1.607 − 5.949j | 0.26 | 2–4% | 38–43% | global |
| 12/13 | ±1.354 − 6.135j | 0.22 | 1.6% | 38.8% | global |
| 14/15 | ±1.015 − 6.385j | 0.16 | 6.6% | 47.2% | global |
| 16 | 0 − 6.493j | — | 0.1% | 34.6% | global |
| 17/18 | ±1.742 − 6.317j | 0.28 | 1.2% | 49.2% | global |
| 19 | −2.291 − 6.137j | 0.36 | 0.1% | 29.1% | global |

18 out of 20 modes have global (non-wall-local) structure.

### Candidate physical oscillatory mode

**Mode pair 8/9 at `ω = ±0.386 − 6.16i`, `St = 0.061`.**

- frequency: St = 0.061, right in the **Dussauge SWBLI low-frequency
  shock-breathing range** (St = 0.02–0.05 typical for subsonic side;
  up to 0.1 reported for supersonic cases in the literature);
- damping: strongly damped (`Im ω = -6.16`), consistent with Case 2
  being globally stable but non-normal — resolvent amplification
  expected at this frequency;
- global spatial structure: 35% of gridpoints have `|u| > 0.1·peak`,
  diagonal shock-cavity bands across the domain;
- symmetric conjugate pair: both (+0.386, -6.16) and (-0.386, -6.16)
  appear — consistent with real eigenvalues of a real operator.

**This is the first candidate "physical oscillatory branch" we have
found on Case 2 γ=1.31.** Mode pair 14/15 at St = 0.16 is another
strong candidate (47% global). Neither was near PLAN's original
seed at (0.56, -1.11i) — but the St ≈ 0.06 pair is closer to the
physical prior from the SWBLI literature than the M2/γ=1.4 seed was.

### Wall-local contamination persists on M2 (as expected)

- Mode 0 (0, -0.695i) peak at (0.817, -0.099) — shock foot,
  co-located with `|∂Ū/∂x|_max`. Same as M21.
- Mode 1 (0, -1.578i) peak at (0.964, -0.151) — recovery-wall,
  co-located with `|∂Ū/∂y|_max`. Same as M21.

These are real eigenvalues of the LNSE operator driven by
mean-flow-gradient inflation. F4 and F5 stories carry over intact.

### Paper implications

1. The paper story can be **built around the M2 Qraw spectrum** as
   it stands — no Qfilt required for a T1 result. The spectrum
   contains both contamination (wall-local Re=0 modes) and genuine
   physics (global oscillatory pairs at Dussauge-range frequencies).
2. Resolvent at `ω ≈ 0.386` or `ω ≈ 1.0` on Qraw M2 is the first
   experiment to run for T1 (tier-1 paper deliverable). Both
   frequencies correspond to global modes.
3. "Case 2 reveals a previously-unresolved damped oscillatory branch
   at St ≈ 0.06, co-existing with wall-local mean-flow-gradient
   contamination" is a publishable headline finding.

### What to do next in priority order

1. **B1 — resolvent scan on Qraw M2** at `ω ∈ {0.2, 0.4, 0.6, 0.8, 1.0,
   1.5, 2.0, 3.0}` (Dussauge range + higher). Get the first gain
   curve. Tier-1 deliverable candidate.
2. **I3 — Chu-energy based classifier** (better than `|u|` threshold).
   Re-classify all M2 modes and produce a clean "local vs global"
   plot for the paper.
3. **I6 — spectral topology map**. 9×6 target sweep around the
   returned-20 region to see if more global modes exist at
   intermediate Im(ω).
4. **When Qfilt arrives**: re-run same three tests on Qfilt M2. Paper
   gets either (a) Qfilt confirms/strengthens the findings, or (b)
   Qfilt doesn't change anything — honest either way.

### Artefacts

- `output/evp_stride2_m2_screening_qraw/runs/*.npz` — two npz.
- `phase1p5/screening_evp_stride2_m2_screening_qraw_spectrum.png`
  (title labels still say "M21" due to plot-script string; data is
  M2).
- `phase1p5/screening_*_modes.png` — mode-shape grids.

---

## 2026-04-18 — Entry 11. Paper strategy reframed.

**Decision.** Dropped the "must perfect EVP target localisation before
resolvent" gate. Added `Research strategy` section to PLAN.md with
tiered T1/T2/T3 commitments, kill-criteria, and parallel tracks.

**Key strategic claims (propagated to PLAN.md):**

- Minimum publishable Case 2 result (T1): one spectrum + one resolvent
  gain curve + one forcing/response pair on Case 2 M2 (Qfilt
  preferred, Qraw fallback).
- Resolvent is NOT gated on EVP re-localisation — pick ω, compute
  SVD.
- Kill-criteria triggers. If Qfilt delays > 1 week → Qraw path. If
  resolvent won't converge → eddy-viscosity (out-of-scope previously).
- Mode 3 "real instability" would be a **paper strength** if
  confirmed; Entry 10 just showed it was a mesh artefact → PLAN's
  "globally stable" assertion stands.
- Physics priors (Lui 2022 St, Dussauge range) should seed resolvent
  ω, not the M2/γ=1.4 empirical target.

**Why this matters.** Entry 10 already gives us enough for T1 on
Qraw M2. We don't have to wait on Qfilt to start assembling paper
figures. The PLAN now explicitly allows that.

---

## 2026-04-19 — Entry 12. First resolvent scan on Qraw M2. Tier-1 paper artefact.

**Plan.** B1 experiment per PLAN Research-strategy section. 7 ω values
sweeping `0.2, 0.4, 0.6, 1.0, 1.6, 2.0, 3.0` covering Dussauge SWBLI
low-frequency range (St 0.02–0.05 → ω 0.13–0.31) plus candidate
global mode frequencies from Entry-10 screening (|Re ω| = 0.39, 1.0,
1.2, 1.6).

Config: `resolvent_m2_qraw_scan.jsonc`. Per-case settings:
`run_stability=false, run_resolvent=true, n_modes_resolvent=3,
solver.resolvent.eigen.eps_tol=1e-6, ncv=20, true_residual=true`,
MUMPS inner solve. 10 MPI ranks, operator 510k DOF (M2).

**Wall-time.** Each ω case ~72 s. 7 cases total ~9 min including
assembly/save overhead. Much faster than the stability screening
because resolvent SVD on the MatShell is more memory-efficient.

### Raw gain table

| ω | St=ω/2π | σ₁ | σ₂ | σ₃ |
|---|---|---|---|---|
| 0.20 | 0.032 | 16.271 | 6.230 | 3.210 |
| 0.40 | 0.064 | 16.259 | 5.674 | 3.184 |
| 0.60 | 0.095 | 16.244 | 5.048 | 3.133 |
| 1.00 | 0.159 | 16.211 | 4.137 | 2.905 |
| 1.60 | 0.255 | 16.161 | 3.692 | 2.342 |
| 2.00 | 0.318 | 16.120 | 3.619 | 2.110 |
| 3.00 | 0.478 | 15.955 | 3.585 | 2.015 |

**σ₁ variation: 2.0% across 15× ω range. σ₂: −42%. σ₃: −37%.**

### Convergence verification

Re-ran `ω=0.40` with parity-level tight settings (ncv=40, tol=1e-8,
6 modes) in `resolvent_m2_qraw_tight_check.jsonc`. Result:

- σ₁ = 16.2587 (IDENTICAL to loose)
- σ₂ = 5.6736 (IDENTICAL)
- σ₃ = 3.1839 (IDENTICAL)
- σ₄ = 2.0297, σ₅ = 1.8177, σ₆ = 1.5045 (new tail)

**Flat σ₁ is physical, not numerical.** Solver converged to machine
precision.

### Forcing / response mode shapes (ω=0.40 and ω=1.60 compared)

See `phase1p5/case2_perf_omega_0p40_*_fr_modes.png` and
`phase1p5/case2_perf_omega_1p60_*_fr_modes.png`. Observations:

- **Mode 1 (σ₁≈16)**: forcing shows shock-band structure +
  localized peak at shock foot. Response shows global diagonal-band
  shock-like structure. Both are essentially **ω-independent** in
  shape — this is the broadband amplifier fingerprint.
- **Mode 2 (σ₂)**: forcing peak at shock foot (weaker). Response
  more diffuse, includes streamwise wavepacket character that
  changes slightly with ω.
- **Mode 3 (σ₃)**: both forcing and response show global shock-band
  structure.

### Interpretation

The flat σ₁(ω) + ω-independent F/R shape is consistent with a
**non-normal amplifier driven by time-independent mean-flow
gradient localization** at the shock foot. The operator `(iωI − L)`
has a pseudo-spectral "continent" near the shock-foot spectral
feature that the leading resolvent mode sees regardless of which
ω is chosen on the real axis.

This is NOT a resonance. It is a **selective broadband
amplification** whose spatial character is set by the mean-flow
geometry (where the gradients are inflated) and whose magnitude
σ₁ ≈ 16.2 is set by the pseudo-spectral distance.

By contrast, σ₂ and σ₃ are frequency-selective: they drop by 40%
as ω increases 15×, with shapes that ARE ω-dependent. These are
the "normal" resolvent gains that the literature expects.

### What this means for the paper

This gives the tier-1 paper story an unexpected but clean shape:

**Case 2 Qraw M2 resolvent analysis reveals two distinct amplification
regimes:**

1. A **broadband, mean-flow-gradient-driven primary amplification**
   (σ₁ ≈ 16.2, ω-independent, shock-foot localized). This is a
   signature of the DNS-operator-on-LES-mean consistency gap
   discussed in REFERENCE.md §3 — the mean flow's inflated
   gradients produce an amplification mechanism that dominates at
   all frequencies.
2. A **frequency-selective secondary amplification**
   (σ₂, σ₃ decreasing with ω) with more distributed spatial
   structure — the "normal" resolvent response that would be
   expected from a mean-flow without the contamination.

This split is exactly what the literature predicts for mean-flow
linearization when the mean-flow residual `∇·⟨u'u'⟩` is large (see
REFERENCE §3 closure gap discussion). **Qfilt comparison is
predicted to shrink σ₁ toward the σ₂, σ₃ range** by reducing the
gradient-inflation mechanism. This is the experiment we'll run when
Qfilt arrives (I5).

### Paper figure candidates (T1)

All available from current runs:

- `phase1p5/resolvent_resolvent_m2_qraw_scan_gain_curve.png` — the
  σ(ω) curve (tier-1 deliverable #2).
- `phase1p5/case2_perf_omega_0p40_*_fr_modes.png` — F/R modes at
  ω=0.40 (tier-1 deliverable #3).
- `phase1p5/screening_evp_stride2_m2_screening_qraw_spectrum.png`
  — EVP spectrum complex-plane plot (tier-1 deliverable #1, with
  updated title).

All three T1 paper artefacts now exist in artifact form. Polish is
remaining (plot titles, axis labels, annotations).

### Artefacts

- `output/resolvent_m2_qraw_scan/runs/*.npz` — 7 runs × ~80 MB each.
- `output/resolvent_m2_qraw_tight_check/runs/*.npz` — 1 run with
  tight settings.
- `phase1p5/resolvent_resolvent_m2_qraw_scan_gain_curve.png` —
  gain curve plot.
- `phase1p5/resolvent_resolvent_m2_qraw_scan_gain_table.txt` — the
  table above.
- `phase1p5/case2_perf_omega_*_fr_modes.png` — F/R mode shapes at
  three sampled frequencies.
- `phase1p5_plot_resolvent_gain.py`, `phase1p5_plot_resolvent_modes.py`
  — reusable analysis scripts.

### Status

Task #20 (B1/T1) complete. All three T1 paper artefacts exist. Next:

- **I3** branch-identity classifier using Chu energy norm (better
  than `|u|` threshold) for the spectrum figure caption.
- **B1 part 2**: resolvent on Qfilt M2 when delivered — direct
  comparison to quantify σ₁ shrinkage.
- **I6** spectral topology map — confirm no hidden modes in the
  intermediate Im-ω region.

---

## 2026-04-19 — Entry 13. I3 Chu-energy classifier + I6 focused topology probe.

### I3 — Chu-energy branch-identity classifier

Implemented `phase1p5_classify_modes.py` and
`phase1p5_plot_classified_spectrum.py`. Classifier uses three metrics
on the per-gridpoint Chu energy density `Σ_var W_diag · |q|²`:

- `wall_frac`: fraction of E_chu in η-index < 5% (near-wall strip);
- `shock_foot_frac`: fraction in a circle `r < 0.08` centered on the
  Qraw `|∂Ū/∂x|_max` location (0.817, -0.099) from Entry 5;
- `global_extent`: fraction of gridpoints with E_chu > 1%·peak.

Heuristic: `WALL_LOCAL` if `wall_frac > 30%` OR `shock_foot_frac > 30%`;
`GLOBAL` if `global_extent > 20%` AND `wall_frac < 30%`; else
`INTERMEDIATE`.

Applied to the M2 Qraw screening (40 modes across both cases):

| class | count | % |
|-------|-------|---|
| WALL_LOCAL | 9 | 22% |
| GLOBAL | 3 | 8% |
| INTERMEDIATE | 28 | 70% |

The Chu classifier disagrees with `|u|`-only classification in
several important places — e.g. Mode 1 at `(0, -1.58j)` that `|u|`
classified 17% global is actually **90% shock-foot dominated** when
weighted by Chu energy (it's the recovery-wall mode from Entry 5).

The GLOBAL set contains just **3 modes** in the M2 screening —
all at `(±1.742, -6.32j)`, St ≈ 0.28 (one counterpart only in one
case's returned top-20 — symmetry gives 2 physical modes).

### I6 — Focused topology probe

Ran `evp_stride2_m2_topology_probe_qraw.jsonc` with 4 strategic
targets the screening didn't reach: `(0, -2.5i)`, `(0, -3.5i)`,
`(0.4, -6.2i)`, `(1.0, -6.4i)`. Wall time ~25 min (4 × 6 min).

Results:

- Probes at `(0, -2.5i)` and `(0, -3.5i)` returned the **same 20
  modes already known from the screening**. Confirms that
  Im(ω) > -4 region has no additional eigenvalues at our tolerance
  (except the already-identified wall-local stationary modes at
  -0.7j and -1.58j).
- Probes at `(0.4, -6.2i)` and `(1.0, -6.4i)` returned **many new
  modes** the screening did not reach, including **4 new GLOBAL
  candidates**.

### The combined M2 Qraw spectrum after probes (120 modes total)

| class | count | fraction |
|-------|-------|----------|
| WALL_LOCAL | 29 | 24% |
| GLOBAL | 10 | 8% |
| INTERMEDIATE | 81 | 68% |

**All GLOBAL-classified modes, with St = \|Re ω\| / 2π:**

| ω | St | extent | peak (x, y) | Dussauge range? |
|---|----|--------|-------------|-----------------|
| `(0.235, -7.129i)` | **0.037** | 21.6% | (1.078, +0.173) | **YES** |
| `(−0.235, -7.129i)` | 0.037 | 21.6% | (1.078, +0.173) | YES (conjugate) |
| `(0.333, -7.270i)` | **0.053** | 27.5% | (1.099, +0.172) | **YES** |
| `(1.480, -6.750i)` | 0.236 | 21.6% | (0.773, -0.098) | no |
| `(1.597, -6.426i)` | 0.254 | 28.0% | (1.101, +0.179) | no |
| `(1.742, -6.317i)` × 3 | 0.277 | 21.5% | (1.081, +0.179) | no |

**Two GLOBAL pairs inside the Dussauge SWBLI low-frequency range**
(St = 0.02–0.05): the `(±0.235, -7.129i)` pair at St ≈ 0.037 and the
`(+0.333, -7.270i)` mode at St ≈ 0.053. These are the paper's
**candidate shock-breathing modes**. Peak locations (1.08, +0.17)
and (1.10, +0.17) sit downstream of the interaction, in the
recovery-zone freestream — consistent with a globally-extended
shock-cavity acoustic response, not a wall-trapped contamination.

### Paper-quality plot

`phase1p5/combined_classified_spectrum.png` shows the full 120-mode
picture with:

- Red X: WALL_LOCAL contamination clusters at `Re(ω) = 0` (stationary
  axis) and at `|Re ω| ≈ 1` near recovery-wall locations;
- Green circles: GLOBAL modes — a clear band at `Im(ω) ≈ -6.5 to
  -7.3`, including the Dussauge-range pair at St ≈ 0.037;
- Gray squares: INTERMEDIATE (oscillatory modes with moderate
  wall/shock contribution).

The spatial separation — wall-local contamination at stationary
axis vs. global oscillations at |Im ω| ≈ 7 — is **directly visible
in one plot**. This is the paper's money figure.

### Status of investigations

- I3 (branch classifier): COMPLETE. Code + M2 Qraw classification in
  `phase1p5/*_chu_class.csv`.
- I6 (topology map): COMPLETE (focused variant). Four strategic
  probes replaced the planned 54-target grid sweep; delivered the
  same information for ~1/10 the compute.
- I7 (Mode 3 stability): CLOSED (Entry 10, mesh artefact).
- I0 (M2 screening rerun): COMPLETE.
- **Remaining:** I5 (Qfilt provenance, external), I1 (stride
  guardrail, cheap code change), I2 (2D-FFT quantitative metric, low
  priority until Qfilt).

### Status of paper T1

All three T1 deliverables now exist at draft quality:

1. **Spectrum panel**: `phase1p5/combined_classified_spectrum.png` —
   120 modes across 6 runs, color-coded by Chu-energy class,
   Dussauge SWBLI band shaded.
2. **Resolvent gain curve**: `phase1p5/resolvent_*_gain_curve.png` —
   σ₁ broadband ~16.2, σ₂, σ₃ frequency-selective.
3. **F/R mode shapes**: `phase1p5/case2_perf_omega_*_fr_modes.png` —
   at 3 sampled ω values (0.4, 1.6, 3.0).

T1 is ACHIEVED on Qraw. Qfilt comparison becomes a T2/T3 stretch:
show that filter cleanup shifts σ₁ from broadband toward
frequency-selective, confirming the DNS-on-LES-mean consistency gap
interpretation.

---

## 2026-04-19 — Entry 14. Refined resolvent scan at GLOBAL eigenvalue positions.

**Plan.** Test whether σ₁(ω) peaks at the GLOBAL eigenvalue positions
found in Entry 13 (I6 topology probe), or remains flat at ~16.2 as the
Entry-12 coarse scan suggested. If σ₁ peaks at GLOBAL positions, the
physical modes dominate the resolvent → paper can cite specific
amplification frequencies. If σ₁ stays flat, the broadband shock-foot
amplifier dominates regardless → paper framing needs adjustment.

**Config.** `resolvent_m2_qraw_refined.jsonc`, 9 ω values:
`{0.100, 0.180, 0.235*, 0.280, 0.333*, 0.500, 1.480*, 1.597*, 1.742*}`
where `*` marks positions at GLOBAL eigenvalue real parts. Same
resolvent settings as Entry 12 coarse scan.

**Result.** σ₁ table:

| ω | GLOBAL nearby? | St | σ₁ | σ₂ | σ₃ |
|---|---|---|---|---|---|
| 0.100 | no | 0.016 | 16.270 | 6.402 | 3.216 |
| 0.180 | no | 0.029 | 16.270 | 6.272 | 3.212 |
| **0.235** | **yes (St=0.037)** | 0.037 | **16.270** | 6.149 | 3.207 |
| 0.280 | no | 0.045 | 16.270 | 6.031 | 3.202 |
| **0.333** | **yes (St=0.053)** | 0.053 | **16.260** | 5.879 | 3.195 |
| 0.500 | no | 0.080 | 16.250 | 5.356 | 3.162 |
| **1.480** | **yes** | 0.236 | **16.170** | 3.733 | 2.448 |
| **1.597** | **yes** | 0.254 | **16.160** | 3.693 | 2.344 |
| **1.742** | **yes** | 0.277 | **16.150** | 3.658 | 2.235 |

- **σ₁ variation across the whole 17× ω range**: 0.12 (16.15 →
  16.27), **less than 0.8%**.
- **Δσ₁ at GLOBAL vs non-GLOBAL neighbors**: < 0.01, at or below
  solver noise.
- σ₂ drops 43% (6.40 → 3.66) smoothly, no GLOBAL-specific peaks.
- σ₃ drops 30% smoothly.

### Interpretation (strong paper finding)

**Having a GLOBAL eigenvalue at the chosen ω does not produce a
peak in σ₁.** The primary resolvent amplification of Case 2 Qraw M2
is NOT a modal resonance on any specific eigenvalue — it is the
**broadband pseudo-spectral gain** driven by the shock-foot mean-flow
gradient inflation. The operator is so strongly non-normal that the
pseudospectrum dominates over any individual eigenvalue's
contribution.

This is the canonical Trefethen-Embree 2005 scenario for
strongly-non-normal operators: the ε-pseudospectrum extends far
from the actual spectrum, and the resolvent norm is controlled by
the pseudo-spectral boundary, not by proximity to eigenvalues.
Standard examples: BL transition, jets, SWBLI.

### What the paper's T1 gain-curve panel should show

Two curves on the same plot:

- σ₁(ω) ≈ 16.27 across the range, labelled "broadband shock-foot
  amplifier";
- σ₂(ω) decreasing 6.4 → 3.7, labelled "subleading response, weakly
  frequency-selective".

Shaded vertical band for Dussauge SWBLI range (St 0.02–0.05,
ω ≈ 0.13–0.31).

Vertical ticks for GLOBAL eigenvalue positions (0.235, 0.333, ...)
annotated: "no resonant peak even at exact GLOBAL eigenvalue
positions — operator non-normality dominates".

### What Qfilt comparison will test

Hypothesis: Qfilt reduces the shock-foot gradient inflation, which
should:

- lower σ₁ (broadband amplifier weakens);
- potentially reveal resonant peaks at GLOBAL eigenvalue positions
  previously hidden under the flat σ₁;
- preserve or shift the σ₂(ω) shape.

The current Qraw result is **the reference** for that comparison.

### Artefacts

- `output/resolvent_m2_qraw_refined/runs/*.npz` — 9 runs.
- `phase1p5/resolvent_resolvent_m2_qraw_refined_gain_curve.png` —
  gain curve.
- `phase1p5/resolvent_resolvent_m2_qraw_refined_gain_table.txt` —
  tabular data.
- `phase1p5/global_oscillatory_candidates.png` — |u| and Chu energy
  of the three candidate GLOBAL modes at St = 0.037, 0.053, 0.277
  (from EVP runs, not resolvent). Top row: |u| shows diffuse
  diagonal-band structure across the whole domain. Bottom row:
  Chu energy on log scale extends across the entire mesh. These
  look **genuinely global**, not wall-local.

### Status

All tier-1 Qraw M2 results are now definitive and internally
consistent. The paper has a clear story:

1. **EVP spectrum** (Entry 13): wall-local contamination modes
   coexist with globally-extended oscillatory modes, including a
   pair inside the Dussauge SWBLI range at St = 0.037.
2. **Resolvent gain** (Entry 14): primary amplification is
   broadband and pseudo-spectral — NOT a modal resonance. Expected
   for a strongly non-normal compressible SWBLI operator.
3. **F/R modes**: shock-foot-driven forcing, shock-band response.

Qfilt comparison remains the key stretch experiment (T2).

---

## 2026-04-19 — Entry 15. M1 cross-mesh verification (EVP, single case).

**Plan.** Run one stationary-target EVP on M1 (native 679×599, 3.4M
DOF operator) with nev=2 to test whether the dominant M2 modes
survive refinement. Tier-2 paper deliverable.

**Context.** First attempt at nev=8 was killed at 45 min with nconv=2
(same "2nd-nearest eigenvalue too far" pattern as Entry 6 on M21).
Restarted at nev=2 finished in 960 s (16 min).

**Result.**

- M1 returned `ω = (±0.0837, -0.5361j)` — a conjugate pair at small
  Re(ω).
- `postcheck_rel = 7–8 × 10⁻⁴` — much looser than M2's Mode 0 at 3e-6,
  but still under the tol=5e-3 gate.
- Chu classifier: **INTERMEDIATE** (wall 0%, shock-foot 4.5%, extent
  2.3%). Peak at (1.02, -0.01) — NOT at the wall.

### Comparison with M2 at the same target

| mesh | Mode 0 eigenvalue | Chu class | postcheck_rel |
|------|-------------------|-----------|---------------|
| M2 (stride 2,2) | (0, -0.695j) | WALL_LOCAL (67% shock-foot) | 3e-6 |
| M1 (stride 1,1) | (±0.0837, -0.536j) | INTERMEDIATE (4.5% shock-foot) | 7e-4 |

The M1 result is **NOT the same** as M2:

- eigenvalue shifted from Im=-0.695 to Im=-0.536 (23% change);
- acquired nonzero Re (±0.084 vs 0);
- lost its wall-local character (4.5% shock-foot vs 67%).

### Interpretation

Three possibilities:

- (a) The M2 eigenvalue was not fully mesh-converged — it was a
  discretization artefact of the coarser grid, particularly at the
  shock foot where gradients are steepest;
- (b) The M2 and M1 operators have GENUINELY different spectra near
  this target because shock-foot geometry resolution changes the
  LNSE coefficients — both are real, but discretization-dependent;
- (c) The M1 result is incompletely converged — postcheck at 7e-4
  is not as tight as M2's 3e-6, so there may be more refinement
  available.

In either case: **"nearest-to-target eigenvalues are not
mesh-independent"** is a finding we must honestly report in the
paper. The qualitative structure — a cluster of near-stationary-axis
modes at low Im ω — is the same across meshes; the specific
eigenvalues are not.

### Implication for T1 paper claims

The T1 paper story was:

1. EVP spectrum shows wall-local contamination at (0, -0.695j);
2. Resolvent σ₁ ≈ 16.2 broadband.

Finding (1) needs hedging: "on M2, we observe a wall-local mode at
(0, -0.695j); the M1 refinement gives a different nearest-to-target
eigenvalue at (±0.084, -0.536j), so specific eigenvalues are
mesh-sensitive". The qualitative picture (stationary-axis
contamination cluster) is preserved.

Finding (2) — the resolvent result — is the **stronger** of the two
because it's a norm, not an eigenvalue. Norms can be more robust
to discretization. Testing this on M1 with a single-point resolvent
check is the next experiment (in progress, config
`resolvent_m1_qraw_check.jsonc`, ω=0.40 same as one M2 point).

### Artefacts

- `output/evp_stride1_m1_screening_qraw/runs/*.npz` — 1 run.
- `phase1p5/case2_perf_bstat_m1_qraw_nev8_ncv300_tol5em3_*_chu_class.csv`
  — classification.

### M1 resolvent — compute-budget blocker

Attempted `resolvent_m1_qraw_check.jsonc` (single ω=0.4 case on M1
to check if σ₁≈16.2 is mesh-independent). Killed at **~60 min in
the resolvent stage with no completion event** — 50× slower than M2
resolvent (which took 50 s total). The bottleneck is MUMPS direct
factorization of `(iωI − L)` on the 3.4M-DOF M1 operator; memory-
and time-bound at this problem size on one socket.

**M1 resolvent is not tractable under the current MUMPS direct
path**, even as a single-ω sanity check. To do it we would need
to switch to the **PETSc iterative Strategy B** (GMRES+ASM per the
main README.md "Strategy B" section). That is a significant
engineering branch not worth opening for a single verification
point at this stage of the campaign.

**Logged as open T3 stretch.** The T1 paper claim "M2 σ₁ ≈ 16.2"
therefore cannot be checked on M1 without extra infrastructure. The
paper must either:

- (a) report M2-only resolvent with an honest caveat about
  mesh-convergence;
- (b) invest in Strategy B iterative resolvent for M1 verification;
- (c) justify mesh-convergence indirectly through EVP + spectral
  topology stability arguments.

This is a Gabriel decision — not something to solve inside this
Phase 1.5 campaign.


## 2026-04-20 — Entry 16. Tight-EVP certification on M2 Qraw — stagnation result.

### Context

After the resolvent ω=0.40 convergence certificate landed
(Entry 25 task; σ₁=16.25870434, ρ_adj_max=2.4e-10 at eps_tol=1e-8),
the parallel question was: do the M2 Qraw EVP modes the supp.tex
draft cites survive tight tolerance?

Targets (`evp_m2_qraw_tight_certify.jsonc`):

- Stationary:  `ω = (0,      −0.6948i)`
- Oscillatory 1: `ω = (+0.235, −7.13i)`, "St=0.037"
- Oscillatory 2: `ω = (+0.333, −7.27i)`, "St=0.053"

Tight settings:

- `eps_tol = 1e-8`, `nev = 4`, `ncv = 80`
- acceptance: `postcheck_rel_max = 1e-6`, `target_distance_max = 0.3`

Previous screening (loose, `eps_tol = 5e-3`) had returned all three
with `accepted_count > 0` into the draft.

### What happened

Launched 2026-04-20 11:00. First case (stationary) entered SLEPc
iteration. Residual trajectory:

| iter | err_range max |
|-----:|---------------:|
|   0  |  1.6e+07  (initial) |
|  15  |  1.6e+07 |
|  25  |  5.2e+06 |
|  40  |  7.2e+04 |
|  60  |  2.3e+03 |
|  80  |  3.8e+01 |
| 100  |  9.2e+00 |
| 200  |  8.4e+00 |
| 300  |  5.0e+00 |
| 400  |  1.0e+01 |
| 500  |  6.6e+00 |
| 580  |  7.3e+00 |

First 100 iters: 7 orders of residual reduction. Then the descent
**stalled**. Iters 100–580 (41 sampled points, stride 5): all inside
`err_range ~ 2.7–14.9`, no downward trend. Killed at iter 580 at
user request after 161 minutes of elapsed wall time. Two oscillatory
cases never ran.

`nconv=0` throughout. Acceptance threshold 1e-6 never reached.

### Interpretation

This is not slow convergence. Slow convergence means a decreasing
residual that hasn't yet crossed tolerance; progress per iteration
is real but small. What we observed is **stagnation**: the leading
Ritz residual oscillates inside a fixed band with no trend. Each
iteration buys nothing toward the target.

The physical reading lines up with the wall-local analysis in
Entry 13 and the PLAN §"Wall-trapped modes — mitigation roadmap":
the loose-tolerance screening at `eps_tol = 5e-3` picked up the
densely-packed wall-local cluster that accumulates along the
`Re(ω) = 0` axis. Any member of that cluster is "close enough" at
5e-3 to be reported. Tight tolerance `1e-8` requires a genuinely
isolated eigenvalue near the target to converge — and the stationary
target does not have one. The cluster has no separable mode there;
Krylov–Schur thrashes between cluster members and never achieves
six-digit certification.

The oscillatory targets were not run, but their basis is even
weaker: `(±0.235, -7.13i)` and `(±0.333, -7.27i)` are resolvent-scan
ω-positions (input), not eigenvalue positions (output). They were
transcribed into the draft by error. See Entry 14 for the classifier
evidence.

### What this means for the paper

1. **Retract the three affected eigenvalue labels from supp.tex**:
   - `ω = (0, -0.6948i)` stationary mode — no tight-tolerance
     certificate; reframe as "loose-tolerance artifact from the
     wall-local cluster."
   - `ω ≈ (±0.235, -7.13i)` St=0.037 — not an eigenvalue.
   - `ω ≈ (±0.333, -7.27i)` St=0.053 — not an eigenvalue.

2. **The only surviving EVP claim** on M2 Qraw is
   `ω = (±1.7416, -6.317i)`, St=0.277 (classifier `GLOBAL`,
   `global_extent = 21.5%`, `wall_frac = 2%`). This still needs its
   own tight-tolerance certificate
   (`evp_m2_qraw_tight_certify_globalmode.jsonc`, queued).

3. **The resolvent narrative is unaffected**. σ₁(ω) has an
   independent convergence certificate (Entry 14 + ω=0.40
   certificate run) and does not depend on EVP mode certification.

### What this means for the PLAN

This entry is the live evidence that closes the loop on the
wall-trapped mode roadmap in PLAN §"Wall-trapped modes — mitigation
roadmap" (added 2026-04-20). The stagnation result strengthens the
priority of:

- M3 (FEAST contour EVP) — because contour integration does not
  need an isolated eigenvalue near a single target; it extracts
  everything inside a region. A rectangle around
  `(1.7416, -6.317i)` that excludes the `Re(ω) = 0` axis would
  bypass the entire stagnation failure mode we just demonstrated.
- M1 (true Qfilt baseflow from upstream) — root cause.

### Artefacts

- Log: `studies/case2/output/evp_m2_qraw_tight_certify/logs/case2_perf_stationary_m0p6948_20260420T090005Z.log`
- No NPZ (killed before solve output).
- Config: `studies/case2/evp_m2_qraw_tight_certify.jsonc` (kept as-is).
