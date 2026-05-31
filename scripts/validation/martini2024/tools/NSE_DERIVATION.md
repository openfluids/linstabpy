# Independent LNSE Derivation: Governing Equations and Conventions

This document records the governing equations, nondimensionalisation, and
linearisation procedure used to build `cartesian_lnse_coefficients_from_nse`
in `symbolic_compressible_nse.py`.  The derivation is independent of
`coefficients_reference_base.py` and `coefficients.py`; it serves as a
first-principles check.

---

## Coordinate system and variable ordering

Physical coordinates:
- x: streamwise
- y: wall-normal
- z: spanwise (Fourier-transformed direction)

Primitive state vector: **q = [ρ, u, v, w, T]** where u, v, w are the
streamwise, wall-normal, and spanwise velocity components.

Base flow (capital letters): Q̄ = (RHO, U, V, W, T, MU, P).
Perturbation (primed): q' = (ρ', u', v', w', T').

The base flow is **2-D**: all z-derivatives of base-flow quantities vanish,
i.e. ∂U/∂z = ∂V/∂z = ∂W/∂z = ∂RHO/∂z = ∂T/∂z = 0.

---

## Nondimensionalisation

- Length: L_ref (reference length, e.g. chord or bubble length)
- Velocity: U_inf (free-stream velocity)
- Density: ρ_inf
- Temperature: T_inf
- Pressure: ρ_inf U_inf²  (some texts use ½ρ_inf U_inf², absorbed in kappa·Ma²)

Resulting nondimensional parameters: Re, Ma, Pr, κ.

Ideal-gas equation of state:

    p = ρ T / (κ Ma²)

So the base pressure P = RHO·T/(κ·Ma²).

---

## Viscous stress tensor (variable viscosity)

    τ_ij = μ (∂u_i/∂x_j + ∂u_j/∂x_i) − (2/3) μ δ_ij div(u)

where div(u) = ∂u/∂x + ∂v/∂y + ∂w/∂z.

Viscosity depends on temperature: μ = μ(T), with dμ/dT = dmudT and
d²μ/dT² = d2mudT2 (both provided as base-flow fields).

---

## Governing equations (nondimensional)

**Continuity:**
    ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y + ∂(ρw)/∂z = 0

**Momentum (x-component):**
    ρ Du/Dt = −∂p/∂x + (1/Re) [∂τ_xx/∂x + ∂τ_xy/∂y + ∂τ_xz/∂z]

**Momentum (y-component):**
    ρ Dv/Dt = −∂p/∂y + (1/Re) [∂τ_xy/∂x + ∂τ_yy/∂y + ∂τ_yz/∂z]

**Momentum (z-component):**
    ρ Dw/Dt = −∂p/∂z + (1/Re) [∂τ_xz/∂x + ∂τ_yz/∂y + ∂τ_zz/∂z]

**Energy (temperature form):**
    ρ DT/Dt = −(κ−1) T ρ div(u)
            + κ/(Re·Pr) [∂(μ ∂T/∂x)/∂x + ∂(μ ∂T/∂y)/∂y + ∂(μ ∂T/∂z)/∂z]
            + κ(κ−1) Ma²/Re · Φ

where the viscous dissipation function is:

    Φ = μ [2(∂u/∂x)² + 2(∂v/∂y)² + 2(∂w/∂z)²
           + (∂u/∂y + ∂v/∂x)² + (∂u/∂z + ∂w/∂x)² + (∂v/∂z + ∂w/∂y)²
           − (2/3)(div u)²]

---

## Linearisation

Reynolds decomposition: q(x,y,z,t) = Q̄(x,y) + q'(x,y,z,t).

Substitute into the NSE, retain O(q') terms.  Base-flow z-derivatives vanish,
so no base-flow advection in z appears.

**Fourier ansatz in z:**
    q'(x,y,z,t) = q̂(x,y) exp(i·kx·z − i·ω·t)

After applying the ansatz:
- ∂/∂z on perturbations → i·kx
- ∂/∂t on perturbations → −i·ω

Here `kx` is the **z-spanwise wavenumber** (symbol name matches the code).

---

## System convention

The LNSE is cast as:

    B ∂q'/∂t = L q'

where L collects all spatial operators acting on q'.  In the code:

    B[i,j] = i · (coefficient of ∂q_j'/∂t in equation i)

The factor i = sp.I comes from the EVP convention: substituting
q' ∼ exp(−iωt) gives B(−iω)q̂ = Lq̂, so the EVP is L q̂ = −iω B q̂.
For the continuity row, the coefficient of ∂ρ'/∂t is 1, hence B[0,0] = i.

---

## Coefficient blocks

For each equation row i and perturbation column j (j = 0=ρ, 1=u, 2=v, 3=w, 4=T):

    eq_i = Σ_j [A0[i,j] q̂_j + Ax[i,j] D_x q̂_j + Ay[i,j] D_y q̂_j
               + Axx[i,j] D_xx q̂_j + Ayy[i,j] D_yy q̂_j + Axy[i,j] D_xy q̂_j]
           = −iω Σ_j B[i,j] q̂_j

The i·kx Fourier substitution converts any ∂/∂z on q̂ to multiplication by
i·kx; the result enters A0 (zeroth-order in spatial operators).

---

## MATLAB internal ↔ physical coordinate mapping

The MATLAB `LinStab2D` code (and its Python translation `coefficients.py`)
uses an **internal** coordinate system that differs from the physical one:

| Physical      | MATLAB internal |
|---------------|----------------|
| x (streamwise)| z_c            |
| y (wall-normal)| r_c           |
| z (spanwise)  | −x_c           |
| U (streamwise)| W_c            |
| V (wall-normal)| V_c           |
| W (spanwise)  | −U_c           |
| kx (phys, spanwise wavenumber) | −kz_c |

The production code maps physical inputs to the MATLAB internal system,
computes coefficients, then remaps back via a permutation matrix T and a
sign flip.  The `cartesian_lnse_coefficients_from_nse` function works
**entirely in the physical coordinate system** and never uses the internal
mapping — making it a genuinely independent check.

---

## Continuity row as anchor

The entries returned by `continuity_operator_coefficients` in
`symbolic_compressible_nse.py` are the anchor for verifying the derivation
framework.  Any mismatch between `cartesian_lnse_coefficients_from_nse` and
the anchor in the continuity row indicates an error in the conventions
(sign, coordinate system, or Fourier ansatz) of the new derivation, not a
disagreement with the production code.

---

## Honesty note

If `cartesian_lnse_coefficients_from_nse` disagrees numerically with
`compute_cartesian_coefficients` on any of the 175 entries, the test
`test_cartesian_coefficients_match_independent_nse_derivation` reports the
exact (block, i, j, symbolic expression, numeric values) and fails.  No
entry is silently adjusted to force agreement; disagreements are surfaced as
potential code or upstream-MATLAB errors.
