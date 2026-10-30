# Example 5 — `D2y_asymm` mesh deform fix

This note records why **Example 5 (paraboloid resolvent)** required a change in
`linstabpy` itself, not just Martini parity YAML knobs.

## Context

Example 5 runs axisymmetric LNSE with azimuthal mode **`m = 5`**. For `m ≥ 2`,
radial derivatives in `L₀` use **only the antisymmetric SBP blocks** (`Dy_asymm`,
`D2y_asymm`, …), not the full unsplit `D2y`. The paraboloid map is a **non-separable**
2D conformal deformation, so Python takes the `_deform_full_2d` path in
`Mesh.deform`.

## Finding (translation bug, not a design choice)

**MATLAB** (`DeformMesh.m`, lines 150–161) computes the mapping metric derivatives
(`dJ`, hence the `Q` diagonal terms) **once** with the **full** first- and
second-order η operators (`Dy`, `D2y`), then reuses those scalar fields when
updating **both** symmetric and antisymmetric operator families.

**Python (before this fix)** recomputed `dJ`/`Q` inside
`_transform_second_derivatives` using whichever **family operators were passed in**.
For the antisymmetric branch that meant `Dy_asymm` and `D2y_asymm` — pole
stencils applied to the coordinate fields. Physical coordinates have **no η parity**,
so those family stencils produced wrong metric derivatives near the axis. That
error propagated into **`D2y_asymm`**, the operator that actually enters `L₀` at
`m = 5`.

Measured on the deformed paraboloid mesh (100×100, FD4), against the LinStab2D
derivative matrices:

| Matrix | Frobenius rel. error (Python vs MATLAB, before fix) |
|--------|-----------------------------------------------------|
| `Dy`, `Dy_asymm`, full `D2y` | ~10⁻¹⁰ |
| **`D2y_asymm`** | **~4.0×10⁻⁵** |
| `Dxy_asymm`, `Dyx_asymm` | ~3.7×10⁻⁶ |

Patching **only** the metric-term source to use full `Dy`/`D2y` (keeping family
matrices elsewhere) collapsed `D2y_asymm` error to **~2.6×10⁻¹⁰**.

Downstream impact before fix: `‖L₀_py − L₀_mat‖_F / ‖L₀_mat‖_F ≈ 6×10⁻⁶`; max
resolvent gain mismatch ~**0.043%** vs LinStab2D fixtures.

## What we changed in this repository

**File:** `src/linstabpy/mesh/mesh.py`

1. `_transform_second_derivatives` gained optional `Dy_metric` / `D2y_metric`
   arguments used **only** for the `dJ`/`Q` metric-gradient terms.
2. In `_deform_full_2d`, both symmetric and antisymmetric family updates now pass
   the **full** `Dy` and `D2y` as metric operators (MATLAB parity), while the
   family-specific stencils still enter the operator combination as before.

**Test:** `tests/test_mesh.py` —
`test_deform_full_2d_family_metric_terms_use_full_operators` rebuilds the
expected `D2y_{symm,asymm}` from explicit metric diagonals and fails on unfixed
code (130 pole-row mismatches on a toy deform).

Example 5 drivers and plots in `examples/linstab2d/` **do not pin MATLAB derivative
matrices**. They exercise the shipped mesh layer with this fix.

## Reference data

`examples/linstab2d/LinStab2DResults/example_5_Paraboloid_results_*.npz` holds the
bundled LinStab2D reference fixtures for this example.

After applying the mesh fix, recompute `example_5_paraboloid.npz` before treating
gain/mode overlays as authoritative (`rm example_5_paraboloid.npz && python example_5_paraboloid.py`).
