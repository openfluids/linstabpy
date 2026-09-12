# Structured mesh

Owner: `linstabpy.mesh.mesh`. `Mesh` is the structured-grid object.

It holds geometry, SBP derivative operators, filters, dissipation state,
and sponge state. Deformation and masking must preserve numerical behavior.
Geometry-only optimizations are allowed only when tests prove that
derivative operators, filters, weights, and saved outputs remain equivalent.

## Files

| File | Role |
|---|---|
| `mesh.py` | `Mesh.create`, index maps, masking |
| `sbp.py` | Summation-by-parts operators in use |
| `sbp_generator.py` | Operator generation for orders 2–8 |
| `filters.py` | Mattsson alpha-filter |
| `fdweights.py` | Finite-difference weights |

SBP orders 2, 4, 6, 8. The Nyquist mode on curvilinear meshes is damped by
the alpha-filter (always on in the driver) and, where the metric
cross-terms mix grid-scale content, by operator dissipation. Dissipation
on a rectilinear mesh is a different claim; see
`driver.preflight.metric_cross_coupling`.
