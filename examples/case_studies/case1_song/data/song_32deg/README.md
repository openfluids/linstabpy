# Song & Hao SWBLI base flow, shock angle 32° (601×301 grid)

Base flow and reference eigenvalue data for the oblique
shock-wave/laminar-boundary-layer interaction of Song & Hao,
*Physics of Fluids* 35(8):084121 (2023), at shock angle θ = 32° —
above the critical angle, globally unstable.

Data courtesy of Ziming Song, redistributed with permission. Please cite
the paper above when using it.

## Contents

| File | What it is |
|---|---|
| `grid.grd` | Plot3D ASCII grid, 601 × 301 × 1 nodes (600 × 300 cells). |
| `mean_flow.dat` | Mean-flow field associated with that grid (ASCII). |
| `omega_i.csv` | Reference EVP dispersion relation: column 1 = spanwise wavenumber β, column 2 = temporal growth rate ω_i. |
| `converted/*.npy` | The primitive fields (ρ, u, v, w, T), coordinates (x, y), and sponge mask interpolated to the nodal grid used by the centred finite-difference operators. |

Freestream normalization: ρ, u, T equal one in the freestream.
