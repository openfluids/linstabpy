# Song SWBLI base flow, shock angle 31.6° (600×200 cells)

Base flow and reference global-stability results for the same
interaction as `../song_32deg/`, at shock angle θ = 31.6° — below the
critical angle, globally stable. Reference for the sub-threshold row of
the stability-bracket comparison.

Data courtesy of Ziming Song, redistributed with permission. Please cite
Song & Hao, *Physics of Fluids* 35(8):084121 (2023).

## Contents

| File | What it is |
|---|---|
| `grid.grd` | Grid-point coordinates, 600 × 200 cells. |
| `mean_flow.dat` | Cell-centre mean flow with one ghost layer, ordered `((RHO(I,J),I=0,NX),J=0,NY)` then U, V, P, T, MU blocks. |
| `eigenvalues.dat` | Reference eigenvalue spectrum at spanwise wavenumber β = 0.238/mm (βX_sh = 19.04 in the X_sh = 80 mm scaling; ω_i > 0 = unstable, mm-based units — multiply by 80 for the X_sh-based scaling). |
| `eigenmode1_real.dat`, `eigenmode1_imag.dat` | Real and imaginary parts of the leading (least-damped, stationary) global eigenmode. |
| `converted/*.npy` | The primitive fields, coordinates, and sponge mask on the grid used by the centred finite-difference operators. |
