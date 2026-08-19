# Song (2023) reference data — θ = 32°

Redistributed with permission from Z. Song. Cite Song & Hao, *Phys. Fluids* **35**, 084121 (2023).

| Path | Contents |
|------|----------|
| `baseflow/song_32deg.npz` | Nodal mesh and mean fields (ρ, u, v, w, T, sponge). Velocity / *c*∞, *T* / *T*∞; *u*<sub>max</sub> = Ma = 2.15. Grid 599 × 301. |
| `eigenmode/song_32deg_leading_mode.npz` | Leading eigenmode primitives on the Song grid (real parts in packed arrays). |
| `spectrum/eigenvalues.dat` | 20 reference eigenvalues at β = 19.2 (ω<sub>r</sub>, ω<sub>i</sub> pairs; multiply by 80 for LinStabPy nondimensionalization). |
| `spectrum/omega_i.csv` | Song β–ω<sub>i</sub> curve (λ<sub>z</sub>, ω<sub>i</sub> pairs). |

LinStabPy EVP outputs (`output/`, `inputs/baseflow/`) are **regenerated locally**
per `examples/song/README.md` and are gitignored. Validation figures under
`figures/` are shipped with the example.
