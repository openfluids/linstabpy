# Song / Robinet SWBLI

Base flows and reference eigenvalues from Z. Song, redistributed with permission.
Cite Song & Hao, Physics of Fluids 35:084121 (2023).

## Convention

The 32 deg field in `song_32deg.npz` is velocity / c_inf and temperature / T_inf:
u_max = Ma = 2.15, T_min = 1. The 31.6 deg field in `song_316deg.npz` is the
freestream-1 copy as received (u_max = 1, T_min = 1). The nodal grid is 599 x 301
(32 deg) and 599 x 201 (31.6 deg). The raw Plot3D grids are 601 x 301 and 601 x 201.

## Figures

```
.venv/bin/python examples/song/song_fig1_baseflow.py
.venv/bin/python examples/song/song_fig2_spectrum.py
.venv/bin/python examples/song/song_fig3_eigenmode.py
```

Fig S2 plots the lead pair quoted in the supplement and Song's growth-rate curve.
The full 20-mode table at beta = 19.2 is not in the machine-readable drop.
Fig S3 plots Song's leading mode. The linstabpy partner field from that run is
not in this repository.

## EVP

```
.venv/bin/python -m examples.song.song
.venv/bin/python examples/song/compare_evp_to_song316.py examples/song/song_316_eigenvalues.dat <run.npz>
```
