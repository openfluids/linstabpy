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

The repo ships the spectrum table and a few mode fields, not the full EVP dump
(`song_32deg_paper.npz` is local, regenerable, gitignored).

```
mpiexec -n 4 python -m examples.song.song --config examples/song/song_32deg_paper.yaml
.venv/bin/python examples/song/pack_run.py
```

`song_32deg_spectrum.csv` is the 20-mode table from that extract.
`song_32deg_modes.npz` holds the three least-damped modes (real primitive
fields on the run grid). Fig S2 plots Song's quoted lead and that table.
Fig S3 plots Song's leading mode beside the packed linstabpy lead.

The supplement quotes a matched lead pair in `song_32deg_spectrum_quoted.csv`
(`0.05417 i` vs `0.05696 i`). This extract's least-damped root is `0.438 i`.

## EVP

```
.venv/bin/python -m examples.song.song
.venv/bin/python examples/song/compare_evp_to_song316.py examples/song/song_316_eigenvalues.dat <run.npz>
```
