Place Song (2023) finite-volume exports here:

- `grid.grd` — Plot3D vertex coordinates (601 × 301)
- `mean_flow.dat` — cell-centre primitives

These files are not redistributed with LinStabPy. After copying them, run
``prepare_baseflow.py`` from the parent ``examples/song/`` directory.
Without them, ``prepare_baseflow --from-reference`` starts from the shipped
nodal base flow in ``reference/baseflow/song_32deg.npz``.
