# Packed fields and base flow

Owner for ingest: `linstabpy.baseflow.grid_fields`.
Owner for `BaseFlow`: `linstabpy.baseflow`.

## Eight keys

`x`, `y`, `rho`, `u`, `v`, `w`, `T`, `sponge`. YAML names the file. The
loader does not guess a Song filename.

`load_case_fields` reads a directory of eight `.npy` files or one `.npz`
with the same keys. After an optional transpose the arrays are `(ny, nx)`.
`stride_eta` walks the first index; `stride_ksi` walks the second.

## Preprocess

Filter, boundary sponge, and shock cleanup live in `baseflow.preprocess`.
Sponge generation (`constant`, `parabolic`, `localized`, `smooth_boundary`,
`smooth_localized`) lives in `baseflow.sponge`. Viscosity in
`baseflow.viscosity`.

Martini `.mat` constructors live under
`examples/martini/matlab_baseflow.py`, not in this package path.
