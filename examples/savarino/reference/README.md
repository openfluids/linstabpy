# Savarino resolvent reference data — θ = 30.8°

Cite Zenodo [10.5281/zenodo.18854407](https://doi.org/10.5281/zenodo.18854407).

## `baseflow/`

Savarino's published Tecplot mean-flow file — not tracked (the Zenodo record
states no redistribution license). Download with `python fetch_zenodo.py baseflow`,
then convert with `python savarino_to_npy.py` → `output/baseflow/` (gitignored).

## Files in `reference/`

| File | Contents |
|------|----------|
| `savarino_grid.csv` | (f, β) grid recovered from Zenodo mode listing. |
| `savarino_forcing_mask_measured.csv` | Forcing-region bounds from published modes. |

The 200-point gain-map run record lives at `../savarino_map3_200pt.csv`.

Figures go to `output/figures/` (not shipped here).

## Built locally, not tracked

| File | Contents |
|------|----------|
| `savarino_response_mack.npz` | Savarino resolvent response at (f, β) = (2, 45). |
| `savarino_response_liftup.npz` | Same at (f, β) = (1e-4, 163). |

Both come from the Zenodo modes, so they are not shipped. Build them with
`python fetch_zenodo.py modes --f 2 --beta 45`,
`python fetch_zenodo.py modes --f 0.0001 --beta 163`, then
`python pack_reference_modes.py`. The Tecplot mode files land in `zenodo/`
(gitignored).
