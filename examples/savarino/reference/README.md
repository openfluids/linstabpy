# Savarino resolvent reference data — θ = 30.8°

Cite Zenodo [10.5281/zenodo.18854407](https://doi.org/10.5281/zenodo.18854407).

## `baseflow/`

Savarino's published Tecplot mean-flow file. Convert with `python savarino_to_npy.py`
→ `output/baseflow/` (gitignored).

## Files in `reference/`

| File | Contents |
|------|----------|
| `savarino_response_mack.npz` | Savarino resolvent response at (f, β) = (2, 45). |
| `savarino_response_liftup.npz` | Same at (f, β) = (1e-4, 163). |
| `savarino_grid.csv` | (f, β) grid recovered from Zenodo mode listing. |
| `savarino_forcing_mask_measured.csv` | Forcing-region bounds from published modes. |

The 200-point gain-map run record lives at `../output/savarino_map2_200pt.csv`.

Figures go to `output/figures/` (not shipped here).

## Optional: `zenodo/` (gitignored)

Tecplot mode archives from `fetch_zenodo.py modes` — for `pack_reference_modes.py` only.
