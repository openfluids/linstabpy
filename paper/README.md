# Paper workspace (`paper`)

This directory is the active paper workspace for the current two-file split:

- `brief.tex`: front-paper brief communication
- `supp.tex`: Supplementary Material
- `refs.bib`: shared bibliography for both targets

## Files

- `brief.tex`: concise framework paper with references to the Supplementary Material.
- `supp.tex`: technical derivations, fidelity/validation structure, solver details, and reproducibility map.
- `refs.bib`: shared bibliography.
- `figs/example_grid.py` and `figs/example_grid.png`: maintained local figure assets still used by `supp.tex`.

Legacy draft sources from the pre-split paper workflow have been removed from
this directory. The active paper workspace now consists only of the brief/supp
pair, their shared bibliography, and the figure assets still referenced by the
supplement.

## Build

Build both active targets:

```bash
cd paper
make
```

Build a single target:

```bash
cd paper
make brief
make supp
```

## Notes

- `supp.tex` points directly at existing artifact paths under `examples/`,
  `validation/`, `results/`, and `paper/figs/` instead of duplicating figures
  under the paper workspace.
- Missing future artifacts should remain explicit placeholders in the TeX
  source until the generating workflow is finalized.
