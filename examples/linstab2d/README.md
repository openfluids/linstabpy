# Martini example family

Six LinStab2D examples used for Table S1.

Run from the repository root:

```
./run_all_examples.sh
.venv/bin/python examples/linstab2d/compare_martini_examples.py
```

MATLAB reference values sit in `*_values.npz`. Mode arrays use **complex128** (same as linstabpy and full campaign fixtures).

Regenerate MATLAB fixtures (requires `LINSTAB2D_ROOT` + MATLAB):

```bash
bash examples/linstab2d/regenerate_martini_fixtures.sh
```

The supplement SOURCE line for Example 3 spatial uses the paper spatial file.
By default ``example_3_streaky_bl.py`` runs all four paper YAML campaigns.
``example_3_plots.py`` writes the presentation figures from those NPZ files,
using ``LinStab2DResults/example_3_paper_*`` MATLAB fixtures. Each YAML pins
``floquet_shift: matlab_parity`` so spectra/modes match LinStab2D; the library
default elsewhere is continuum-correct Floquet FD.

```
.venv/bin/python examples/linstab2d/example_3_streaky_bl.py
.venv/bin/python examples/linstab2d/example_3_plots.py
```

Single-case override:

```
LINSTABPY_EXAMPLE_CONFIG=examples/linstab2d/example_3_paper_spatial_Lxpi_nP2.yaml \
LINSTABPY_EXAMPLE_STEM=example_3_paper_spatial_Lxpi_nP2 \
.venv/bin/python examples/linstab2d/example_3_streaky_bl.py
```

Examples 1, 2 and 5 run a resolvent and compare complex modes with the LinStab2D
MATLAB references. Their YAML files set `solver.resolvent_sign_convention: matlab_parity`,
the LinStab2D sign `L0 - (gamma + i*omega) I`, so the modes match the references.
The library default is `physical`, `(gamma - i*omega) I - L0`; see
`wiki/resolvent-sign-convention.md`. Gains do not depend on the choice here: the base flows of all three
have no spanwise or azimuthal velocity (`W = 0`). Examples 3 and 4 run no resolvent.

Validation audit CLIs (assembly/symbolic parity) live under `tests/tools/`; see
`tests/tools/README.md`.
