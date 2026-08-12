# Martini example family

Six LinStab2D examples used for Table S1.

Run from the repository root:

```
./run_all_examples.sh
.venv/bin/python examples/martini/compare_martini_examples.py
```

MATLAB reference values sit in `*_values.npz`. Mode arrays sit in `*_modes.npz` at complex64. The measured max relative deviation after that cast is in `martini_c64_max_rel.txt`.

The supplement SOURCE line for Example 3 spatial uses the paper spatial file:

```
LINSTABPY_EXAMPLE_CONFIG=examples/martini/example_3_paper_spatial_Lxpi_nP2.yaml \
LINSTABPY_EXAMPLE_STEM=example_3_paper_spatial_Lxpi_nP2 \
.venv/bin/python examples/martini/example_3_streaky_bl.py
```
