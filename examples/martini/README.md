# Martini example family

Six LinStab2D examples used for Table S1.

Run from the repository root:

```
./run_all_examples.sh
.venv/bin/python examples/martini/compare_martini_examples.py
```

MATLAB reference values sit in `*_values.npz`. Mode arrays sit in `*_modes.npz` at complex64. The measured max relative deviation after that cast is in `martini_c64_max_rel.txt`.
