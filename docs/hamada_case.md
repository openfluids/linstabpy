# Hamada Case Integration (Python)

## Objective
Run Hamada's compressible case with `linstab2d` and compare temporal spectrum against MATLAB reference output.

## Inputs
- Case arrays: `hamada_test/*.npy`
- MATLAB reference spectrum: `hamada_test/MATLAB_results/spectrum.mat`

## Run the Python example
```bash
.venv/bin/python -m linstab2d.examples.example_6_hamada
```

Optional environment overrides:
- `LINSTAB2D_HAMADA_CASE_DIR` (default: `hamada_test`)
- `LINSTAB2D_HAMADA_RUN_RESOLVENT` (default: `false`)
- `LINSTAB2D_HAMADA_RUN_STABILITY` (default: `true`)
- `LINSTAB2D_HAMADA_N_MODES` (default: `100`)
- `LINSTAB2D_HAMADA_N_MODES_RESOLVENT` (default: `10`)
- `LINSTAB2D_HAMADA_STRIDE_KSI`, `LINSTAB2D_HAMADA_STRIDE_ETA` (default: `1`)
- `LINSTAB2D_HAMADA_BC_PENALTY` (default: `1e8`)

Output:
- `linstab2d/examples/example_6_hamada.npz`

## Compare Python vs MATLAB spectrum
```bash
.venv/bin/python tools/parity/check_hamada_spectrum.py \
  --python-npz linstab2d/examples/example_6_hamada.npz \
  --matlab-spectrum hamada_test/MATLAB_results/spectrum.mat
```

Parity artifact:
- `artifacts/hamada_spectrum_parity.json`

## Smoke test
```bash
.venv/bin/python -m pytest tests/test_hamada_case_smoke.py -q
```
