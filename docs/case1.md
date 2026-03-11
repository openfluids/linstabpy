# Case 1 (SWBLI) Integration (Python)

## Objective
Run Case 1 compressible case with `linstab2d` and compare temporal spectrum against MATLAB reference output.

## Inputs
- Case arrays: `case1/*.npy`
- MATLAB reference spectrum: `case1/MATLAB_results/spectrum.mat`

## Run the Python example
```bash
.venv/bin/python case1.py
```

Optional environment overrides:
- `LINSTAB2D_CASE1_CASE_DIR` (default: `case1`)
- `LINSTAB2D_CASE1_RUN_RESOLVENT` (default: `false`)
- `LINSTAB2D_CASE1_RUN_STABILITY` (default: `true`)
- `LINSTAB2D_CASE1_N_MODES` (default: `100`)
- `LINSTAB2D_CASE1_N_MODES_RESOLVENT` (default: `10`)
- `LINSTAB2D_CASE1_STRIDE_KSI`, `LINSTAB2D_CASE1_STRIDE_ETA` (default: `1`)
- `LINSTAB2D_CASE1_BC_PENALTY` (default: `1e8`)

Output:
- `case1.npz`

## Compare Python vs MATLAB spectrum
```bash
.venv/bin/python check_spectrum.py \
  --python-npz case1.npz \
  --matlab-spectrum case1/MATLAB_results/spectrum.mat
```

Parity artifact:
- `results/case1/output/case1_spectrum_parity.json`

## Smoke test
```bash
.venv/bin/python -m pytest tests/test_case1.py -q
```
