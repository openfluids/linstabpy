# MATLAB vs Python Example Parity

Reference MATLAB examples:
- `linstab2d/data/baseflows/example_0_TestCurvilinearDerivative.m`
- `linstab2d/data/baseflows/example_1_Blasius.m`
- `linstab2d/data/baseflows/example_2_Jet.m`
- `linstab2d/data/baseflows/example_3_Streaky_BL.m`
- `linstab2d/data/baseflows/example_4_Cavity.m`
- `linstab2d/data/baseflows/example_5_Paraboloid.m`

Python counterparts:
- `examples/example_0_curvilinear_validation.py`
- `examples/example_1_blasius.py`
- `examples/example_2_jet.py`
- `examples/example_3_streaky_bl.py`
- `examples/example_4_cavity.py`
- `examples/example_5_paraboloid.py`

Coverage matrix:

| MATLAB example | Python example | Status | Required artifacts |
|---|---|---|---|
| `example_0_TestCurvilinearDerivative.m` | `example_0_curvilinear_validation.py` | `implemented` | `example_0_curvilinear_validation.npz`, `example_0_curvilinear_validation.png`, `artifacts/curvilinear_derivative_parity.json` |
| `example_1_Blasius.m` | `example_1_blasius.py` | `implemented` | `example_1_blasius.npz`, `example_1_blasius.png` |
| `example_2_Jet.m` | `example_2_jet.py` | `implemented` | `example_2_jet.npz`, `example_2_jet.png` |
| `example_3_Streaky_BL.m` | `example_3_streaky_bl.py` | `implemented` | `example_3_streaky_bl.npz`, `example_3_streaky_bl.png` |
| `example_4_Cavity.m` | `example_4_cavity.py` | `implemented` | `example_4_cavity.npz`, `example_4_cavity.png` |
| `example_5_Paraboloid.m` | `example_5_paraboloid.py` | `implemented` | `example_5_paraboloid.npz`, `example_5_paraboloid.png` |

Parity execution command:
```bash
.venv/bin/python tools/parity/compare_martini_examples.py
```
