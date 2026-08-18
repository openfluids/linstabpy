# Test validation tools (`tests/tools/`)

## Decision (2026-08-17) — please review before merge

These scripts lived under `examples/martini/` because parity work started there.
They are **not** Martini runnable examples (no Table S1 driver/plot pair).

**Relocated to `tests/tools/`** so that:

| Concern | Location |
|---------|----------|
| User-facing Martini examples | `examples/martini/example_*.py`, plotters, fixtures |
| Library audit / parity CLIs used by pytest | **`tests/tools/`** (this directory) |

### Files moved

| File | Role |
|------|------|
| `assembly_parity.py` | Compare scipy vs PETSc assembly (CLI) |
| `assembly_metrics_diff.py` | Regression diff on assembly metric JSON |
| `lnse_equivalence_harness.py` | Dual-path assembly safety harness |
| `matlab_source_parity.py` | Symbolic audit vs MATLAB coefficient `.m` sources |
| `symbolic_compressible_nse.py` | SymPy LNSE templates + export CLI |
| `coefficients_reference_base.py` | Frozen pre-fusion coefficient reference |
| `NSE_DERIVATION.md` | Human-readable derivation notes for symbolic checks |

### Tests that invoke this directory

- `tests/test_assembly_parity_tool.py`
- `tests/test_assembly_metrics_diff_tool.py`
- `tests/test_lnse_equivalence_harness.py`
- `tests/test_matlab_source_parity.py`
- `tests/test_symbolic_equation_parity.py`

`tests/conftest.py` prepends this directory to `sys.path` so imports resolve as
`from matlab_source_parity import …` / `from symbolic_compressible_nse import …`.

### Running a tool manually

From the repository root:

```bash
.venv/bin/python tests/tools/assembly_parity.py pair --model 2D --skip-eigs
.venv/bin/python tests/tools/symbolic_compressible_nse.py --mode all
```

Default JSON outputs go under `artifacts/validation/` (gitignored), not beside
the scripts.

### Owner checklist

- [ ] Accept `tests/tools/` as the permanent home for validation CLIs
- [ ] Confirm no external docs/CI still point at removed `examples/martini/` audit scripts

**Reviewer:** ___________________  
**Date:** ___________________  
**Decision:** ☐ Accept ☐ Accept with edits ☐ Reject — prefer: ___________
