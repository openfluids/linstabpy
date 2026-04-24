# Operator Fixture Schema

Operator fixtures preserve MATLAB-backed validation without keeping large raw
exports in Git.

## File Types

- Dense arrays use NumPy `.npz`.
- Sparse matrices use SciPy `save_npz()` output with `.npz`.
- Every data file has a sibling `.jsonc` metadata file with the same stem.

## Metadata Keys

Required `.jsonc` keys:

- `fixture_id`: stable snake-case identifier.
- `source_path`: MATLAB script, export, or Python converter that produced the
  fixture.
- `source_commit`: Git commit of the source checkout, or `unknown` when the
  source predates tracking.
- `generated_at`: ISO-8601 generation timestamp.
- `physical_parameters`: Reynolds number, Mach number, Prandtl number,
  heat-capacity ratio, reference temperature, wavenumbers, and solver target
  when applicable.
- `shape`: array or sparse-matrix shape.
- `dtype`: stored dtype.
- `ordering`: vectorization convention, usually Fortran order for MATLAB
  parity.
- `tolerance`: absolute and relative tolerances for parity checks.
- `contents`: short description of each array key.

## Size Policy

Tracked fixture files should stay below 10 MiB. A larger artifact needs either
a smaller regression slice or a regeneration script under `scripts/`.

No tracked GitHub-facing file may exceed 50 MiB. The repository guardrail is:

```bash
python scripts/check_tracked_file_size.py
```
