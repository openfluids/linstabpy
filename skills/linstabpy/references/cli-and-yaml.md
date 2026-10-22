# CLI and YAML cases

## CLI: env vars, no flags

`linstabpy` has exactly one flag: `--version` (light: prints the version,
never initializes PETSc or MPI). Everything else travels by environment —
full generated table in `generated/cli-env.md`. One unique output path per
concurrent run.

```bash
export LINSTABPY_CASE_CONFIG=cases/my_case.yaml
export LINSTABPY_CASE_OUTPUT_NPZ=results/my_case.npz
mpiexec -n 8 linstabpy
```

Trap: the CLI uses `parse_known_args`, so `linstabpy run case.yaml`
does not error — it warns on stderr, then runs the env config and drops
your words. The exit code stays 0. If a run used the wrong case, check
the env first. Single-dash PETSc options (`-ksp_monitor`, ...) pass
through to the solver stack; there is no `--config` flag. Unknown
`LINSTABPY_*` names in the environment likewise warn on stderr with a
did-you-mean hint (registry: `src/linstabpy/config/env_registry.py`) and are ignored.

Cache: rank 0 loads the existing output NPZ (if any) and compares its
stored config hash + git SHA. Match prints `Cache valid` and skips the
compute. In sweeps, use one output path per run or export
`LINSTABPY_FORCE_RECOMPUTE=1` deliberately.

## YAML: schema and validation

- Suffix must be `.yaml` or `.yml`; top-level value must be a mapping.
- Top-level keys must be inside `TOP_LEVEL_ALLOWED` — full generated
  list in `generated/yaml-keys.md`. Unknown keys fail `validate_config`
  (with a did-you-mean hint when the typo is close).
- The `linstabpy` command reads physics, numerics, bc and analysis values
  from their nested blocks (`physics.Re`, `numerics.fd_order`,
  `analysis.run_stability`, ...). Some flat keys pass `validate_config`
  because example scripts read them; the command refuses them with the
  nested key to use.
- Complex values as `[real, imag]` lists (also plain numbers or strings
  parseable by `complex()`); unsigned scientific floats like `1e5` are
  accepted.
- SBP `fd_order` is 2, 4, 6, or 8 only — anything else fails validation.

Always validate before running:

```python
from linstabpy.config.schema import validate_config
from linstabpy.utils.yaml_io import load_config

cfg = load_config("cases/my_case.yaml")
validate_config(cfg, source="cases/my_case.yaml")
```

This needs only PyYAML — no PETSc, no MPI. Start new cases from
`../assets/minimal-case.yaml`.
