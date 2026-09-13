# CLI and YAML cases

## CLI: env vars, no flags

`linstabpy` has exactly one flag: `--version` (light: prints the version,
never initializes PETSc or MPI). Everything else travels by environment:

| Variable | Default | Meaning |
|---|---|---|
| `LINSTABPY_CASE_CONFIG` | `./configs/parity_config.yaml` | YAML case file. Legacy aliases `LINSTABPY_CASE2_CONFIG`, `LINSTABPY_CASE1_CONFIG` also work. |
| `LINSTABPY_CASE_OUTPUT_NPZ` | `{prefix}_{config_md5}_{git_sha8}.npz` in cwd | Result archive. Per-run aliases exist too. One unique path per concurrent run. |
| `LINSTABPY_CASE_LOG_PREFIX` | `linstabpy` | Log-line prefix and default output stem. |
| `LINSTABPY_FORCE_RECOMPUTE` | unset | `1`/`true`/`yes`/`on` skips the cache check and recomputes. |
| `LINSTABPY_STAGE_HEARTBEAT_SECONDS` | `30` | Progress heartbeat for long stages. |

```bash
export LINSTABPY_CASE_CONFIG=cases/my_case.yaml
export LINSTABPY_CASE_OUTPUT_NPZ=results/my_case.npz
mpiexec -n 8 linstabpy
```

Trap: the CLI uses `parse_known_args`, so `linstabpy run case.yaml`
does not error — it runs the env config and drops your words. If a run
used the wrong case, check the env first.

Cache: rank 0 loads the existing output NPZ (if any) and compares its
stored config hash + git SHA. Match prints `Cache valid` and skips the
compute. In sweeps, use one output path per run or export
`LINSTABPY_FORCE_RECOMPUTE=1` deliberately.

## YAML: schema and validation

- Suffix must be `.yaml` or `.yml`; top-level value must be a mapping.
- Top-level keys must be inside `TOP_LEVEL_ALLOWED` — full generated
  list in `generated/yaml-keys.md`. Unknown keys fail `validate_config`.
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
