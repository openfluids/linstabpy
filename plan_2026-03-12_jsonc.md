# Plan 2026-03-12: Remaining Work to Finish the `py` + `jsonc` Pattern

The shared JSONC loader, BC recipe applicator, example JSONC files, and example migrations already exist. This file now tracks only the unfinished cleanup.

## 1. Finish Case 1 de-environmentalization

1. Remove remaining parameter-priority logic from `results/case1/case1.sh`.
2. Keep environment variables only where they select the config path or solver/runtime infrastructure.
3. Make the JSONC file the sole source of Case 1 scientific parameters.

### Acceptance

- Case 1 no longer needs environment-variable parameter overrides for physics, grid, or analysis settings.

## 2. Convert the performance harness to config-overrides JSONC generation

1. Replace env-mapping logic in `results/case1/perf_memory.py`.
2. Replace per-case `env` parameter overrides in `results/case1/perf_memory_cases.jsonc` with explicit config overrides.
3. Keep launcher, MPI, and runtime controls separate from scientific configuration overrides.

### Acceptance

- Each perf case is produced from a merged JSONC configuration, not a parameter-to-env translation layer.

## 3. Remove remaining Song-specific config duplication

1. Replace the leftover Song-specific BC duplication with the shared `apply_bc_recipe(...)` path.
2. Remove the remaining inline fallback recipe logic once the JSONC contract is complete.

### Acceptance

- `validation/song_2023` uses the shared JSONC + BC recipe path without a case-local BC helper.

## 4. Verify the remaining JSONC-driven workflows

1. Re-run Case 1 through its canonical path after the cleanup.
2. Re-run the perf harness on at least one representative case.
3. Re-run the Song 2023 case through its JSONC driver after the BC cleanup.

### Acceptance

- Case 1, the perf harness, and Song 2023 all run cleanly from their JSONC-driven entry points.
