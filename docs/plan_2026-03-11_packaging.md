# Plan 2026-03-11: Remaining Packaging Alignment Work

The project already uses a `src/` layout, dependency groups, and `linstabpy` distribution metadata. This file now tracks only the remaining packaging cleanup.

## 1. Remove repo-root import assumptions

Audit and eliminate remaining repo-root import hacks in workflows that should run against the installed package:

- `run_all_examples.sh`
- `results/case1/case1.sh`
- `results/case1/perf_memory.py`
- parity tooling under `tools/parity/`

Keep repository-relative paths for data and artifacts only where those scripts are intentionally repo-local.

### Acceptance

- Supported Python entry points no longer depend on `PYTHONPATH` pointing at the repository root.
- Repo-only scripts are clearly distinguished from installable package behavior.

## 2. Finish package-data and resource loading cleanup

1. Verify which baseflow assets are truly required at runtime.
2. Make installed-package resource loading the default path for those assets.
3. Keep reference-only MATLAB/parity material outside the installed-package contract.

### Acceptance

- A wheel install can load required runtime assets without depending on a repo checkout or submodule-specific fallback paths.

## 3. Add build and wheel verification

1. Add a canonical build check with `uv build`.
2. Verify clean-environment wheel installation.
3. Test that an installed wheel can import `linstab2d` and load packaged baseflow assets.

### Acceptance

- sdist and wheel build successfully.
- A clean install works without repo-root import assumptions.

## 4. Decide whether to ship a public CLI

1. If a supported CLI is needed, define a small stable command surface.
2. If not, keep case-study and parity scripts explicitly repo-local.

### Acceptance

- The repository clearly distinguishes library interfaces from repo-maintenance or research scripts.
