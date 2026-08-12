# Development and Release Workflow

linstabpy is developed in small, verified increments. The public target is
`1.0.0`: one field loader, one YAML shape, and no compatibility shims.

## Branches

| Branch | Role |
|---|---|
| `main` | Stable. Always passes the full suite. Every release is tagged here. **No direct pushes** — updated only by merging a reviewed PR from `dev`. |
| `dev` | Integration branch for ongoing work. Default base for new branches. |
| Topic branches | `feat/*`, `fix/*`, `refactor/*`, `test/*`, `docs/*`. Short-lived; branch from `dev`, PR back into `dev`. |

## Versioning

- **Patch `0.0.x` / `1.0.x`** — one small, verified increment.
- **Minor `x.y.0`** — a completed milestone.
- **Major `1.0.0`** — first committed-stable public API.

`pyproject.toml`'s `version` on `main` always equals the latest tag. `dev`
carries the **next** target with a `.devN` suffix so an install from `dev`
never claims to be a release.

## The micro-release loop

1. **Pick a slice** that forms one coherent piece of work.
2. **Implement** on `dev` (or a topic branch → PR into `dev`).
3. **Verify** — the merge gate below. Keep it green at every commit.
4. **Changelog** — add entries under `[Unreleased]` in `CHANGELOG.md` as the
   work lands.
5. **Promote** — when the slice is coherent and green, open a PR `dev → main`.
6. **Tag** — on merge, cut the next tag on `main`.

## Merge gate — what must hold before a PR into `main`

1. **Full suite green on a machine with PETSc/SLEPc** —
   `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`.
2. **PETSc-free lane green** — loader, YAML, I/O, schema, and operator
   tests that collect without `mpi4py`.
3. **Frozen-numerics intact** — the hash-frozen validated-math files
   (operator coefficients, energy norm, SBP operators/generator) are unchanged.
   Any genuine math change carries an independent `< 1e-10` oracle parity,
   and the freeze hash is never re-baselined to pass.
4. **Lint/format clean** — `ruff check .` and `ruff format --check .`.
5. **No stray changes** — only files that trace to the slice; named-file staging
   only (never `git add -A`).
6. **Changelog updated** — the slice's user-visible changes are recorded.

GitHub Actions runs the PETSc-free lane and the tracked-file-size check.
It does not compile PETSc. The full suite is the workstation merge gate.

## Where verification runs

The development machines are memory-limited; the full merge gate runs on a
big-memory verification workstation. The loop:

1. Commit on the development machine and push the slice to `dev`.
2. On the verification workstation, pull `dev` (`git fetch && git pull
   --ff-only`) and run the full merge gate there — the suite, the PETSc-free
   lane, and lint.
3. Open the `dev → main` PR only after the gate is green on the verification
   workstation. A state that is green only on a development laptop is never
   promoted.

Never copy edited sources to the verification machine by hand (no rsync/scp):
it runs exactly what `dev` has, so every result stays traceable to a commit
SHA.

## Releasing — cutting a tag on `main`

1. In `CHANGELOG.md`, move the `[Unreleased]` entries into a new dated section.
2. Set `pyproject.toml` `version` to that tag.
3. Merge `dev → main` via the gated PR.
4. Annotated tag: `git tag -a vX.Y.Z <commit> -m "…"` then
   `git push origin vX.Y.Z`. **Never move a published tag**.
5. On `dev`, bump `pyproject.toml` to the next `.devN` and open a fresh
   `[Unreleased]` heading in the changelog.

## CHANGELOG discipline

- [Keep a Changelog](https://keepachangelog.com/) format.
- Every user-visible change gets an entry under `[Unreleased]` **as it lands**.
- Sections, in order: **Added / Changed / Fixed / Removed / Verified**.
- A withdrawn version number is never reused.

## Notes

- Plans and the task graph live in the private working repository, not here.
- Example PNGs under `examples/**/*.png` are tracked.
- Coverage on GitHub is the omit list (`src/` minus `solvers/`, `linalg/`,
  and the PETSc helpers). Do not set `fail_under` on the whole package.
