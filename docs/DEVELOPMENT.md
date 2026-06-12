# Development and Release Workflow

linstabpy is developed in small, frequently-tagged increments. The rule of
thumb: **a few beads of work → prove it still works → cut the next patch tag on
`main`.** Keep versions granular; only jump the minor number for a real
milestone.

## Branches

| Branch | Role |
|---|---|
| `main` | Stable. Always passes the full suite. Every release is tagged here. **No direct pushes** — updated only by merging a reviewed PR from `dev`. |
| `dev` | Integration branch for ongoing work. Default base for new branches. |
| Topic branches | `feat/*`, `fix/*`, `refactor/*`, `test/*`, `docs/*`. Short-lived; branch from `dev`, PR back into `dev`. |

## Versioning (pre-1.0, deliberately granular)

While the version is `< 1.0`:

- **Patch `0.0.x`** — one small, verified increment (a few beads' worth). Each
  green slice gets its own tag: `0.0.3 → 0.0.4 → 0.0.5 → …`. This is the normal
  cadence.
- **Minor `0.x.0`** — a larger milestone: a completed epic, or the finished
  clean-code revamp. e.g. the `0.0.x` series rolls up into `0.1.0`.
- **Major `1.0.0`** — first committed-stable public API.

`pyproject.toml`'s `version` on `main` always equals the latest tag. `dev`
carries the **next** target with a `.devN` suffix (e.g. `0.0.4.dev0`) so an
install from `dev` never claims to be a release.

## The micro-release loop

1. **Pick a slice** — a few `dev` beads that form one coherent piece of work.
2. **Implement** on `dev` (or a topic branch → PR into `dev`).
3. **Verify** — the merge gate below. Keep it green at *every* commit, not just
   at the end.
4. **Changelog** — add entries under `[Unreleased]` in `CHANGELOG.md` as the
   work lands.
5. **Promote** — when the slice is coherent and green, open a PR `dev → main`.
6. **Tag** — on merge, cut the next patch tag on `main` (release steps below).

Repeat. Each loop is one small, defensible step forward with its own tag and
changelog entry.

## Merge gate — what must hold before a PR into `main`

1. **Full suite green** — `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
   (currently 749 passed / 14 skipped).
2. **No-MPI pure-math lane green** — the single-rank stub path that a local
   mpi4py install hides. (Run the dissipation-gate files with
   `mpi4py`/`petsc4py`/`slepc4py` blocked; see the repo's no-MPI repro pattern.)
3. **Frozen-numerics intact** — the hash-frozen validated-math files
   (operator coefficients, energy norm, SBP operators/generator) are unchanged.
   Any genuine math change carries an **independent** `< 1e-10` oracle parity
   (first-principles derivation, never a self-referential snapshot), and the
   freeze hash is never re-baselined to pass.
4. **Lint/format clean** — `ruff check .` and `ruff format --check .`.
5. **No stray changes** — only files that trace to the slice; named-file staging
   only (never `git add -A`).
6. **Changelog updated** — the slice's user-visible changes are recorded.

## Where verification runs

The development machines are memory-limited; the full merge gate runs on a
big-memory verification workstation. The loop:

1. Commit on the development machine and push the slice to `dev`.
2. On the verification workstation, pull `dev` (`git fetch && git pull
   --ff-only`) and run the full merge gate there — the suite, the no-MPI
   lane, and lint.
3. Open the `dev → main` PR only after the gate is green on the verification
   workstation. A state that is green only on a development laptop is never
   promoted.

Never copy edited sources to the verification machine by hand (no rsync/scp):
it runs exactly what `dev` has, so every result stays traceable to a commit
SHA.

## Releasing — cutting a patch tag on `main`

1. In `CHANGELOG.md`, move the `[Unreleased]` entries into a new
   `[0.0.x] - YYYY-MM-DD` section.
2. Set `pyproject.toml` `version = "0.0.x"`.
   Freeze the solver-stack constraints: `bash scripts/tools/freeze_env.sh v0.0.x`.
3. Merge `dev → main` via the gated PR.
4. Annotated tag: `git tag -a v0.0.x <commit> -m "…"` then
   `git push origin v0.0.x`. **Never move a published tag** — if a tag was cut on
   a bad commit, withdraw it and cut the *next* number, don't re-point it.
5. On `dev`, bump `pyproject.toml` to the next `.devN` and open a fresh
   `[Unreleased]` heading in the changelog.

## CHANGELOG discipline

- [Keep a Changelog](https://keepachangelog.com/) format.
- Every user-visible change gets an entry under `[Unreleased]` **as it lands** —
  not reconstructed at release time.
- Sections, in order: **Added / Changed / Fixed / Removed / Verified**.
- A release moves `[Unreleased]` → a dated `[0.0.x]` section and links the
  compare range.
- A withdrawn version number is never reused; leave a one-line note explaining
  the gap (see the 0.0.2 note).

## Notes

- Per-push cloud CI and dependency-bump automation are **paused** while the code
  stabilizes; verification is local. Reintroduce a lean CI (single Linux runner,
  no expensive macOS lane) at public release.
- Plans and the bead-tracked task graph live in the private working repository,
  not here. This document is the public-facing process contract.
