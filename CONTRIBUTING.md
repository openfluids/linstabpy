<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Contributing to linstabpy

Contributions are genuinely welcome, and that includes the ones that are not
code. A bug report, a confusing docstring, a README paragraph that turned out to
be wrong, a question that took you an hour to answer yourself — all of those are
worth opening an [issue](https://github.com/openfluids/linstabpy/issues) for.

Install problems especially. The PETSc / SLEPc / MPI layering is the hardest
part of using linstabpy, and every report about it makes the next person's
install easier. If you are unsure whether something is worth reporting, it
probably is. Open the issue.

## Getting set up

```bash
git clone https://github.com/openfluids/linstabpy.git
cd linstabpy
bash install.sh
bash install.sh --check
```

`install.sh` creates `.venv` and builds or installs the Python/PETSc/SLEPc
stack; `--check` verifies that `petsc4py` and `slepc4py` import and that the
runtime libraries loaded in the right order. Run `bash diagnose.sh` first if
your machine is unusual — it reports what is already present before anything is
built.

Expect the first install to take a while. This is normal, not a hang.

## Before you open a pull request

The same checks CI runs:

```bash
ruff check .
ruff format --check .
uv sync --group test
.venv/bin/python -m pytest tests/
python scripts/check_tracked_file_size.py
```

The dissipation guardrail tests are the ones that matter most, and CI runs them
separately for that reason:

```bash
.venv/bin/python -m pytest \
  tests/test_sbp_dissipation.py tests/test_sbp_generator.py \
  tests/test_mesh_dissipation.py tests/test_assembly_dissipation.py
```

If one fails for a reason you think is unrelated to your change, say so in the
pull request rather than working around it — that is useful information, and
sometimes it is CI that is wrong.

## What makes a pull request easy to review

The [pull request template](.github/pull_request_template.md) has the full
checklist. In short:

- **One thing at a time.** A focused change gets reviewed quickly. A change that
  also reformats fifty unrelated lines is hard to read and slow to merge.
- **Say what you verified.** A pasted command and its output is worth more than
  "tested locally". For anything touching the operators, a convergence rate or a
  residual is the useful evidence.
- **Ask early.** For anything substantial, open an issue first. It is much
  better to disagree about an approach before you have written it than after.
- **Draft PRs are fine.** Opening one early to ask "is this the right
  direction?" is welcome and costs nothing.

Reviews may take a few days — one maintainer, research alongside. A nudge on a
quiet pull request is welcome, not annoying.

## Conventions

Only the ones that are actually enforced:

- **Numerics need a falsifiable check, not just a passing test.** Observed
  convergence order under refinement, a conservation or symmetry residual, or a
  known analytical limit — Blasius, Stokes, Poiseuille `Re_c = 5772.22`. A
  regression test against current output will not catch a subtly wrong operator.
- **Ported LinStab2D code keeps its provenance.** Anything derived from
  Martini & Schmidt (2024) says so in the docstring, naming the MATLAB routine
  it came from.
- Summation-by-parts structure is not an implementation detail. A change that
  breaks the SBP property fails the dissipation guardrails, and that is the
  guardrail working.
- No tracked file over 50 MiB; CI enforces it. Large fields belong in a release
  asset or generated on demand.
- Use `np.random.default_rng(seed)`, not the legacy module-level `np.random`
  calls.
- Formatting and import order are handled by `ruff` — do not hand-tune them.
- Every source file carries its `SPDX-License-Identifier: GPL-3.0-only` header.

## Conduct and licence

Everyone taking part is asked to follow the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).
It is short.

linstabpy is licensed under **GPL-3.0-only** — unlike the other openfluids
packages, which are Apache-2.0 — and contributions are accepted under the same
licence. If you are porting code in from elsewhere, check that its licence is
compatible before opening the pull request. See `LICENSE` and `NOTICE`.

Found a security problem? Please do not open a public issue — see the
[openfluids security policy](https://github.com/openfluids/.github/blob/main/SECURITY.md).
