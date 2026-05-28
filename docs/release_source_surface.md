# Release Source-Surface Map

This map defines the public `linstabpy` surfaces that must be reviewed before a
release. It is generated from tracked files, not from local caches or the parent
research workspace.

Snapshot:

- Inventory base SHA: `3483caf`
- Inventory command: `git ls-files`
- Tracked files after adding this map: 642
- Top-level tracked groups:

| Group | Files |
|---|---:|
| `scripts/` | 297 |
| `examples/` | 218 |
| `tests/` | 52 |
| `src/` | 43 |
| `docs/` | 12 |
| `.github/` | 5 |
| root package/install files | 15 |

Regenerate the top-level count with:

```bash
git ls-files | awk -F/ '{print $1}' | sort | uniq -c | sort -nr
```

## Review Actions

| Surface | File list command | Count | Required release action |
|---|---|---:|---|
| Package source | `git ls-files 'src/linstabpy/*.py' 'src/linstabpy/**/*.py'` | 42 | Public docstring coverage, selected private invariant comments, provenance scan, focused tests. |
| Tests | `git ls-files 'tests/*.py' 'tests/**/*.py'` | 51 | Assertion/tolerance readability audit and provenance scan. No tutorial comments. |
| Examples | `git ls-files 'examples/*.py' 'examples/**/*.py'` | 53 | Source comments for cache behavior, optional data, MPI usage, and output paths; example smoke or tracked blocker. |
| Validation scripts | `git ls-files 'scripts/*.py' 'scripts/**/*.py'` | 65 | CLI failure-message audit, public-path scan, and focused script tests where available. |
| Public docs | `git ls-files '*.md' 'docs/**/*.tex' 'docs/**/*.pdf' 'README.md' 'CITATION.cff' 'NOTICE' 'LICENSE'` | 29 | Render/link check, install/MPI/env clarity, public/private boundary scan. |
| Install and diagnostics | `git ls-files install.sh diagnose.sh benchmark.sh run_all_examples.sh scripts/ci` | 10 | Clean install/check behavior, MPI-stack reporting, thread-env guidance, actionable errors. |
| CI and packaging metadata | `git ls-files '.github/**' pyproject.toml uv.lock ruff.toml .pre-commit-config.yaml Dockerfile .python-version` | 11 | Release-command dry run, dependency policy check, package-data boundary audit. |
| Tracked generated/reference assets | `git ls-files | rg '\\.(npz|npy|mat|png|pdf|json|jsonc|csv|txt|dat|h5|hdf5)$'` | 319 | Artifact-size, fixture provenance, license, and regeneration/ignore-policy audit. |

## Source Package Modules

The package source is the main docstring and invariant-comment surface.

| Package area | File count | Required action |
|---|---:|---|
| `src/linstabpy/mesh` | 9 | Mesh mutation, deformation, filter, and operator-state comments. |
| `src/linstabpy/utils` | 8 | PETSc/MPI lifecycle, solver config, and backend helper comments. |
| `src/linstabpy/solvers` | 7 | Solver acceptance, PETSc/SLEPc setup, and resolvent linear-solve comments. |
| `src/linstabpy/baseflow` | 7 | Loader contract, optional data behavior, and public import docstrings. |
| `src/linstabpy/operators` | 5 | LNSE assembly order, coefficient preparation, sparse backend, and parity comments. |
| `src/linstabpy/boundary_conditions` | 4 | BC recipe, effective rows, column-zeroed rows, and backend-dispatch comments. |
| `src/linstabpy/results_runtime.py` | 1 | Runtime stage ownership, postcheck acceptance, and NPZ schema comments. |
| `src/linstabpy/config` | 1 | Public configuration API docstrings. |
| `src/linstabpy/__init__.py` | 1 | Public package import surface. |

Regenerate this table with:

```bash
git ls-files src/linstabpy |
  awk -F/ '{print $1"/"$2"/"$3}' |
  sort | uniq -c | sort -nr
```

## Examples And Case Studies

| Example surface | File count | Required action |
|---|---:|---|
| `examples/case_studies/case2_lui` | 115 | Public case-study docs and scripts; no private host paths; cache/MPI behavior clear. |
| `examples/case_studies/case1_song` | 84 | Same as above, plus optional reference-data behavior. |
| Top-level examples `example_0` through `example_5` | 14 | Smoke/regeneration behavior, cache paths, and output files clear. |
| Shared helpers and package markers | 4 | Public import and helper behavior documented only where non-obvious. |
| `examples/fix_examples_figures.md` | 1 | Documentation/provenance scan. |

Regenerate with:

```bash
git ls-files examples |
  awk -F/ '{if ($3=="") print $1"/"$2; else print $1"/"$2"/"$3}' |
  sort | uniq -c | sort -nr
```

## Scripts And Validation Data

| Script surface | File count | Required action |
|---|---:|---|
| `scripts/validation/song_2023` | 256 | Third-party provenance, fixture boundary, and optional-data audit. |
| `scripts/validation/martini2024` | 19 | Martini fixture provenance and parity-regeneration audit. |
| Other root `scripts/validation` entries | 10 | Matrix, equivalence, and example-run UX checks. |
| `scripts/tools` | 5 | Public usefulness, path safety, and error-message audit. |
| `scripts/ci` | 6 | Install/diagnose parity with root scripts and README. |
| `scripts/check_tracked_file_size.py` | 1 | Artifact-size gate support. |

Regenerate with:

```bash
git ls-files scripts |
  awk -F/ '{if ($3=="") print $1"/"$2; else print $1"/"$2"/"$3}' |
  sort | uniq -c | sort -nr
```

## Generated And Reference Assets

Tracked data-like files need provenance and size review before release.

| Extension | Count |
|---|---:|
| `npy` | 123 |
| `mat` | 74 |
| `jsonc` | 66 |
| `png` | 34 |
| `csv` | 8 |
| `json` | 6 |
| `txt` | 4 |
| `dat` | 3 |
| `pdf` | 1 |

Regenerate with:

```bash
git ls-files |
  rg '\.(npz|npy|mat|png|pdf|json|jsonc|csv|txt|dat|h5|hdf5)$' |
  awk -F. '{print $NF}' | sort | uniq -c | sort -nr
```

These files are not targets for docstring coverage. They are targets for the
artifact-size, license/provenance, and regeneration-policy audits.

Current artifact-size audit:

```bash
.venv/bin/python scripts/check_tracked_file_size.py
```

Result at release-audit time: no tracked file exceeds the 50 MiB policy. The
largest tracked data files are MATLAB/Numpy reference fixtures under
`examples/case_studies/case1_song/reference/` and
`scripts/validation/song_2023/reference/`, with the largest entries below
50 MiB. They remain source-repository fixtures, not package data.

Current source-distribution boundary:

```bash
uv build --sdist --out-dir artifacts/release_audit/dist_check
tar -tzf artifacts/release_audit/dist_check/linstabpy-0.1.0.tar.gz |
  rg '(^|/)(examples|scripts|docs|artifacts|data/baseflows|\.venv)/|\.(mat|npy|npz|png|pdf|dat)$'
```

Result at release-audit time: the built sdist is about 324 KiB and the tar
content scan above reports no examples, validation scripts, docs subtrees,
ignored artifacts, `.venv`, optional baseflows, or binary fixture extensions.
The public Git repository may contain validation fixtures; the installable
package should stay code-only unless a future release intentionally adds small
package data with a dedicated manifest and license note.

## Out Of Scope

The public release surface excludes:

- untracked planning or review files;
- ignored caches and generated reports under `artifacts/`;
- parent-workspace manuscript files, campaign notes, and machine runbooks;
- local virtual environments, build directories, and editor state.

If any excluded file is needed for a public command, move the reproducible
command or minimal fixture into the tracked public tree and record the
provenance in the relevant release audit.
