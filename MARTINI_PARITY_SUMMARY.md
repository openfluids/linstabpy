# Martini examples — parity summary

**Repository:** `linstabpy-examples-update` (this tree)  
**Scope:** Martini (2024) Examples **0–5** under `examples/martini/`  
**Status:** Audit **CLOSED 2026-08-18** — all six examples signed off (`examples/martini/AUDIT_PLAN.md`)  
**Ground truth for numbers:** Martini2024 campaign tree (read-only); repo ships slim MATLAB fixtures + local linstabpy driver outputs.

This document is the **single summary** of modifications, parity knobs, bugs found, and accepted residuals when porting the Martini supplement examples into this repository. It consolidates `examples/martini/AUDIT_PLAN.md`, `physics_audit_campaign_vs_repo.json`, per-example YAML/driver notes, and the deeper audits for Examples **1**, **4**, and **5**.

---

## 1. What changed in this repo (global)

| Topic | Before / issue | After / decision |
|-------|----------------|------------------|
| **MATLAB fixtures** | Split `*_values.npz` / `*_modes.npz` at `examples/martini/` root | Slim bundles in **`examples/martini/LinStab2DResults/*.npz`** (`complex128`, compact u+v or u+v+w where applicable) |
| **Python driver NPZ** | Some committed at repo root | **`example_*.npz` gitignored** — regenerate locally (`run_all_examples.sh` / per-case drivers); several exceed the 60 MiB tracked-file guard |
| **Raw MATLAB `.mat`** | — | **`martini_examples/`** at repo root is gitignored scratch only (never commit) |
| **Baseflow data** | `examples/martini/data/` | **`examples/martini/baseflows/`** (`BL_BF.mat`, `Step_BF.mat`, `M09jet.mat`, `Paraboloid_BF.mat`) |
| **Plot layer** | Campaign-only presentation scripts | **`example_*_plots.py`** + **`plot_compare_utils.py`** — thesis-style mosaics (LinStab2D \| linstabpy \| peak-normalized % error) |
| **MATLAB export** | External only | Bundled **`examples/martini/linstab2D_scripts/`** + **`regenerate_martini_fixtures.sh`** |
| **Run logs** | — | **`examples/martini/_run_logs/`** gitignored |
| **Validation CLIs** | Under `examples/martini/` | Moved to **`tests/tools/`** (assembly/symbolic parity — not part of the public example path) |

**Comparator:** `compare_martini_examples.py` + `martini_cases.yaml` (tolerances, stems). Latest JSON: `martini_parity_latest.json`.

---

## 2. Per-example summary

| Ex | Case | Main parity knobs / fixes | Outcome |
|----|------|---------------------------|---------|
| **0** | Curvilinear derivative test | None — analytic reference | Campaign vs repo NPZ rel ≤ 10⁻¹⁴; plots PASS |
| **1** | Blasius resolvent | **`zero_dmudT_martini_parity: true`** in YAML | See §3.1 — MATLAB 2D drops μ(T) coupling; parity when pinned |
| **2** | Jet resolvent | **`bc_mask_layout: kron`** (repo metadata); **`jet_resolvent_response_c_diag`** in plots | Slim fixtures **complex128**; mosaic error 0%; `C` mask order matched campaign |
| **3** | Streaky BL (4 paper NPZ) | **`floquet_shift: matlab_parity`** in each paper YAML | Temporal modes match campaign; Floquet/BC per Martini recipe |
| **4** | Open cavity EVP | **`Pr: 0.7`**, **`zero_dmudT: true`**; plot gridding/scaling fixes | Operators machine-identical; see §3.2 |
| **5** | Paraboloid resolvent | **`D2y_asymm` mesh fix** in `src/` | See §3.3 — translation bug in full 2D deform |

---

## 3. Examples with non-trivial findings

### 3.1 Example 1 — Blasius (`zero_dmudT` / MATLAB `dMUdT` bug)

**Symptom:** With full Sutherland viscosity coupling, leading resolvent gain differs by **~0.54%** (mode 1 only); subleading modes and mode shapes are much closer.

**Root cause (MATLAB reference, not linstabpy):**

- LinStab2D **2D Cartesian** coefficients use **`dMUdT`** / **`d2MUdT2`** (`getCoeffsLaminar_Cartesian.m`).
- `sutherland_air.m` only fills lowercase **`dmudT`** / **`d2mudT2`**.
- `GetLinProblem.m` defaults missing capitalised fields to **zero** → MATLAB **silently drops all T→momentum viscosity coupling** in 2D.
- linstabpy uses the populated `dmudT` — physically fuller behaviour.
- Axisymmetric cases (Ex. 2, 5) use a different coefficient file with lowercase `dmudT` and are **not** affected.

**Parity policy in this repo:**

```yaml
# example_1_blasius.yaml
zero_dmudT_martini_parity: true   # zeros dmudT after sutherland_air — matches LinStab2D 2D
```

With this flag: `L0` Frobenius vs MATLAB **~2.5×10⁻¹⁵**; gains match to **10 digits**. Production physics (flag off) keeps Sutherland coupling and the 0.54% leading-gain gap vs MATLAB.

**Also documented:** Ex. 1 is **not** sponge-free (mild three-edge sponge on ~34% of points). Resolvent backend differs (UMFPACK/`eigs` vs MUMPS/SLEPc) but is **exonerated** when MATLAB `L0` is fed to Python resolvent.

---

### 3.2 Example 4 — Cavity (`Pr`, `zero_dmudT`, operator + plot audit)

**Parity knobs (repo YAML):**

```yaml
# example_4_cavity.yaml
Pr: 0.7              # MATLAB hardcodes 0.7; Step_BF.mat stores 0.71
zero_dmudT: true       # same dMUdT=0 class as Ex. 1 for 2D Cartesian
```

**Operator audit (closed):**

| Issue | Resolution |
|-------|------------|
| **Pr 0.71 (py from baseflow) vs 0.70 (MATLAB)** | Pin **`Pr: 0.7`** in driver after `load_cavity` |
| **Python `dmudT` vs MATLAB `dMUdT=0`** | Pin **`zero_dmudT: true`** — removes +503k nnz superset in u←T / v←T blocks |
| **T-map / CS1 vs folded Python assembly** | Compare post-sponge stages with MATLAB **T** applied — overlap **~0.18%** pre-fix; **machine-identical** after Pr + dmudT pins |
| **Sponge / BC / Chu** | Identical DOF sets and fields — ruled out |
| **Strict 50-mode pairing** | Worst L₂ **0.028%** direct / **0.11%** adjoint |

**Plot layer (repo mosaics):**

- Fixes: `used_ind` **0-based** gridding, **`compact_uvw`** scaling with LinStab2D idx labels, slim MATLAB u+v+w fixtures.
- Peak-normalized mosaic error **≤ 0.49%** (adjoint **u**); **not** floor corners — **mouth/lid-adjacent** band.

**Accepted residual (not a linstabpy bug):**

- On the **same** assembled operator, stored **MATLAB eigenvectors** have eigen-residual **~10⁻³** (direct) / **~10⁻²** (adjoint) localized at cavity lips; **Python modes** satisfy the operator to **~10⁻¹²**.
- The remaining **0.3–0.5%** mosaic patches are **MATLAB `eigs` convergence error** in the reference fixtures, not operator, BC, sponge, or compare-layer bugs. No further linstabpy fix removes them; only tighter MATLAB regen or accepting Python modes as reference.

**Mask geometry note:** mask threshold `x < 0.34` vs cavity wall `x_cavity_start ≈ 0.34029` leaves a 50-cell **mouth strip** — plot residuals correlate with that band, not implementation error.

---

### 3.3 Example 5 — Paraboloid (`D2y_asymm` mesh fix)

**Symptom:** At azimuthal mode **m = 5**, radial operators use **`D2y_asymm` only**. Before the fix, `‖D2y_asymm_py − D2y_asymm_mat‖_F / ‖·‖_F ≈ 4×10⁻⁵` → `L0` gap **~6×10⁻⁶**, max gain mismatch **~0.043%**.

**Root cause (linstabpy translation bug):**

- MATLAB `DeformMesh.m` computes mapping metric derivatives **`dJ` / `Q` once** with **full** `Dy` / `D2y`, then updates **both** symm and asymm families.
- Python `_transform_second_derivatives` previously recomputed metrics with **family** operators (`Dy_asymm`, `D2y_asymm`). Coordinates have **no η parity**, so pole metric derivatives were wrong.

**Fix in `src/linstabpy/mesh/mesh.py`:**

- Optional `Dy_metric` / `D2y_metric` for metric terms only.
- `_deform_full_2d` passes **full** `Dy` / `D2y` for metrics; family stencils unchanged elsewhere.
- Regression: `tests/test_mesh.py` → `test_deform_full_2d_family_metric_terms_use_full_operators`.

Details: `examples/martini/example_5_D2y_asymm_fix.md`.

**Physics check before acceptance:** The audit closes **numerical parity** with LinStab2D (`D2y_asymm`, `L0`, resolvent gains). That is **not** the same as a standalone proof that the corrected metric path is the right continuum limit for deformed polar/axisymmetric meshes. Before merging `mesh.py`, the owner should confirm we are **not introducing wrong physics** — e.g. independent review of `DeformMesh.m` vs `_deform_full_2d`, spot checks on metric terms away from the Martini paraboloid case, and sanity on other azimuthal modes / geometries that use the asymm family. Do not accept the `D2y` / `Dy_metric` change on parity alone.

---

## 4. Examples 0, 2, 3 — shorter notes

### Example 0

- No parity knobs. Curvilinear FD test; repo NPZ matches campaign.

### Example 2

- **`bc_mask_layout: kron`** documented in repo YAML (matches MATLAB `kron(B, I_5)` observation mask).
- Plot path: **`jet_resolvent_response_c_diag`** applies `C` on MATLAB responses before error panels (matches campaign).
- Slim fixtures regenerated as **`complex128`** (was `complex64`) for bit-identical u modes vs full MATLAB trim.
- Campaign vs repo NPZ: gains rel **~2×10⁻¹³**; mosaics **0%** peak-norm after plot fixes.

### Example 3

- Four paper YAMLs pin **`floquet_shift: matlab_parity`** (library default elsewhere is **`continuum`** — see §5).
- Repo stores **`floquet_exp`** in NPZ keys; campaign comparison exact on temporal spectra/modes.
- `BL_BF.mat` under `baseflows/`.

---

## 5. Library changes (`src/`) — owner review required

The examples branch includes **unreviewed changes** under `src/linstabpy/`. **Do not merge to the primary library until the owner signs off below.**

| File | Change | Motivation |
|------|--------|------------|
| **`mesh/mesh.py`** | `Dy_metric` / `D2y_metric` in `_transform_second_derivatives`; full `Dy`/`D2y` for metric terms in `_deform_full_2d` | Ex. 5 `D2y_asymm` parity (§3.3) |
| **`operators/assembly.py`** | Default Floquet shift mode **`continuum`** (was **`matlab_parity`**) | Library default = physically continuum-correct FD; Martini Ex. 3 YAMLs still set **`floquet_shift: matlab_parity`** explicitly |

**What we believe:**

1. **Mesh fix** — bug fix matching MATLAB `DeformMesh.m`; not a Martini-only hack. **Requires a physics sanity check** (§3.3) — parity with LinStab2D is necessary but not sufficient.
2. **Floquet default** — examples that need LinStab2D discrete algebra must **opt in** via YAML (`floquet_shift: matlab_parity`); Ex. 3 already does.

**What we do *not* claim:**

- That every Martini residual is zero (Ex. 1 production physics, Ex. 4 MATLAB eigenvector tail, Ex. 2 structural gain tolerance in `martini_cases.yaml`).
- That overlay / pinned-operator experiments replace shipped mesh code (Ex. 5 diagnostic overlay NPZ stays outside production).

---

## 6. Ruled out (do not re-chase)

| Claim | Status |
|-------|--------|
| Wrong sponge recipe (Ex. 4) | Ruled out — field and diagonal match |
| Wrong BC DOF set / penalty (Ex. 4) | Ruled out — 2720 = 2720 |
| Floor corners as mosaic hotspot (Ex. 4) | Ruled out — mouth/lid band only |
| Ex. 2 `kron` mask as operator bug | Documented layout; plots matched with `C` recipe |
| Ex. 5 full unsplit `D2y` mismatch | Ruled out — only **`D2y_asymm`** metric path was wrong |
| Compare-layer gridding alone explains Ex. 4 tail | Ruled out after `used_ind` / `compact_uvw` fixes |

---

## 7. Related files in this repository

| File | Contents |
|------|----------|
| `examples/martini/AUDIT_PLAN.md` | Step A→E workflow, per-example PASS table, commands |
| `examples/martini/physics_audit_campaign_vs_repo.json` | Machine YAML/NPZ diff vs campaign |
| `examples/martini/martini_fixtures_README.md` | Fixture regen, commit policy |
| `examples/martini/example_5_D2y_asymm_fix.md` | Ex. 5 mesh fix + owner question (duplicate of §5) |
| `examples/martini/example_{1,4}_*.yaml` | Parity knob source of truth |
| `CHANGELOG.md` | Release-level Martini bullet under `[Unreleased]` |

---

## 8. Owner sign-off — `src/` modifications

> **Before accepting or merging the changes in `src/linstabpy/` from this examples work, the repository owner should review:**
>
> 1. **`mesh/mesh.py`** — metric operators in `_deform_full_2d` (§3.3, `example_5_D2y_asymm_fix.md`). **Verify the `D2y_asymm` / `Dy_metric` change is physically correct**, not only MATLAB-parity correct, before accepting (see §3.3 physics check).
> 2. **`operators/assembly.py`** — default Floquet mode **`continuum`** vs explicit **`matlab_parity`** in Ex. 3 YAMLs (§5).
> 3. Any accompanying tests (`tests/test_mesh.py`, Floquet-related tests) that assume the new defaults.
>
> Confirm that these library changes are acceptable for the **main** `linstabpy` package, not only for Martini validation in this examples fork.

| Reviewer | Date | Decision |
|----------|------|----------|
| | | ☐ Accept `src/` changes as-is &nbsp; ☐ Accept with edits &nbsp; ☐ Reject — use: _________ |

**Signature / notes:**

---

*Last updated: 2026-08-18*
