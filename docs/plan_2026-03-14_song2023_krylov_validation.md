# Song2023 / Robinet Krylov Validation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the Song & Hao (2023) / Robinet (2007) oblique SWBLI temporal eigenvalue problem through a four-stage ladder: SciPy+MUMPS baseline, PETSc-assembly+MUMPS parity, Krylov+LU-PC bridge, lighter-PC exploration.

**Architecture:** The Song2023 case is a 2D LNSE temporal stability analysis on a curvilinear mesh (deformed flat plate with oblique shock impingement). The solver chain is: baseflow loading, mesh construction, `build_lnse_system` (operator + BCs), `temporal_eigenvalues` (SLEPc shift-invert EVP). Each validation stage changes only the assembly backend and/or the ST linear solver config via JSONC, keeping the driver `song_2023.py` and comparison tools untouched.

**Tech Stack:** PETSc/SLEPc via petsc4py/slepc4py, SciPy sparse, MUMPS direct solver, JSONC configuration, NumPy/Matplotlib for comparison metrics and plots.

---

## File Structure

| File | Role |
|------|------|
| `validation/song_2023/song_2023.jsonc` | Base config (currently `scipy/auto + mumps`). Gets variant copies per stage. |
| `validation/song_2023/configs/scipy_assembly__mumps.jsonc` | Stage 1 explicit config |
| `validation/song_2023/configs/petsc_assembly__mumps.jsonc` | Stage 2 explicit config |
| `validation/song_2023/configs/petsc_assembly__krylov_lu_pc.jsonc` | Stage 3 explicit config |
| `validation/song_2023/configs/petsc_assembly__krylov_asm.jsonc` | Stage 4 candidate config |
| `validation/song_2023/run_ladder.py` | Orchestrator: runs all stage configs, captures metrics, produces summary table |
| `validation/song_2023/run_ladder.jsonc` | Ladder orchestrator config (stage list, NPZ output paths, comparison thresholds) |
| `validation/song_2023/song_2023.py` | Existing driver — no changes planned unless a bug surfaces |
| `validation/song_2023/compare_spectra.py` | Existing audit tool — no changes planned |
| `validation/song_2023/compare_L0.py` | Existing operator comparison — no changes planned |

## Guardrails

- All solver settings in JSONC — no hardcoded choices in `.py`.
- Do not change `paper/`.
- Do not claim `petsc_assembly__krylov` is production-ready until validated.
- Preserve the evidence order: baseline, PETSc parity, Krylov bridge, lighter PC.
- Each stage must complete and produce comparison metrics before moving to the next.

---

## Chunk 1: Stage 1 — SciPy + MUMPS Baseline

### Task 1: Create explicit Stage 1 config

**Files:**
- Create: `validation/song_2023/configs/scipy_assembly__mumps.jsonc`

- [ ] **Step 1: Create the configs directory**

```bash
mkdir -p validation/song_2023/configs
```

- [ ] **Step 2: Write the Stage 1 JSONC config**

Copy from `song_2023.jsonc` with explicit values for the baseline path.
The only differences from the base config are making implicit defaults explicit:
- `"assembly_backend": "scipy"` (base has `"auto"`)
- `"petsc_matrix_mode": "local"` (base has `"auto"`)

All flow parameters, BC recipe, and baseflow preprocessing remain identical.

- [ ] **Step 3: Verify the config loads without errors**

Run:
```bash
.venv/bin/python -c "
from linstab2d.utils.jsonc import load_jsonc
cfg = load_jsonc('validation/song_2023/configs/scipy_assembly__mumps.jsonc')
print('assembly_backend:', cfg.get('assembly_backend'))
print('solver:', cfg.get('solver'))
print('OK')
"
```
Expected: prints `assembly_backend: scipy`, solver dict, and `OK`.

- [ ] **Step 4: Commit**

```bash
git add validation/song_2023/configs/scipy_assembly__mumps.jsonc
git commit -m "validation: add explicit Stage 1 scipy+mumps config for Song2023 ladder"
```

### Task 2: Run Stage 1 and capture baseline NPZ

**Files:**
- Output: `validation/song_2023/configs/scipy_assembly__mumps.npz` (gitignored)

- [ ] **Step 1: Delete any existing NPZ so the case recomputes**

```bash
rm -f validation/song_2023/song_2023.npz
```

- [ ] **Step 2: Run the case with the Stage 1 config**

```bash
LINSTAB2D_SONG2023_CONFIG=validation/song_2023/configs/scipy_assembly__mumps.jsonc \
  .venv/bin/python -m validation.song_2023.song_2023
```

Observe:
- The case completes without errors.
- `song_2023.npz` is written.
- Timing and memory stats are printed.

- [ ] **Step 3: Copy the NPZ to the stage-specific name**

```bash
cp validation/song_2023/song_2023.npz \
   validation/song_2023/configs/scipy_assembly__mumps.npz
```

- [ ] **Step 4: Run spectra comparison against MATLAB reference**

```bash
.venv/bin/python -m validation.song_2023.compare_spectra \
  --python-npz validation/song_2023/configs/scipy_assembly__mumps.npz
```

Observe: the audit report prints. Record the `max abs error` and `rms abs error` for MATLAB vs Python. These are the baseline metrics.

- [ ] **Step 5: Record baseline metrics in a short summary**

Create `validation/song_2023/configs/stage1_baseline.log` with the comparison output.

---

## Chunk 2: Stage 2 — PETSc Assembly + MUMPS Parity

### Task 3: Create Stage 2 config

**Files:**
- Create: `validation/song_2023/configs/petsc_assembly__mumps.jsonc`

- [ ] **Step 1: Write the Stage 2 JSONC**

Identical to Stage 1 except:
- `"assembly_backend": "petsc"` (was `"scipy"`)

All other fields are identical to `scipy_assembly__mumps.jsonc`.

Purpose: separate assembly-backend effects from solver effects. If this stage drifts from Stage 1, the PETSc assembly path has a parity bug.

- [ ] **Step 2: Verify config loads**

```bash
.venv/bin/python -c "
from linstab2d.utils.jsonc import load_jsonc
cfg = load_jsonc('validation/song_2023/configs/petsc_assembly__mumps.jsonc')
assert cfg['assembly_backend'] == 'petsc'
assert cfg['solver']['petsc_matrix_mode'] == 'local'
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add validation/song_2023/configs/petsc_assembly__mumps.jsonc
git commit -m "validation: add Stage 2 petsc+mumps config for Song2023 ladder"
```

### Task 4: Run Stage 2 and compare against Stage 1

- [ ] **Step 1: Run the case with the Stage 2 config**

```bash
rm -f validation/song_2023/song_2023.npz
LINSTAB2D_SONG2023_CONFIG=validation/song_2023/configs/petsc_assembly__mumps.jsonc \
  .venv/bin/python -m validation.song_2023.song_2023
cp validation/song_2023/song_2023.npz \
   validation/song_2023/configs/petsc_assembly__mumps.npz
```

- [ ] **Step 2: Compare Stage 2 eigenvalues against Stage 1**

Load both NPZs, sort eigenvalues by imaginary part, compute max and mean |delta omega|.

**Decision gate:** If max |delta omega| > 1e-4, stop and fix PETSc assembly parity before proceeding. The issue is in the assembly path, not the solver.

- [ ] **Step 3: Run spectra comparison against MATLAB**

```bash
.venv/bin/python -m validation.song_2023.compare_spectra \
  --python-npz validation/song_2023/configs/petsc_assembly__mumps.npz
```

Record metrics. They should be essentially identical to Stage 1.

- [ ] **Step 4: Record Stage 2 metrics**

Append to or create `validation/song_2023/configs/stage2_petsc_parity.log`.

---

## Chunk 3: Stage 3 — Krylov + LU Preconditioner Bridge

### Task 5: Create Stage 3 config

**Files:**
- Create: `validation/song_2023/configs/petsc_assembly__krylov_lu_pc.jsonc`

- [ ] **Step 1: Write the Stage 3 JSONC**

Identical to Stage 2 plus a `solver.eigen` section:

```jsonc
"solver": {
    "petsc_matrix_mode": "local",
    "linsolve": { "type": "mumps" },
    "eigen": {
        "st_solver": "iterative",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_pc_side": "right",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 20000
    }
}
```

Purpose: first Krylov EVP path using the known-working gmres+lu-pc fallback. This is a Krylov-owned solve with a strong LU preconditioner — not yet the final memory-light target, but the correct bridge between direct LU and lighter preconditioners.

- [ ] **Step 2: Verify config loads and has the eigen section**

```bash
.venv/bin/python -c "
from linstab2d.utils.jsonc import load_jsonc
cfg = load_jsonc('validation/song_2023/configs/petsc_assembly__krylov_lu_pc.jsonc')
eigen = cfg['solver']['eigen']
assert eigen['st_solver'] == 'iterative'
assert eigen['pc_type'] == 'lu'
print('OK:', eigen)
"
```

- [ ] **Step 3: Commit**

```bash
git add validation/song_2023/configs/petsc_assembly__krylov_lu_pc.jsonc
git commit -m "validation: add Stage 3 krylov+lu-pc config for Song2023 ladder"
```

### Task 6: Run Stage 3 and compare against Stages 1-2

- [ ] **Step 1: Run the case with the Stage 3 config**

```bash
rm -f validation/song_2023/song_2023.npz
LINSTAB2D_SONG2023_CONFIG=validation/song_2023/configs/petsc_assembly__krylov_lu_pc.jsonc \
  .venv/bin/python -m validation.song_2023.song_2023
cp validation/song_2023/song_2023.npz \
   validation/song_2023/configs/petsc_assembly__krylov_lu_pc.npz
```

Observe:
- The case converges without PETSc/SLEPc divergence errors.
- If it fails with KSP divergence, record the error and investigate before giving up.

- [ ] **Step 2: Compare Stage 3 eigenvalues against baseline**

Load both NPZs, sort eigenvalues, compute max and mean |delta omega| vs Stage 1.

- [ ] **Step 3: Record Stage 3 metrics including timing and memory**

From the `song_2023.log` output, record wall-clock time and peak memory for comparison with Stage 1.

- [ ] **Step 4: Run spectra comparison against MATLAB**

```bash
.venv/bin/python -m validation.song_2023.compare_spectra \
  --python-npz validation/song_2023/configs/petsc_assembly__krylov_lu_pc.npz
```

---

## Chunk 4: Stage 4 — Lighter Preconditioner Exploration

### Task 7: Create Stage 4 candidate configs

**Files:**
- Create: `validation/song_2023/configs/petsc_assembly__krylov_asm.jsonc`

This stage is exploratory. The first candidate is ASM (additive Schwarz) with `fgmres`, which is the default Krylov PC in the existing tuning matrix (`validation/krylov_evp_tuning.jsonc`).

- [ ] **Step 1: Write the ASM candidate JSONC**

Identical to Stage 3 except:

```jsonc
"eigen": {
    "st_solver": "iterative",
    "ksp_type": "fgmres",
    "pc_type": "asm",
    "ksp_pc_side": "right",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 20000,
    "asm_overlap": 2,
    "sub_ksp_type": "preonly",
    "sub_pc_type": "lu"
}
```

- [ ] **Step 2: Commit**

```bash
git add validation/song_2023/configs/petsc_assembly__krylov_asm.jsonc
git commit -m "validation: add Stage 4 krylov+asm candidate config for Song2023"
```

### Task 8: Run Stage 4 candidates and record results

- [ ] **Step 1: Run the ASM candidate**

```bash
rm -f validation/song_2023/song_2023.npz
LINSTAB2D_SONG2023_CONFIG=validation/song_2023/configs/petsc_assembly__krylov_asm.jsonc \
  .venv/bin/python -m validation.song_2023.song_2023
```

If the run:
- **Converges**: copy NPZ, compare eigenvalues against Stages 1-3, record timing/memory.
- **Diverges** (KSP or EPS error): record the error message. This is expected behavior for lighter PCs — do not try to force convergence.

- [ ] **Step 2: Compare (if converged) against baseline**

Same comparison pattern as Task 6, Step 2.

- [ ] **Step 3: Record all Stage 4 results in a log**

For each candidate, record: config name, converged (yes/no), n_eigenvalues, max |delta omega| vs baseline, wall time, peak memory.

---

## Chunk 5: Ladder Summary and Orchestrator

### Task 9: Create the ladder orchestrator config

**Files:**
- Create: `validation/song_2023/run_ladder.jsonc`

- [ ] **Step 1: Write the ladder JSONC**

Contains: `case_dir`, list of `stages` (name, config path, output NPZ path, description, is_baseline flag), comparison thresholds (`parity_threshold: 1e-8`, `krylov_threshold: 1e-4`), and `n_eigenvalues_compare: 20`.

- [ ] **Step 2: Commit**

```bash
git add validation/song_2023/run_ladder.jsonc
git commit -m "validation: add Song2023 ladder orchestrator config"
```

### Task 10: Write the ladder orchestrator script

**Files:**
- Create: `validation/song_2023/run_ladder.py`

- [ ] **Step 1: Write the orchestrator**

The script should:

1. Load `run_ladder.jsonc`
2. For each stage in order:
   - Delete the existing `song_2023.npz` so recomputation is forced
   - Set `LINSTAB2D_SONG2023_CONFIG` to the stage config path
   - Run `song_2023.py` via subprocess (to isolate PETSc state)
   - Copy the output NPZ to the stage-specific path
   - Parse the `.log` file for timing and memory
3. After all stages:
   - Load all NPZ files
   - Compare eigenvalues between each stage and the baseline
   - Print a summary table with columns:
     `stage | converged | n_eigenvalues | max|delta omega| vs baseline | wall_time_s | peak_memory_mib`
   - Save the table to `validation/song_2023/configs/ladder_summary.txt`

- [ ] **Step 2: Verify syntax**

```bash
.venv/bin/python -c "import ast; ast.parse(open('validation/song_2023/run_ladder.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add validation/song_2023/run_ladder.py
git commit -m "validation: add Song2023 ladder orchestrator script"
```

### Task 11: Run the full ladder and produce the summary table

- [ ] **Step 1: Run the ladder**

```bash
.venv/bin/python -m validation.song_2023.run_ladder
```

This will take several minutes per stage.

- [ ] **Step 2: Review the summary table**

Check `validation/song_2023/configs/ladder_summary.txt` for:
- Stage 1 (baseline): converged, eigenvalues present
- Stage 2 (PETSc parity): max |delta omega| < 1e-8 vs baseline
- Stage 3 (Krylov+LU-PC): max |delta omega| < 1e-4 vs baseline
- Stage 4 (Krylov+ASM): converged or explicitly recorded as divergent

- [ ] **Step 3: Commit summary**

```bash
git add validation/song_2023/configs/ladder_summary.txt
git commit -m "validation: Song2023 ladder results — Stage 1-4 comparison table"
```

---

## Definition of Done

This plan is done when we can truthfully say:

- Song2023 closes on `scipy_assembly__mumps` (Stage 1 baseline)
- Song2023 closes on `petsc_assembly__mumps` (Stage 2 parity)
- Song2023 closes on `petsc_assembly__krylov_lu_pc` (Stage 3 bridge)
- Any lighter-preconditioner Krylov result (Stage 4) is either validated or explicitly left as "diverged — unresolved"
- A summary table exists with convergence, eigenvalue agreement, timing, and memory for all stages

## Out of Scope

- Distributed MPI runs of Song2023 (covered by the separate MPI Option B plan)
- Resolvent analysis on Song2023 (`run_resolvent: false`)
- Changes to `paper/`
- Source code changes to `song_2023.py` or the comparison tools
- 8+ rank scaling
