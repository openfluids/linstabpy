# Example Matrix Validation Plan

Date: 2026-03-14

## Goal

Re-execute the canonical examples and then stress the supported solver/backend
combinations so we can show:

- the examples still run after the recent PETSc/MPI/test cleanup work
- MATLAB-facing accuracy is preserved where reference bundles exist
- the numerics remain stable across supported backend variations

This plan excludes any changes under `paper/`.

## Terms

- `SciPy assembly`
  - build the operator first as a SciPy sparse matrix in Python
  - then convert it into PETSc for solving
- `PETSc assembly`
  - build the operator directly as a PETSc sparse matrix
  - no SciPy operator is used as the primary assembled matrix
- `MUMPS`
  - sparse direct LU factorization backend used through PETSc
  - usually the most robust option, but more memory-heavy
- `Krylov`
  - iterative PETSc solve path (`gmres`/`fgmres` family with preconditioning)
  - useful to test flexibility and scalability, but not always a drop-in replacement for direct LU

## Variant naming rule

Use:

`<assembly_path>__<solver_family>`

Examples:

- `scipy_assembly__mumps`
- `scipy_assembly__krylov`
- `petsc_assembly__mumps`
- `petsc_assembly__krylov`

This naming separates:

- where the operator comes from
- how the solve or shift-invert stage is performed

When a solver-family qualifier needs one extra level of precision, append the
PC choice:

- `scipy_assembly__krylov_lu_pc`
- `petsc_assembly__krylov_lu_pc`

Use that only when the distinction is important. For the current EVP tuning,
it is important, because the first stable Krylov-EVP runs required LU as the
preconditioner.

## Validation philosophy

We want three layers, in this order:

1. Baseline reproducibility
   - rerun the canonical example configs as-shipped
2. Backend equivalence
   - compare SciPy-assembled operators that later convert to PETSc against
     native PETSc-assembled operators where that path is supported
3. Solver robustness
   - compare MUMPS against iterative Krylov configurations where the example
     path actually uses the relevant solver

## Matrix tiers

### Core reference matrix

Purpose:
- prove that the examples still run
- prove that SciPy-assembly and native-PETSc assembly agree where supported
- keep runtime and interpretation manageable

Variants:

- `baseline`
- `scipy_assembly__mumps`
- `scipy_assembly__krylov`
- `petsc_assembly__mumps`

### Stretch robustness matrix

Purpose:
- test flexibility beyond the direct-LU baseline
- identify whether Krylov-based alternatives are practical, accurate, and stable

Variants:

- `scipy_assembly__krylov`
- `petsc_assembly__krylov`

Current EVP working fallback:

- `*_assembly__krylov_lu_pc`
  - keep the outer solve iterative
  - keep PETSc/SLEPc in Krylov mode
  - use LU as the preconditioner because lighter PCs are not yet robust enough

## Scope by example

### Example 0: curvilinear derivative validation

- baseline only
- no PETSc/backend matrix needed

### Example 1: Blasius resolvent

- Core:
  - `baseline`
  - `scipy_assembly__mumps`
  - `petsc_assembly__mumps`
- Stretch:
  - `petsc_assembly__krylov`

### Example 2: jet resolvent

- Core:
  - `baseline`
  - `scipy_assembly__mumps`
  - `petsc_assembly__mumps`
- Stretch:
  - `petsc_assembly__krylov`

### Example 3: streaky BL temporal/spatial EVP

- Core:
  - `baseline`
  - `scipy_assembly__mumps`
  - `petsc_assembly__mumps`
- Stretch:
  - `scipy_assembly__krylov_lu_pc`
  - `petsc_assembly__krylov_lu_pc`
- this example now becomes both an assembly comparison case and an eigensolver robustness case

### Example 4: cavity direct/adjoint EVP

- Core:
  - `baseline`
  - `scipy_assembly__mumps`
- Stretch:
  - `scipy_assembly__krylov_lu_pc`
- do not force native PETSc assembly here; the example explicitly documents
  that SciPy assembly is required for the current adjoint workflow

### Example 5: paraboloid resolvent

- Core:
  - `baseline`
  - `scipy_assembly__mumps`
  - `petsc_assembly__mumps`
- Stretch:
  - `petsc_assembly__krylov`

## Implementation approach

Use a dedicated JSONC-driven harness under `validation/` that:

- loads the base example JSONC
- applies variant overrides by deep merge
- redirects the example output NPZ to a variant-specific validation folder
- calls the example compute function directly
- records summary metrics in JSON

Do not overwrite the canonical `examples/example_*.npz` outputs during the
variant runs.

## Metrics to record

### For resolvent examples

- gains
- MATLAB gain comparison when available
- max SVD residuals if present in the output NPZ
- success/failure and runtime

### For temporal/spatial eigenvalue examples

- leading eigenvalues
- nearest-reference spectral mismatch when available
- max residuals if present in the output NPZ
- success/failure and runtime

## Variant policy

Only run combinations that the code already claims to support.

### Supported targets

- `baseline`
- `scipy_assembly__mumps`
- `scipy_assembly__krylov`
- `petsc_assembly__mumps`
- `petsc_assembly__krylov`

### Unsupported-by-design targets

- native PETSc assembly for Example 4 cavity adjoint workflow
- distributed PETSc matrix modes in these single-process example reruns

Unsupported combinations should be reported as skipped-by-design, not forced.

## Execution phases

### Phase 1: baseline rerun

- rerun the canonical example sequence
- confirm each example completes
- confirm reference comparisons still pass where the example already enforces them

### Phase 2: build the matrix harness

- add `validation/examples_matrix.jsonc`
- add `validation/run_examples_matrix.py`
- make the harness emit one summary JSON/CSV per run

### Phase 3: run supported variants

1. run the Core reference matrix
2. compare gains/eigenvalues/residuals against the canonical baseline
3. then run the Stretch robustness matrix
4. stop and record if a Krylov variant is accurate but impractically slow
5. stop and record if a Krylov variant fails to converge cleanly

Current knowledge:

- resolvent Krylov remains a legitimate stretch target
- for the EVP examples, the first working Krylov tradeoff is `gmres + lu` PC
- lighter PCs (`none`, `jacobi`, `ilu`, `asm`, `bjacobi`, current `fieldsplit`)
  remain unresolved and should stay in the dedicated tuning matrix until they
  actually converge

### Phase 4: assess

- identify parity-preserving variants
- identify variants that are numerically fragile, too slow, or unsupported
- keep the report evidence with runtimes and mismatch metrics

## Definition of done

This work is done when we can state, with artifacts:

- the canonical examples still run
- the Core reference matrix has been exercised on all supported examples
- the Stretch robustness matrix has either been exercised or explicitly classified as too slow / unsupported
- MATLAB-reference accuracy is preserved where the reference bundles exist
- any unsupported or fragile combinations are clearly named rather than hidden
