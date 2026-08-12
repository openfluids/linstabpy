# cuDSS Direct C-API Integration Design Note

This note decides whether `linstabpy` should integrate NVIDIA cuDSS as a GPU
sparse-direct factor backend for the shift-invert EVP and resolvent paths. It is
a design note only. It does not wire cuDSS into any solver and does not claim a
benchmarked result.

## Established constraint: no released PETSc cuDSS interface

cuDSS has no released PETSc solver-registry interface in the facts checked for
this bead: `MATSOLVERCUDSS` returned 404, the PETSc 3.25 changelog did not list a
cuDSS solver, and PETSc `config/packages` did not contain `cudss.py`. Therefore a
cuDSS path is not a PETSc options one-liner such as:

```text
-pc_factor_mat_solver_type cudss
```

Any `linstabpy` cuDSS experiment would be a bespoke direct-C-API integration
outside PETSc's `MATSOLVER` registry and outside the current factor-backend seam
based on `pc_factor_mat_solver_type`, probe-based availability checks, and
fail-loud PETSc setup.

## Approach: direct C-API path

A cuDSS R&D spike would need a new SLEPc ST shell path rather than a normal
PETSc `PCLU` factor-backend selection. The concrete path is:

1. Build the shifted matrix with the existing PETSc/SLEPc operator assembly, then
   extract the AIJ matrix into CSR arrays. The extracted data must preserve the
   non-symmetric complex operator values used by the temporal EVP, spatial EVP,
   and resolvent shift-invert paths.
2. Create cuDSS matrix descriptors for a non-symmetric complex LU solve. The
   spike should test both `CUDSS_ALG_1` and `CUDSS_ALG_2` because no repository
   measurement currently identifies which algorithm is more robust for these
   shifted operators.
3. Run cuDSS analyze and factor once per shift. Reuse that factorization for all
   RHS solves at the same shift. Refactor only when the shift, matrix structure,
   ordering, precision, or numerical options change.
4. Implement solve-per-RHS calls behind an ST shell action. SLEPc would still own
   the outer eigen/resolvent iteration, but the shell would call cuDSS for each
   linear solve instead of PETSc `KSPPREONLY+PCLU` or PETSc `FGMRES+PCLU`.
5. Define vector marshalling explicitly. PETSc owns the host `Vec` objects at the
   SLEPc boundary. The cuDSS adapter owns the device buffers for CSR data,
   factors, and RHS/solution work arrays. For each ST shell application, copy or
   map the PETSc input vector into the adapter-owned device RHS buffer, call
   cuDSS solve, then copy or map the device solution back into the PETSc output
   `Vec`. If a future PETSc device-vector route is used, ownership still remains
   explicit: PETSc owns the `Vec`; the cuDSS adapter borrows or mirrors its data
   only for the duration of the solve.
6. Record timings and bytes for AIJ-to-CSR extraction, host-device transfers,
   analyze, factor, and each solve. Without those records, the path remains a
   code experiment rather than an adoption candidate.

This approach keeps CPU MUMPS unchanged as the reference path. It also keeps the
existing PETSc factor-backend seam intact for MUMPS, STRUMPACK, and other PETSc
registered solvers.

## Tradeoffs vs STRUMPACK

STRUMPACK is already reachable through PETSc's solver registry. In the current
backend notes, the useful STRUMPACK route is `KSP=fgmres`, `PC=lu`,
`PCFactorSetMatSolverType=strumpack`, right-preconditioned solves, and the
recorded small shifted-EVP comparison against the MUMPS reference tolerances.
That route stays inside PETSc/SLEPc setup, options handling, availability probes,
and ordinary failure paths.

cuDSS might buy a simpler single-GPU sparse-direct factor/solve execution model
and lower per-RHS solve time after a factorization is resident on the GPU
[assumed]. It might also be attractive for repeated RHS solves at one shift if
factorization reuse dominates transfer and setup costs [assumed].

The cost is that cuDSS does not enter through PETSc's factor backend registry.
`linstabpy` would own a new non-PETSc integration surface: CSR extraction,
cuDSS descriptors, device-memory lifetime, vector transfers, error handling,
complex non-symmetric LU option choices, and SLEPc ST-shell semantics. The
largest numerical risk is complex-LU behavior on the shifted non-normal
operators; the repository has no cuDSS measurement showing acceptable residuals,
accepted modes, or resolvent gains. The largest engineering risk is maintaining a
separate solve path whose behavior cannot be checked by the existing PETSc
factor-backend probe alone.

## Maintenance cost

Owning a direct-C-API backend outside PETSc means `linstabpy` would need to
maintain:

- build and import detection for CUDA and cuDSS without changing the default CPU
  environment;
- C/Cython/ctypes or extension-module bindings for cuDSS handles, descriptors,
  status codes, streams, and data types;
- a CSR extraction layer from PETSc AIJ matrices, including integer-width and
  complex-scalar conventions;
- GPU memory ownership rules for matrix, factor, RHS, and solution buffers;
- ST-shell integration, including repeated RHS calls and cleanup on SLEPc errors;
- parity, residual, target-distance, and transfer-overhead records that are
  comparable to the CPU MUMPS reference and the STRUMPACK diagnostic route;
- version drift handling for cuDSS and CUDA; and
- documentation explaining that cuDSS is not selected with
  `pc_factor_mat_solver_type` unless PETSc later adds such a registry entry.

This is materially more expensive than a PETSc-registered factor backend because
PETSc would not be responsible for the cuDSS setup, option database, vector
movement, or solver lifecycle.

## Effort estimate

A working R&D spike is roughly 2--4 person-weeks [assumed]:

| Work item | Rough effort |
|---|---:|
| cuDSS binding skeleton, build detection, and first solve | 3--5 person-days [assumed] |
| PETSc AIJ to CSR extraction with complex data and ownership tests | 2--4 person-days [assumed] |
| SLEPc ST shell action with host-device vector marshalling | 4--7 person-days [assumed] |
| Analyze/factor reuse per shift and solve-per-RHS timing records | 2--4 person-days [assumed] |
| Parity, residual, memory, transfer, and failure-path tests | 4--7 person-days [assumed] |
| Documentation and go/no-go report | 1--2 person-days [assumed] |

This estimate assumes one developer already familiar with PETSc, SLEPc ST
semantics, CUDA device memory, and `linstabpy`'s acceptance gates [assumed]. It
excludes productization, packaging, multi-GPU work, and any upstream PETSc work.

## Go/no-go criteria

All criteria are measured against the CPU MUMPS reference at the same matrix,
shift, requested modes, RHS count, acceptance settings, and hardware class where
applicable. Quality must pass before cost is considered.

| Criterion | Metric | Threshold | Measurement method | If it fails |
|---|---:|---:|---|---|
| Parity quality | `postcheck_residual_rel_max`, `target_distance_max`, and conjugate-aware matched eigenvalue distance versus the MUMPS reference | `postcheck_residual_rel_max <= 1e-6`; `target_distance_max <= 0.11`; `matched_eigenvalue_distance_max <= 1e-6` | Reuse the acceptance-anchored parity harness or an equivalent run record with full-operator postchecks | No adoption. PETSc/SLEPc iteration convergence is not enough. |
| Complex LU robustness | Fraction of tested shifts and RHS batches that factor and solve without NaN/Inf, cuDSS error status, or residual rejection | 100% for the selected test and target cases [assumed] | Record cuDSS status, solve residuals, rejected-mode reasons, and matrix statistics for each tested shift | Keep as R&D only; do not expose as a user-selectable path. |
| Factor memory/capacity | Peak device memory for CSR, factors, and work buffers; largest matrix that factors on the target GPU | Candidate must fit the target case with at least 20% headroom [assumed], or must factor a case that the CPU reference cannot fit [repo-recorded measurement when available] | CUDA/cuDSS memory records plus matrix size, nonzeros, precision, algorithm, and GPU memory capacity [vendor spec for capacity] | Do not justify the path as a capacity route. |
| Wall time at equal mode quality | End-to-end setup plus solve wall time relative to CPU MUMPS and to STRUMPACK diagnostic route after all pass the quality gate | `<= 0.80` of CPU MUMPS [assumed] for a speed/cost win; `<= 1.10` [assumed] only if capacity also passes | Same requested modes and acceptance settings; record analyze, factor, transfer, and solve times separately | Do not claim lower wall time. |
| Factorization-reuse economics | Useful RHS solves per factorization and total sweep time versus refactor-per-shift alternatives | Reuse must reduce measured total sweep time by `>= 2.0x` [assumed] for the tested frequency/RHS pattern | Count factor setups, RHS solves, shifts, solve residuals, and accepted modes/gains | Restrict cuDSS to cases where reuse dominates, or no-go. |
| Transfer overhead | Host-device transfer time divided by total cuDSS-path wall time | `<= 0.10` [assumed] | Timers around matrix upload, RHS upload, solution download, and any PETSc `Vec` staging | The path is not GPU-resident enough; redesign residency before adoption. |
| Maintenance burden | Lines of binding code, new failure paths, and required CI/hardware checks | Small enough for one maintainer to test and update per release [assumed] | Review after the spike with the concrete diff and required test matrix | No-go if the path cannot be maintained without weakening the CPU reference workflow. |

The cuDSS path is a **go** only if parity quality passes, complex-LU robustness
passes for the selected cases, at least one of capacity or wall-time criteria
passes, transfer overhead is below threshold, and the maintenance burden is
acceptable. It is **research-only** if it factors and solves some cases but does
not pass the quality and cost gates. It is **no-go** if complex-LU behavior,
transfer overhead, or maintenance cost fails the table above.

## Recommendation

Current recommendation: **conditional no-go for product integration; allow only a
contained R&D spike after the STRUMPACK route and CPU MUMPS reference remain
unchanged**.

The rationale is simple: cuDSS has no released PETSc interface, so the work would
create a second solver integration stack rather than reuse the existing PETSc
factor-backend seam. The possible benefit is single-GPU factor/solve throughput
and repeated-RHS reuse [assumed], but the repository has no cuDSS measurements
for complex non-symmetric LU quality, memory capacity, transfer overhead, or
end-to-end wall time.

Reopen product integration only when a spike records all of the following:

- accepted EVP or resolvent results at the same acceptance-anchored tolerances as
  the CPU MUMPS reference;
- device memory records showing the target matrices fit with the headroom stated
  in the go/no-go table;
- timing records separating extraction, transfer, analyze, factor, and repeated
  solves;
- a demonstrated factorization-reuse case where the total measured sweep cost
  passes the reuse threshold; and
- a maintainability review showing that the direct-C-API path does not weaken the
  default CPU workflow or duplicate PETSc behavior in fragile ways.

Until those records exist, cuDSS should not be advertised as a selectable
`linstabpy` factor backend. The only honest near-term output is an R&D spike
plan and a measurement report, not solver wiring.

## Honesty tags and claim boundaries

All non-measured numerical thresholds and effort estimates in this note are
marked `[assumed]`. Hardware capability values must be marked `[vendor spec]` if
added later. Repository run outputs must be marked as recorded measurements only
when they come from saved `linstabpy` commands or artifacts.

This note makes no benchmark claim. It does not claim that cuDSS is integrated,
that it lowers wall time, or that it passes `linstabpy` acceptance gates. It also
does not change the CPU MUMPS reference path or the STRUMPACK diagnostic route.
