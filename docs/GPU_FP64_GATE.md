# GPU FP64 Gate for Line 3

This document is the decision gate for committing Line-3 GPU-resident work. The
CPU MUMPS path remains the reference path. Line 3 may proceed only when measured
quality and cost data show that a GPU backend is useful as a capacity,
memory-bandwidth, or mixed-precision path. It is not justified as an FP64-FLOPS
path.

## Framing: FP64 FLOPS is the wrong axis

The shifted EVP and resolvent paths in `linstabpy` are sparse-direct workloads.
For shift-invert, the expensive step is sparse factorization plus repeated sparse
triangular or preconditioned solves. Those phases are dominated by memory
capacity, fill, data movement, ordering, pivoting, and reuse economics. Peak
FP64 FLOPS is therefore a weak predictor of useful performance.

This matters most for consumer and Grace-class NVIDIA targets. GB10-class parts
are treated here as having an FP64:FP32 throughput ratio near `1:64` [vendor
spec]. Under that ratio, any argument that depends on high FP64 peak throughput
is rejected before implementation. Blackwell-Ultra-direction parts are also
framed here as deprioritizing FP64 throughput [vendor spec]. That does not make
them irrelevant for `linstabpy`; it means the case must be made through larger
problem capacity, better sustained memory movement, lower factor memory, or a
mixed-precision path whose final modes pass the same acceptance gate as the CPU
reference.

No speedup, production support, or accuracy claim is made by this document.
Those claims require recorded measurements from the gates below.

## Go/no-go criteria

All criteria are measured against the CPU MUMPS reference at equal requested
problem, target, number of modes, and acceptance settings. A Line-3 backend must
pass the quality criteria first; cost wins do not compensate for rejected modes.

| Criterion | Metric | Threshold | Measurement method | If it fails |
|---|---:|---:|---|---|
| Mixed-precision / backend parity quality | Candidate `postcheck_residual_rel_max`, reference `postcheck_residual_rel_max`, candidate/reference `target_distance_max`, and conjugate-aware matched eigenvalue distance | `postcheck_residual_rel_max <= 1e-6`; `target_distance_max <= 0.11`; `matched_eigenvalue_distance_max <= 1e-6` | Run `tests/test_parity_vs_mumps.py` and collect the emitted `PARITY_RECORD`; these are the acceptance-anchored tolerances already encoded by that harness [repo-recorded measurement when emitted] | No Line-3 commit for that backend/precision. Treat PETSc/SLEPc convergence alone as insufficient. |
| Factor memory capacity | Candidate `factor_memory_bytes / reference_factor_memory_bytes` on the same matrix, plus matrix size and nonzeros | `<= 0.75` [assumed] for a capacity win, or candidate succeeds on a matrix for which the CPU reference reports out-of-memory/factorization failure [repo-recorded measurement] | Extend or reuse the parity/factor probe record so both candidate and MUMPS factor memory are recorded from PETSc factor `MatGetInfo`; record `matrix_size`, `matrix_nnz`, ranks, and backend options | Do not justify Line 3 as a capacity path. Keep it as research-only unless another criterion independently passes. |
| Sustained memory-bandwidth use | Sustained bytes moved per second divided by the platform's documented peak memory bandwidth | `>= 0.50` of documented peak [assumed] during the dominant factor/solve phase | Use a profiler or backend log that records bytes moved and elapsed time; tag the peak bandwidth source as `[vendor spec]` and the run output as `[repo-recorded measurement]` | Do not argue from bandwidth. Investigate data layout, transfer overhead, or backend maturity before Line-3 productization. |
| Wall time at equal mode quality | Candidate end-to-end wall time divided by CPU MUMPS reference wall time, after both pass the parity quality gate | `<= 0.80` [assumed] for a speed win; `<= 1.10` [assumed] is acceptable only when the candidate also passes the capacity criterion | Use the parity `PARITY_RECORD` wall-time fields (`eps_setup`, `eps_solve`) or an equivalent run sidecar, with identical requested modes and acceptance settings | Do not claim faster. If only capacity passes, document Line 3 as capacity-only. |
| Factorization reuse economics | Number of useful RHS/solve applications per factorization, and candidate repeated-solve wall time relative to refactorizing each frequency | Reuse must reduce total measured wall time by `>= 2.0x` [assumed] over per-frequency refactorization for the tested sweep | Use the existing anchor/factor-reuse sweep records: count factor setups, forward/adjoint actions, GMRES iterations, and certified residuals per frequency [repo-recorded measurement when emitted] | Do not use Line 3 for resolvent sweeps that refactor each frequency. Defer until reuse dominates. |
| Data-transfer overhead | Host-device transfer time divided by total candidate wall time | `<= 0.10` [assumed] for GPU-resident Line 3 | Profiler or explicit timers around matrix/vector transfers and backend solve phases; include transfer bytes and rank/GPU count in the record | The implementation is not GPU-resident enough. Fix residency before claiming Line 3 benefit. |

A backend is a **go** for Line-3 productization only when:

1. the parity quality criterion passes;
2. at least one of capacity, bandwidth, or wall-time criteria passes with a
   recorded measurement; and
3. transfer overhead does not dominate the candidate run.

Otherwise the verdict is **no-go** or **research-only**, even if the backend can
be selected through PETSc.

## Recorded recommendation

Current verdict: commit Line 3 only as a capacity / memory-bandwidth /
mixed-precision effort. The FP64-FLOPS framing is rejected for this workload and
hardware class.

The evidence available in the repository today is quality-gate infrastructure,
not a GPU production measurement. The parity harness records acceptance-anchored
quality and cost fields through `PARITY_RECORD`, and the backend documentation
records that STRUMPACK with right-preconditioned FGMRES can recover a small
shifted EVP comparison while SuperLU_DIST complex CUDA solves were not reliable
in the tested build. No general speedup or production support follows from that.

Reopen the FP64-FLOPS framing only if a target GPU records both of these facts:

- hardware FP64 throughput ratio is at least `1:4` relative to FP32 [assumed
  reopen threshold], with the source tagged as `[vendor spec]` or
  `[repo-recorded measurement]`; and
- an end-to-end `linstabpy` sparse-direct shift-invert or resolvent run is at
  least `2.0x` faster than the CPU MUMPS reference at equal accepted mode
  quality [assumed reopen threshold].

Until then, every Line-3 proposal must lead with measured capacity, bandwidth,
residency, mixed-precision parity, or factorization-reuse economics, not FP64
peak FLOPS.
