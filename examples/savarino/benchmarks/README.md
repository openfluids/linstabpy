# Backend benchmarks: controlled single-stage measurements

These are the measurements that decided the cuDSS backend, kept so every
factorization-time and precision claim in the paper traces to a recorded
run. The production campaign records (whole resolvent points, end to end)
are the `savarino_*.csv` files one directory up; a benchmark here isolates
one stage at a time on the same 450k-unknown operator, on an otherwise
idle node.

Every CSV starts with comment lines recording machine, job id, environment,
and the exact solver settings of that run. All runs: dual Xeon Silver 4210R
node, one NVIDIA RTX A6000 per GPU job, linstabpy commit in the header.

## Workflow

1. Dump the assembled operator once from the production path:

   ```
   python dump_operator.py --help
   ```

   This writes `K` (and the shift data for a frequency line) for a chosen
   `(f, beta)` point of the Savarino case, so the benchmark scripts measure
   the solvers alone, not assembly.

2. Run a benchmark script against the dump. Each script prints and appends
   its CSV.

| Record | Script | Question it answers |
|--------|--------|---------------------|
| `fp32_preconditioner_*.csv` | `bench_fp32.py` | Does an FP32 LU precondition complex128 FGMRES on this operator? (condition estimate, equilibration, matching) |
| `cudss_factor_solve_*.csv` | `bench_cudss.py` | cuDSS factor/solve times for K and K^H against the MUMPS host reference — the 5.7–9.0 s vs 24–33 s numbers |
| `strumpack_blr_transpose_*.csv` | `bench_strumpack.py` | STRUMPACK GPU factor times (41–55 s), BLR as a preconditioner, and the transpose-solve probe |
| `rtol_ladder_*.csv` | production driver (configs one directory up, `type: iterative`, `ksp_rtol` swept) | Gain error vs FGMRES tolerance on the CPU path — the working-accuracy ladder |
| `cudss_line_point_*.csv` | `bench_cudss_line.py` | Complete cuDSS frequency-line point economics (refactor + solves), FP64 and FP32 |

## Relation to the paper

The Supplementary Material quotes from these files: the per-backend
factorization times, the single-precision iteration counts and failures
(confirmed at scale by `../savarino_cudss_farm_fp32.csv`), and the
tolerance ladder. Production `factor_s` columns in the campaign CSVs are
end-to-end per-point values on a loaded node and include both factors; the
benchmark numbers are per-factorization on an idle node — the two are
complementary, not interchangeable.
