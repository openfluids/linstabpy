# Song2023 Order Memory Preflight Plan

## Goal
Estimate the cost of increasing Song2023 finite-difference order on the full grid before running expensive eigensolves.

## Why
- Current full corrected Song baseline already used about `7.5 GiB` peak memory at `fd_order = 2`.
- On a `16 GB` MacBook Pro, full `fd_order = 4/6/8` plus MUMPS factorization may exceed usable memory.
- We need a data-based stop/go decision before launching full stability solves or MPI stress tests.

## Preflight Scope
Run the full Song case with:
- `fd_order = 2`
- `fd_order = 4`
- `fd_order = 6`
- `fd_order = 8`

but with:
- `run_stability = false`
- `run_resolvent = false`

so the run stops after:
- baseflow load
- mesh build
- operator + BC assembly
- Chu norm
- save

This gives:
- assembly wall time
- operator nnz
- peak memory during assembly-only workflow
- per-step timings

## Deliverables
1. JSONC benchmark config for the assembly-only order sweep.
2. Runner support to choose a custom Song matrix JSONC.
3. Log output that records:
   - operator shape
   - operator nnz
   - total time
   - peak memory
   - per-step timings
4. A recommendation for which full-order solves are safe to attempt on this laptop.

## Decision Rule
- If assembly-only memory at a given order already approaches the machine limit, do not run the full eigensolve at that order on this laptop.
- If assembly-only memory stays moderate, full stability remains a candidate.

## Next Step After Preflight
- Keep `fd_order = 2` full result as baseline.
- Attempt full `fd_order = 4`, then `6`, then `8` only if the preflight trend stays acceptable.
