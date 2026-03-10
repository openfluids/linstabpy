# Example 3 Temporal Blocker 2026-03-10

## What is already known
1. The Example 3 setup path is not the primary bug.
   - The Python mesh, quadrature, sponge, baseflow fields, and baseflow
     derivative inputs match the exported MATLAB base bundle to machine
     precision.
   - The exported MATLAB temporal modes satisfy the current Python temporal
     operator at low residual.
2. The blocker is therefore in the temporal eigensolve / extraction path.
   - The current Python/SLEPc run is able to move parts of the spectrum into
     the right neighborhood.
   - It does not yet recover the full exported 10-mode temporal set cleanly.
3. Runtime evidence shows this is a real Krylov workload, not a deadlock.
   - Stack sampling during the temporal solve showed time dominated by
     `EPSSolve_KrylovSchur_Default`, `BVMatArnoldi`, repeated `STApply`, and
     `MatSolve_MUMPS` calls.

## What is not yet understood
1. Which temporal extraction policy reproduces the exported MATLAB 10-mode set.
2. Whether a single target shift is sufficient, or whether Example 3 needs a
   deliberate multi-window harvest strategy to cover the low pair and the
   denser `0.62-0.63` cluster.
3. Why MATLAB `eigs(A, k, sigma)` with the saved target returns the exported set
   while the current SLEPc path does not recover the same subset reliably.

## Working conclusion
Example 3 is no longer a mesh/baseflow/operator debugging problem. It is a
solver-selection problem in a crowded target neighborhood.

## Attack plan
1. Keep Example 3 temporal and spatial runs staged separately.
2. Treat temporal closure as the active blocker.
3. Instrument temporal runs with:
   - saved raw converged temporal eigenvalues
   - target(s) used
   - residuals of the extracted eigenpairs
   - runtime and memory metadata
4. Test targeted temporal harvest strategies in this order:
   - single explicit target with explicit `ST` shift
   - smaller `nev` with larger `ncv`
   - multiple small target windows with deterministic merge policy
5. Only after a fresh temporal artifact is trustworthy, run the spatial stage.
