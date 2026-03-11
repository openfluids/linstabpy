# Martini Gain Audit 2026-03-07

## Scope
Audit the remaining resolvent gain-only mismatches for:
- Example 1: Blasius
- Example 2: Jet
- Example 5: Paraboloid

The goal is to decide whether the remaining `SOFT_FAIL` status is more likely a
Python implementation bug or a limitation/inconsistency in the saved MATLAB
gain reference.

## Verified findings
1. The paper definition is not the issue.
   - The gain definition in `martini2024.md` matches the implementation used in
     `linstab2d`: weighted resolvent singular values with the same forcing and
     response norm by default.
2. The remaining non-3 mismatches are gain-only.
   - After the clean rerun baseline, Examples 1, 2, and 5 have forcing and
     response modes that are already close to MATLAB, but the gain scalars still
     miss the configured `gain_rel` thresholds.
3. Solver tightening did not move the Python gains.
   - Larger `eps_ncv`, tighter `eps_tol`, and longer Krylov runs did not change
     the Python gain values for Example 1.
   - Recomputing the shell eigensolve with SciPy/ARPACK on the same operator
     also returned the Python gains.
4. Rayleigh-quotient refinement did not move the Python gains.
   - Re-evaluating gains from the converged forcing vectors on the current
     operator reproduced the stored Python gains.
5. The exported MATLAB gain scalar is not a strict gold value at current
   tolerances.
   - For Example 1, applying the exported MATLAB forcing modes to the current
     Python resolvent operator reproduces the Python gains, not the exported
     MATLAB gain scalar.
   - Example 2 and Example 5 show the same tendency, though less cleanly.
6. The saved MATLAB bundles are not fully self-consistent enough to justify
   using gain scalars as a hard closure criterion.
   - Example 1 and Example 5 bundles include exported `L0`, `W`, and `invW`.
   - The saved gain/mode triplets are not cleanly recovered from those exported
     operators in a direct audit.

## Interpretation policy
1. Do not silently relax tolerances and call the cases closed.
2. Do not use the saved MATLAB gain scalar alone as a hard gold reference for
   Examples 1, 2, and 5.
3. Treat mode-shape agreement as the stronger verification signal for these
   three resolvent cases until the MATLAB-side gain export path is audited more
   deeply.
4. Keep the current status as `SOFT_FAIL` while this caveat remains unresolved.

## Consequence for execution order
1. Example 0 and Example 4 are closed.
2. Example 3 is the next primary code target.
3. Any future appendix wording must state that the remaining Example 1/2/5
   gain-only mismatches are under reference audit, not hidden or waived.
