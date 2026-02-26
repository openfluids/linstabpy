# Gabriel task: MATLAB resolvent full export checklist

Date: 2026-02-26  
Scope: export all intermediate resolvent artifacts from MATLAB so Python parity can be checked end-to-end.

## Objective

Produce a deterministic MATLAB export pack that lets us compare Python vs MATLAB at every resolvent stage, not only final gains/modes.

Reference MATLAB implementation:
- `linstab2d/data/baseflows/aux_matlab/resolvent.m`
- `linstab2d/data/baseflows/aux_matlab/AuxiliaryFunctions/GetInverseFunction.m`

## Freeze one parity case first

- [ ] Select one case and freeze it (recommended first case: Blasius or Hamada resolvent setup).
- [ ] Record exact numerical settings: `Re`, `Ma`, `Pr`, `kappa`, `T0`, `omega`, `nEig`, `fd_order`, mesh size, BC penalty, model (`2D`/`axy`), `m`, `kx/beta`.
- [ ] Use deterministic RNG: `rng(20260226,'twister')`.

## Export folder and naming

- [ ] Create case-local folder: `<case_dir>/MATLAB_exports_resolvent/`.
- [ ] Save every artifact with `-v7.3` (so Python `h5py` can load sparse complex matrices).
- [ ] Use the exact filenames and variable names below.

## Mandatory artifacts to export

- [ ] `res_config.mat` with variable `res_config` containing all scalar settings and metadata.
- [ ] `L0_postBC.mat` with variable `L0_postBC` (the exact operator passed to `resolvent(...)`).
- [ ] `W.mat` with variable `W` (matrix or diagonal sparse matrix used in MATLAB call).
- [ ] `invW.mat` with variable `invW`.
- [ ] `W_diag.mat` with variable `W_diag` (explicit vector diagonal, always provided even if `W` is sparse matrix).
- [ ] `invW_diag.mat` with variable `invW_diag`.
- [ ] `B.mat` with variable `B` (as used in call to `resolvent`, after any vector-to-diagonal conversion).
- [ ] `C.mat` with variable `C`.
- [ ] `LsI.mat` with variable `LsI` where `LsI = L0_postBC - 1i*omega*I`.
- [ ] `LU_factors.mat` with variables `LL`, `UU`, `pp`, `qq`, `rr` from `lu(LsI)` when `GetInverseFunction` uses `opts.type='lu'`.
- [ ] `res_probe_vectors.mat` with variable `V_probe` (complex probes, size `nDOF x nProbe`, recommended `nProbe=8`).
- [ ] `res_probe_filter.mat` with variables `FILTER_V_probe`, `FILTERct_V_probe`.
- [ ] `res_probe_actions.mat` with variables `H_V_probe`, `Htr_V_probe`, `HtrH_V_probe`.
- [ ] `res_svd_raw.mat` with variables `OMEGA_raw`, `V_raw` from `eigs(HtrH,...)` before normalization loop.
- [ ] `res_svd_final.mat` with variables `S`, `U`, `V` after normalization/output loop.
- [ ] `res_checks.mat` with variables `gram_V`, `gram_U`, `residual_sigma` for quick numerical sanity.

## Exact operator-action definitions to export

- [ ] Export `H(v) = FILTER(C * R(B * FILTER(v)))`.
- [ ] Export `Htr(v) = FILTER_ct(B' * R_T(C' * FILTER_ct(v)))`.
- [ ] Export `HtrH(v) = invW * Htr(W * H(v))`.
- [ ] Evaluate and save those on every probe vector column in `V_probe`.

## Required metadata inside `res_config`

- [ ] Case name and script name used in MATLAB.
- [ ] Git commit hash of MATLAB repo and export date/time.
- [ ] `nDOF`, `omega`, `nEig`, solver type in `GetInverseFunction` (`lu`, `builtin`, `\`, `ilu`).
- [ ] If filtered run: filter function names and whether `FILTER`/`FILTER_ct` were active.
- [ ] Shape and nnz of `L0_postBC`, `LsI`, `W`, `invW`, `B`, `C`.

## Minimal MATLAB snippet requirements

- [ ] Build or capture `L0_postBC`, `W`, `invW`, `B`, `C` exactly as used in the production call.
- [ ] Recompute `LsI = L0_postBC - 1i*omega*speye(nDOF)`.
- [ ] Force LU branch once for export comparability:
- [ ] `[LL,UU,pp,qq,rr] = lu(LsI);`
- [ ] Define `R` and `R_T` exactly like `GetInverseFunction` LU branch.
- [ ] Generate probes with frozen RNG and save `H`, `Htr`, `HtrH` actions.
- [ ] Save raw eigs output (`V_raw`, `OMEGA_raw`) and final (`S`,`U`,`V`).

## Acceptance checklist before handing off

- [ ] All mandatory files exist in `<case_dir>/MATLAB_exports_resolvent/`.
- [ ] All sparse matrices are saved as sparse (not accidentally full).
- [ ] `nDOF` matches across every artifact.
- [ ] `S` is real-positive (imaginary part near machine precision).
- [ ] `V' * W * V` and `U' * W * U` are close to identity in `res_checks.mat`.
- [ ] One README note in folder describing exactly how export was produced.

## Why this is enough for full parity

With these artifacts we can compare:
- matrix-level equivalence (`L0_postBC`, `LsI`, `B`, `C`, `W`, `invW`),
- linear-solve path equivalence (LU factors and probe actions),
- operator-action equivalence (`H`, `Htr`, `HtrH` on deterministic probes),
- final SVD outputs (`S`, `U`, `V`) and normalization consistency.
