# Task for Gabriel (MATLAB Exports for Python Parity)

## Context
We are matching one Hamada reference case between:
- MATLAB workflow (`example_6_hamada/workflow/main.m`)
- Python/PETSc workflow (`example_6_hamada.py`)

Current mismatch is mainly in operator equivalence.  
To debug efficiently, we need exact MATLAB intermediate matrices and index sets.

## Goal
Please run **one fixed case** and export the exact artifacts below so we can compare MATLAB vs Python step-by-step.

## Important
- Use the same parameters as your reference run (Re, Ma, Pr, kappa, T0, FDorder, St, beta, penalty, BC setup).
- Save all files under:
- `example_6_hamada/MATLAB_exports/`
- Use MATLAB `save(..., '-v7.3')` for large sparse matrices.
- Keep variable names exactly as requested.

## Required Deliverables
1. `run_metadata.mat`
- `Re, Ma, Pr, kappa, T_0, FDorder, St, beta, penalty`
- `omega_target, nEig_GS, nEig`
- `ratio_Uinfty_to_characVelocity, ratio_L_to_Lx`
- `savepath`

2. `case_inputs_used.mat`
- `x, y`
- `sponge_aux` (or final sponge used)
- `baseFlow.RHO, baseFlow.U, baseFlow.V, baseFlow.W, baseFlow.T`
- `baseFlow.MU, baseFlow.dmudT, baseFlow.d2mudT2`

3. `indices_and_bcs.mat`
- full `idx` struct from `GetLinProblem`
- `idx_dirichlet_lt`
- `idx_dirichlet_b`
- `idx_neumann_b`
- `idx_neumann_r`

4. Operator snapshots (sparse matrices, exact workflow order)
- `L0_preBC.mat` with variable `L0_preBC` (right after `GetLinProblem`, before any BC)
- `L0_after_dirichlet_lt.mat` with variable `L0_after_dirichlet_lt`
- `L0_after_dirichlet_b.mat` with variable `L0_after_dirichlet_b`
- `L0_after_neumann_bT.mat` with variable `L0_after_neumann_bT`
- `L0_postBC.mat` with variable `L0_postBC` (after all BCs)

5. Spectrum/mode references actually used in MATLAB run
- `spectrum.mat` (variable `omega`)
- `eigenvector_7.mat` (same format you already use)

6. Convention check
- `operator_convention.mat` with:
- `LO_saved` (exact matrix used in your current `LO.mat`)
- `L0_postBC`
- `scale_note` (string, for example `"LO_saved = L0_postBC"` or `"LO_saved = L0_postBC/(-1i)"`)

## Minimal MATLAB snippet (guide)
```matlab
export_dir = '../example_6_hamada/MATLAB_exports';
if ~exist(export_dir,'dir'); mkdir(export_dir); end

save(fullfile(export_dir,'run_metadata.mat'), ...
    'Re','Ma','Pr','kappa','T_0','FDorder','St','beta','penalty', ...
    'omega_target','nEig_GS','nEig', ...
    'ratio_Uinfty_to_characVelocity','ratio_L_to_Lx','savepath', '-v7.3');

% After readbaseflow + sutherland + sponge
save(fullfile(export_dir,'case_inputs_used.mat'), ...
    'x','y','sponge_aux','baseFlow','-v7.3');

% After GetLinProblem
L0_preBC = L0;
save(fullfile(export_dir,'L0_preBC.mat'),'L0_preBC','-v7.3');

% After each BC step
L0_after_dirichlet_lt = L0;
save(fullfile(export_dir,'L0_after_dirichlet_lt.mat'),'L0_after_dirichlet_lt','-v7.3');

L0_after_dirichlet_b = L0;
save(fullfile(export_dir,'L0_after_dirichlet_b.mat'),'L0_after_dirichlet_b','-v7.3');

L0_after_neumann_bT = L0;
save(fullfile(export_dir,'L0_after_neumann_bT.mat'),'L0_after_neumann_bT','-v7.3');

L0_postBC = L0;
save(fullfile(export_dir,'L0_postBC.mat'),'L0_postBC','-v7.3');

save(fullfile(export_dir,'indices_and_bcs.mat'), ...
    'idx','idx_dirichlet_lt','idx_dirichlet_b','idx_neumann_b','idx_neumann_r','-v7.3');
```

## What we will do with this
- Compare matrix parity at each stage (`preBC`, each BC, `postBC`)
- Isolate whether mismatch comes from:
- linearized operator formation
- BC row construction/indexing
- convention/scaling mismatch

This will let us close MATLAB/Python equivalence before scaling to larger realistic cases.
