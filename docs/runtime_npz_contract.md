# Runtime NPZ Output Contract

Case-style drivers that call `linstabpy.results_runtime.compute()` write one
compressed `.npz` file per run. The archive is the handoff between long
computations, parity checks, plotting scripts, and campaign summaries, so key
names should be treated as a compatibility surface.

The checked contract lives in `linstabpy.results_runtime.RUNTIME_NPZ_CONTRACT_KEYS`.
Runtime outputs may include extra arrays, but the checked keys should not be
renamed or removed without a migration note and an update to downstream readers.

## Groups

Core spectra and resolvent:

- `omega`, `direct_modes`
- `gains`, `responses`, `forcings`
- `resolvent_rho_fwd`, `resolvent_rho_adj`
- `omega_target`, `St`, `beta`

Mesh, state, and weights:

- `X`, `Y`, `U`, `sponge`
- `W_diag`, `invW_diag`
- `fd_order`, `alpha_filter`

Boundary-condition provenance:

- `bc_indices`, `bc_total_indices`
- `bc_effective_dirichlet_indices`
- `bc_column_zeroed_dirichlet_indices`
- `bc_effective_neumann_indices`, `bc_overlap_indices`

Stability solver metadata:

- `stability_eps_nev`, `stability_eps_ncv`, `stability_eps_nconv`
- `stability_eps_iterations`, `stability_eps_reason`, `stability_ksp_reason`
- `stability_eps_conv_test`, `stability_eps_tol`, `stability_eps_max_it`
- `stability_true_residual`, `stability_reduce_dirichlet`
- `stability_residual_estimates`, `stability_residual_absolute`
- `stability_residual_relative`, `stability_residual_backward`

Stability postcheck and acceptance:

- `stability_postcheck_residual_abs`
- `stability_postcheck_residual_rel`
- `stability_postcheck_target_distance`
- `stability_all_omega`
- `stability_all_postcheck_rel`
- `stability_all_target_distance`
- `stability_all_reject_reason_code`
- `stability_acceptance_postcheck_rel_max`
- `stability_acceptance_target_distance_max`
- `stability_acceptance_omega_real_abs_max`
- `stability_acceptance_omega_imag_abs_max`
- `stability_acceptance_conjugate_aware`
- `stability_accepted_count`
- `stability_accepted_indices`
- `stability_rejected_indices`
- `stability_missing_conjugate_partner_indices`
- `stability_partner_rejected_indices`

Numerical policy and provenance:

- `numerics_operator_dissipation_enabled`
- `numerics_operator_dissipation_epsilon`
- `numerics_operator_dissipation_order_diss`
- `numerics_operator_dissipation_apply_to_stability`
- `numerics_operator_dissipation_apply_to_resolvent`
- `solver_eigen_st_solver_used`
- `solver_apply_to_resolvent_effective`
- `provenance_config_hash`
- `provenance_linstabpy_git_sha`

## Rules For Changes

Additive keys are fine when a new diagnostic or provenance field is useful.
Renames and removals are not silent cleanup: update this document, update
`RUNTIME_NPZ_CONTRACT_KEYS`, update tests, and adjust every reader that consumes
the old name.

Acceptance arrays keep the pre-acceptance and post-acceptance views separate.
`stability_all_*` arrays describe every converged mode before filtering.
`stability_accepted_indices` and `stability_rejected_indices` describe the
verdict after residual, target-distance, and omega-family gates.
