# SPDX-License-Identifier: GPL-3.0-only
"""Assemble the runtime NPZ payload in contract groups."""

from __future__ import annotations

from typing import Any

RUNTIME_NPZ_CONTRACT_KEYS = (
    "omega",
    "direct_modes",
    "gains",
    "responses",
    "forcings",
    "resolvent_rho_fwd",
    "resolvent_rho_adj",
    "omega_target",
    "St",
    "beta",
    "fd_order",
    "alpha_filter",
    "n_modes_stability_requested",
    "n_modes_resolvent_requested",
    "X",
    "Y",
    "U",
    "sponge",
    "bc_indices",
    "bc_total_indices",
    "bc_effective_dirichlet_indices",
    "bc_column_zeroed_dirichlet_indices",
    "bc_effective_neumann_indices",
    "bc_overlap_indices",
    "W_diag",
    "invW_diag",
    "stability_eps_nev",
    "stability_eps_ncv",
    "stability_eps_nconv",
    "stability_eps_iterations",
    "stability_eps_reason",
    "stability_ksp_reason",
    "stability_eps_conv_test",
    "stability_eps_tol",
    "stability_eps_max_it",
    "stability_true_residual",
    "stability_reduce_dirichlet",
    "stability_residual_estimates",
    "stability_residual_absolute",
    "stability_residual_relative",
    "stability_residual_backward",
    "stability_postcheck_residual_abs",
    "stability_postcheck_residual_rel",
    "stability_postcheck_target_distance",
    "stability_all_omega",
    "stability_all_postcheck_rel",
    "stability_all_target_distance",
    "stability_all_reject_reason_code",
    "stability_acceptance_postcheck_rel_max",
    "stability_acceptance_target_distance_max",
    "stability_acceptance_omega_real_abs_max",
    "stability_acceptance_omega_imag_abs_max",
    "stability_acceptance_conjugate_aware",
    "stability_accepted_count",
    "stability_accepted_indices",
    "stability_rejected_indices",
    "stability_missing_conjugate_partner_indices",
    "stability_partner_rejected_indices",
    "numerics_operator_dissipation_enabled",
    "numerics_operator_dissipation_epsilon",
    "numerics_operator_dissipation_order_diss",
    "numerics_operator_dissipation_apply_to_stability",
    "numerics_operator_dissipation_apply_to_resolvent",
    "solver_eigen_st_solver_used",
    "solver_apply_to_resolvent_effective",
    "provenance_config_hash",
    "provenance_linstabpy_git_sha",
    "resolvent_sign_convention",
    "provenance_factor_backend",
    "provenance_factor_precision",
    "git_sha",
    "config_md5",
    "hostname",
    "compute_mode",
    "mpi_stack",
    "cost_wall_total_s",
    "cost_wall_assemble_s",
    "cost_wall_factor_s",
    "cost_wall_solve_s",
    "cost_peak_rss_bytes",
    "cost_peak_gpu_mem_bytes",
    "cost_mpi_ranks",
    "cost_gpus",
    "cost_petsc_options",
    "cost_hardware_tag",
    "cost_git_sha",
)


def build_runtime_payload(
    *,
    spectra: dict[str, Any],
    problem: dict[str, Any],
    fields: dict[str, Any],
    stability: dict[str, Any],
    dissipation: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Merge the runtime groups into one NPZ payload.

    Groups follow the metrics sidecar: spectra, problem, fields,
    stability, dissipation, and provenance. ``save_results`` adds cost keys later.
    by ``save_results``.
    """
    payload: dict[str, Any] = {}
    payload.update(spectra)
    payload.update(problem)
    payload.update(fields)
    payload.update(stability)
    payload.update(dissipation)
    payload.update(provenance)
    return payload
