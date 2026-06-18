# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from linstabpy.solvers.router.anchors import (
    EVP_DISTANCE_TRIGGER,
    FGMRES_ITERATIONS_TRIGGER,
    FGMRES_RESIDUAL_TRIGGER,
    IterProbeResult,
    choose_anchors_adaptive,
)
from linstabpy.solvers.router.band_marking import BandMarks, NEAR_POLE, STABLE
from linstabpy.solvers.router import stable_sweep
from linstabpy.solvers.router.certification import certified_rsvd

LOGGER = logging.getLogger(__name__)


def _marks(omegas, distances, radius):
    omegas = np.asarray(omegas, dtype=float)
    distances = np.asarray(distances, dtype=float)
    labels = np.where(distances < radius, NEAR_POLE, STABLE).astype(object)
    return BandMarks(omegas=omegas, labels=labels, distances=distances, danger_radius=float(radius))


def _operator():
    diagonal = np.array([2.0 + 0.1j, 2.7 - 0.2j, 3.4 + 0.3j, 4.2 - 0.1j, 5.0 + 0.2j])
    coupling = np.array([0.03 - 0.02j, -0.04 + 0.01j, 0.02 + 0.03j, -0.01j])
    return sp.diags([coupling, diagonal, 0.5 * coupling], [-1, 0, 1], format="csc", dtype=complex)


def _sweep_cfg(n):
    probes_a = (np.eye(n, dtype=complex)[0], np.eye(n, dtype=complex)[-1])
    probes_b = (np.eye(n, dtype=complex)[1], np.eye(n, dtype=complex)[-2])
    return {
        "sigma_tol": 1.0e-7,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (4, 5, 6),
        "probes_a": probes_a,
        "probes_b": probes_b,
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-9,
        "reuse_symbolic": False,
    }


def _direct_mumps_gain_map(L0, omegas, cfg):
    gains = []
    for omega in omegas:
        operator = stable_sweep.shifted_operator(L0, omega)
        factor = stable_sweep.factorize(
            operator,
            env_prefix="LINSTABPY_RESOLVENT",
            solver_cfg=cfg.get("solver_cfg"),
            cfg=cfg.get("factorize_cfg"),
        )
        try:

            def apply_h(rhs, factor=factor):
                return factor.solve(rhs)

            def apply_h_adj(rhs, factor=factor):
                return factor.solve_adjoint(rhs)

            direct_cfg = dict(cfg)
            direct_cfg.update(
                {
                    "forward_residuals": (0.0, 0.0),
                    "adjoint_residuals": (0.0, 0.0),
                    "approx_solve": factor.solve,
                    "approx_solve_adjoint": factor.solve_adjoint,
                }
            )
            omega_gains, *_ = certified_rsvd(apply_h, apply_h_adj, cfg["W"], cfg["invW"], direct_cfg)
            gains.append(float(omega_gains[0]))
        finally:
            factor.destroy()
    return np.asarray(gains)


def test_adaptive_inserts_more_anchors_near_hard_subband_than_easy_bulk():
    omegas = np.arange(9, dtype=float)
    distances = np.full(omegas.shape, 1.0)
    distances[(omegas >= 4.0) & (omegas <= 6.0)] = 0.04
    marks = _marks(omegas, distances, radius=0.1)

    anchors, records = choose_anchors_adaptive(
        omegas,
        marks,
        {
            "coarse_anchor_count": 2,
            "fgmres_iter_threshold": np.inf,
            "fgmres_residual_threshold": np.inf,
        },
    )

    inserted = [record for record in records if record.inserted]
    hard_inserted = [record for record in inserted if 4.0 <= record.omega.real <= 6.0]
    easy_inserted = [record for record in inserted if record not in hard_inserted]

    assert len(hard_inserted) == 3
    assert len(easy_inserted) == 0
    assert all(EVP_DISTANCE_TRIGGER in record.triggers for record in hard_inserted)
    np.testing.assert_array_equal(anchors.real, np.array([0.0, 4.0, 5.0, 6.0, 8.0]))


def test_s6_distance_signal_forces_near_pole_insertion():
    omegas = np.arange(5, dtype=float)
    marks = _marks(omegas, [1.0, 1.0, 0.01, 1.0, 1.0], radius=0.05)

    anchors, records = choose_anchors_adaptive(
        omegas,
        marks,
        {
            "coarse_anchor_count": 2,
            "fgmres_iter_threshold": np.inf,
            "fgmres_residual_threshold": np.inf,
        },
    )

    assert np.any(np.isclose(anchors.real, 2.0))
    near_pole = next(record for record in records if np.isclose(record.omega.real, 2.0))
    assert near_pole.inserted
    assert EVP_DISTANCE_TRIGGER in near_pole.triggers
    assert near_pole.evp_distance == 0.01
    assert near_pole.evp_radius == 0.05


def test_injected_fgmres_iter_and_residual_triggers_insert_anchors():
    omegas = np.arange(5, dtype=float)
    marks = _marks(omegas, np.ones_like(omegas), radius=0.0)

    def probe(omega, _anchors, _cfg):
        if np.isclose(omega.real, 1.0):
            return IterProbeResult(iterations=2, residual=2.0e-3)
        if np.isclose(omega.real, 3.0):
            return IterProbeResult(iterations=25, residual=1.0e-12)
        return IterProbeResult(iterations=2, residual=1.0e-12)

    anchors, records = choose_anchors_adaptive(
        omegas,
        marks,
        {
            "coarse_anchor_count": 2,
            "iter_probe": probe,
            "fgmres_iter_threshold": 10,
            "fgmres_residual_threshold": 1.0e-3,
        },
    )

    assert np.any(np.isclose(anchors.real, 1.0))
    assert np.any(np.isclose(anchors.real, 3.0))
    residual_record = next(record for record in records if np.isclose(record.omega.real, 1.0))
    iter_record = next(record for record in records if np.isclose(record.omega.real, 3.0))
    assert residual_record.inserted
    assert FGMRES_RESIDUAL_TRIGGER in residual_record.triggers
    assert iter_record.inserted
    assert FGMRES_ITERATIONS_TRIGGER in iter_record.triggers
    assert iter_record.fgmres_iterations == 25


def test_default_cheap_probe_reads_measured_iteration_arrays():
    omegas = np.array([0.0, 1.0, 2.0])
    marks = _marks(omegas, np.ones_like(omegas), radius=0.0)

    anchors, records = choose_anchors_adaptive(
        omegas,
        marks,
        {
            "coarse_anchor_count": 2,
            "fgmres_iterations": [1, 30, 1],
            "fgmres_residuals": [0.0, 0.0, 0.0],
            "fgmres_iter_threshold": 10,
            "fgmres_residual_threshold": np.inf,
        },
    )

    assert np.any(np.isclose(anchors.real, 1.0))
    middle = next(record for record in records if np.isclose(record.omega.real, 1.0))
    assert middle.inserted
    assert FGMRES_ITERATIONS_TRIGGER in middle.triggers


def test_adaptive_anchors_feed_stable_sweep_and_log_fixed_baseline_measurement(caplog):
    L0 = _operator()
    omegas = np.array([0.75, 0.9, 1.0, 1.1, 1.25], dtype=complex)
    marks = _marks(omegas.real, [1.0, 1.0, 0.02, 1.0, 1.0], radius=0.05)
    cfg = _sweep_cfg(L0.shape[0])

    adaptive_anchors, records = choose_anchors_adaptive(
        omegas,
        marks,
        {
            "coarse_anchor_count": 2,
            "fgmres_iter_threshold": np.inf,
            "fgmres_residual_threshold": np.inf,
        },
    )
    fixed_uniform_baseline = np.linspace(float(omegas.real.min()), float(omegas.real.max()), omegas.size)

    with caplog.at_level(logging.INFO):
        LOGGER.info(
            "adaptive_anchor_measurement adaptive_factorizations=%d fixed_uniform_factorizations=%d triggers=%s",
            len(adaptive_anchors),
            len(fixed_uniform_baseline),
            [record.trigger for record in records],
        )

    result = stable_sweep.sweep_stable_regime(L0, omegas, adaptive_anchors, cfg)
    direct = _direct_mumps_gain_map(L0, omegas, cfg)

    np.testing.assert_allclose(result.gain, direct, rtol=cfg["sigma_tol"], atol=cfg["sigma_tol"])
    assert int(result.metrics["factorize_calls"]) == len(adaptive_anchors)
    assert len(adaptive_anchors) == 3
    assert len(fixed_uniform_baseline) == 5
    assert any(record.inserted and EVP_DISTANCE_TRIGGER in record.triggers for record in records)
    assert "adaptive_anchor_measurement adaptive_factorizations=3 fixed_uniform_factorizations=5" in caplog.text
