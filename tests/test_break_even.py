# SPDX-License-Identifier: GPL-3.0-only
"""Break-even router tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router.break_even import (
    BreakEvenProbeResult,
    _timing_probe,
    break_even_decision,
)


def _pattern(n: int = 6) -> sp.csc_matrix:
    diagonal = 4.0 + 0.1j * np.arange(n)
    lower = -0.2 + 0.03j * np.arange(n - 1)
    upper = 0.1 - 0.02j * np.arange(n - 1)
    return sp.diags([lower, diagonal, upper], [-1, 0, 1], format="csc", dtype=complex)


def _fake_probe(
    *,
    t_fact_seconds: float,
    t_apply_seconds: float,
    iterations: int,
    calls: list[tuple[tuple[int, int], int]] | None = None,
):
    def probe(pattern: sp.csc_matrix, _cfg: dict[str, Any]) -> BreakEvenProbeResult:
        if calls is not None:
            calls.append((tuple(pattern.shape), int(pattern.nnz)))
        return BreakEvenProbeResult(
            t_fact_seconds=t_fact_seconds,
            t_apply_seconds=t_apply_seconds,
            iterations=iterations,
            n=int(pattern.shape[0]),
            nnz=int(pattern.nnz),
            reorder="colamd",
            backend="fake-probe",
        )

    return probe


def test_break_even_is_reproducible_and_probe_timings_drive_crossover() -> None:
    pattern = _pattern()
    calls: list[tuple[tuple[int, int], int]] = []
    cfg = {
        "k_anchors": 3,
        "break_even_probe": _fake_probe(
            t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=2, calls=calls
        ),
    }

    decision = break_even_decision(pattern, 3, cfg)
    repeat = break_even_decision(pattern, 3, cfg)

    assert decision == repeat
    assert calls == [((6, 6), int(pattern.nnz)), ((6, 6), int(pattern.nnz))]
    assert decision.path == "direct"
    assert decision.t_fact_seconds == 10.0
    assert decision.t_apply_seconds == 1.0
    assert decision.iterations == 2
    assert decision.k_anchors == 3
    assert decision.crossover_nf == pytest.approx(3.0 / (1.0 - (2.0 * 1.0 / 10.0)))
    assert "fake-probe" in decision.note

    # Same pattern and anchor count, but a different measured apply timing moves
    # the crossover.  A hardcoded constant would not follow this fake probe.
    cheaper_apply = break_even_decision(
        pattern,
        3,
        {
            "k_anchors": 3,
            "break_even_probe": _fake_probe(
                t_fact_seconds=10.0, t_apply_seconds=0.25, iterations=2
            ),
        },
    )
    assert cheaper_apply.crossover_nf == pytest.approx(3.0 / (1.0 - (2.0 * 0.25 / 10.0)))
    assert cheaper_apply.crossover_nf != pytest.approx(decision.crossover_nf)


def test_small_and_large_frequency_counts_route_across_same_measured_crossover() -> None:
    pattern = _pattern()
    cfg = {
        "k_anchors": 3,
        "break_even_probe": _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=2),
    }

    small = break_even_decision(pattern, 3, cfg)
    large = break_even_decision(pattern, 4, cfg)

    assert small.crossover_nf == pytest.approx(3.75)
    assert large.crossover_nf == pytest.approx(small.crossover_nf)
    assert small.path == "direct"
    assert large.path == "fgmres-sweep"
    assert "path=fgmres-sweep" in large.note


def test_degenerate_apply_cost_keeps_direct_for_all_frequency_counts() -> None:
    pattern = _pattern()
    cfg = {
        "k_anchors": 1,
        "break_even_probe": _fake_probe(t_fact_seconds=1.0, t_apply_seconds=0.5, iterations=2),
    }

    one_frequency = break_even_decision(pattern, 1, cfg)
    many_frequencies = break_even_decision(pattern, 1_000_000, cfg)

    assert one_frequency.crossover_nf == float("inf")
    assert many_frequencies.crossover_nf == float("inf")
    assert one_frequency.path == "direct"
    assert many_frequencies.path == "direct"
    assert "sweep never amortizes" in many_frequencies.note


def test_real_splu_probe_measures_factorization_and_representative_solve() -> None:
    pattern = _pattern(20)

    probe = _timing_probe(pattern, {"representative_iterations": 1})

    assert probe.backend == "scipy-splu"
    assert probe.n == pattern.shape[0]
    assert probe.nnz == pattern.nnz
    assert probe.t_fact_seconds > 0.0
    assert probe.t_apply_seconds > 0.0
    assert probe.iterations == 1

    decision = break_even_decision(pattern, 1, {"k_anchors": 1, "representative_iterations": 1})
    expected = decision.k_anchors / (
        1.0 - decision.iterations * decision.t_apply_seconds / decision.t_fact_seconds
    )
    assert decision.crossover_nf == pytest.approx(expected)
    assert "scipy-splu" in decision.note


def test_break_even_at_exact_crossover_selects_sweep() -> None:
    pattern = _pattern()
    cfg = {
        "k_anchors": 4,
        "break_even_probe": _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=2),
    }
    at_crossover = break_even_decision(pattern, 5, cfg)
    below = break_even_decision(pattern, 4, cfg)
    assert at_crossover.crossover_nf == pytest.approx(5.0)
    assert at_crossover.path == "fgmres-sweep"
    assert below.path == "direct"
    assert "at/above" in at_crossover.note
    assert "below" in below.note


def test_break_even_reads_anchor_count_and_rejects_non_positive_probe() -> None:
    pattern = _pattern()
    probe = _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=2)
    from_list = break_even_decision(pattern, 3, {"anchors": [0.0, 1.0, 2.0], "break_even_probe": probe})
    assert from_list.k_anchors == 3
    default_k = break_even_decision(pattern, 3, {"break_even_probe": probe})
    assert default_k.k_anchors == 1
    zero_iters = _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=0)
    allowed = break_even_decision(pattern, 3, {"k_anchors": 2, "break_even_probe": zero_iters})
    assert allowed.iterations == 0
    with pytest.raises(ValueError, match="T_fact"):
        break_even_decision(
            pattern, 2, {"break_even_probe": _fake_probe(t_fact_seconds=0.0, t_apply_seconds=1.0, iterations=1)}
        )
    with pytest.raises(ValueError, match="T_apply"):
        break_even_decision(
            pattern, 2, {"break_even_probe": _fake_probe(t_fact_seconds=1.0, t_apply_seconds=0.0, iterations=1)}
        )


def test_timing_probe_defaults_iterations_and_rejects_zero_frequency_count() -> None:
    pattern = _pattern(8)
    probe = _timing_probe(pattern, {})
    assert probe.iterations == 1
    assert probe.t_fact_seconds > 0.0
    cfg = {"k_anchors": 1, "break_even_probe": _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=1)}
    with pytest.raises(ValueError, match="n_frequencies"):
        break_even_decision(pattern, 0, cfg)
    with pytest.raises(ValueError, match="k_anchors"):
        break_even_decision(pattern, 2, {"k_anchors": 0, "break_even_probe": cfg["break_even_probe"]})
    with pytest.raises(ValueError, match="iterations"):
        break_even_decision(
            pattern,
            2,
            {"k_anchors": 1, "break_even_probe": _fake_probe(t_fact_seconds=10.0, t_apply_seconds=1.0, iterations=-1)},
        )
