# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from examples import figure_style


def _reset_style() -> None:
    figure_style._APPLIED = False
    matplotlib.rcParams["text.usetex"] = False


def test_save_exits_zero_when_probe_is_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(figure_style, "latex_available", lambda: False)
    _reset_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel(r"$\beta$")
    out = tmp_path / "no_tex.png"
    result = figure_style.save(fig, out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0
    assert matplotlib.rcParams["text.usetex"] is False


def test_save_retries_mathtext_after_usetex_savefig_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(figure_style, "latex_available", lambda: True)
    _reset_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel(r"$\beta$")
    real_savefig = fig.savefig
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("usetex missing dvipng")
        return real_savefig(*args, **kwargs)

    fig.savefig = boom
    out = tmp_path / "retry.png"
    figure_style.save(fig, out)
    assert calls["n"] == 2
    assert out.is_file()
    assert out.stat().st_size > 0
    assert matplotlib.rcParams["text.usetex"] is False


def test_latex_available_is_false_when_savefig_raises(monkeypatch) -> None:
    import matplotlib.figure

    def fail_savefig(self, *args, **kwargs):
        raise RuntimeError("no dvipng")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_savefig)
    assert figure_style.latex_available() is False


def test_check_plot_environment_requires_latex(monkeypatch) -> None:
    monkeypatch.setattr(figure_style, "latex_available", lambda: False)
    _reset_style()
    with pytest.raises(RuntimeError, match="usetex"):
        figure_style.check_plot_environment(require_latex=True)
