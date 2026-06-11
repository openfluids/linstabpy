# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for results-runtime stage heartbeat logging."""

from __future__ import annotations

import importlib
import time

results_runtime = importlib.import_module("linstabpy.driver.run")


def test_stage_heartbeat_emits_periodic_logs(monkeypatch):
    """Heartbeat context emits repeated 'still alive' logs while active."""
    messages: list[str] = []
    monkeypatch.setattr(results_runtime, "_log", messages.append)

    with results_runtime._stage_heartbeat("build_mesh", interval_seconds=0.01):
        time.sleep(0.035)

    heartbeat_messages = [msg for msg in messages if "stage heartbeat:" in msg]
    assert heartbeat_messages
    assert all("stage=build_mesh" in msg for msg in heartbeat_messages)


def test_stage_heartbeat_disabled_when_interval_non_positive(monkeypatch):
    """Non-positive intervals disable heartbeat logging."""
    messages: list[str] = []
    monkeypatch.setattr(results_runtime, "_log", messages.append)

    with results_runtime._stage_heartbeat("assemble_operator_bc", interval_seconds=0.0):
        time.sleep(0.02)

    assert not messages


def test_log_stage_includes_peak_rss_delta(monkeypatch):
    """Stage completion logs carry an RSS delta when started with the runtime helper."""
    messages: list[str] = []
    monkeypatch.setattr(results_runtime, "_log", messages.append)
    monkeypatch.setattr(results_runtime, "_peak_rss_bytes", lambda: 2048)

    results_runtime._log_stage("stability", (10.0, 1024))

    assert len(messages) == 1
    assert "stage complete: stage=stability" in messages[0]
    assert "rss_peak_delta_bytes=1024" in messages[0]
