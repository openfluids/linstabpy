"""Unit tests for results-runtime stage heartbeat logging."""

from __future__ import annotations

import time

from linstab2d import results_runtime


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
