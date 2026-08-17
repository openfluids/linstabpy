# SPDX-License-Identifier: GPL-3.0-only
"""Write one map point as soon as it finishes. Resume from that file."""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_PAIR_ATOL = 1.0e-12


def append_row(path: Path, fields: Sequence[str], row: Sequence[object]) -> None:
    """Append one flushed row. Write the header when the file is new."""
    if len(row) != len(fields):
        raise ValueError(f"row has {len(row)} values, header has {len(fields)} fields")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not dest.exists()) or dest.stat().st_size == 0
    with dest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(list(fields))
        writer.writerow(list(row))
        handle.flush()
        os.fsync(handle.fileno())


def finished_pairs(
    path: Path,
    *,
    beta_field: str,
    f_field: str,
) -> set[tuple[float, float]]:
    """Return (beta, frequency) pairs already stored in the CSV."""
    dest = Path(path)
    if not dest.exists() or dest.stat().st_size == 0:
        return set()
    pairs: set[tuple[float, float]] = set()
    with dest.open(encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            if rec.get(beta_field) is None or rec.get(f_field) is None:
                continue
            pairs.add((float(rec[beta_field]), float(rec[f_field])))
    return pairs


def remaining(
    values: Iterable[float],
    done: set[tuple[float, float]],
    *,
    beta: float,
    atol: float = _PAIR_ATOL,
) -> list[float]:
    """Return frequencies for this beta that are not in the finished set."""
    left: list[float] = []
    for freq in values:
        freq_f = float(freq)
        if any(
            abs(pair[0] - float(beta)) <= atol and abs(pair[1] - freq_f) <= atol for pair in done
        ):
            continue
        left.append(freq_f)
    return left


def atomic_savez(path: Path, **payload: Any) -> None:
    """Write a compressed npz, then replace the old file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, dest)


def last_factor_seconds(path: Path, field: str = "factor_s") -> float:
    """Return the last positive factor wall from the CSV, or 0."""
    dest = Path(path)
    if not dest.exists() or dest.stat().st_size == 0:
        return 0.0
    last = 0.0
    with dest.open(encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            raw = rec.get(field)
            if raw in (None, ""):
                continue
            val = float(raw)
            if val > 0.0:
                last = val
    return last


def job_remaining_s() -> float | None:
    """Return seconds left in the job, or None if no deadline is set."""
    raw = os.environ.get("LINSTABPY_JOB_DEADLINE") or os.environ.get("SLURM_JOB_END_TIME")
    if raw in (None, ""):
        return None
    try:
        end = float(raw)
    except ValueError:
        return None
    if end > 1.0e11:
        end = end / 1000.0
    return max(0.0, end - time.time())


def time_left_for_factor(remaining_s: float | None, need_s: float) -> bool:
    """Return True if the job can still pay one factor, or no deadline exists."""
    if remaining_s is None:
        return True
    if float(need_s) <= 0.0:
        return True
    return float(remaining_s) > float(need_s)
