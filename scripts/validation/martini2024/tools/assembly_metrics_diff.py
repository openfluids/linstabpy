#!/usr/bin/env python3
"""Compare two LNSE assembly-metrics sidecars stage-by-stage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _relative_delta(before: float | int | None, after: float | int | None) -> float | None:
    if before is None or after is None:
        return None
    before_f = float(before)
    after_f = float(after)
    if math.isclose(before_f, 0.0, abs_tol=1e-30):
        if math.isclose(after_f, 0.0, abs_tol=1e-30):
            return 0.0
        return None
    return (after_f - before_f) / before_f


def _metric_triplet(
    before: float | int | None, after: float | int | None
) -> dict[str, float | None]:
    delta = None if before is None or after is None else float(after) - float(before)
    return {
        "before": None if before is None else float(before),
        "after": None if after is None else float(after),
        "delta": delta,
        "delta_ratio": _relative_delta(before, after),
    }


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:+.1f}%"


def _format_float(value: float | None, *, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value / scale:.3f}{suffix}"


def _compare_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_stages = before.get("stages", {})
    after_stages = after.get("stages", {})
    stage_names = sorted(set(before_stages) | set(after_stages))

    stage_report: dict[str, Any] = {}
    for stage in stage_names:
        left = before_stages.get(stage, {})
        right = after_stages.get(stage, {})
        stage_report[stage] = {
            "seconds": _metric_triplet(left.get("seconds"), right.get("seconds")),
            "rss_peak_delta_bytes": _metric_triplet(
                left.get("rss_peak_delta_bytes"),
                right.get("rss_peak_delta_bytes"),
            ),
            "nnz": _metric_triplet(left.get("nnz"), right.get("nnz")),
            "skipped_before": bool(left.get("skipped", False)),
            "skipped_after": bool(right.get("skipped", False)),
        }

    return {
        "baseline": {
            "path": str(before.get("_path", "")),
            "nnz": before.get("nnz"),
            "coeff_tensor_bytes_total": before.get("coeff_tensor_bytes_total"),
            "ndof": before.get("ndof"),
        },
        "candidate": {
            "path": str(after.get("_path", "")),
            "nnz": after.get("nnz"),
            "coeff_tensor_bytes_total": after.get("coeff_tensor_bytes_total"),
            "ndof": after.get("ndof"),
        },
        "summary": {
            "nnz": _metric_triplet(before.get("nnz"), after.get("nnz")),
            "coeff_tensor_bytes_total": _metric_triplet(
                before.get("coeff_tensor_bytes_total"),
                after.get("coeff_tensor_bytes_total"),
            ),
            "wall_seconds": _metric_triplet(before.get("wall_seconds"), after.get("wall_seconds")),
        },
        "stages": stage_report,
    }


def _ratio_regressed(delta_ratio: float | None, threshold: float) -> bool:
    return delta_ratio is not None and delta_ratio > threshold


def _collect_regressions(
    report: dict[str, Any],
    *,
    time_threshold: float,
    memory_threshold: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    summary = report["summary"]

    if _ratio_regressed(summary["wall_seconds"]["delta_ratio"], time_threshold):
        findings.append(
            {
                "scope": "summary",
                "metric": "wall_seconds",
                "delta_ratio": summary["wall_seconds"]["delta_ratio"],
                "threshold": time_threshold,
            }
        )
    if _ratio_regressed(summary["coeff_tensor_bytes_total"]["delta_ratio"], memory_threshold):
        findings.append(
            {
                "scope": "summary",
                "metric": "coeff_tensor_bytes_total",
                "delta_ratio": summary["coeff_tensor_bytes_total"]["delta_ratio"],
                "threshold": memory_threshold,
            }
        )

    for stage, payload in report["stages"].items():
        if payload["skipped_before"] or payload["skipped_after"]:
            continue

        seconds = payload["seconds"]
        if _ratio_regressed(seconds["delta_ratio"], time_threshold):
            findings.append(
                {
                    "scope": stage,
                    "metric": "seconds",
                    "delta_ratio": seconds["delta_ratio"],
                    "threshold": time_threshold,
                }
            )

        rss = payload["rss_peak_delta_bytes"]
        if _ratio_regressed(rss["delta_ratio"], memory_threshold):
            findings.append(
                {
                    "scope": stage,
                    "metric": "rss_peak_delta_bytes",
                    "delta_ratio": rss["delta_ratio"],
                    "threshold": memory_threshold,
                }
            )

    return findings


def _render_text(report: dict[str, Any]) -> str:
    lines = []
    summary = report["summary"]
    lines.append("Assembly metrics diff")
    lines.append(
        "summary: "
        f"wall={_format_ratio(summary['wall_seconds']['delta_ratio'])}, "
        f"coeff_bytes={_format_ratio(summary['coeff_tensor_bytes_total']['delta_ratio'])}, "
        f"nnz={_format_ratio(summary['nnz']['delta_ratio'])}"
    )
    lines.append("")
    lines.append("stage                           seconds             delta        rss_delta")
    lines.append("--------------------------------------------------------------------------")
    for stage, payload in report["stages"].items():
        seconds = payload["seconds"]
        rss = payload["rss_peak_delta_bytes"]
        suffix = ""
        if payload["skipped_before"] or payload["skipped_after"]:
            suffix = " skipped"
        lines.append(
            f"{stage:28s} "
            f"{_format_float(seconds['before'], suffix='s'):>9s} -> "
            f"{_format_float(seconds['after'], suffix='s'):>9s} "
            f"{_format_ratio(seconds['delta_ratio']):>10s} "
            f"{_format_ratio(rss['delta_ratio']):>10s}{suffix}"
        )
    regressions = report.get("regressions", [])
    if regressions:
        lines.append("")
        lines.append("regressions:")
        for item in regressions:
            lines.append(
                f"- {item['scope']} {item['metric']} "
                f"{_format_ratio(item['delta_ratio'])} "
                f"(threshold {_format_ratio(item['threshold'])})"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline assembly metrics JSON")
    parser.add_argument("candidate", type=Path, help="Candidate assembly metrics JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the comparison report as JSON instead of a text table",
    )
    parser.add_argument(
        "--fail-if-regression",
        action="store_true",
        help="Return a non-zero exit code if any stage or summary metric regresses beyond the thresholds",
    )
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=0.0,
        help="Allowed fractional increase in timing metrics before failing (0.20 = 20%%)",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.0,
        help="Allowed fractional increase in memory metrics before failing (0.20 = 20%%)",
    )
    args = parser.parse_args(argv)
    if args.time_threshold < 0.0:
        raise ValueError("--time-threshold must be >= 0")
    if args.memory_threshold < 0.0:
        raise ValueError("--memory-threshold must be >= 0")

    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    baseline["_path"] = str(args.baseline.resolve())
    candidate["_path"] = str(args.candidate.resolve())

    report = _compare_metrics(baseline, candidate)
    report["regressions"] = _collect_regressions(
        report,
        time_threshold=args.time_threshold,
        memory_threshold=args.memory_threshold,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_render_text(report))
        sys.stdout.write("\n")
    if args.fail_if_regression and report["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
