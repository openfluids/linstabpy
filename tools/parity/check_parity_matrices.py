#!/usr/bin/env python3
"""Check operator/branch parity matrices and emit gating artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BRANCH_KEYS = ["model", "solver", "bc", "deformation", "backend"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _validate_status(status: str, allowed: set[str], where: str) -> None:
    if status not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown status '{status}' at {where}; allowed: {allowed_list}")


def _operator_report(spec: dict[str, Any]) -> dict[str, Any]:
    entries = spec.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("operator matrix 'entries' must be a list")

    allowed = set(spec.get("status_values", []))
    required_status = str(spec.get("required_status", "validated"))
    if not allowed:
        raise ValueError("operator matrix 'status_values' must be non-empty")
    _validate_status(required_status, allowed, "operator.required_status")

    missing_required: list[dict[str, Any]] = []
    counts = {status: 0 for status in sorted(allowed)}

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"operator entry {i} is not an object")
        status = str(entry.get("status", "pending"))
        _validate_status(status, allowed, f"operator.entries[{i}].status")
        counts[status] = counts.get(status, 0) + 1

        required = bool(entry.get("required", True))
        if required and status != required_status:
            missing_required.append(
                {
                    "id": entry.get("id"),
                    "matlab": entry.get("matlab"),
                    "python": entry.get("python"),
                    "status": status,
                    "required_status": required_status,
                }
            )

    return {
        "generated_at_utc": _now_utc(),
        "required_status": required_status,
        "counts": counts,
        "total_entries": len(entries),
        "missing_required_count": len(missing_required),
        "missing_required": missing_required,
        "pass": len(missing_required) == 0,
    }


def _branch_combo_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(str(record[k]) for k in BRANCH_KEYS)  # type: ignore[return-value]


def _branch_report(spec: dict[str, Any]) -> dict[str, Any]:
    dims = spec.get("dimensions", {})
    if not isinstance(dims, dict):
        raise ValueError("branch matrix 'dimensions' must be an object")
    for key in BRANCH_KEYS:
        if key not in dims:
            raise ValueError(f"branch matrix missing dimensions.{key}")
        if not isinstance(dims[key], list) or not dims[key]:
            raise ValueError(f"branch matrix dimensions.{key} must be a non-empty list")

    allowed = set(spec.get("status_values", []))
    required_status = str(spec.get("required_status", "validated"))
    if not allowed:
        raise ValueError("branch matrix 'status_values' must be non-empty")
    _validate_status(required_status, allowed, "branch.required_status")

    exclusions_raw = spec.get("required_exclusions", [])
    if not isinstance(exclusions_raw, list):
        raise ValueError("branch matrix 'required_exclusions' must be a list")
    exclusions: set[tuple[str, str, str, str, str]] = set()
    for i, ex in enumerate(exclusions_raw):
        if not isinstance(ex, dict):
            raise ValueError(f"required_exclusions[{i}] must be an object")
        missing = [k for k in BRANCH_KEYS if k not in ex]
        if missing:
            raise ValueError(f"required_exclusions[{i}] missing keys: {', '.join(missing)}")
        exclusions.add(_branch_combo_key(ex))

    records = spec.get("records", [])
    if not isinstance(records, list):
        raise ValueError("branch matrix 'records' must be a list")

    record_map: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"branch record {i} is not an object")
        missing = [k for k in BRANCH_KEYS if k not in rec]
        if missing:
            raise ValueError(f"branch record {i} missing keys: {', '.join(missing)}")

        for k in BRANCH_KEYS:
            if rec[k] not in dims[k]:
                raise ValueError(
                    f"branch record {i} has invalid {k}='{rec[k]}'; "
                    f"allowed: {dims[k]}"
                )

        status = str(rec.get("status", "pending"))
        _validate_status(status, allowed, f"branch.records[{i}].status")

        key = _branch_combo_key(rec)
        if key in record_map:
            raise ValueError(f"duplicate branch record for combo {key}")
        record_map[key] = rec

    combos: list[dict[str, Any]] = []
    counts = {status: 0 for status in sorted(allowed)}
    required_missing: list[dict[str, Any]] = []

    for combo in itertools.product(
        dims["model"], dims["solver"], dims["bc"], dims["deformation"], dims["backend"]
    ):
        key = tuple(str(x) for x in combo)
        rec = record_map.get(key)
        status = str(rec["status"]) if rec is not None else "pending"
        _validate_status(status, allowed, f"branch.combo[{key}].status")
        counts[status] = counts.get(status, 0) + 1

        combo_entry = {
            "model": key[0],
            "solver": key[1],
            "bc": key[2],
            "deformation": key[3],
            "backend": key[4],
            "status": status,
        }
        if rec and "notes" in rec:
            combo_entry["notes"] = rec["notes"]
        combos.append(combo_entry)

        required = key not in exclusions
        if required and status != required_status:
            required_missing.append(
                {
                    "model": key[0],
                    "solver": key[1],
                    "bc": key[2],
                    "deformation": key[3],
                    "backend": key[4],
                    "status": status,
                    "required_status": required_status,
                }
            )

    return {
        "generated_at_utc": _now_utc(),
        "required_status": required_status,
        "dimensions": dims,
        "total_combinations": len(combos),
        "counts": counts,
        "required_exclusions_count": len(exclusions),
        "missing_required_count": len(required_missing),
        "missing_required": required_missing,
        "pass": len(required_missing) == 0,
        "combinations": combos,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-matrix",
        type=Path,
        default=Path("tools/parity/operator_matrix.json"),
        help="Path to operator matrix JSON",
    )
    parser.add_argument(
        "--branch-matrix",
        type=Path,
        default=Path("tools/parity/branch_matrix.json"),
        help="Path to branch matrix JSON",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where report JSON files are written",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success even when required entries are not fully validated",
    )
    args = parser.parse_args()

    operator_spec = _load_json(args.operator_matrix)
    branch_spec = _load_json(args.branch_matrix)

    operator_report = _operator_report(operator_spec)
    branch_report = _branch_report(branch_spec)

    _write_json(args.artifacts_dir / "operator_matrix_report.json", operator_report)
    _write_json(args.artifacts_dir / "branch_matrix_report.json", branch_report)

    overall_pass = bool(operator_report["pass"] and branch_report["pass"])
    summary = {
        "generated_at_utc": _now_utc(),
        "operator_pass": operator_report["pass"],
        "branch_pass": branch_report["pass"],
        "overall_pass": overall_pass,
        "operator_missing_required_count": operator_report["missing_required_count"],
        "branch_missing_required_count": branch_report["missing_required_count"],
    }
    _write_json(args.artifacts_dir / "parity_gate_summary.json", summary)

    print(
        "Parity matrix check:",
        f"operator_pass={operator_report['pass']}",
        f"branch_pass={branch_report['pass']}",
        f"overall_pass={overall_pass}",
    )
    print(
        "Missing required:",
        f"operator={operator_report['missing_required_count']}",
        f"branch={branch_report['missing_required_count']}",
    )

    if not overall_pass and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
