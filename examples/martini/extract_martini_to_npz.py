#!/usr/bin/env python3
"""Extract Martini MATLAB HDF5 exports into normalized NPZ bundles."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstabpy.utils.jsonc import load_config
from examples.martini.matlab_hdf5 import extract_file

DEFAULT_INPUT_DIR = ROOT / "martini_examples"
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "martini"
DEFAULT_MANIFEST = (
    ROOT / "examples" / "martini" / "martini_manifest.json"
)
DEFAULT_CONFIG = ROOT / "examples" / "martini" / "martini_cases.yaml"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _expected_case_notes(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    cfg = load_config(config_path)
    notes: list[dict] = []
    for case in cfg.get("cases", []):
        if str(case.get("comparison_kind", "")) != "unavailable":
            continue
        notes.append(
            {
                "case_id": str(case.get("case_id", "unknown")),
                "label": str(case.get("label", "")),
                "status": "UNAVAILABLE",
                "note": str(case.get("note", "Configured as unavailable.")),
            }
        )
    return notes


def expected_export_stems(config_path: Path) -> list[str]:
    """Return MATLAB export stems required by the Martini parity registry."""
    cfg = load_config(config_path)
    stems: list[str] = []
    seen: set[str] = set()
    for case in cfg.get("cases", []):
        for key, value in case.items():
            if not key.startswith("matlab_") or not key.endswith("_stem"):
                continue
            stem = str(value)
            if stem in seen:
                continue
            stems.append(stem)
            seen.add(stem)
    return stems


def fixture_status(input_dir: Path, output_dir: Path, config_path: Path) -> dict:
    """Inventory raw MATLAB exports and extracted NPZ fixtures."""
    stems = expected_export_stems(config_path)
    raw_missing = [stem for stem in stems if not (input_dir / f"{stem}.mat").exists()]
    npz_missing = [stem for stem in stems if not (output_dir / f"{stem}.npz").exists()]
    return {
        "input_dir": _display_path(input_dir),
        "output_dir": _display_path(output_dir),
        "config": _display_path(config_path),
        "required_count": len(stems),
        "required_stems": stems,
        "raw_mat_missing": raw_missing,
        "npz_missing": npz_missing,
        "raw_mat_present_count": len(stems) - len(raw_missing),
        "npz_present_count": len(stems) - len(npz_missing),
    }


def _print_fixture_status(status: dict) -> None:
    print(
        "Martini fixture preflight:",
        f"raw_mat={status['raw_mat_present_count']}/{status['required_count']}",
        f"npz={status['npz_present_count']}/{status['required_count']}",
    )
    if status["raw_mat_missing"]:
        print(f"Missing raw MATLAB exports in {status['input_dir']}:")
        for stem in status["raw_mat_missing"]:
            print(f"  - {stem}.mat")
    if status["npz_missing"]:
        print(f"Missing extracted fixtures in {status['output_dir']}:")
        for stem in status["npz_missing"]:
            print(f"  - {stem}.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-fixtures",
        action="store_true",
        help="List missing raw MATLAB exports and extracted NPZ fixtures without extracting.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    config_path = args.config.resolve()

    if args.check_fixtures:
        status = fixture_status(input_dir, output_dir, config_path)
        _print_fixture_status(status)
        return 1 if status["raw_mat_missing"] or status["npz_missing"] else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    mat_paths = sorted(input_dir.glob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")

    for mat_path in mat_paths:
        flat, inventory = extract_file(mat_path)
        out_path = output_dir / f"{mat_path.stem}.npz"

        payload: dict[str, np.ndarray] = {
            "__meta_schema_version": np.asarray(["martini_extract_v1"], dtype=str),
            "__meta_source_file": np.asarray([_display_path(mat_path)], dtype=str),
            "__meta_generated_at_utc": np.asarray([_now_utc()], dtype=str),
            "__meta_field_paths": np.asarray([field.path for field in inventory], dtype=str),
            "__meta_field_dtypes": np.asarray([field.dtype for field in inventory], dtype=str),
            "__meta_field_shapes": np.asarray(
                [
                    "x".join(str(dim) for dim in field.shape) if field.shape else "scalar"
                    for field in inventory
                ],
                dtype=str,
            ),
        }
        payload.update(flat)
        np.savez_compressed(out_path, **payload)

        entries.append(
            {
                "stem": mat_path.stem,
                "source_file": _display_path(mat_path),
                "npz_file": _display_path(out_path),
                "n_fields": len(inventory),
                "field_paths": [field.path for field in inventory],
            }
        )

    manifest = {
        "generated_at_utc": _now_utc(),
        "input_dir": _display_path(input_dir),
        "output_dir": _display_path(output_dir),
        "files": entries,
        "notes": _expected_case_notes(config_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "Martini extraction:",
        f"files={len(entries)}",
        f"output_dir={_display_path(output_dir)}",
        f"manifest={_display_path(manifest_path)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
