#!/usr/bin/env python3
"""Run reduced-grid Krylov EVP tuning trials from JSONC."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from linstabpy.utils.jsonc import load_jsonc

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
CONFIG_PATH = SCRIPT.with_name("krylov_evp_tuning.jsonc")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SRC_PATH = REPO / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


def _patch_outputs(module: Any, run_dir: Path) -> tuple[dict[str, Path], list[str]]:
    originals: dict[str, Path] = {}
    patched: list[str] = []
    for name, value in vars(module).items():
        if not isinstance(value, Path):
            continue
        if not name.startswith("OUTPUT_"):
            continue
        originals[name] = value
        setattr(module, name, run_dir / value.name)
        patched.append(name)
    return originals, patched


def _restore_outputs(module: Any, originals: dict[str, Path]) -> None:
    for name, value in originals.items():
        setattr(module, name, value)


def _summarize(example_name: str, npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        files = set(data.files)
        summary: dict[str, Any] = {}
        if "temporal_residuals" in files:
            summary["temporal_residual_max"] = float(
                np.max(np.asarray(data["temporal_residuals"], dtype=float))
            )
        if "spatial_residuals" in files:
            summary["spatial_residual_max"] = float(
                np.max(np.asarray(data["spatial_residuals"], dtype=float))
            )
        if "direct_residuals" in files:
            summary["direct_residual_max"] = float(
                np.max(np.asarray(data["direct_residuals"], dtype=float))
            )
        if "adjoint_residuals" in files:
            summary["adjoint_residual_max"] = float(
                np.max(np.asarray(data["adjoint_residuals"], dtype=float))
            )
        return summary


def _run_single_trial(cfg: dict[str, Any], example_name: str, trial_name: str) -> int:
    example = next(item for item in cfg["examples"] if item["name"] == example_name)
    trial = next(item for item in cfg["trials"] if item["name"] == trial_name)

    output_root = REPO / cfg["output_root"]
    run_dir = output_root / example["name"] / trial["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"

    module = importlib.import_module(example["module"])
    compute = getattr(module, example["compute_fn"])
    base_cfg = load_jsonc(REPO / example["config_path"])
    base_cfg = _deep_merge(base_cfg, example["reduced_overrides"])
    cfg_run = _deep_merge(base_cfg, trial)

    originals, _ = _patch_outputs(module, run_dir)
    output_npz = run_dir / originals["OUTPUT_NPZ"].name

    print(f"[trial] {example['name']} :: {trial['name']}")
    result: dict[str, Any] = {
        "example": example["name"],
        "trial": trial["name"],
        "output_npz": str(output_npz.relative_to(REPO)),
        "status": "failed",
    }

    t0 = time.perf_counter()
    try:
        compute(cfg_run)
        result["runtime_s"] = float(time.perf_counter() - t0)
        result["summary"] = _summarize(example["name"], output_npz)
        result["status"] = "ok"
    except BaseException as exc:
        result["runtime_s"] = float(time.perf_counter() - t0)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"[fail] {example['name']} :: {trial['name']}: {type(exc).__name__}: {exc}")
    finally:
        _restore_outputs(module, originals)

    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, indent=2, sort_keys=True)
        fh.write("\n")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-example",
        default="",
        help="Internal: run one example trial in an isolated subprocess.",
    )
    parser.add_argument(
        "--worker-trial",
        default="",
        help="Internal: run one example trial in an isolated subprocess.",
    )
    parser.add_argument(
        "--examples", default="", help="Comma-separated example names to run from the tuning JSONC."
    )
    parser.add_argument(
        "--trials", default="", help="Comma-separated trial names to run from the tuning JSONC."
    )
    args = parser.parse_args()

    cfg = load_jsonc(CONFIG_PATH)
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    if args.worker_example and args.worker_trial:
        raise SystemExit(_run_single_trial(cfg, args.worker_example, args.worker_trial))

    selected_examples = {item.strip() for item in args.examples.split(",") if item.strip()}
    selected_trials = {item.strip() for item in args.trials.split(",") if item.strip()}

    results: list[dict[str, Any]] = []
    for example in cfg["examples"]:
        if selected_examples and example["name"] not in selected_examples:
            continue
        for trial in cfg["trials"]:
            if selected_trials and trial["name"] not in selected_trials:
                continue
            run_dir = output_root / example["name"] / trial["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "result.json"
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--worker-example",
                example["name"],
                "--worker-trial",
                trial["name"],
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                capture_output=True,
                text=True,
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {
                    "example": example["name"],
                    "trial": trial["name"],
                    "status": "failed",
                    "error_type": "SubprocessError",
                    "error": f"Worker exited with code {proc.returncode} without result.json",
                }
            result["worker_returncode"] = int(proc.returncode)
            result["stdout_log"] = str(stdout_path.relative_to(REPO))
            result["stderr_log"] = str(stderr_path.relative_to(REPO))
            results.append(result)

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, sort_keys=True)
        fh.write("\n")

    rows = []
    for item in results:
        row = {
            "example": item["example"],
            "trial": item["trial"],
            "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "output_npz": item["output_npz"],
            "error_type": item.get("error_type"),
            "error": item.get("error"),
        }
        for key, value in item.get("summary", {}).items():
            row[key] = _json_safe(value)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(item["status"] == "ok" for item in results)
    n_fail = len(results) - n_ok
    print()
    print(f"Krylov tuning summary: ok={n_ok}, failed={n_fail}")
    print(f"JSON summary: {summary_json.relative_to(REPO)}")
    print(f"CSV summary:  {summary_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
