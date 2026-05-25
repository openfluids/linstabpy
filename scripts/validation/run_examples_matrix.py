#!/usr/bin/env python3
"""Run example backend/solver variants without overwriting canonical outputs."""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
import traceback
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from linstabpy.utils.jsonc import load_jsonc


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
CONFIG_PATH = SCRIPT.with_name("examples_matrix.jsonc")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SRC_PATH = REPO / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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


@contextmanager
def _patched_output_paths(module: Any, run_dir: Path):
    originals: dict[str, Any] = {}
    try:
        for name, value in list(vars(module).items()):
            if not isinstance(value, Path):
                continue
            if not (name.startswith("OUTPUT_") or name == "ARTIFACT_JSON"):
                continue
            originals[name] = value
            setattr(module, name, run_dir / value.name)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _nearest_reference_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float] | None:
    cand = np.asarray(candidate, dtype=complex).reshape(-1)
    ref = np.asarray(reference, dtype=complex).reshape(-1)
    if cand.size == 0 or ref.size == 0:
        return None
    dists = np.array([np.min(np.abs(cand - item)) for item in ref], dtype=float)
    return {
        "nearest_ref_max_abs": float(np.max(dists)),
        "nearest_ref_median_abs": float(np.median(dists)),
        "nearest_ref_min_abs": float(np.min(dists)),
    }


def _load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _summarize_curvilinear(data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "df_linf": float(np.asarray(data["df_linf"]).reshape(-1)[0]),
        "df_rel_l2": float(np.asarray(data["df_rel_l2"]).reshape(-1)[0]),
        "overall_pass": bool(np.asarray(data["overall_pass"]).reshape(-1)[0]),
    }


def _summarize_resolvent(module: Any, data: dict[str, np.ndarray]) -> dict[str, Any]:
    gains = np.asarray(data["gains"], dtype=float).reshape(-1)
    summary: dict[str, Any] = {
        "gains": gains.tolist(),
        "gain_monotone": bool(np.all(np.diff(gains) <= 1e-12)),
    }
    if "svd_residuals_fwd" in data:
        summary["svd_residual_fwd_max"] = float(np.max(np.asarray(data["svd_residuals_fwd"], dtype=float)))
    if "svd_residuals_adj" in data:
        summary["svd_residual_adj_max"] = float(np.max(np.asarray(data["svd_residuals_adj"], dtype=float)))

    ref_loader = getattr(module, "_load_matlab_gains", None)
    if callable(ref_loader):
        ref = ref_loader()
        if ref is not None:
            ref = np.asarray(ref, dtype=float).reshape(-1)
            n = min(gains.size, ref.size)
            if n > 0:
                rel = np.abs(gains[:n] - ref[:n]) / np.maximum(np.abs(ref[:n]), 1e-30)
                summary["matlab_gain_ref_n"] = int(n)
                summary["matlab_gain_rel_max"] = float(np.max(rel))
                summary["matlab_gain_rel_median"] = float(np.median(rel))
    return summary


def _summarize_temporal_spatial(module: Any, data: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "temporal_omega" in data:
        temporal = np.asarray(data["temporal_omega"], dtype=complex).reshape(-1)
        summary["temporal_count"] = int(temporal.size)
        summary["temporal_head"] = [complex(v) for v in temporal[:5]]
        if "temporal_residuals" in data:
            summary["temporal_residual_max"] = float(np.max(np.asarray(data["temporal_residuals"], dtype=float)))
        loader = getattr(module, "_load_matlab_temporal_omega", None)
        if callable(loader):
            ref = loader()
            if ref is not None:
                stats = _nearest_reference_stats(temporal, ref)
                if stats is not None:
                    summary["temporal_reference_stats"] = stats

    if "spatial_alpha" in data:
        spatial = np.asarray(data["spatial_alpha"], dtype=complex).reshape(-1)
        summary["spatial_count"] = int(spatial.size)
        summary["spatial_head"] = [complex(v) for v in spatial[:5]]
        if "spatial_residuals" in data:
            summary["spatial_residual_max"] = float(np.max(np.asarray(data["spatial_residuals"], dtype=float)))
        loader = getattr(module, "_load_matlab_spatial_alpha", None)
        if callable(loader):
            ref = loader()
            if ref is not None:
                stats = _nearest_reference_stats(spatial, ref)
                if stats is not None:
                    summary["spatial_reference_stats"] = stats
    return summary


def _summarize_cavity(module: Any, data: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "direct_count": int(np.asarray(data["omega"], dtype=complex).size),
        "adjoint_count": int(np.asarray(data["omega_adj"], dtype=complex).size),
        "direct_residual_max": float(np.max(np.asarray(data["direct_residuals"], dtype=float))),
        "adjoint_residual_max": float(np.max(np.asarray(data["adjoint_residuals"], dtype=float))),
    }
    loader = getattr(module, "_load_matlab_spectra", None)
    if callable(loader):
        direct_ref, adjoint_ref = loader()
        direct = np.asarray(data["omega"], dtype=complex).reshape(-1)
        adjoint = np.asarray(data["omega_adj"], dtype=complex).reshape(-1)
        if direct_ref is not None:
            stats = _nearest_reference_stats(direct, direct_ref)
            if stats is not None:
                summary["direct_reference_stats"] = stats
        if adjoint_ref is not None:
            stats = _nearest_reference_stats(adjoint, adjoint_ref)
            if stats is not None:
                summary["adjoint_reference_stats"] = stats
    return summary


def _summarize(kind: str, module: Any, npz_path: Path) -> dict[str, Any]:
    data = _load_npz(npz_path)
    if kind == "curvilinear":
        return _summarize_curvilinear(data)
    if kind == "resolvent":
        return _summarize_resolvent(module, data)
    if kind == "temporal_spatial":
        return _summarize_temporal_spatial(module, data)
    if kind == "cavity_evp":
        return _summarize_cavity(module, data)
    raise ValueError(f"Unsupported example kind: {kind}")


def _run_one(example: dict[str, Any], variant_name: str, variant_overrides: dict[str, Any], output_root: Path) -> dict[str, Any]:
    module = importlib.import_module(example["module"])
    compute = getattr(module, example["compute_fn"])
    base_cfg = load_jsonc(REPO / example["config_path"])
    cfg = _deep_merge(base_cfg, variant_overrides)

    run_dir = output_root / example["name"] / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    original_output_npz = getattr(module, "OUTPUT_NPZ", None)
    if not isinstance(original_output_npz, Path):
        raise AttributeError(f"{example['module']} does not expose OUTPUT_NPZ as a pathlib.Path")
    npz_path = run_dir / original_output_npz.name

    print(f"[run] {example['name']} :: {variant_name}")
    print(f"      output -> {npz_path.relative_to(REPO)}")

    result: dict[str, Any] = {
        "example": example["name"],
        "variant": variant_name,
        "module": example["module"],
        "config_path": example["config_path"],
        "output_npz": npz_path.relative_to(REPO),
        "status": "failed",
    }

    t0 = time.perf_counter()
    with _patched_output_paths(module, run_dir):
        try:
            compute(cfg)
            result["runtime_s"] = float(time.perf_counter() - t0)
            result["summary"] = _summarize(example["kind"], module, npz_path)
            result["status"] = "ok"
        except Exception as exc:
            result["runtime_s"] = float(time.perf_counter() - t0)
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            print(f"[fail] {example['name']} :: {variant_name}: {type(exc).__name__}: {exc}")
    return result


def _variant_in_tier(variant_name: str, tier: str) -> bool:
    if tier == "all":
        return True
    if tier == "core":
        return "krylov" not in variant_name
    if tier == "stretch":
        return "krylov" in variant_name
    raise ValueError(f"Unsupported tier: {tier}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("core", "stretch", "all"),
        default="all",
        help="Run only the Core reference matrix, only the Stretch Krylov matrix, or all variants.",
    )
    parser.add_argument(
        "--examples",
        default="",
        help="Comma-separated example names to run from the JSONC matrix (default: all).",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names to run from the JSONC matrix (default: all in selected tier).",
    )
    args = parser.parse_args()

    cfg = load_jsonc(CONFIG_PATH)
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    variants = cfg["variants"]
    selected_examples = {
        item.strip() for item in args.examples.split(",") if item.strip()
    }
    selected_variants = {
        item.strip() for item in args.variants.split(",") if item.strip()
    }
    results: list[dict[str, Any]] = []
    for example in cfg["examples"]:
        if selected_examples and example["name"] not in selected_examples:
            continue
        for variant_name in example["variants"]:
            if not _variant_in_tier(variant_name, args.tier):
                continue
            if selected_variants and variant_name not in selected_variants:
                continue
            variant_overrides = variants[variant_name]
            results.append(_run_one(example, variant_name, variant_overrides, output_root))

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, sort_keys=True)
        fh.write("\n")

    flat_rows: list[dict[str, Any]] = []
    for item in results:
        row = {
            "example": item["example"],
            "variant": item["variant"],
            "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "output_npz": item["output_npz"],
            "error_type": item.get("error_type"),
            "error": item.get("error"),
        }
        summary = item.get("summary", {})
        for key, value in summary.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    row[f"{key}.{sub_key}"] = _json_safe(sub_val)
            else:
                row[key] = _json_safe(value)
        flat_rows.append(row)

    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    n_ok = sum(item["status"] == "ok" for item in results)
    n_fail = len(results) - n_ok
    print()
    print(f"Matrix summary ({args.tier}): ok={n_ok}, failed={n_fail}")
    print(f"JSON summary: {summary_json.relative_to(REPO)}")
    print(f"CSV summary:  {summary_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
