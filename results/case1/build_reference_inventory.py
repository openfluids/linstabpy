#!/usr/bin/env python3
"""Build a hash + shape inventory for Case 1 MATLAB parity exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstab2d.utils.jsonc import load_jsonc


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize_dataset(ds: h5py.Dataset) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in ds.shape],
        "dtype": str(ds.dtype),
        "is_complex": bool(np.issubdtype(ds.dtype, np.complexfloating)),
    }


def _summarize_hdf5(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "mat_v7_3_hdf5", "keys": {}}
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            node = handle[key]
            if isinstance(node, h5py.Dataset):
                out["keys"][key] = {
                    "type": "dataset",
                    **_summarize_dataset(node),
                }
                continue

            if isinstance(node, h5py.Group):
                child_keys = sorted(node.keys())
                # MATLAB sparse export convention (CSC triplets in group).
                if {"data", "ir", "jc"}.issubset(set(child_keys)):
                    data = np.asarray(node["data"])
                    jc = np.asarray(node["jc"])
                    n = int(jc.size - 1)
                    out["keys"][key] = {
                        "type": "sparse_csc_triplet",
                        "shape": [n, n],
                        "nnz": int(data.size),
                        "children": {
                            "data": _summarize_dataset(node["data"]),
                            "ir": _summarize_dataset(node["ir"]),
                            "jc": _summarize_dataset(node["jc"]),
                        },
                    }
                else:
                    grp_info: dict[str, Any] = {"type": "group", "children": {}}
                    for ck in child_keys:
                        child = node[ck]
                        if isinstance(child, h5py.Dataset):
                            grp_info["children"][ck] = {
                                "type": "dataset",
                                **_summarize_dataset(child),
                            }
                        else:
                            grp_info["children"][ck] = {
                                "type": "group",
                                "keys": sorted(child.keys()),
                            }
                    out["keys"][key] = grp_info
                continue

            out["keys"][key] = {"type": type(node).__name__}
    return out


def _summarize_mat_v5(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "mat_v5", "keys": {}}
    mat = loadmat(path)
    for key in sorted(k for k in mat.keys() if not k.startswith("__")):
        val = mat[key]
        if sp.issparse(val):
            arr = val.tocsc()
            out["keys"][key] = {
                "type": "sparse",
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "nnz": int(arr.nnz),
                "dtype": str(arr.dtype),
                "is_complex": bool(np.issubdtype(arr.dtype, np.complexfloating)),
            }
            continue
        arr = np.asarray(val)
        out["keys"][key] = {
            "type": "ndarray",
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "is_complex": bool(np.iscomplexobj(arr)),
        }
    return out


def _summarize_mat(path: Path) -> dict[str, Any]:
    try:
        return _summarize_hdf5(path)
    except OSError:
        return _summarize_mat_v5(path)


def _cfg_get(cfg: dict[str, Any], path: str, default):
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    if args.config.exists():
        config = load_jsonc(args.config)

    if args.dirs:
        roots = [Path(d) for d in args.dirs]
    else:
        roots = [
            Path(_cfg_get(config, "paths.matlab_exports_stability_dir", "case1/MATLAB_exports")),
            Path(_cfg_get(config, "paths.matlab_exports_resolvent_dir", "case1/MATLAB_exports_resolvent")),
        ]

    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.glob("*.mat")))

    inventory_files = []
    for path in sorted(files):
        summary = _summarize_mat(path)
        inventory_files.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
                "summary": summary,
            }
        )

    return {
        "generated_at_utc": _now_utc(),
        "config_path": str(args.config),
        "source_dirs": [str(p) for p in roots],
        "n_files": len(inventory_files),
        "files": inventory_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("case1/parity_config.jsonc"),
        help="Optional canonical parity config JSONC.",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        help="Override source directories containing MATLAB .mat files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("case1/parity_reference_inventory.json"),
        help="Inventory artifact path.",
    )
    args = parser.parse_args()

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "Case 1 reference inventory:",
        f"files={report['n_files']}",
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
