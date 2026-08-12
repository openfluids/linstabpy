#!/usr/bin/env python3
"""Compare two LNSE assembly paths on deterministic safety cases.

The default ``baseline`` and ``candidate`` paths both call the current assembly
implementation. That is deliberate: before P1/P2/P3 land, this script proves the
harness itself and writes the comparison contract that future paths must satisfy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
MARTINI_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(MARTINI_TOOLS))
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "architecture_refactor"
    / "safety_infrastructure"
    / "lnse_equivalence_harness.json"
)

from assembly_parity import (  # noqa: E402
    _case_setup,
    _compare_eigenvalues,
    _compare_sparse_matrices,
)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    model: str
    seed: int
    floquet_exp: float
    bc_recipe: list[dict[str, object]]
    eig_target: complex


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _complex_parts(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def _array_digest(array: np.ndarray) -> str:
    import hashlib

    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _case_specs(selected: list[str]) -> list[CaseSpec]:
    specs = {
        "2D": CaseSpec(
            label="2D",
            model="2D",
            seed=1234,
            floquet_exp=0.15,
            bc_recipe=[
                {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
                {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
                {"type": "neumann", "borders": "b", "variables": "T"},
                {"type": "neumann", "borders": "r", "variables": "ruvwT"},
            ],
            eig_target=-0.05j,
        ),
        "axy": CaseSpec(
            label="axy",
            model="axy",
            seed=2234,
            floquet_exp=0.0,
            bc_recipe=[
                {"type": "dirichlet", "borders": "l", "variables": "ruvwT"},
                {"type": "neumann", "borders": "r", "variables": "T"},
            ],
            eig_target=-0.03j,
        ),
    }
    return [specs[name] for name in selected]


def _raw_operator_from_prepared(prep: dict[str, Any], *, floquet_exp: float) -> sp.csr_matrix:
    from linstabpy.operators.assembly import _apply_floquet_shift, _assemble_block_operator

    ngp = prep["ngp"]
    dz = prep["DZ"]
    d2z = prep["D2Z"]
    if floquet_exp != 0.0:
        dz, d2z = _apply_floquet_shift(dz, d2z, 1j * floquet_exp, ngp)
    raw = _assemble_block_operator(
        ngp,
        prep["coeffs"],
        prep["DR"],
        prep["D2R"],
        dz,
        d2z,
        prep["D2RZ"],
    )
    return sp.csr_matrix(raw, dtype=complex)


def _build_current_path(spec: CaseSpec, *, path_label: str) -> dict[str, Any]:
    from linstabpy.operators import build_lnse_system
    from linstabpy.operators.assembly import (
        _COEFF_TENSOR_NAMES,
        _assemble_from_prepared,
        _prepare_coefficients,
    )

    mesh, baseflow, opts = _case_setup(spec.model, spec.seed)
    prep = _prepare_coefficients(
        mesh,
        baseflow,
        kx=float(opts["kx"]),
        model=str(opts["model"]),
        m=float(opts["m"]),
    )
    raw = _raw_operator_from_prepared(prep, floquet_exp=spec.floquet_exp)
    pre_bc = sp.csr_matrix(_assemble_from_prepared(prep, floquet_exp=spec.floquet_exp))
    post_bc, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=float(opts["kx"]),
        model=str(opts["model"]),
        m=float(opts["m"]),
        floquet_exp=spec.floquet_exp,
        bc_recipe=spec.bc_recipe,
        bc_penalty=1.0e4,
        backend="scipy",
    )

    coeffs = prep["coeffs"]
    coefficient_tensors = {
        name: np.asarray(getattr(coeffs, name), dtype=complex) for name in _COEFF_TENSOR_NAMES
    }
    return {
        "label": path_label,
        "metadata": {
            "path": "current",
            "model": spec.model,
            "seed": int(spec.seed),
            "kx": float(opts["kx"]),
            "m": float(opts["m"]),
            "floquet_exp": float(spec.floquet_exp),
            "ngp": int(mesh.ngp),
            "ndof": int(5 * mesh.ngp),
        },
        "coefficients": coefficient_tensors,
        "raw_operator": raw,
        "pre_bc_operator": pre_bc,
        "post_bc_operator": sp.csr_matrix(post_bc, dtype=complex),
        "bc_indices": np.asarray(bc_indices, dtype=np.int64),
        "idx_keys": sorted(idx),
    }


def _compare_arrays(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    *,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    names = sorted(set(before) | set(after))
    entries = {}
    passed = True
    for name in names:
        if name not in before or name not in after:
            entries[name] = {"pass": False, "reason": "missing tensor"}  # nosec B105
            passed = False
            continue
        left = np.asarray(before[name])
        right = np.asarray(after[name])
        if left.shape != right.shape:
            entries[name] = {
                "pass": False,  # nosec B105
                "reason": "shape mismatch",
                "before_shape": list(left.shape),
                "after_shape": list(right.shape),
            }
            passed = False
            continue
        delta = right - left
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        denom = max(float(np.max(np.abs(left))) if left.size else 0.0, 1.0e-30)
        max_rel = float(max_abs / denom)
        item_pass = bool(max_abs <= abs_tol and max_rel <= rel_tol)
        passed = passed and item_pass
        entries[name] = {
            "pass": item_pass,
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "before_sha256": _array_digest(left),
            "after_sha256": _array_digest(right),
            "max_abs": max_abs,
            "max_rel": max_rel,
            "abs_tol": float(abs_tol),
            "rel_tol": float(rel_tol),
        }
    return {"pass": bool(passed), "tensors": entries}


def _compare_bc_indices(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    left = np.asarray(before, dtype=np.int64)
    right = np.asarray(after, dtype=np.int64)
    match = bool(np.array_equal(left, right))
    mismatch_preview = {
        "before": [] if match else left[:20].astype(int).tolist(),
        "after": [] if match else right[:20].astype(int).tolist(),
    }
    return {
        "pass": match,
        "count_before": int(left.size),
        "count_after": int(right.size),
        "before_sha256": _array_digest(left),
        "after_sha256": _array_digest(right),
        "first_mismatches": mismatch_preview,
    }


def _compare_case(spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any]:
    before = _build_current_path(spec, path_label="baseline")
    after = _build_current_path(spec, path_label="candidate")
    raw = _compare_sparse_matrices(
        before["raw_operator"],
        after["raw_operator"],
        abs_tol=args.sparse_abs_tol,
        rel_tol=args.sparse_rel_tol,
        max_terms=args.max_terms,
    )
    pre_bc = _compare_sparse_matrices(
        before["pre_bc_operator"],
        after["pre_bc_operator"],
        abs_tol=args.sparse_abs_tol,
        rel_tol=args.sparse_rel_tol,
        max_terms=args.max_terms,
    )
    post_bc = _compare_sparse_matrices(
        before["post_bc_operator"],
        after["post_bc_operator"],
        abs_tol=args.sparse_abs_tol,
        rel_tol=args.sparse_rel_tol,
        max_terms=args.max_terms,
    )
    coefficients = _compare_arrays(
        before["coefficients"],
        after["coefficients"],
        abs_tol=args.coeff_abs_tol,
        rel_tol=args.coeff_rel_tol,
    )
    bc_indices = _compare_bc_indices(before["bc_indices"], after["bc_indices"])
    eigs = _compare_eigenvalues(
        before["post_bc_operator"],
        after["post_bc_operator"],
        n_modes=args.eig_modes,
        target=spec.eig_target,
        abs_tol=args.eig_abs_tol,
        rel_tol=args.eig_rel_tol,
        max_ndof=args.max_eig_ndof,
        skip=args.skip_eigs,
    )
    eig_pass = bool(eigs.get("pass", True)) if eigs.get("status") == "ok" else True
    passed = all(
        [
            coefficients["pass"],
            raw["pass"],
            raw["structure_match"],
            pre_bc["pass"],
            pre_bc["structure_match"],
            post_bc["pass"],
            post_bc["structure_match"],
            bc_indices["pass"],
            eig_pass,
        ]
    )
    return {
        "case": spec.label,
        "pass": bool(passed),
        "baseline_metadata": before["metadata"],
        "candidate_metadata": after["metadata"],
        "coefficients": coefficients,
        "raw_operator": raw,
        "pre_bc_operator": pre_bc,
        "post_bc_operator": post_bc,
        "bc_indices": bc_indices,
        "eigen_sanity": eigs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=("2D", "axy"), dest="cases")
    parser.add_argument("--coeff-abs-tol", type=float, default=0.0)
    parser.add_argument("--coeff-rel-tol", type=float, default=0.0)
    parser.add_argument("--sparse-abs-tol", type=float, default=0.0)
    parser.add_argument("--sparse-rel-tol", type=float, default=0.0)
    parser.add_argument("--eig-abs-tol", type=float, default=1e-10)
    parser.add_argument("--eig-rel-tol", type=float, default=1e-10)
    parser.add_argument("--eig-modes", type=int, default=4)
    parser.add_argument("--max-eig-ndof", type=int, default=1500)
    parser.add_argument("--max-terms", type=int, default=20)
    parser.add_argument("--skip-eigs", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)

    dual_path_enabled = _env_flag("LINSTABPY_DUAL_PATH")
    if args.output is not None:
        output = args.output
    elif "LINSTABPY_DUAL_PATH_REPORT" in os.environ:
        output = Path(os.environ["LINSTABPY_DUAL_PATH_REPORT"])
    else:
        output = DEFAULT_OUTPUT
    selected = args.cases or ["2D", "axy"]
    case_reports = [_compare_case(spec, args) for spec in _case_specs(selected)]
    report = {
        "generated_at_utc": _now_utc(),
        "mode": "dual-path-current-vs-current" if dual_path_enabled else "current-vs-current",
        "dual_path_enabled": dual_path_enabled,
        "enabled_by": "LINSTABPY_DUAL_PATH" if dual_path_enabled else None,
        "description": (
            "Baseline and candidate paths both use the current implementation; "
            "future P1/P2/P3 paths must satisfy the same comparison contract."
        ),
        "pass": bool(all(case["pass"] for case in case_reports)),
        "cases": case_reports,
    }
    _write_json(output, report)
    print(
        "LNSE equivalence harness:",
        f"pass={report['pass']}",
        f"cases={','.join(selected)}",
        f"dual_path_enabled={dual_path_enabled}",
        f"output={output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
