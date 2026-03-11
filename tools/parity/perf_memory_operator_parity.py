#!/usr/bin/env python3
"""PETSc/SLEPc-oriented operator perf/memory parity harness."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_bytes() -> int:
    # Linux returns KiB, macOS returns bytes.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _parse_meshes(spec: str) -> list[tuple[int, int]]:
    meshes: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" not in chunk:
            raise ValueError(f"Invalid mesh chunk '{chunk}', expected NxXNy format")
        nx_s, ny_s = chunk.split("x", maxsplit=1)
        nx, ny = int(nx_s), int(ny_s)
        if nx < 3 or ny < 3:
            raise ValueError("Each mesh must have nx>=3 and ny>=3")
        meshes.append((nx, ny))
    if not meshes:
        raise ValueError("No mesh sizes provided")
    return meshes


def _parse_complex(text: str) -> complex:
    return complex(text.replace("i", "j"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _stage(case: dict[str, Any], name: str, fn):
    rss0 = _peak_rss_bytes()
    py_cur0, py_peak0 = tracemalloc.get_traced_memory()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    rss1 = _peak_rss_bytes()
    py_cur1, py_peak1 = tracemalloc.get_traced_memory()
    case["metrics"][name] = {
        "seconds": t1 - t0,
        "rss_bytes_before": rss0,
        "rss_bytes_after": rss1,
        "rss_delta_bytes": max(0, rss1 - rss0),
        "py_current_bytes_before": py_cur0,
        "py_current_bytes_after": py_cur1,
        "py_peak_bytes_before": py_peak0,
        "py_peak_bytes_after": py_peak1,
        "py_peak_delta_bytes": max(0, py_peak1 - py_peak0),
    }
    return result


def _build_uniform_baseflow(mesh, BaseFlow, sutherland_air):
    ny, nx = mesh.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    bf = BaseFlow(
        Re=1e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=ones.copy(),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(bf)
    return bf


def _evaluate_case(case: dict[str, Any], args, modules):
    Mesh = modules["Mesh"]
    BaseFlow = modules["BaseFlow"]
    sutherland_air = modules["sutherland_air"]
    build_lnse_operator = modules["build_lnse_operator"]
    build_spatial_operator = modules["build_spatial_operator"]
    compute_chu_norm = modules["compute_chu_norm"]
    apply_dirichlet = modules["apply_dirichlet"]
    apply_dirichlet_spatial = modules["apply_dirichlet_spatial"]
    temporal_eigenvalues = modules["temporal_eigenvalues"]
    spatial_eigenvalues = modules["spatial_eigenvalues"]
    resolvent_analysis = modules["resolvent_analysis"]
    make_linear_solver = modules["make_linear_solver"]

    nx = case["mesh"]["nx"]
    ny = case["mesh"]["ny"]

    def _assembly():
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=args.fd_order,
        )
        bf = _build_uniform_baseflow(mesh, BaseFlow, sutherland_air)
        L0, idx = build_lnse_operator(mesh, bf, kx=args.kx, model="2D")
        L0, bc_indices = apply_dirichlet(
            L0, idx, args.borders, args.variables, mesh.ngp, penalty=args.penalty
        )

        p = mesh.used_ind
        W_int = mesh.W.ravel(order="F")[p]
        rho_p = bf.RHO.ravel(order="F")[p]
        temp_p = bf.T.ravel(order="F")[p]
        W_diag, invW_diag = compute_chu_norm(
            W_int,
            rho_p,
            temp_p,
            bf.kappa,
            bf.Ma,
            mesh.ngp,
            model="2D",
        )
        return mesh, bf, L0, idx, bc_indices, W_diag, invW_diag

    mesh, bf, L0, idx, bc_indices, W_diag, invW_diag = _stage(case, "assembly", _assembly)

    ndof = int(L0.shape[0])
    case["operator_stats"] = {
        "ndof": ndof,
        "ngp": int(mesh.ngp),
        "nnz": int(L0.nnz),
        "density": float(L0.nnz / (ndof * ndof)),
        "bc_count": int(np.asarray(bc_indices).size),
        "is_sparse": bool(sp.isspmatrix(L0)),
    }

    rng = np.random.default_rng(args.seed + nx * 1000 + ny)

    def _linear_solve():
        L_sigma = L0.copy()
        L_sigma.setdiag(L_sigma.diagonal() - 1j * args.omega)
        solver = make_linear_solver(L_sigma)
        x_true = rng.standard_normal(ndof) + 1j * rng.standard_normal(ndof)
        rhs = L_sigma @ x_true
        sol = solver.solve(rhs)
        rhs_norm = max(np.linalg.norm(rhs), 1e-30)
        resid = np.linalg.norm(L_sigma @ sol - rhs) / rhs_norm

        refine_steps_used = 0
        for _ in range(max(int(args.linear_refine_steps), 0)):
            if resid <= args.max_linear_residual:
                break
            corr_rhs = rhs - (L_sigma @ sol)
            delta = solver.solve(corr_rhs)
            sol = sol + delta
            resid = np.linalg.norm(L_sigma @ sol - rhs) / rhs_norm
            refine_steps_used += 1

        return {
            "relative_residual": float(resid),
            "relative_solution_error": float(
                np.linalg.norm(sol - x_true) / max(np.linalg.norm(x_true), 1e-30)
            ),
            "refine_steps_used": int(refine_steps_used),
        }

    case["results"]["linear_solve"] = _stage(case, "linear_solve", _linear_solve)

    def _temporal():
        eigvals, _ = temporal_eigenvalues(
            L0,
            n_modes=args.n_temporal,
            target=args.temporal_target,
            W_diag=W_diag,
        )
        return {
            "n_converged": int(eigvals.size),
            "target": str(args.temporal_target),
            "max_abs_eig": float(np.max(np.abs(eigvals))) if eigvals.size else 0.0,
            "dense_fallback_path": bool(ndof <= 2 or args.n_temporal >= ndof - 1),
        }

    case["results"]["temporal_eigs"] = _stage(case, "temporal_eigs", _temporal)

    def _spatial():
        Lc, R, Lw, idx_sp = build_spatial_operator(
            mesh,
            bf,
            model="2D",
            kx=args.kx,
            delta=args.delta,
        )
        Lc, R, bc_indices_sp = apply_dirichlet_spatial(
            Lc,
            R,
            idx_sp,
            args.borders,
            args.variables,
            mesh.ngp,
            penalty=args.penalty,
        )
        eigvals, _ = spatial_eigenvalues(
            Lc,
            Lw,
            R,
            omega=args.omega,
            n_modes=args.n_spatial,
            target=args.spatial_target,
        )
        ndof_sp = int(Lc.shape[0])
        return {
            "n_converged": int(eigvals.size),
            "target": str(args.spatial_target),
            "Lc_nnz": int(Lc.nnz),
            "R_nnz": int(R.nnz),
            "bc_count": int(np.asarray(bc_indices_sp).size),
            "dense_fallback_path": bool(ndof_sp <= 2 or args.n_spatial >= ndof_sp - 1),
        }

    case["results"]["spatial_eigs"] = _stage(case, "spatial_eigs", _spatial)

    def _resolvent():
        gains, _, _ = resolvent_analysis(
            L0,
            omega=args.omega,
            n_modes=args.n_resolvent,
            W_diag=W_diag,
        )
        return {
            "n_converged": int(gains.size),
            "max_gain": float(np.max(gains)) if gains.size else 0.0,
        }

    case["results"]["resolvent"] = _stage(case, "resolvent", _resolvent)


def _evaluate_checks(case: dict[str, Any], args) -> None:
    strict_gate_names = {
        "assembly_time_per_mnnz",
        "linear_solve_seconds",
        "case_total_seconds",
        "memory_rss_growth_bytes",
        "rss_growth_per_mnnz",
    }

    checks: dict[str, dict[str, Any]] = {}

    nnz = case["operator_stats"].get("nnz", 0)
    mnnz = max(float(nnz) / 1e6, 1e-12)
    assembly_seconds = case["metrics"]["assembly"]["seconds"]

    checks["sparse_operator"] = {
        "pass": bool(case["operator_stats"].get("is_sparse", False)),
        "value": case["operator_stats"].get("is_sparse", False),
    }

    checks["assembly_time_per_mnnz"] = {
        "pass": True if args.max_assembly_s_per_mnnz is None else (assembly_seconds / mnnz) <= args.max_assembly_s_per_mnnz,
        "value": assembly_seconds / mnnz,
        "threshold": args.max_assembly_s_per_mnnz,
    }

    checks["linear_solve_seconds"] = {
        "pass": True if args.max_linear_solve_seconds is None else case["metrics"]["linear_solve"]["seconds"] <= args.max_linear_solve_seconds,
        "value": case["metrics"]["linear_solve"]["seconds"],
        "threshold": args.max_linear_solve_seconds,
    }

    case_runtime = sum(
        stage["seconds"]
        for stage in (
            case["metrics"].get("assembly", {}),
            case["metrics"].get("linear_solve", {}),
            case["metrics"].get("temporal_eigs", {}),
            case["metrics"].get("spatial_eigs", {}),
            case["metrics"].get("resolvent", {}),
        )
        if isinstance(stage, dict) and "seconds" in stage
    )
    checks["case_total_seconds"] = {
        "pass": True if args.max_case_runtime_seconds is None else case_runtime <= args.max_case_runtime_seconds,
        "value": case_runtime,
        "threshold": args.max_case_runtime_seconds,
    }

    rss_delta = case["metrics"]["resolvent"]["rss_bytes_after"] - case["metrics"]["assembly"]["rss_bytes_before"]
    checks["memory_rss_growth_bytes"] = {
        "pass": True if args.max_rss_growth_bytes is None else rss_delta <= args.max_rss_growth_bytes,
        "value": rss_delta,
        "threshold": args.max_rss_growth_bytes,
    }
    checks["rss_growth_per_mnnz"] = {
        "pass": True if args.max_rss_bytes_per_mnnz is None else (rss_delta / mnnz) <= args.max_rss_bytes_per_mnnz,
        "value": rss_delta / mnnz,
        "threshold": args.max_rss_bytes_per_mnnz,
    }

    lin_resid = case["results"]["linear_solve"]["relative_residual"]
    checks["linear_solve_residual"] = {
        "pass": lin_resid <= args.max_linear_residual,
        "value": lin_resid,
        "threshold": args.max_linear_residual,
    }

    dense_path = bool(case["results"]["temporal_eigs"]["dense_fallback_path"]) or bool(
        case["results"]["spatial_eigs"]["dense_fallback_path"]
    )
    checks["no_dense_fallback"] = {"pass": not dense_path, "value": dense_path}

    case["strict_gates"] = {
        name: checks[name]
        for name in strict_gate_names
        if name in checks
    }
    case["checks"] = checks
    case["status"] = "pass" if all(item["pass"] for item in checks.values()) else "fail"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="24x16,32x24", help="Comma-separated mesh list (e.g., 24x16,32x24)")
    parser.add_argument("--fd-order", type=int, default=4)
    parser.add_argument("--kx", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--n-temporal", type=int, default=4)
    parser.add_argument("--n-spatial", type=int, default=3)
    parser.add_argument("--n-resolvent", type=int, default=2)
    parser.add_argument("--temporal-target", type=_parse_complex, default=0.1 + 0.0j)
    parser.add_argument("--spatial-target", type=_parse_complex, default=0.1 + 0.0j)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--borders", default="bt")
    parser.add_argument("--variables", default="uvwT")
    parser.add_argument("--penalty", type=float, default=1e4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-linear-residual", type=float, default=1e-8)
    parser.add_argument("--linear-refine-steps", type=int, default=8)
    parser.add_argument("--max-assembly-s-per-mnnz", type=float, default=None)
    parser.add_argument("--max-linear-solve-seconds", type=float, default=None)
    parser.add_argument("--max-case-runtime-seconds", type=float, default=None)
    parser.add_argument("--max-rss-bytes-per-mnnz", type=float, default=None)
    parser.add_argument("--max-rss-growth-bytes", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("tools/parity/output/operator_perf_memory.json"))
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if one or more checks fail",
    )
    args = parser.parse_args()

    meshes = _parse_meshes(args.meshes)

    try:
        from petsc4py import PETSc  # type: ignore
        from slepc4py import SLEPc  # type: ignore

        from linstab2d.baseflow import BaseFlow, sutherland_air
        from linstab2d.boundary_conditions import apply_dirichlet, apply_dirichlet_spatial
        from linstab2d.mesh import Mesh
        from linstab2d.operators import build_lnse_operator, build_spatial_operator, compute_chu_norm
        from linstab2d.solvers import resolvent_analysis, spatial_eigenvalues, temporal_eigenvalues
        from linstab2d.solvers.linsolve import make_linear_solver
    except Exception as exc:  # pragma: no cover - environment dependent
        report = {
            "generated_at_utc": _now_utc(),
            "status": "blocked",
            "error": str(exc),
            "message": "PETSc/SLEPc or linstab2d imports unavailable",
        }
        _write_json(args.output, report)
        print(report["message"], report["error"])
        return 1

    modules = {
        "Mesh": Mesh,
        "BaseFlow": BaseFlow,
        "sutherland_air": sutherland_air,
        "build_lnse_operator": build_lnse_operator,
        "build_spatial_operator": build_spatial_operator,
        "compute_chu_norm": compute_chu_norm,
        "apply_dirichlet": apply_dirichlet,
        "apply_dirichlet_spatial": apply_dirichlet_spatial,
        "temporal_eigenvalues": temporal_eigenvalues,
        "spatial_eigenvalues": spatial_eigenvalues,
        "resolvent_analysis": resolvent_analysis,
        "make_linear_solver": make_linear_solver,
    }

    tracemalloc.start()
    cases: list[dict[str, Any]] = []
    for nx, ny in meshes:
        case: dict[str, Any] = {
            "mesh": {"nx": nx, "ny": ny},
            "metrics": {},
            "results": {},
            "checks": {},
            "status": "fail",
            "error": None,
        }
        try:
            _evaluate_case(case, args, modules)
            _evaluate_checks(case, args)
        except Exception as exc:  # pragma: no cover - runtime path
            case["status"] = "fail"
            case["error"] = str(exc)
        cases.append(case)

    report = {
        "generated_at_utc": _now_utc(),
        "backend": {
            "petsc_version": list(PETSc.Sys.getVersion()),
            "slepc_version": list(SLEPc.Sys.getVersion()),
            "petsc_scalar_type": str(PETSc.ScalarType.__name__),
        },
        "config": {
            "meshes": [{"nx": nx, "ny": ny} for nx, ny in meshes],
            "fd_order": args.fd_order,
            "kx": args.kx,
            "omega": args.omega,
            "n_temporal": args.n_temporal,
            "n_spatial": args.n_spatial,
            "n_resolvent": args.n_resolvent,
            "temporal_target": str(args.temporal_target),
            "spatial_target": str(args.spatial_target),
            "delta": args.delta,
            "borders": args.borders,
            "variables": args.variables,
            "penalty": args.penalty,
            "seed": args.seed,
            "max_linear_residual": args.max_linear_residual,
            "max_linear_solve_seconds": args.max_linear_solve_seconds,
            "max_case_runtime_seconds": args.max_case_runtime_seconds,
            "max_assembly_s_per_mnnz": args.max_assembly_s_per_mnnz,
            "max_rss_growth_bytes": args.max_rss_growth_bytes,
            "max_rss_bytes_per_mnnz": args.max_rss_bytes_per_mnnz,
        },
        "cases": cases,
    }
    report["strict_gates"] = {
        "enabled": any(
            value is not None
            for value in (
                args.max_linear_solve_seconds,
                args.max_case_runtime_seconds,
                args.max_assembly_s_per_mnnz,
                args.max_rss_growth_bytes,
                args.max_rss_bytes_per_mnnz,
            )
        ),
        "assembly_time_per_mnnz": {
            "threshold": args.max_assembly_s_per_mnnz,
            "pass_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("assembly_time_per_mnnz", {}).get("threshold") is not None
                and case.get("strict_gates", {}).get("assembly_time_per_mnnz", {}).get("pass", False)
            ),
            "fail_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("assembly_time_per_mnnz", {}).get("threshold") is not None
                and not case.get("strict_gates", {}).get("assembly_time_per_mnnz", {}).get("pass", False)
            ),
        },
        "linear_solve_seconds": {
            "threshold": args.max_linear_solve_seconds,
            "pass_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("linear_solve_seconds", {}).get("threshold") is not None
                and case.get("strict_gates", {}).get("linear_solve_seconds", {}).get("pass", False)
            ),
            "fail_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("linear_solve_seconds", {}).get("threshold") is not None
                and not case.get("strict_gates", {}).get("linear_solve_seconds", {}).get("pass", False)
            ),
        },
        "case_total_seconds": {
            "threshold": args.max_case_runtime_seconds,
            "pass_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("case_total_seconds", {}).get("threshold") is not None
                and case.get("strict_gates", {}).get("case_total_seconds", {}).get("pass", False)
            ),
            "fail_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("case_total_seconds", {}).get("threshold") is not None
                and not case.get("strict_gates", {}).get("case_total_seconds", {}).get("pass", False)
            ),
        },
        "memory_rss_growth_bytes": {
            "threshold": args.max_rss_growth_bytes,
            "pass_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("memory_rss_growth_bytes", {}).get("threshold") is not None
                and case.get("strict_gates", {}).get("memory_rss_growth_bytes", {}).get("pass", False)
            ),
            "fail_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("memory_rss_growth_bytes", {}).get("threshold") is not None
                and not case.get("strict_gates", {}).get("memory_rss_growth_bytes", {}).get("pass", False)
            ),
        },
        "rss_growth_per_mnnz": {
            "threshold": args.max_rss_bytes_per_mnnz,
            "pass_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("rss_growth_per_mnnz", {}).get("threshold") is not None
                and case.get("strict_gates", {}).get("rss_growth_per_mnnz", {}).get("pass", False)
            ),
            "fail_count": sum(
                1
                for case in cases
                if case.get("strict_gates", {}).get("rss_growth_per_mnnz", {}).get("threshold") is not None
                and not case.get("strict_gates", {}).get("rss_growth_per_mnnz", {}).get("pass", False)
            ),
        },
        "overall_strict_pass": all(
            (args.max_assembly_s_per_mnnz is None or case.get("strict_gates", {}).get("assembly_time_per_mnnz", {}).get("pass", False))
            and (args.max_linear_solve_seconds is None or case.get("strict_gates", {}).get("linear_solve_seconds", {}).get("pass", False))
            and (args.max_case_runtime_seconds is None or case.get("strict_gates", {}).get("case_total_seconds", {}).get("pass", False))
            and (args.max_rss_growth_bytes is None or case.get("strict_gates", {}).get("memory_rss_growth_bytes", {}).get("pass", False))
            and (args.max_rss_bytes_per_mnnz is None or case.get("strict_gates", {}).get("rss_growth_per_mnnz", {}).get("pass", False))
            for case in cases
        ),
    }
    overall_pass = all(case["status"] == "pass" for case in cases)
    report["status"] = "pass" if overall_pass else "fail"
    report["pass_count"] = sum(1 for case in cases if case["status"] == "pass")
    report["fail_count"] = sum(1 for case in cases if case["status"] != "pass")

    _write_json(args.output, report)
    print(
        "Perf/memory parity:",
        f"status={report['status']}",
        f"pass={report['pass_count']}",
        f"fail={report['fail_count']}",
        f"output={args.output}",
    )

    if report["status"] != "pass" and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
