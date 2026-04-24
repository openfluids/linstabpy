#!/usr/bin/env python3
"""Run direct shifted-KSP experiments for Song2023 from JSONC."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from linstab2d.utils.jsonc import load_jsonc, parse_complex


SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parents[1]
REPO = SCRIPT.parents[3]
CONFIG_PATH = SONG_DIR / "configs" / "shifted_ksp_matrix.jsonc"

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


def _worker_run(cfg: dict[str, Any], backend: str, trial_name: str) -> int:
    from petsc4py import PETSc

    from validation.song_2023.baseflow_loader import load_my_case
    from linstab2d.mesh import Mesh
    from linstab2d.operators.assembly import build_lnse_system
    from linstab2d.solvers.petsc_utils import scipy_to_petsc

    output_root = REPO / cfg["output_root"]
    run_dir = output_root / backend / trial_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"

    base_cfg = load_jsonc(REPO / cfg["base_config"])
    base_cfg = _deep_merge(base_cfg, cfg["reduced_overrides"])
    omega_target = parse_complex(cfg["omega_target"])
    case_dir = SONG_DIR

    t0 = time.perf_counter()
    result: dict[str, Any] = {
        "backend": backend,
        "trial": trial_name,
        "omega_target": omega_target,
        "status": "failed",
    }

    try:
        X, Y, baseflow, sponge, _ = load_my_case(
            case_dir,
            Re=base_cfg["Re"],
            Ma=base_cfg["Ma"],
            Pr=base_cfg["Pr"],
            kappa=base_cfg["kappa"],
            T_0=base_cfg["T_0"],
            stride_ksi=base_cfg["stride_ksi"],
            stride_eta=base_cfg["stride_eta"],
            preprocess_cfg=base_cfg.get("baseflow_preprocess"),
        )
        ny, nx = X.shape
        mesh = Mesh.create(
            (0.0, 2.0),
            (0.0, 1.0),
            nx=nx,
            ny=ny,
            fd_order=base_cfg["fd_order"],
            alpha_filter=base_cfg.get("alpha_filter", "none"),
        )
        mesh = mesh.deform(X_new=X, Y_new=Y)
        mesh.sponge = sponge

        solver_cfg = {"petsc_matrix_mode": "local"}
        L0, _, _ = build_lnse_system(
            mesh,
            baseflow,
            kx=base_cfg["beta"],
            model="2D",
            bc_recipe=base_cfg["bc"]["recipe"],
            bc_penalty=base_cfg["bc"]["penalty"],
            backend=backend,
            solver_cfg=solver_cfg,
        )
        if not hasattr(L0, "getInfo"):
            temporal_op = (1j * L0).tocsc()
            A = scipy_to_petsc(temporal_op, solver_cfg)
        else:
            A = L0.copy()
            A.scale(1j)
        A.shift(-omega_target)
        A.assemble()

        kcfg = next(item["ksp"] for item in cfg["trials"] if item["name"] == trial_name)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType(kcfg["type"])
        side = kcfg.get("pc_side")
        if side == "right":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        elif side == "left":
            ksp.setPCSide(PETSc.PC.Side.LEFT)
        pc = ksp.getPC()
        pc.setType(kcfg["pc_type"])
        if kcfg["pc_type"] == "lu":
            pc.setFactorSolverType(kcfg.get("pc_factor_solver_type", "mumps"))
        if kcfg["type"] != "preonly":
            ksp.setTolerances(rtol=float(kcfg.get("rtol", 1e-8)), max_it=int(kcfg.get("max_it", 20000)))
            ksp.setErrorIfNotConverged(True)
        ksp.setUp()

        b = A.createVecLeft()
        x = b.duplicate()
        rng = np.random.default_rng(123)
        rhs = rng.standard_normal(A.getSize()[0]) + 1j * rng.standard_normal(A.getSize()[0])
        rows = np.arange(*b.getOwnershipRange(), dtype=np.int32)
        if rows.size:
            b.setValues(rows, rhs[rows])
        b.assemblyBegin()
        b.assemblyEnd()
        ksp.solve(b, x)

        r = b.duplicate()
        Ax = b.duplicate()
        A.mult(x, Ax)
        r.waxpy(-1.0, Ax, b)
        true_residual = float(r.norm() / max(b.norm(), 1e-30))
        r.destroy()
        Ax.destroy()

        result["status"] = "ok"
        result["runtime_s"] = float(time.perf_counter() - t0)
        result["ksp_reason"] = int(ksp.getConvergedReason())
        result["ksp_its"] = int(ksp.getIterationNumber())
        result["true_residual"] = true_residual
        result["matrix_nnz"] = int(A.getInfo()["nz_used"])
        result["zero_diag_count"] = int(np.count_nonzero(np.abs(np.array(A.getDiagonal().array_r, copy=True)) == 0))

        b.destroy()
        x.destroy()
        ksp.destroy()
        A.destroy()
    except BaseException as exc:
        result["runtime_s"] = float(time.perf_counter() - t0)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result)))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="", help="Optional single assembly backend to run.")
    parser.add_argument("--trials", default="", help="Comma-separated trial names to run.")
    parser.add_argument("--worker-backend", default="", help="Internal subprocess backend selector.")
    parser.add_argument("--worker-trial", default="", help="Internal subprocess trial selector.")
    args = parser.parse_args()

    cfg = load_jsonc(CONFIG_PATH)
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    if args.worker_backend and args.worker_trial:
        raise SystemExit(_worker_run(cfg, args.worker_backend, args.worker_trial))

    selected_backend = args.backend.strip()
    selected_trials = {item.strip() for item in args.trials.split(",") if item.strip()}

    results: list[dict[str, Any]] = []
    for backend in cfg["assembly_backends"]:
        if selected_backend and backend != selected_backend:
            continue
        for trial in cfg["trials"]:
            if selected_trials and trial["name"] not in selected_trials:
                continue
            run_dir = output_root / backend / trial["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "result.json"
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--worker-backend",
                backend,
                "--worker-trial",
                trial["name"],
            ]
            proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {
                    "backend": backend,
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
    summary_json.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for item in results:
        row = {
            "backend": item["backend"],
            "trial": item["trial"],
            "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "ksp_reason": item.get("ksp_reason"),
            "ksp_its": item.get("ksp_its"),
            "true_residual": item.get("true_residual"),
            "matrix_nnz": item.get("matrix_nnz"),
            "zero_diag_count": item.get("zero_diag_count"),
            "error_type": item.get("error_type"),
            "error": item.get("error"),
            "stdout_log": item["stdout_log"],
            "stderr_log": item["stderr_log"],
        }
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(item["status"] == "ok" for item in results)
    n_fail = len(results) - n_ok
    print(f"Song shifted-KSP summary: ok={n_ok}, failed={n_fail}")
    print(f"JSON summary: {summary_json.relative_to(REPO)}")
    print(f"CSV summary:  {summary_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
