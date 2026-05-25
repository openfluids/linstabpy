#!/usr/bin/env python3
"""Config-driven timing and memory harness for Case 2 runs.

Launches the canonical ``case2.compute()`` pipeline in a child
process, monitors wall time and process-tree RSS, parses key operator metadata
from the emitted log stream, and writes JSON/CSV summary artifacts for later
efficiency studies.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstabpy.utils.jsonc import load_jsonc

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent
DEFAULT_CONFIG = (
    CASE_DIR / "configs" / "evp" / "evp_stride2x1_oscillatory_target_ncv500_tol1e4.jsonc"
)
DEFAULT_JSON = CASE_DIR / "output" / "case2_perf_memory_latest.json"
DEFAULT_CSV = CASE_DIR / "output" / "case2_perf_memory_table.csv"
DEFAULT_LOG_DIR = CASE_DIR / "output" / "logs"
DEFAULT_RUN_DIR = CASE_DIR / "output" / "runs"
DEFAULT_OUTPUT_NPZ = CASE_DIR / "case2.npz"

_RE_CASE_SHAPE = re.compile(r"shape=\((\d+),\s*(\d+)\), points=(\d+)")
_RE_OPERATOR = re.compile(r"operator assembled(?: with BCs)?: shape=\((\d+),\s*(\d+)\), nnz=(\d+)")
_RE_BC = re.compile(r"boundary conditions applied: bc_dofs=(\d+), nnz=(\d+)")
_RE_STAB = re.compile(r"stability analysis finished: converged_modes=(\d+)")
_RE_RES = re.compile(r"running resolvent: omega_target=([^\s,]+), modes=(\d+)")
_RE_FINISH = re.compile(r"compute finished in ([0-9.]+)s")
_RE_STAGE = re.compile(r"stage complete: stage=([a-z_]+) elapsed=([0-9.]+)s")
_RE_SOLVER = re.compile(
    r"solver setup: mpi_ranks=(\d+) st_solver=([^\s]+) "
    r"resolvent_solver=([^\s]+) root_assembly=(True|False) matrix_mode=([^\s]+)"
)

_LEGACY_CONFIG_FIELD_MAP = {
    "case_dir": "paths.case_dir",
    "case_inputs_mat": "paths.case_inputs_mat",
    "stride_ksi": "numerics.stride_ksi",
    "stride_eta": "numerics.stride_eta",
    "fd_order": "numerics.fd_order",
    "beta": "numerics.beta",
    "alpha_filter": "numerics.alpha_filter",
    "bc_penalty": "bc.penalty",
    "run_stability": "analysis.run_stability",
    "run_resolvent": "analysis.run_resolvent",
    "n_modes_stability": "analysis.n_modes_stability",
    "n_modes_resolvent": "analysis.n_modes_resolvent",
    "use_raw_eigs": "analysis.use_raw_eigs",
    "use_matlab_sponge_aux": "physics.use_matlab_sponge_aux",
    "zero_mu_t_derivs": "physics.zero_mu_t_derivs",
    "zero_spanwise": "physics.zero_spanwise",
    "matrix_mode": "solver.petsc_matrix_mode",
    "st_solver": "solver.eigen.st_solver",
    "resolvent_solver": "solver.linsolve.type",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_jsonc_with_includes(
    path: Path,
    _seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Load JSONC; if it has a top-level ``$include``, deep-merge with current winning (chainable, cycle-guarded)."""
    resolved = path.resolve()
    if resolved in _seen:
        raise RecursionError(f"Cyclic $include chain at {resolved}")
    cfg = load_jsonc(resolved)
    include_raw = cfg.pop("$include", None)
    if include_raw is None:
        return cfg
    include_path = Path(include_raw)
    if not include_path.is_absolute():
        include_path = (resolved.parent / include_path).resolve()
    base = _load_jsonc_with_includes(include_path, _seen | {resolved})
    return _deep_merge(base, cfg)


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_input_path(primary_dir: Path, raw: str | Path, *fallback_dirs: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [primary_dir, *fallback_dirs]
    for base_dir in candidates:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (primary_dir / path).resolve()


def _resolve_executable(base_dir: Path, raw: str, *fallback_dirs: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if path.parent != Path("."):
        return str(_resolve_input_path(base_dir, raw, *fallback_dirs))
    return raw


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    node = mapping
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _legacy_case_overrides(case: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, dotted_path in _LEGACY_CONFIG_FIELD_MAP.items():
        if key in case:
            _set_nested(overrides, dotted_path, case[key])
    return overrides


def _build_case_config(base_config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    cfg = _deep_merge(base_config, _legacy_case_overrides(case))
    explicit = case.get("config_overrides", {})
    if isinstance(explicit, dict):
        cfg = _deep_merge(cfg, explicit)
    return cfg


def _ps_table() -> dict[int, tuple[int, int]]:
    """Return {pid: (ppid, rss_bytes)} for current processes."""
    out = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,rss="],
        text=True,
    )
    table: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kib = (int(parts[0]), int(parts[1]), int(parts[2]))
        table[pid] = (ppid, rss_kib * 1024)
    return table


def _process_tree_rss_bytes(root_pid: int) -> tuple[int, int]:
    """Return (rss_sum_bytes, n_processes) for root pid and descendants."""
    try:
        table = _ps_table()
    except Exception:
        return 0, 0

    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)

    total_rss = 0
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        node = table.get(pid)
        if node is not None:
            total_rss += node[1]
        stack.extend(children.get(pid, []))

    return total_rss, len(seen)


def _read_output_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, Any] = {}
        if "omega" in data:
            omega = np.asarray(data["omega"]).reshape(-1)
            result["n_omega"] = int(omega.size)
            if omega.size:
                result["leading_omega_real"] = float(np.real(omega[0]))
                result["leading_omega_imag"] = float(np.imag(omega[0]))
        if "gains" in data:
            gains = np.asarray(data["gains"], dtype=float).reshape(-1)
            result["n_gains"] = int(gains.size)
            if gains.size:
                result["max_gain"] = float(np.max(gains))
                result["leading_gain"] = float(gains[0])
        for key_in, prefix in (
            ("resolvent_rho_fwd", "resolvent_rho_fwd"),
            ("resolvent_rho_adj", "resolvent_rho_adj"),
        ):
            if key_in in data:
                arr = np.asarray(data[key_in], dtype=float).reshape(-1)
                finite = arr[np.isfinite(arr)]
                result[f"{prefix}_count"] = int(arr.size)
                if finite.size:
                    result[f"{prefix}_max"] = float(np.max(finite))
                    result[f"{prefix}_median"] = float(np.median(finite))
                    result[f"{prefix}_leading"] = float(arr[0])
        if "omega_target" in data:
            omega_target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
            result["omega_target_real"] = float(np.real(omega_target))
            result["omega_target_imag"] = float(np.imag(omega_target))
        if "X" in data:
            shape = np.asarray(data["X"]).shape
            result["mesh_shape"] = [int(shape[0]), int(shape[1])]
        if "fd_order" in data:
            result["fd_order_saved"] = int(np.asarray(data["fd_order"]).reshape(-1)[0])
        if "alpha_filter" in data:
            result["alpha_filter_saved"] = str(np.asarray(data["alpha_filter"]).reshape(-1)[0])
        if "use_matlab_sponge_aux" in data:
            result["use_matlab_sponge_aux_saved"] = bool(
                np.asarray(data["use_matlab_sponge_aux"]).reshape(-1)[0]
            )
        if "sponge_enabled" in data:
            result["sponge_enabled"] = bool(np.asarray(data["sponge_enabled"]).reshape(-1)[0])
        if "sponge_source" in data:
            result["sponge_source"] = str(np.asarray(data["sponge_source"]).reshape(-1)[0])
        if "sponge_preset" in data:
            result["sponge_preset"] = str(np.asarray(data["sponge_preset"]).reshape(-1)[0])
        if "n_modes_stability_requested" in data:
            result["n_modes_stability_requested"] = int(
                np.asarray(data["n_modes_stability_requested"]).reshape(-1)[0]
            )
        if "stability_eps_nev" in data:
            result["stability_eps_nev"] = int(np.asarray(data["stability_eps_nev"]).reshape(-1)[0])
        if "stability_eps_ncv" in data:
            result["stability_eps_ncv"] = int(np.asarray(data["stability_eps_ncv"]).reshape(-1)[0])
        if "stability_eps_nconv" in data:
            result["stability_eps_nconv"] = int(
                np.asarray(data["stability_eps_nconv"]).reshape(-1)[0]
            )
        if "stability_eps_iterations" in data:
            result["stability_eps_iterations"] = int(
                np.asarray(data["stability_eps_iterations"]).reshape(-1)[0]
            )
        if "stability_eps_conv_test" in data:
            result["stability_eps_conv_test"] = str(
                np.asarray(data["stability_eps_conv_test"]).reshape(-1)[0]
            )
        if "stability_eps_tol" in data:
            result["stability_eps_tol"] = float(
                np.asarray(data["stability_eps_tol"]).reshape(-1)[0]
            )
        if "stability_eps_setup_seconds" in data:
            result["stability_eps_setup_seconds"] = float(
                np.asarray(data["stability_eps_setup_seconds"]).reshape(-1)[0]
            )
        if "stability_eps_solve_seconds" in data:
            result["stability_eps_solve_seconds"] = float(
                np.asarray(data["stability_eps_solve_seconds"]).reshape(-1)[0]
            )
        if "stability_true_residual" in data:
            result["stability_true_residual"] = bool(
                np.asarray(data["stability_true_residual"]).reshape(-1)[0]
            )
        if "stability_reduce_dirichlet" in data:
            result["stability_reduce_dirichlet"] = bool(
                np.asarray(data["stability_reduce_dirichlet"]).reshape(-1)[0]
            )
        if "stability_reduce_dirichlet_set" in data:
            result["stability_reduce_dirichlet_set"] = str(
                np.asarray(data["stability_reduce_dirichlet_set"]).reshape(-1)[0]
            )
        if "stability_constrained_dirichlet_count" in data:
            result["stability_constrained_dirichlet_count"] = int(
                np.asarray(data["stability_constrained_dirichlet_count"]).reshape(-1)[0]
            )
        if "stability_residual_error_type" in data:
            result["stability_residual_error_type"] = str(
                np.asarray(data["stability_residual_error_type"]).reshape(-1)[0]
            )
        if "stability_residual_estimates" in data:
            residuals = np.asarray(data["stability_residual_estimates"], dtype=float).reshape(-1)
            finite = residuals[np.isfinite(residuals)]
            result["stability_residual_count"] = int(residuals.size)
            if finite.size:
                result["stability_residual_max"] = float(np.max(finite))
                result["stability_residual_median"] = float(np.median(finite))
                result["stability_residual_first10_max"] = float(
                    np.max(finite[: min(10, finite.size)])
                )
        if "stability_residual_relative" in data:
            residuals_rel = np.asarray(data["stability_residual_relative"], dtype=float).reshape(-1)
            finite_rel = residuals_rel[np.isfinite(residuals_rel)]
            result["stability_residual_relative_count"] = int(residuals_rel.size)
            if finite_rel.size:
                result["stability_residual_relative_min"] = float(np.min(finite_rel))
                result["stability_residual_relative_max"] = float(np.max(finite_rel))
                result["stability_residual_relative_median"] = float(np.median(finite_rel))
                result["stability_residual_relative_count_le_1e6"] = int(np.sum(finite_rel <= 1e-6))
                result["stability_residual_relative_count_le_1e4"] = int(np.sum(finite_rel <= 1e-4))
                result["stability_residual_relative_count_le_1e3"] = int(np.sum(finite_rel <= 1e-3))
        if "stability_postcheck_residual_rel" in data:
            post_rel = np.asarray(data["stability_postcheck_residual_rel"], dtype=float).reshape(-1)
            finite_post_rel = post_rel[np.isfinite(post_rel)]
            result["stability_postcheck_residual_rel_count"] = int(post_rel.size)
            if finite_post_rel.size:
                result["stability_postcheck_residual_rel_min"] = float(np.min(finite_post_rel))
                result["stability_postcheck_residual_rel_max"] = float(np.max(finite_post_rel))
                result["stability_postcheck_residual_rel_median"] = float(
                    np.median(finite_post_rel)
                )
                result["stability_postcheck_residual_rel_count_le_1e6"] = int(
                    np.sum(finite_post_rel <= 1e-6)
                )
                result["stability_postcheck_residual_rel_count_le_1e4"] = int(
                    np.sum(finite_post_rel <= 1e-4)
                )
                result["stability_postcheck_residual_rel_count_le_1e3"] = int(
                    np.sum(finite_post_rel <= 1e-3)
                )
        if "stability_postcheck_target_distance" in data:
            post_dist = np.asarray(
                data["stability_postcheck_target_distance"], dtype=float
            ).reshape(-1)
            finite_post_dist = post_dist[np.isfinite(post_dist)]
            result["stability_postcheck_target_distance_count"] = int(post_dist.size)
            if finite_post_dist.size:
                result["stability_postcheck_target_distance_min"] = float(np.min(finite_post_dist))
                result["stability_postcheck_target_distance_max"] = float(np.max(finite_post_dist))
                result["stability_postcheck_target_distance_median"] = float(
                    np.median(finite_post_dist)
                )
        if "stability_acceptance_postcheck_rel_max" in data:
            result["stability_acceptance_postcheck_rel_max"] = float(
                np.asarray(data["stability_acceptance_postcheck_rel_max"]).reshape(-1)[0]
            )
        if "stability_acceptance_target_distance_max" in data:
            result["stability_acceptance_target_distance_max"] = float(
                np.asarray(data["stability_acceptance_target_distance_max"]).reshape(-1)[0]
            )
        if "stability_acceptance_omega_real_abs_max" in data:
            result["stability_acceptance_omega_real_abs_max"] = float(
                np.asarray(data["stability_acceptance_omega_real_abs_max"]).reshape(-1)[0]
            )
        if "stability_acceptance_omega_imag_abs_max" in data:
            result["stability_acceptance_omega_imag_abs_max"] = float(
                np.asarray(data["stability_acceptance_omega_imag_abs_max"]).reshape(-1)[0]
            )
        if "stability_accepted_count" in data:
            result["stability_accepted_count"] = int(
                np.asarray(data["stability_accepted_count"]).reshape(-1)[0]
            )
        if "stability_accepted_postcheck_rel_min" in data:
            result["stability_accepted_postcheck_rel_min"] = float(
                np.asarray(data["stability_accepted_postcheck_rel_min"]).reshape(-1)[0]
            )
        if "stability_accepted_postcheck_rel_median" in data:
            result["stability_accepted_postcheck_rel_median"] = float(
                np.asarray(data["stability_accepted_postcheck_rel_median"]).reshape(-1)[0]
            )
        if "stability_accepted_target_distance_min" in data:
            result["stability_accepted_target_distance_min"] = float(
                np.asarray(data["stability_accepted_target_distance_min"]).reshape(-1)[0]
            )
        if "stability_accepted_target_distance_median" in data:
            result["stability_accepted_target_distance_median"] = float(
                np.asarray(data["stability_accepted_target_distance_median"]).reshape(-1)[0]
            )
        if "stability_accepted_omega_real_min" in data:
            result["stability_accepted_omega_real_min"] = float(
                np.asarray(data["stability_accepted_omega_real_min"]).reshape(-1)[0]
            )
        if "stability_accepted_omega_real_max" in data:
            result["stability_accepted_omega_real_max"] = float(
                np.asarray(data["stability_accepted_omega_real_max"]).reshape(-1)[0]
            )
        if "stability_accepted_omega_imag_min" in data:
            result["stability_accepted_omega_imag_min"] = float(
                np.asarray(data["stability_accepted_omega_imag_min"]).reshape(-1)[0]
            )
        if "stability_accepted_omega_imag_max" in data:
            result["stability_accepted_omega_imag_max"] = float(
                np.asarray(data["stability_accepted_omega_imag_max"]).reshape(-1)[0]
            )
        if "stability_postcheck_bc_leak_max" in data:
            leak = np.asarray(data["stability_postcheck_bc_leak_max"], dtype=float).reshape(-1)
            finite_leak = leak[np.isfinite(leak)]
            result["stability_postcheck_bc_leak_count"] = int(leak.size)
            if finite_leak.size:
                result["stability_postcheck_bc_leak_max"] = float(np.max(finite_leak))
                result["stability_postcheck_bc_leak_median"] = float(np.median(finite_leak))
        if "stability_postcheck_dirichlet_leak_max" in data:
            leak = np.asarray(data["stability_postcheck_dirichlet_leak_max"], dtype=float).reshape(
                -1
            )
            finite_leak = leak[np.isfinite(leak)]
            result["stability_postcheck_dirichlet_leak_count"] = int(leak.size)
            if finite_leak.size:
                result["stability_postcheck_dirichlet_leak_max"] = float(np.max(finite_leak))
                result["stability_postcheck_dirichlet_leak_median"] = float(np.median(finite_leak))
        return result


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _parse_log_metrics(lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    stage_timings: dict[str, float] = {}
    for line in lines:
        if "loaded case arrays:" in line:
            match = _RE_CASE_SHAPE.search(line)
            if match:
                metrics["case_shape"] = [int(match.group(1)), int(match.group(2))]
                metrics["case_points"] = int(match.group(3))
        else:
            match = _RE_OPERATOR.search(line)
            if match:
                metrics["operator_shape"] = [int(match.group(1)), int(match.group(2))]
                metrics["ndof"] = int(match.group(1))
                metrics["nnz_pre_bc"] = int(match.group(3))
                continue
        if "boundary conditions applied:" in line:
            match = _RE_BC.search(line)
            if match:
                metrics["bc_dofs"] = int(match.group(1))
                metrics["nnz_post_bc"] = int(match.group(2))
        elif "stability analysis finished:" in line:
            match = _RE_STAB.search(line)
            if match:
                metrics["converged_modes_stability"] = int(match.group(1))
        elif "running resolvent:" in line:
            match = _RE_RES.search(line)
            if match:
                try:
                    _omega_target = complex(match.group(1))
                    metrics["omega_target_from_log_real"] = float(_omega_target.real)
                    metrics["omega_target_from_log_imag"] = float(_omega_target.imag)
                except ValueError:
                    pass
                metrics["requested_resolvent_modes"] = int(match.group(2))
        elif "solver setup:" in line:
            match = _RE_SOLVER.search(line)
            if match:
                metrics["mpi_ranks"] = int(match.group(1))
                metrics["st_solver"] = match.group(2)
                metrics["resolvent_solver"] = match.group(3)
                metrics["root_assembly"] = match.group(4) == "True"
                metrics["matrix_mode"] = match.group(5)
        elif "stage complete:" in line:
            match = _RE_STAGE.search(line)
            if match:
                stage_timings[match.group(1)] = float(match.group(2))
        elif "compute finished in" in line:
            match = _RE_FINISH.search(line)
            if match:
                metrics["compute_elapsed_seconds_reported"] = float(match.group(1))
    if stage_timings:
        metrics["stage_timings_seconds"] = stage_timings
    return metrics


def _build_env(default_env: dict[str, str], case: dict[str, Any]) -> dict[str, str]:
    env = dict(default_env)
    for key, value in dict(case.get("env", {})).items():
        env[str(key)] = str(value)
    return env


def _build_command(python_exe: str, launcher_cfg: dict[str, Any]) -> list[str]:
    mpi_procs = int(launcher_cfg.get("mpi_procs", 1))
    launcher = str(launcher_cfg.get("launcher", "")).strip()
    case_module = os.getenv("LINSTABPY_CASE_MODULE", "case2").strip() or "case2"
    command = [python_exe, "-u", "-c", f"from {case_module} import compute; compute()"]
    if mpi_procs <= 1:
        return command
    if not launcher:
        launcher = "mpiexec"
    launcher_parts = shlex.split(launcher)
    return launcher_parts + ["-n", str(mpi_procs)] + command


def _monitor_process(
    command: list[str],
    env: dict[str, str],
    *,
    cwd: Path,
    log_path: Path,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                log_fh.write(line)
                log_fh.flush()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        peak_rss_bytes = 0
        peak_proc_count = 0
        t0 = time.perf_counter()
        while proc.poll() is None:
            rss_bytes, proc_count = _process_tree_rss_bytes(proc.pid)
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            peak_proc_count = max(peak_proc_count, proc_count)
            time.sleep(max(sample_interval_seconds, 0.1))

        reader.join(timeout=5.0)
        elapsed = time.perf_counter() - t0
        rss_bytes, proc_count = _process_tree_rss_bytes(proc.pid)
        peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
        peak_proc_count = max(peak_proc_count, proc_count)
        return {
            "returncode": int(proc.returncode),
            "elapsed_seconds": float(elapsed),
            "peak_rss_bytes": int(peak_rss_bytes),
            "peak_process_count": int(peak_proc_count),
            "stdout_lines": lines,
        }


def _case_row(case: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(case.get("metrics", {}))
    parsed = dict(case.get("parsed", {}))
    output = dict(case.get("output", {}))
    effective = dict(case.get("effective_case", {}))
    resolved_cfg = dict(case.get("resolved_config", {}))
    stage_timings = dict(parsed.get("stage_timings_seconds", {}))
    return {
        "case_id": case["case_id"],
        "status": case["status"],
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "peak_rss_bytes": metrics.get("peak_rss_bytes"),
        "peak_rss_gib": (
            float(metrics["peak_rss_bytes"]) / float(1024**3)
            if metrics.get("peak_rss_bytes") is not None
            else None
        ),
        "peak_process_count": metrics.get("peak_process_count"),
        "fd_order": _cfg_get(resolved_cfg, "numerics.fd_order", effective.get("fd_order")),
        "alpha_filter": _cfg_get(resolved_cfg, "numerics.alpha_filter"),
        "sponge_enabled": _cfg_get(
            resolved_cfg,
            "physics.sponge.enabled",
            output.get("sponge_enabled"),
        ),
        "use_matlab_sponge_aux": _cfg_get(
            resolved_cfg,
            "physics.use_matlab_sponge_aux",
            effective.get("use_matlab_sponge_aux"),
        ),
        "stride_ksi": _cfg_get(resolved_cfg, "numerics.stride_ksi", effective.get("stride_ksi")),
        "stride_eta": _cfg_get(resolved_cfg, "numerics.stride_eta", effective.get("stride_eta")),
        "run_stability": _cfg_get(
            resolved_cfg, "analysis.run_stability", effective.get("run_stability")
        ),
        "run_resolvent": _cfg_get(
            resolved_cfg, "analysis.run_resolvent", effective.get("run_resolvent")
        ),
        "n_modes_stability_requested": output.get("n_modes_stability_requested"),
        "stability_eps_nev": output.get("stability_eps_nev"),
        "stability_eps_ncv": output.get("stability_eps_ncv"),
        "stability_eps_nconv": output.get("stability_eps_nconv"),
        "stability_eps_iterations": output.get("stability_eps_iterations"),
        "stability_eps_conv_test": output.get("stability_eps_conv_test"),
        "stability_eps_tol": output.get("stability_eps_tol"),
        "stability_eps_setup_seconds": output.get("stability_eps_setup_seconds"),
        "stability_eps_solve_seconds": output.get("stability_eps_solve_seconds"),
        "stability_true_residual": output.get("stability_true_residual"),
        "stability_reduce_dirichlet": output.get("stability_reduce_dirichlet"),
        "stability_reduce_dirichlet_set": output.get("stability_reduce_dirichlet_set"),
        "stability_constrained_dirichlet_count": output.get(
            "stability_constrained_dirichlet_count"
        ),
        "stability_residual_error_type": output.get("stability_residual_error_type"),
        "stability_residual_max": output.get("stability_residual_max"),
        "stability_residual_median": output.get("stability_residual_median"),
        "stability_residual_first10_max": output.get("stability_residual_first10_max"),
        "stability_residual_relative_min": output.get("stability_residual_relative_min"),
        "stability_residual_relative_max": output.get("stability_residual_relative_max"),
        "stability_residual_relative_median": output.get("stability_residual_relative_median"),
        "stability_residual_relative_count_le_1e6": output.get(
            "stability_residual_relative_count_le_1e6"
        ),
        "stability_residual_relative_count_le_1e4": output.get(
            "stability_residual_relative_count_le_1e4"
        ),
        "stability_residual_relative_count_le_1e3": output.get(
            "stability_residual_relative_count_le_1e3"
        ),
        "stability_postcheck_residual_rel_min": output.get("stability_postcheck_residual_rel_min"),
        "stability_postcheck_residual_rel_max": output.get("stability_postcheck_residual_rel_max"),
        "stability_postcheck_residual_rel_median": output.get(
            "stability_postcheck_residual_rel_median"
        ),
        "stability_postcheck_residual_rel_count_le_1e6": output.get(
            "stability_postcheck_residual_rel_count_le_1e6"
        ),
        "stability_postcheck_residual_rel_count_le_1e4": output.get(
            "stability_postcheck_residual_rel_count_le_1e4"
        ),
        "stability_postcheck_residual_rel_count_le_1e3": output.get(
            "stability_postcheck_residual_rel_count_le_1e3"
        ),
        "stability_postcheck_target_distance_min": output.get(
            "stability_postcheck_target_distance_min"
        ),
        "stability_postcheck_target_distance_max": output.get(
            "stability_postcheck_target_distance_max"
        ),
        "stability_postcheck_target_distance_median": output.get(
            "stability_postcheck_target_distance_median"
        ),
        "stability_acceptance_postcheck_rel_max": output.get(
            "stability_acceptance_postcheck_rel_max"
        ),
        "stability_acceptance_target_distance_max": output.get(
            "stability_acceptance_target_distance_max"
        ),
        "stability_acceptance_omega_real_abs_max": output.get(
            "stability_acceptance_omega_real_abs_max"
        ),
        "stability_acceptance_omega_imag_abs_max": output.get(
            "stability_acceptance_omega_imag_abs_max"
        ),
        "stability_accepted_count": output.get("stability_accepted_count"),
        "stability_accepted_postcheck_rel_min": output.get("stability_accepted_postcheck_rel_min"),
        "stability_accepted_postcheck_rel_median": output.get(
            "stability_accepted_postcheck_rel_median"
        ),
        "stability_accepted_target_distance_min": output.get(
            "stability_accepted_target_distance_min"
        ),
        "stability_accepted_target_distance_median": output.get(
            "stability_accepted_target_distance_median"
        ),
        "stability_accepted_omega_real_min": output.get("stability_accepted_omega_real_min"),
        "stability_accepted_omega_real_max": output.get("stability_accepted_omega_real_max"),
        "stability_accepted_omega_imag_min": output.get("stability_accepted_omega_imag_min"),
        "stability_accepted_omega_imag_max": output.get("stability_accepted_omega_imag_max"),
        "stability_postcheck_bc_leak_max": output.get("stability_postcheck_bc_leak_max"),
        "stability_postcheck_bc_leak_median": output.get("stability_postcheck_bc_leak_median"),
        "stability_postcheck_dirichlet_leak_max": output.get(
            "stability_postcheck_dirichlet_leak_max"
        ),
        "stability_postcheck_dirichlet_leak_median": output.get(
            "stability_postcheck_dirichlet_leak_median"
        ),
        "mpi_ranks": parsed.get("mpi_ranks"),
        "st_solver": parsed.get("st_solver"),
        "resolvent_solver": parsed.get("resolvent_solver"),
        "matrix_mode": parsed.get("matrix_mode"),
        "root_assembly": parsed.get("root_assembly"),
        "ndof": parsed.get("ndof"),
        "nnz_pre_bc": parsed.get("nnz_pre_bc"),
        "nnz_post_bc": parsed.get("nnz_post_bc"),
        "bc_dofs": parsed.get("bc_dofs"),
        "load_case_seconds": stage_timings.get("load_case"),
        "build_mesh_seconds": stage_timings.get("build_mesh"),
        "assemble_operator_seconds": stage_timings.get("assemble_operator_bc"),
        "assemble_operator_bc_seconds": stage_timings.get("assemble_operator_bc"),
        "apply_bc_seconds": stage_timings.get("apply_bc"),
        "compute_chu_norm_seconds": stage_timings.get("compute_chu_norm"),
        "stability_seconds": stage_timings.get("stability"),
        "resolvent_seconds": stage_timings.get("resolvent"),
        "save_results_seconds": stage_timings.get("save_results"),
        "compute_elapsed_seconds_reported": parsed.get("compute_elapsed_seconds_reported"),
        "leading_omega_real": output.get("leading_omega_real"),
        "leading_omega_imag": output.get("leading_omega_imag"),
        "leading_gain": output.get("leading_gain"),
        "max_gain": output.get("max_gain"),
        "resolvent_rho_fwd_max": output.get("resolvent_rho_fwd_max"),
        "resolvent_rho_fwd_median": output.get("resolvent_rho_fwd_median"),
        "resolvent_rho_fwd_leading": output.get("resolvent_rho_fwd_leading"),
        "resolvent_rho_adj_max": output.get("resolvent_rho_adj_max"),
        "resolvent_rho_adj_median": output.get("resolvent_rho_adj_median"),
        "resolvent_rho_adj_leading": output.get("resolvent_rho_adj_leading"),
        "case_config_path": case.get("case_config_path"),
        "log_path": case.get("log_path"),
        "artifact_npz": case.get("artifact_npz"),
        "error": case.get("error"),
    }


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_case_row(case) for case in cases]
    fieldnames = [
        "case_id",
        "status",
        "elapsed_seconds",
        "peak_rss_bytes",
        "peak_rss_gib",
        "peak_process_count",
        "fd_order",
        "alpha_filter",
        "sponge_enabled",
        "use_matlab_sponge_aux",
        "stride_ksi",
        "stride_eta",
        "run_stability",
        "run_resolvent",
        "n_modes_stability_requested",
        "stability_eps_nev",
        "stability_eps_ncv",
        "stability_eps_nconv",
        "stability_eps_iterations",
        "stability_eps_conv_test",
        "stability_eps_tol",
        "stability_eps_setup_seconds",
        "stability_eps_solve_seconds",
        "stability_true_residual",
        "stability_reduce_dirichlet",
        "stability_reduce_dirichlet_set",
        "stability_constrained_dirichlet_count",
        "stability_residual_error_type",
        "stability_residual_max",
        "stability_residual_median",
        "stability_residual_first10_max",
        "stability_residual_relative_min",
        "stability_residual_relative_max",
        "stability_residual_relative_median",
        "stability_residual_relative_count_le_1e6",
        "stability_residual_relative_count_le_1e4",
        "stability_residual_relative_count_le_1e3",
        "stability_postcheck_residual_rel_min",
        "stability_postcheck_residual_rel_max",
        "stability_postcheck_residual_rel_median",
        "stability_postcheck_residual_rel_count_le_1e6",
        "stability_postcheck_residual_rel_count_le_1e4",
        "stability_postcheck_residual_rel_count_le_1e3",
        "stability_postcheck_target_distance_min",
        "stability_postcheck_target_distance_max",
        "stability_postcheck_target_distance_median",
        "stability_acceptance_postcheck_rel_max",
        "stability_acceptance_target_distance_max",
        "stability_acceptance_omega_real_abs_max",
        "stability_acceptance_omega_imag_abs_max",
        "stability_accepted_count",
        "stability_accepted_postcheck_rel_min",
        "stability_accepted_postcheck_rel_median",
        "stability_accepted_target_distance_min",
        "stability_accepted_target_distance_median",
        "stability_accepted_omega_real_min",
        "stability_accepted_omega_real_max",
        "stability_accepted_omega_imag_min",
        "stability_accepted_omega_imag_max",
        "stability_postcheck_bc_leak_max",
        "stability_postcheck_bc_leak_median",
        "stability_postcheck_dirichlet_leak_max",
        "stability_postcheck_dirichlet_leak_median",
        "mpi_ranks",
        "st_solver",
        "resolvent_solver",
        "matrix_mode",
        "root_assembly",
        "ndof",
        "nnz_pre_bc",
        "nnz_post_bc",
        "bc_dofs",
        "load_case_seconds",
        "build_mesh_seconds",
        "assemble_operator_seconds",
        "assemble_operator_bc_seconds",
        "apply_bc_seconds",
        "compute_chu_norm_seconds",
        "stability_seconds",
        "resolvent_seconds",
        "save_results_seconds",
        "compute_elapsed_seconds_reported",
        "leading_omega_real",
        "leading_omega_imag",
        "leading_gain",
        "max_gain",
        "resolvent_rho_fwd_max",
        "resolvent_rho_fwd_median",
        "resolvent_rho_fwd_leading",
        "resolvent_rho_adj_max",
        "resolvent_rho_adj_median",
        "resolvent_rho_adj_leading",
        "case_config_path",
        "log_path",
        "artifact_npz",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_payload(
    *,
    config_path: Path,
    report_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": _now_utc(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "cases": report_cases,
        "ok_count": sum(case["status"] == "ok" for case in report_cases),
        "fail_count": sum(case["status"] != "ok" for case in report_cases),
    }


def _flush_reports(
    *,
    config_path: Path,
    report_json: Path,
    report_csv: Path,
    report_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    report = _report_payload(config_path=config_path, report_cases=report_cases)
    _write_json(report_json, report)
    _write_csv(report_csv, report_cases)
    return report


def main() -> int:
    config_path = Path(os.getenv("LINSTABPY_CASE2_PERF_CONFIG", str(DEFAULT_CONFIG))).resolve()
    cfg = _load_jsonc_with_includes(config_path)
    config_dir = config_path.parent

    outputs = dict(cfg.get("outputs", {}))
    report_json = _resolve_path(CASE_DIR, outputs.get("report_json", str(DEFAULT_JSON)))
    report_csv = _resolve_path(CASE_DIR, outputs.get("report_csv", str(DEFAULT_CSV)))
    log_dir = _resolve_path(CASE_DIR, outputs.get("log_dir", str(DEFAULT_LOG_DIR)))
    run_dir = _resolve_path(CASE_DIR, outputs.get("run_dir", str(DEFAULT_RUN_DIR)))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    defaults = dict(cfg.get("defaults", {}))
    default_env = {str(k): str(v) for k, v in dict(defaults.get("env", {})).items()}
    default_launcher = dict(defaults.get("launcher", {}))
    monitor_interval_seconds = float(defaults.get("monitor_interval_seconds", 0.5))
    python_exe = _resolve_executable(
        config_dir, str(defaults.get("python_exe", sys.executable)), CASE_DIR
    )
    default_base_config = defaults.get(
        "base_config",
        default_env.get("LINSTABPY_CASE2_CONFIG", "configs/parity_config.jsonc"),
    )
    default_case_workdir = _resolve_input_path(
        config_dir,
        defaults.get("case_workdir", str(CASE_DIR)),
        CASE_DIR,
    )

    cases_cfg = cfg.get("cases", [])
    if not isinstance(cases_cfg, list) or not cases_cfg:
        raise ValueError(f"No cases configured in {config_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_cases: list[dict[str, Any]] = []

    for raw_case in cases_cfg:
        if not isinstance(raw_case, dict):
            raise ValueError("Each case entry must be an object")
        effective_case = _deep_merge(defaults.get("case_defaults", {}), raw_case)
        if not bool(effective_case.get("enabled", True)):
            continue

        case_id = str(effective_case["case_id"])
        case_report: dict[str, Any] = {
            "case_id": case_id,
            "label": effective_case.get("label", case_id),
            "status": "failed",
            "effective_case": effective_case,
            "metrics": {},
            "parsed": {},
            "output": {},
            "error": None,
        }
        report_cases.append(case_report)

        case_log = log_dir / f"case2_perf_{case_id}_{timestamp}.log"
        case_npz = run_dir / f"case2_perf_{case_id}_{timestamp}.npz"
        case_cfg_path = run_dir / f"case2_perf_{case_id}_{timestamp}.config.json"
        case_report["log_path"] = str(case_log)
        case_report["artifact_npz"] = str(case_npz)
        case_report["case_config_path"] = str(case_cfg_path)
        case_workdir = _resolve_input_path(
            config_dir,
            effective_case.get("case_workdir", str(default_case_workdir)),
            CASE_DIR,
        )
        case_report["case_workdir"] = str(case_workdir)

        base_config_path = _resolve_input_path(
            config_dir,
            effective_case.get("base_config", default_base_config),
            CASE_DIR / "configs",
            CASE_DIR,
        )
        base_config = load_jsonc(base_config_path)
        case_config = _build_case_config(base_config, effective_case)
        _write_json(case_cfg_path, case_config)
        case_report["resolved_config"] = case_config

        env = os.environ.copy()
        env.update(_build_env(default_env, effective_case))
        env["LINSTABPY_CASE2_CONFIG"] = str(case_cfg_path)
        env["LINSTABPY_CASE_CONFIG"] = str(case_cfg_path)
        env["LINSTABPY_CASE2_OUTPUT_NPZ"] = str(case_npz)
        env["LINSTABPY_CASE_OUTPUT_NPZ"] = str(case_npz)
        pythonpath_entries = [str(ROOT / "src"), str(ROOT)]
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            pythonpath_entries + ([existing_pythonpath] if existing_pythonpath else [])
        )
        launcher_cfg = _deep_merge(default_launcher, dict(effective_case.get("launcher", {})))
        command = _build_command(python_exe, launcher_cfg)
        case_report["command"] = command

        print(f"[case2-perf] running {case_id}: {' '.join(command)}")
        try:
            metrics = _monitor_process(
                command,
                env,
                cwd=case_workdir,
                log_path=case_log,
                sample_interval_seconds=monitor_interval_seconds,
            )
            case_report["metrics"] = {
                "elapsed_seconds": metrics["elapsed_seconds"],
                "peak_rss_bytes": metrics["peak_rss_bytes"],
                "peak_process_count": metrics["peak_process_count"],
                "returncode": metrics["returncode"],
            }
            case_report["parsed"] = _parse_log_metrics(metrics["stdout_lines"])

            output_npz = case_npz if case_npz.exists() else DEFAULT_OUTPUT_NPZ
            if metrics["returncode"] == 0 and output_npz.exists():
                if output_npz != case_npz:
                    shutil.copy2(output_npz, case_npz)
                case_report["output"] = _read_output_npz(case_npz)
                case_report["status"] = "ok"
            else:
                case_report["error"] = (
                    f"child process failed with returncode={metrics['returncode']}"
                )
        except Exception as exc:
            case_report["error"] = str(exc)

        _flush_reports(
            config_path=config_path,
            report_json=report_json,
            report_csv=report_csv,
            report_cases=report_cases,
        )
    report = _flush_reports(
        config_path=config_path,
        report_json=report_json,
        report_csv=report_csv,
        report_cases=report_cases,
    )
    print(
        "[case2-perf]",
        f"ok={report['ok_count']}",
        f"fail={report['fail_count']}",
        f"json={report_json}",
        f"csv={report_csv}",
    )
    return 0 if report["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
