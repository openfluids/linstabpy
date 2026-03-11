#!/usr/bin/env python3
"""Config-driven timing and memory harness for Case 1 runs.

Launches the canonical ``case1.compute()`` pipeline in a child
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._helpers import load_jsonc


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent
DEFAULT_CONFIG = CASE_DIR / "perf_memory_cases.jsonc"
DEFAULT_JSON = CASE_DIR / "output" / "case1_perf_memory_latest.json"
DEFAULT_CSV = CASE_DIR / "output" / "case1_perf_memory_table.csv"
DEFAULT_LOG_DIR = CASE_DIR / "output" / "logs"
DEFAULT_RUN_DIR = CASE_DIR / "output" / "runs"
DEFAULT_OUTPUT_NPZ = CASE_DIR / "case1.npz"

_RE_CASE_SHAPE = re.compile(r"shape=\((\d+),\s*(\d+)\), points=(\d+)")
_RE_OPERATOR = re.compile(r"operator assembled: shape=\((\d+),\s*(\d+)\), nnz=(\d+)")
_RE_BC = re.compile(r"boundary conditions applied: bc_dofs=(\d+), nnz=(\d+)")
_RE_STAB = re.compile(r"stability analysis finished: converged_modes=(\d+)")
_RE_RES = re.compile(r"running resolvent: omega_target=([^\s,]+), modes=(\d+)")
_RE_FINISH = re.compile(r"compute finished in ([0-9.]+)s")
_RE_STAGE = re.compile(r"stage complete: stage=([a-z_]+) elapsed=([0-9.]+)s")
_RE_SOLVER = re.compile(
    r"solver setup: mpi_ranks=(\d+) st_solver=([^\s]+) "
    r"resolvent_solver=([^\s]+) root_assembly=(True|False) matrix_mode=([^\s]+)"
)


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
    with np.load(path, allow_pickle = False) as data:
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
        if "omega_target" in data:
            omega_target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
            result["omega_target_real"] = float(np.real(omega_target))
            result["omega_target_imag"] = float(np.imag(omega_target))
        if "X" in data:
            shape = np.asarray(data["X"]).shape
            result["mesh_shape"] = [int(shape[0]), int(shape[1])]
        return result


def _parse_log_metrics(lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    stage_timings: dict[str, float] = {}
    for line in lines:
        if "loaded case arrays:" in line:
            match = _RE_CASE_SHAPE.search(line)
            if match:
                metrics["case_shape"] = [int(match.group(1)), int(match.group(2))]
                metrics["case_points"] = int(match.group(3))
        elif "operator assembled:" in line:
            match = _RE_OPERATOR.search(line)
            if match:
                metrics["operator_shape"] = [int(match.group(1)), int(match.group(2))]
                metrics["ndof"] = int(match.group(1))
                metrics["nnz_pre_bc"] = int(match.group(3))
        elif "boundary conditions applied:" in line:
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
                metrics["omega_target_from_log"] = float(match.group(1))
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

    mapping = {
        "stride_ksi": "LINSTAB2D_CASE1_STRIDE_KSI",
        "stride_eta": "LINSTAB2D_CASE1_STRIDE_ETA",
        "fd_order": "LINSTAB2D_CASE1_FD_ORDER",
        "beta": "LINSTAB2D_CASE1_BETA",
        "alpha_filter": "LINSTAB2D_CASE1_ALPHA_FILTER",
        "run_stability": "LINSTAB2D_CASE1_RUN_STABILITY",
        "run_resolvent": "LINSTAB2D_CASE1_RUN_RESOLVENT",
        "n_modes_stability": "LINSTAB2D_CASE1_N_MODES",
        "n_modes_resolvent": "LINSTAB2D_CASE1_N_MODES_RESOLVENT",
        "bc_penalty": "LINSTAB2D_CASE1_BC_PENALTY",
        "case_dir": "LINSTAB2D_CASE1_CASE_DIR",
        "case_inputs_mat": "LINSTAB2D_CASE1_CASE_INPUTS_MAT",
    }
    for key, env_name in mapping.items():
        if key in case:
            value = case[key]
            if isinstance(value, bool):
                env[env_name] = "1" if value else "0"
            else:
                env[env_name] = str(value)

    for key, value in dict(case.get("env", {})).items():
        env[str(key)] = str(value)

    return env


def _build_command(python_exe: str, launcher_cfg: dict[str, Any]) -> list[str]:
    mpi_procs = int(launcher_cfg.get("mpi_procs", 1))
    launcher = str(launcher_cfg.get("launcher", "")).strip()
    command = [python_exe, "-u", "-c", "from case1 import compute; compute()"]
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
        "stride_ksi": effective.get("stride_ksi"),
        "stride_eta": effective.get("stride_eta"),
        "run_stability": effective.get("run_stability"),
        "run_resolvent": effective.get("run_resolvent"),
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
        "assemble_operator_seconds": stage_timings.get("assemble_operator"),
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
        "stride_ksi",
        "stride_eta",
        "run_stability",
        "run_resolvent",
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
        "log_path",
        "artifact_npz",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config_path = Path(os.getenv("LINSTAB2D_CASE1_PERF_CONFIG", str(DEFAULT_CONFIG))).resolve()
    cfg = load_jsonc(config_path)

    outputs = dict(cfg.get("outputs", {}))
    report_json = Path(outputs.get("report_json", DEFAULT_JSON)).resolve()
    report_csv = Path(outputs.get("report_csv", DEFAULT_CSV)).resolve()
    log_dir = Path(outputs.get("log_dir", DEFAULT_LOG_DIR)).resolve()
    run_dir = Path(outputs.get("run_dir", DEFAULT_RUN_DIR)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    defaults = dict(cfg.get("defaults", {}))
    default_env = {str(k): str(v) for k, v in dict(defaults.get("env", {})).items()}
    default_launcher = dict(defaults.get("launcher", {}))
    monitor_interval_seconds = float(defaults.get("monitor_interval_seconds", 0.5))
    python_exe = str(defaults.get("python_exe", sys.executable))

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

        case_log = log_dir / f"case1_perf_{case_id}_{timestamp}.log"
        case_npz = run_dir / f"case1_perf_{case_id}_{timestamp}.npz"
        case_report["log_path"] = str(case_log)
        case_report["artifact_npz"] = str(case_npz)

        env = os.environ.copy()
        env.update(_build_env(default_env, effective_case))
        launcher_cfg = _deep_merge(default_launcher, dict(effective_case.get("launcher", {})))
        command = _build_command(python_exe, launcher_cfg)
        case_report["command"] = command

        print(f"[case1-perf] running {case_id}: {' '.join(command)}")
        try:
            metrics = _monitor_process(
                command,
                env,
                cwd=ROOT,
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

            if metrics["returncode"] == 0 and DEFAULT_OUTPUT_NPZ.exists():
                shutil.copy2(DEFAULT_OUTPUT_NPZ, case_npz)
                case_report["output"] = _read_output_npz(case_npz)
                case_report["status"] = "ok"
            else:
                case_report["error"] = f"child process failed with returncode={metrics['returncode']}"
        except Exception as exc:
            case_report["error"] = str(exc)

    report = {
        "generated_at_utc": _now_utc(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "cases": report_cases,
        "ok_count": sum(case["status"] == "ok" for case in report_cases),
        "fail_count": sum(case["status"] != "ok" for case in report_cases),
    }

    _write_json(report_json, report)
    _write_csv(report_csv, report_cases)
    print(
        "[case1-perf]",
        f"ok={report['ok_count']}",
        f"fail={report['fail_count']}",
        f"json={report_json}",
        f"csv={report_csv}",
    )
    return 0 if report["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
