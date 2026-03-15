#!/usr/bin/env python3
"""Sequential Song2023 full benchmark with MPI rank and order ladder."""

from __future__ import annotations

import argparse
import csv
import json
import time
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from linstab2d.utils.jsonc import load_jsonc

from . import run_song_2023_matrix as matrix_runner


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
DEFAULT_CONFIG = SCRIPT.with_name("song_2023_scaling.jsonc")
SWAP_RE = re.compile(r"used = ([0-9.]+)M")
FREE_RE = re.compile(r"System-wide memory free percentage:\s*([0-9.]+)%")


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _read_swap_used_mib() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    match = SWAP_RE.search(out)
    return float(match.group(1)) if match else None


def _read_free_percent() -> float | None:
    try:
        out = subprocess.check_output(["memory_pressure", "-Q"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    match = FREE_RE.search(out)
    return float(match.group(1)) if match else None


def _parse_log_metrics(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "total_time_s",
            "total_time_s_min_rank",
            "total_time_s_mean_rank",
            "peak_memory_mib",
            "peak_memory_mib_rank0",
            "peak_memory_mib_sum_ranks",
            "memory_delta_mib_rank0",
            "operator_nnz",
        }:
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
        elif key in {"mpi_ranks"}:
            try:
                metrics[key] = int(value.split()[0])
            except ValueError:
                metrics[key] = value
        elif key in {"petsc_matrix_mode", "distributed_matrix", "matrix_type", "ownership_ranges"}:
            metrics[key] = value
    return metrics


def _run_guarded(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    swap_before_mib: float | None,
    stop_on_swap_delta_mib: float | None,
    poll_interval_s: float = 5.0,
) -> tuple[subprocess.CompletedProcess[str], float | None, str | None]:
    """Run one command and terminate early if swap growth crosses the guard."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    swap_peak_mib = swap_before_mib
    stop_reason = None

    while True:
        ret = proc.poll()
        current_swap = _read_swap_used_mib()
        if current_swap is not None:
            if swap_peak_mib is None:
                swap_peak_mib = current_swap
            else:
                swap_peak_mib = max(float(swap_peak_mib), float(current_swap))
            if (
                ret is None
                and stop_on_swap_delta_mib is not None
                and swap_before_mib is not None
                and current_swap - swap_before_mib >= float(stop_on_swap_delta_mib)
            ):
                stop_reason = (
                    f"swap_delta_mib={current_swap - swap_before_mib:.2f} >= "
                    f"stop_on_swap_delta_mib={float(stop_on_swap_delta_mib):.2f}"
                )
                proc.terminate()
                break
        if ret is not None:
            break
        time.sleep(poll_interval_s)

    try:
        stdout, stderr = proc.communicate(timeout=15.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=int(proc.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )
    return completed, swap_peak_mib, stop_reason


def _should_stop(result: dict[str, Any], guard_cfg: dict[str, Any]) -> str | None:
    summary = result.get("summary", {})
    peak_max = summary.get("peak_memory_mib")
    peak_sum = summary.get("peak_memory_mib_sum_ranks")
    swap_delta = result.get("swap_delta_mib")

    stop_peak_max = guard_cfg.get("stop_on_peak_memory_mib_max_rank")
    if stop_peak_max is not None and peak_max is not None and float(peak_max) >= float(stop_peak_max):
        return f"peak_memory_mib={peak_max} >= stop_on_peak_memory_mib_max_rank={stop_peak_max}"

    stop_peak_sum = guard_cfg.get("stop_on_peak_memory_mib_sum_ranks")
    if stop_peak_sum is not None and peak_sum is not None and float(peak_sum) >= float(stop_peak_sum):
        return f"peak_memory_mib_sum_ranks={peak_sum} >= stop_on_peak_memory_mib_sum_ranks={stop_peak_sum}"

    stop_swap = guard_cfg.get("stop_on_swap_delta_mib")
    if stop_swap is not None and swap_delta is not None and float(swap_delta) >= float(stop_swap):
        return f"swap_delta_mib={swap_delta} >= stop_on_swap_delta_mib={stop_swap}"

    if result.get("ranks", 1) > 1 and str(summary.get("distributed_matrix", "")).lower() not in {"true", "1"}:
        return "distributed proof missing for ranks > 1"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to Song scaling JSONC config.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated scaling variant names to run (default: all).",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_jsonc(cfg_path)
    base_cfg = load_jsonc(REPO / cfg["base_config"])
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    guard_cfg = cfg.get("memory_guard", {})
    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}

    results: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        if selected_variants and variant["name"] not in selected_variants:
            continue
        name = variant["name"]
        ranks = int(variant["ranks"])
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_run = _deep_merge(base_cfg, variant["overrides"])
        config_path = run_dir / "song_2023.variant.jsonc"
        config_path.write_text(json.dumps(_json_safe(cfg_run), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        env = dict(**subprocess.os.environ)
        env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"

        python_bin = str(REPO / ".venv" / "bin" / "python")
        if ranks == 1:
            cmd = [python_bin, "-m", "validation.song_2023.song_2023", "--config", str(config_path)]
        else:
            cmd = ["mpiexec", "-n", str(ranks), python_bin, "-m", "validation.song_2023.song_2023", "--config", str(config_path)]

        swap_before = _read_swap_used_mib()
        free_before = _read_free_percent()
        print(f"[song-scaling] {name}")
        matrix_runner._clear_canonical_outputs()
        proc, swap_peak_mib, runtime_stop_reason = _run_guarded(
            cmd,
            cwd=str(REPO),
            env=env,
            swap_before_mib=swap_before,
            stop_on_swap_delta_mib=guard_cfg.get("stop_on_swap_delta_mib"),
        )
        swap_after = _read_swap_used_mib()
        free_after = _read_free_percent()

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        target_npz, target_log = matrix_runner._collect_outputs(run_dir)

        result: dict[str, Any] = {
            "variant": name,
            "ranks": ranks,
            "returncode": int(proc.returncode),
            "status": "ok" if proc.returncode == 0 and target_log.exists() else "failed",
            "stdout_log": str(stdout_path.relative_to(REPO)),
            "stderr_log": str(stderr_path.relative_to(REPO)),
            "config_path": str(config_path.relative_to(REPO)),
            "swap_before_mib": swap_before,
            "swap_after_mib": swap_after,
            "swap_peak_mib": swap_peak_mib,
            "swap_delta_mib": (None if swap_before is None or swap_after is None else float(swap_after - swap_before)),
            "free_before_percent": free_before,
            "free_after_percent": free_after,
        }
        if runtime_stop_reason is not None:
            result["stop_reason"] = runtime_stop_reason
        if result["status"] == "ok":
            result["summary"] = _parse_log_metrics(target_log)
            stop_reason = _should_stop(result, guard_cfg)
            if stop_reason is not None:
                result["stop_reason"] = stop_reason
                results.append(result)
                print(f"[song-scaling] stopping after {name}: {stop_reason}")
                break
        else:
            result["error"] = (proc.stderr or proc.stdout).splitlines()[:30]
            results.append(result)
            print(
                f"[song-scaling] stopping after {name}: "
                f"{result.get('stop_reason', 'run failed')}"
            )
            break

        results.append(result)

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "ranks": item["ranks"],
            "status": item["status"],
            "returncode": item["returncode"],
            "swap_before_mib": item.get("swap_before_mib", ""),
            "swap_after_mib": item.get("swap_after_mib", ""),
            "swap_delta_mib": item.get("swap_delta_mib", ""),
            "free_before_percent": item.get("free_before_percent", ""),
            "free_after_percent": item.get("free_after_percent", ""),
            "stop_reason": item.get("stop_reason", ""),
            "stdout_log": item["stdout_log"],
            "stderr_log": item["stderr_log"],
        }
        for key, value in item.get("summary", {}).items():
            row[key] = _json_safe(value)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"JSON summary: {summary_json.relative_to(REPO)}")
    print(f"CSV summary:  {summary_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
