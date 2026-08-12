#!/usr/bin/env python3
"""Sequential Song2023 full benchmark with MPI rank and order ladder."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from linstabpy.utils.jsonc import load_config

from . import run_song_2023_matrix as matrix_runner

SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parent
REPO = SCRIPT.parents[2]
DEFAULT_CONFIG = SONG_DIR / "song_32deg.yaml"
SWAP_RE = re.compile(r"used = ([0-9.]+)M")
FREE_RE = re.compile(r"System-wide memory free percentage:\s*([0-9.]+)%")
MEMINFO_RE = re.compile(r"^([A-Za-z_()]+):\s+([0-9]+)\s+kB$", re.MULTILINE)


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
    meminfo = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and meminfo.exists():
        try:
            text = meminfo.read_text(encoding="utf-8")
        except OSError:
            return None
        fields = {match.group(1): float(match.group(2)) for match in MEMINFO_RE.finditer(text)}
        swap_total = fields.get("SwapTotal")
        swap_free = fields.get("SwapFree")
        if swap_total is None or swap_free is None:
            return None
        return (swap_total - swap_free) / 1024.0

    try:
        out = subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True, stderr=subprocess.STDOUT
        )
    except Exception:
        return None
    match = SWAP_RE.search(out)
    return float(match.group(1)) if match else None


def _read_free_percent() -> float | None:
    meminfo = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and meminfo.exists():
        try:
            text = meminfo.read_text(encoding="utf-8")
        except OSError:
            return None
        fields = {match.group(1): float(match.group(2)) for match in MEMINFO_RE.finditer(text)}
        total = fields.get("MemTotal")
        available = fields.get("MemAvailable")
        if total is None or available is None or total <= 0.0:
            return None
        return 100.0 * available / total

    try:
        out = subprocess.check_output(
            ["memory_pressure", "-Q"], text=True, stderr=subprocess.STDOUT
        )
    except Exception:
        return None
    match = FREE_RE.search(out)
    return float(match.group(1)) if match else None


def _parse_log_metrics(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    step_summary: dict[str, dict[str, float | int]] = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("step_summary "):
            parts = line.split()
            if len(parts) >= 9:
                name = parts[1]
                step_summary[name] = {
                    "time_min": float(parts[2]),
                    "time_mean": float(parts[3]),
                    "time_max": float(parts[4]),
                    "memory_min": float(parts[5]),
                    "memory_mean": float(parts[6]),
                    "memory_max": float(parts[7]),
                    "count": int(parts[8]),
                }
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip().rstrip(",")
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
                    metrics[key] = int(value)
                except ValueError:
                    metrics[key] = value
            elif key in {"petsc_matrix_mode", "distributed_matrix", "matrix_type"}:
                metrics[key] = value
        if line.startswith("ownership_ranges="):
            metrics["ownership_ranges"] = line.split("=", 1)[1].strip()
    if step_summary:
        total_time = metrics.get("total_time_s")
        if isinstance(total_time, (int, float)) and float(total_time) > 0.0:
            for step_name, step_metrics in step_summary.items():
                step_metrics["time_share_total"] = float(step_metrics["time_max"]) / float(
                    total_time
                )
        metrics["step_summary"] = step_summary
    return metrics


def _collect_existing_result(run_dir: Path, *, ranks: int, name: str) -> dict[str, Any] | None:
    target_npz = run_dir / "song_2023.npz"
    target_log = run_dir / "song_2023.log"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    if not target_npz.exists() or not target_log.exists():
        return None

    summary = _parse_log_metrics(target_log)
    result: dict[str, Any] = {
        "variant": name,
        "ranks": ranks,
        "returncode": 0,
        "status": "ok",
        "stdout_log": str(stdout_path.relative_to(REPO)) if stdout_path.exists() else "",
        "stderr_log": str(stderr_path.relative_to(REPO)) if stderr_path.exists() else "",
        "config_path": str((run_dir / "song_2023.variant.yaml").relative_to(REPO)),
        "swap_before_mib": None,
        "swap_after_mib": None,
        "swap_peak_mib": None,
        "swap_delta_mib": None,
        "free_before_percent": None,
        "free_after_percent": None,
        "summary": summary,
        "reused_existing": True,
    }
    return result


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
    if (
        stop_peak_max is not None
        and peak_max is not None
        and float(peak_max) >= float(stop_peak_max)
    ):
        return f"peak_memory_mib={peak_max} >= stop_on_peak_memory_mib_max_rank={stop_peak_max}"

    stop_peak_sum = guard_cfg.get("stop_on_peak_memory_mib_sum_ranks")
    if (
        stop_peak_sum is not None
        and peak_sum is not None
        and float(peak_sum) >= float(stop_peak_sum)
    ):
        return f"peak_memory_mib_sum_ranks={peak_sum} >= stop_on_peak_memory_mib_sum_ranks={stop_peak_sum}"

    stop_swap = guard_cfg.get("stop_on_swap_delta_mib")
    if stop_swap is not None and swap_delta is not None and float(swap_delta) >= float(stop_swap):
        return f"swap_delta_mib={swap_delta} >= stop_on_swap_delta_mib={stop_swap}"

    if result.get("ranks", 1) > 1 and str(summary.get("distributed_matrix", "")).lower() not in {
        "true",
        "1",
    }:
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
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Re-run variants even when song_2023.npz and song_2023.log already exist.",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_config(cfg_path)
    base_cfg = load_config(REPO / cfg["base_config"])
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

        if not args.rerun_existing:
            existing = _collect_existing_result(run_dir, ranks=ranks, name=name)
            if existing is not None:
                print(f"[song-scaling] {name} (reusing existing result)")
                results.append(existing)
                continue

        cfg_run = _deep_merge(base_cfg, variant["overrides"])
        config_path = run_dir / "song_2023.variant.yaml"
        config_path.write_text(
            json.dumps(_json_safe(cfg_run), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        env = dict(**subprocess.os.environ)
        env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"

        python_bin = str(REPO / ".venv" / "bin" / "python")
        if ranks == 1:
            cmd = [
                python_bin,
                "-m",
                "scripts.validation.song_2023.song_2023",
                "--config",
                str(config_path),
            ]
        else:
            cmd = [
                "mpiexec",
                "-n",
                str(ranks),
                python_bin,
                "-m",
                "scripts.validation.song_2023.song_2023",
                "--config",
                str(config_path),
            ]

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
            "swap_delta_mib": (
                None
                if swap_before is None or swap_after is None
                else float(swap_after - swap_before)
            ),
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
                f"[song-scaling] stopping after {name}: {result.get('stop_reason', 'run failed')}"
            )
            break

        results.append(result)

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
