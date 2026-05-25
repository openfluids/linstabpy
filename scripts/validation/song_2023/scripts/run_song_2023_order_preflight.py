#!/usr/bin/env python3
"""Run Song2023 order preflight sequentially with a memory guard."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from linstabpy.utils.jsonc import load_jsonc

from . import run_song_2023_matrix as matrix_runner


SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parents[1]
REPO = SCRIPT.parents[3]
DEFAULT_CONFIG = SONG_DIR / "configs" / "song_2023_order_preflight.jsonc"


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


def _memory_limit_hit(summary: dict[str, Any], threshold_mib: float | None) -> bool:
    if threshold_mib is None:
        return False
    peak = summary.get("peak_memory_mib")
    return peak is not None and float(peak) >= float(threshold_mib)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to Song order-preflight JSONC config.",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_jsonc(cfg_path)
    base_cfg = load_jsonc(REPO / cfg["base_config"])
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    guard_cfg = cfg.get("memory_guard", {})
    threshold_mib = float(guard_cfg["stop_on_peak_memory_mib"]) if "stop_on_peak_memory_mib" in guard_cfg else None

    results: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        name = variant["name"]
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_run = _deep_merge(base_cfg, variant["overrides"])
        config_path = run_dir / "song_2023.variant.jsonc"
        config_path.write_text(json.dumps(_json_safe(cfg_run), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        env = dict(**subprocess.os.environ)
        env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"

        cmd = [
            str(REPO / ".venv" / "bin" / "python"),
            "-m",
            "scripts.validation.song_2023.song_2023",
            "--config",
            str(config_path),
        ]
        print(f"[song-order-preflight] {name}")
        matrix_runner._clear_canonical_outputs()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        target_npz, target_log = matrix_runner._collect_outputs(run_dir)

        result: dict[str, Any] = {
            "variant": name,
            "returncode": int(proc.returncode),
            "status": "ok" if proc.returncode == 0 and target_log.exists() else "failed",
            "stdout_log": str(stdout_path.relative_to(REPO)),
            "stderr_log": str(stderr_path.relative_to(REPO)),
            "config_path": str(config_path.relative_to(REPO)),
        }
        if result["status"] == "ok":
            summary = matrix_runner._summarize(target_npz, target_log)
            result["summary"] = summary
            if _memory_limit_hit(summary, threshold_mib):
                result["stop_reason"] = (
                    f"peak_memory_mib={summary['peak_memory_mib']} >= "
                    f"stop_on_peak_memory_mib={threshold_mib}"
                )
                results.append(result)
                print(f"[song-order-preflight] stopping after {name}: {result['stop_reason']}")
                break
        else:
            result["error"] = (proc.stderr or proc.stdout).splitlines()[:20]
            results.append(result)
            print(f"[song-order-preflight] stopping after {name}: run failed")
            break
        results.append(result)

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "status": item["status"],
            "returncode": item["returncode"],
            "stdout_log": item["stdout_log"],
            "stderr_log": item["stderr_log"],
            "stop_reason": item.get("stop_reason", ""),
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
