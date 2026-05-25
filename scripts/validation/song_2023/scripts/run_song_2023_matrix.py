#!/usr/bin/env python3
"""Run Song2023 validation variants from JSONC without touching canonical outputs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from linstabpy.utils.jsonc import load_jsonc


SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parents[1]
REPO = SCRIPT.parents[3]
CONFIG_PATH = SONG_DIR / "configs" / "song_2023_matrix.jsonc"
CANONICAL_OUTPUTS = (
    SONG_DIR / "song_2023.npz",
    SONG_DIR / "song_2023.log",
    SONG_DIR / "song_2023.png",
    SONG_DIR / "song_2023_baseflow.png",
    SONG_DIR / "song_2023_support_direct_modes.png",
    SONG_DIR / "song_2023_resolvent.png",
)

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


def _summarize(npz_path: Path, log_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    summary: dict[str, Any] = {}
    if "omega" in data:
        omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
        summary["n_omega"] = int(omega.size)
        if omega.size:
            summary["nearest_to_target"] = {
                "real": float(np.real(omega[0])),
                "imag": float(np.imag(omega[0])),
            }
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        step_summary: dict[str, dict[str, float]] = {}
        for line in lines:
            if line.startswith("total_time_s="):
                summary["total_time_s"] = float(line.split("=", 1)[1])
            elif line.startswith("peak_memory_mib="):
                summary["peak_memory_mib"] = float(line.split("=", 1)[1])
            elif line.startswith("step_summary "):
                # Format: step_summary <name> <time_min> <time_mean> <time_max>
                #         <mem_min> <mem_mean> <mem_max> <count>
                parts = line.split()
                if len(parts) >= 9:
                    step_name = parts[1]
                    step_summary[step_name] = {
                        "time_min": float(parts[2]),
                        "time_mean": float(parts[3]),
                        "time_max": float(parts[4]),
                        "memory_min": float(parts[5]),
                        "memory_mean": float(parts[6]),
                        "memory_max": float(parts[7]),
                        "count": int(parts[8]),
                    }
        if step_summary:
            summary["step_summary"] = step_summary
    return summary


def _clear_canonical_outputs() -> None:
    """Remove canonical Song outputs so each matrix variant recomputes cleanly."""
    for path in CANONICAL_OUTPUTS:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _collect_outputs(run_dir: Path) -> tuple[Path, Path]:
    """Move canonical Song outputs into the variant directory."""
    npz_path = SONG_DIR / "song_2023.npz"
    log_path = SONG_DIR / "song_2023.log"
    target_npz = run_dir / "song_2023.npz"
    target_log = run_dir / "song_2023.log"
    if npz_path.exists():
        npz_path.replace(target_npz)
    if log_path.exists():
        log_path.replace(target_log)

    for path in CANONICAL_OUTPUTS:
        if not path.exists() or path.suffix != ".png":
            continue
        path.replace(run_dir / path.name)

    return target_npz, target_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to a Song matrix JSONC file (default: song_2023_matrix.jsonc).",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Apply reduced_overrides from song_2023_matrix.jsonc for a faster diagnostic run.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated Song matrix variant names to run (default: all).",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_jsonc(cfg_path)
    base_cfg = load_jsonc(REPO / cfg["base_config"])
    if args.reduced and "reduced_overrides" in cfg:
        base_cfg = _deep_merge(base_cfg, cfg["reduced_overrides"])
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}

    results: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        if selected_variants and variant["name"] not in selected_variants:
            continue
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
        print(f"[song-matrix] {name}")
        _clear_canonical_outputs()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        target_npz, target_log = _collect_outputs(run_dir)

        result: dict[str, Any] = {
            "variant": name,
            "status": "ok" if proc.returncode == 0 and target_npz.exists() else "failed",
            "returncode": int(proc.returncode),
            "stdout_log": str(stdout_path.relative_to(REPO)),
            "stderr_log": str(stderr_path.relative_to(REPO)),
            "output_npz": str(target_npz.relative_to(REPO)),
            "config_path": str(config_path.relative_to(REPO)),
        }
        if result["status"] == "ok":
            result["summary"] = _summarize(target_npz, target_log)
        else:
            result["error"] = (proc.stderr or proc.stdout).splitlines()[:20]
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
            "output_npz": item["output_npz"],
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

    n_ok = sum(item["status"] == "ok" for item in results)
    n_fail = len(results) - n_ok
    mode_label = "reduced" if args.reduced else "full"
    print(f"Song2023 matrix summary ({mode_label}): ok={n_ok}, failed={n_fail}")
    print(f"JSON summary: {summary_json.relative_to(REPO)}")
    print(f"CSV summary:  {summary_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
