#!/usr/bin/env python3
"""Run Song2023 fd-order convergence study — serial full eigensolves.

Executes fd2, fd4, fd6 eigensolves sequentially with a memory guard.
Produces a summary JSON and CSV suitable for populating the paper's
scaling table (tab:scaling).

Usage (from repo root):
    PYTHONPATH=src:. .venv/bin/python -m scripts.validation.song_2023.scripts.run_song_2023_order_convergence
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from linstabpy.utils.jsonc import load_jsonc

from . import run_song_2023_matrix as matrix_runner
from .run_song_2023_order_preflight import _memory_limit_hit


SCRIPT = Path(__file__).resolve()
SONG_DIR = SCRIPT.parents[1]
REPO = SCRIPT.parents[3]
DEFAULT_CONFIG = SONG_DIR / "configs" / "order_convergence.jsonc"


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


def _compare_with_matlab(npz_path: Path) -> dict[str, Any]:
    """Compare eigenvalues from an NPZ with the MATLAB reference spectrum."""
    matlab_ref = REPO / "scripts/validation/song_2023/reference/matlab/matlab_reference/EigModes/spectrum.mat"
    if not npz_path.exists():
        return {"status": "missing_npz"}

    data = np.load(npz_path, allow_pickle = False)
    if "omega" not in data:
        return {"status": "no_omega_in_npz"}

    omega_py = data["omega"]

    try:
        import scipy.io as sio
        mat = sio.loadmat(str(matlab_ref))
        omega_ml = mat["omega"].ravel()
    except Exception as exc:
        return {"status": f"matlab_load_error: {exc}"}

    # Nearest-neighbour distance: for each Python eigenvalue, distance to
    # the closest MATLAB eigenvalue (and vice versa)
    from scipy.spatial.distance import cdist
    pts_py = np.column_stack([omega_py.real, omega_py.imag])
    pts_ml = np.column_stack([omega_ml.real, omega_ml.imag])
    dist_py2ml = cdist(pts_py, pts_ml).min(axis=1)

    # Leading mode (largest Im(omega))
    idx_lead = np.argmax(omega_py.imag)
    omega_lead = omega_py[idx_lead]
    idx_ml_lead = np.argmax(omega_ml.imag)
    omega_ml_lead = omega_ml[idx_ml_lead]

    return {
        "n_modes_python": int(omega_py.size),
        "n_modes_matlab": int(omega_ml.size),
        "mean_nearest_matlab": float(np.mean(dist_py2ml)),
        "max_nearest_matlab": float(np.max(dist_py2ml)),
        "leading_eigenvalue": {
            "python_re": float(omega_lead.real),
            "python_im": float(omega_lead.imag),
            "matlab_re": float(omega_ml_lead.real),
            "matlab_im": float(omega_ml_lead.imag),
            "im_diff_pct": float(
                abs(omega_lead.imag - omega_ml_lead.imag)
                / max(abs(omega_ml_lead.imag), 1e-15)
                * 100
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to order-convergence JSONC config.",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_jsonc(cfg_path)
    base_cfg = load_jsonc(REPO / cfg["base_config"])
    output_root = REPO / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    guard_cfg = cfg.get("memory_guard", {})
    threshold_mib = (
        float(guard_cfg["stop_on_peak_memory_mib"])
        if "stop_on_peak_memory_mib" in guard_cfg
        else None
    )

    results: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        name = variant["name"]
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_run = _deep_merge(base_cfg, variant["overrides"])
        config_path = run_dir / "song_2023.variant.jsonc"
        config_path.write_text(
            json.dumps(_json_safe(cfg_run), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        env = dict(**subprocess.os.environ)
        env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"

        cmd = [
            str(REPO / ".venv" / "bin" / "python"),
            "-m",
            "scripts.validation.song_2023.song_2023",
            "--config",
            str(config_path),
        ]
        print(f"\n[order-convergence] ─── {name} ───")
        matrix_runner._clear_canonical_outputs()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        target_npz, target_log = matrix_runner._collect_outputs(run_dir)

        result: dict[str, Any] = {
            "variant": name,
            "fd_order": variant["overrides"]["fd_order"],
            "returncode": int(proc.returncode),
            "status": "ok" if proc.returncode == 0 and target_log.exists() else "failed",
        }

        if result["status"] == "ok":
            summary = matrix_runner._summarize(target_npz, target_log)
            result["summary"] = summary
            result["comparison"] = _compare_with_matlab(target_npz)

            if _memory_limit_hit(summary, threshold_mib):
                result["stop_reason"] = (
                    f"peak_memory_mib={summary.get('peak_memory_mib')} >= "
                    f"stop_on_peak_memory_mib={threshold_mib}"
                )
                results.append(result)
                print(f"[order-convergence] stopping: {result['stop_reason']}")
                break
        else:
            result["error"] = (proc.stderr or proc.stdout).splitlines()[:20]
            results.append(result)
            print(f"[order-convergence] stopping: run failed (returncode={proc.returncode})")
            break

        results.append(result)

    # Write summary
    summary_json = output_root / "order_convergence_summary.json"
    summary_json.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # CSV for quick inspection
    summary_csv = output_root / "order_convergence_summary.csv"
    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "fd_order": item.get("fd_order", ""),
            "status": item["status"],
        }
        for key, value in item.get("summary", {}).items():
            if not isinstance(value, dict):
                row[key] = _json_safe(value)
        comp = item.get("comparison", {})
        row["mean_nearest_matlab"] = comp.get("mean_nearest_matlab", "")
        row["max_nearest_matlab"] = comp.get("max_nearest_matlab", "")
        lead = comp.get("leading_eigenvalue", {})
        row["lead_im_python"] = lead.get("python_im", "")
        row["lead_im_matlab"] = lead.get("matlab_im", "")
        row["lead_im_diff_pct"] = lead.get("im_diff_pct", "")
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[order-convergence] JSON: {summary_json.relative_to(REPO)}")
    print(f"[order-convergence] CSV:  {summary_csv.relative_to(REPO)}")

    # Print quick summary table
    print("\n┌─────────┬────────────┬────────────┬───────────────────┐")
    print("│ fd_order │ time (s)   │ peak (MiB) │ mean_dist(MATLAB) │")
    print("├─────────┼────────────┼────────────┼───────────────────┤")
    for item in results:
        if item["status"] != "ok":
            continue
        fd = item.get("fd_order", "?")
        t = item.get("summary", {}).get("total_time_s", "?")
        mem = item.get("summary", {}).get("peak_memory_mib", "?")
        dist = item.get("comparison", {}).get("mean_nearest_matlab", "?")
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else str(t)
        mem_str = f"{mem:.0f}" if isinstance(mem, (int, float)) else str(mem)
        dist_str = f"{dist:.4e}" if isinstance(dist, (int, float)) else str(dist)
        print(f"│ fd{fd:<6}│ {t_str:>10} │ {mem_str:>10} │ {dist_str:>17} │")
    print("└─────────┴────────────┴────────────┴───────────────────┘")


if __name__ == "__main__":
    main()
