#!/usr/bin/env python3
"""Validation driver for manuscript tables/figures.

Runs the configured example modules, checks that expected artifacts were
produced, and writes ``validation_summary.json`` with run metadata and checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "validation_cases.yaml"
DEFAULT_SUMMARY = REPO_ROOT / "validation_summary.json"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _load_cases_config(path: Path) -> dict:
    """Load the validation case registry.

    The file extension is ``.yaml``, but contents are YAML-compatible JSON so
    the loader is dependency-free.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return data


def _run_case(module: str, env_overrides: dict[str, str] | None = None) -> tuple[int, float]:
    start = time.perf_counter()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed


def _weighted_orthogonality_check(npz_path: Path, mode_key: str, weights_key: str, tol: float) -> dict:
    if not npz_path.exists():
        return {
            "status": "missing_artifact",
            "artifact": str(npz_path),
            "expression": "|v^H W v - 1|",
            "pass": False,
        }

    data = np.load(npz_path, allow_pickle = False)
    if mode_key not in data or weights_key not in data:
        return {
            "status": "missing_fields",
            "artifact": str(npz_path),
            "mode_key": mode_key,
            "weights_key": weights_key,
            "expression": "|v^H W v - 1|",
            "pass": False,
        }

    modes = np.asarray(data[mode_key])
    w_diag = np.asarray(data[weights_key]).ravel()
    if modes.ndim != 2 or modes.shape[1] == 0 or modes.shape[0] != w_diag.size:
        return {
            "status": "invalid_shapes",
            "artifact": str(npz_path),
            "modes_shape": list(modes.shape),
            "weights_size": int(w_diag.size),
            "expression": "|v^H W v - 1|",
            "pass": False,
        }

    v = modes[:, 0]
    # Equivalent to np.abs(vh @ W @ v - 1) with vh = v.conj().T and diagonal W.
    error = float(np.abs(np.vdot(v, w_diag * v) - 1.0))
    return {
        "status": "ok",
        "artifact": str(npz_path),
        "expression": "|v^H W v - 1|",
        "value": error,
        "tolerance": float(tol),
        "pass": bool(error < tol),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="matlab_export/", help="MATLAB reference directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Validation case registry")
    parser.add_argument("--output", default=str(DEFAULT_SUMMARY), help="Summary JSON output")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    ref_path = (REPO_ROOT / args.ref).resolve() if not Path(args.ref).is_absolute() else Path(args.ref)

    cfg = _load_cases_config(config_path)
    cases = cfg.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases configured in {config_path}")

    summary: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "reference_dir": str(ref_path),
        "config_file": str(config_path),
        "cases": [],
    }

    for case in cases:
        name = str(case.get("name", "unnamed_case"))
        module = str(case.get("module", "")).strip()
        artifacts = [str(p) for p in case.get("artifacts", [])]
        env_overrides = {str(k): str(v) for k, v in dict(case.get("env", {})).items()}
        if not module:
            summary["cases"].append(
                {
                    "name": name,
                    "module": module,
                    "status": "invalid_config",
                    "artifacts": artifacts,
                }
            )
            continue

        rc, elapsed = _run_case(module, env_overrides=env_overrides)
        artifact_status = []
        all_artifacts_present = True
        for rel in artifacts:
            path = REPO_ROOT / rel
            exists = path.exists()
            artifact_status.append({"path": rel, "exists": bool(exists)})
            all_artifacts_present = all_artifacts_present and exists

        case_status = "ok" if rc == 0 and all_artifacts_present else "failed"
        summary["cases"].append(
            {
                "name": name,
                "module": module,
                "returncode": int(rc),
                "elapsed_seconds": float(elapsed),
                "env": env_overrides,
                "artifacts": artifact_status,
                "status": case_status,
            }
        )

    orth_cfg = cfg.get("orthogonality_check")
    if isinstance(orth_cfg, dict):
        orth = _weighted_orthogonality_check(
            npz_path=REPO_ROOT / str(orth_cfg.get("artifact", "example_6_hamada.npz")),
            mode_key=str(orth_cfg.get("mode_key", "direct_modes")),
            weights_key=str(orth_cfg.get("weights_key", "W_diag")),
            tol=float(orth_cfg.get("tolerance", 1e-12)),
        )
        summary["weighted_mode_orthogonality"] = orth

    summary["all_pass"] = all(
        c.get("status") == "ok" for c in summary["cases"]
    ) and bool(summary.get("weighted_mode_orthogonality", {}).get("pass", True))

    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validation summary written to {output_path}")
    print(f"All pass: {summary['all_pass']}")


if __name__ == "__main__":
    main()
