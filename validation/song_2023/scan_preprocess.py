#!/usr/bin/env python3
"""Scan reduced Song 2023 preprocessing branches.

This utility runs a small set of stability-only reduced cases for the Song 2023
validation problem and compares them with the available MATLAB / Song spectrum
references. The goal is to cheaply identify candidate preprocessing branches
that are numerically stable and not obviously worse than the raw baseline.

Usage from repo root:

    PYTHONPATH=src:. .venv/bin/python -m validation.song_2023.scan_preprocess
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from linstab2d.utils.jsonc import load_jsonc

from .song_2023 import DEFAULT_CONFIG_PATH, _resolve_frequency_target, run_case


@dataclass(frozen=True)
class CaseSpec:
    name: str
    preprocess_cfg: dict[str, Any]
    description: str


def _filter_case(
    *,
    name: str,
    description: str,
    directions: list[str],
    eta_min_index: int,
    fields: list[str] | None = None,
    passes: int = 1,
) -> CaseSpec:
    return CaseSpec(
        name=name,
        preprocess_cfg={
            "enabled": True,
            "sponge": {"scale": 1.0, "boundary": {"enabled": False}},
            "filter": {
                "enabled": True,
                "passes": passes,
                "fields": list(fields or ["rho", "u", "v", "T"]),
                "directions": list(directions),
                "eta_min_index": eta_min_index,
                "eta_max_index": None,
            },
            "shock": {"enabled": False},
        },
        description=description,
    )


def _sponge_case(
    *,
    name: str,
    description: str,
    points: list[int],
    amplitudes: list[float],
) -> CaseSpec:
    return CaseSpec(
        name=name,
        preprocess_cfg={
            "enabled": True,
            "sponge": {
                "scale": 1.0,
                "boundary": {
                    "enabled": True,
                    "mode": "add",
                    "points": list(points),
                    "amplitudes": list(amplitudes),
                },
            },
            "filter": {
                "enabled": False,
                "passes": 0,
                "fields": ["rho", "u", "v", "T"],
                "directions": ["x", "y"],
                "eta_min_index": 0,
                "eta_max_index": None,
            },
            "shock": {"enabled": False},
        },
        description=description,
    )


@dataclass
class CaseResult:
    name: str
    description: str
    success: bool
    preprocess_summary: str
    lead_omega: complex | None = None
    mean_nearest_matlab: float | None = None
    mean_nearest_song: float | None = None
    rough_x: float | None = None
    rough_y: float | None = None
    peak_memory_mib: float | None = None
    total_time_s: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    output_npz: str | None = None
    output_log: str | None = None

    def score(self, raw_baseline: "CaseResult | None") -> float:
        if not self.success:
            return float("inf")
        if raw_baseline is None or not raw_baseline.success:
            parts = [self.mean_nearest_matlab, self.mean_nearest_song, self.rough_x, self.rough_y]
            return float(sum(v for v in parts if v is not None))
        numerators = [
            (self.mean_nearest_matlab, raw_baseline.mean_nearest_matlab),
            (self.mean_nearest_song, raw_baseline.mean_nearest_song),
            (self.rough_x, raw_baseline.rough_x),
            (self.rough_y, raw_baseline.rough_y),
        ]
        total = 0.0
        used = 0
        for value, base in numerators:
            if value is None or base is None:
                continue
            total += value / max(base, 1e-30)
            used += 1
        return float("inf") if used == 0 else total / used


def _load_reference_spectra(base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(base_dir / "matlab_reference" / "EigModes" / "spectrum.mat")["omega"]
    omega_mat = np.diag(mat) if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else mat.reshape(-1)
    omega_mat = np.asarray(omega_mat, dtype=complex).reshape(-1)

    song_vals = np.loadtxt(base_dir / "ref_results" / "eigenvalues_Ziming.dat")
    omega_song = 80.0 * (song_vals[:, 0] + 1j * song_vals[:, 1])
    return omega_mat, np.asarray(omega_song, dtype=complex).reshape(-1)


def _parse_log_metrics(log_path: Path) -> tuple[float | None, float | None]:
    peak_memory_mib = None
    total_time_s = None
    if not log_path.exists():
        return peak_memory_mib, total_time_s
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("peak_memory_mib="):
            peak_memory_mib = float(line.split("=", 1)[1])
        elif line.startswith("total_time_s="):
            total_time_s = float(line.split("=", 1)[1])
    return peak_memory_mib, total_time_s


def _load_case_metrics(npz_path: Path, log_path: Path, omega_target: complex) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
        modes = np.asarray(data["direct_modes"], dtype=complex)
        X = np.asarray(data["X"], dtype=float)
        shape = X.shape
        ngp = shape[0] * shape[1]
        u0 = modes[ngp : 2 * ngp, 0].reshape(shape, order="F")
        rough_x = np.linalg.norm(np.diff(np.real(u0), axis=1)) / max(np.linalg.norm(np.real(u0)), 1e-30)
        rough_y = np.linalg.norm(np.diff(np.real(u0), axis=0)) / max(np.linalg.norm(np.real(u0)), 1e-30)
        preprocess_summary = str(data["preprocess_summary"])

    peak_memory_mib, total_time_s = _parse_log_metrics(log_path)
    lead_omega = omega[int(np.argmin(np.abs(omega - omega_target)))] if omega.size else None
    return {
        "omega": omega,
        "rough_x": float(rough_x),
        "rough_y": float(rough_y),
        "lead_omega": lead_omega,
        "preprocess_summary": preprocess_summary,
        "peak_memory_mib": peak_memory_mib,
        "total_time_s": total_time_s,
    }


def _case_specs() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="raw",
            preprocess_cfg={"enabled": False},
            description="No preprocessing baseline",
        ),
        _filter_case(
            name="filter_y_eta8",
            description="One y-only pass above eta index 8",
            directions=["y"],
            eta_min_index=8,
        ),
        _filter_case(
            name="filter_y_eta12",
            description="One y-only pass above eta index 12",
            directions=["y"],
            eta_min_index=12,
        ),
        _filter_case(
            name="filter_y_eta16",
            description="One y-only pass above eta index 16",
            directions=["y"],
            eta_min_index=16,
        ),
        _filter_case(
            name="filter_y_eta20",
            description="One y-only pass above eta index 20",
            directions=["y"],
            eta_min_index=20,
        ),
        _filter_case(
            name="filter_uvT_y_eta12",
            description="One y-only pass on u,v,T above eta index 12",
            directions=["y"],
            eta_min_index=12,
            fields=["u", "v", "T"],
        ),
        _filter_case(
            name="filter_xy_eta16",
            description="One xy pass higher above wall (eta>=16)",
            directions=["x", "y"],
            eta_min_index=16,
        ),
        _filter_case(
            name="filter_xy_eta20",
            description="One xy pass higher above wall (eta>=20)",
            directions=["x", "y"],
            eta_min_index=20,
        ),
        _sponge_case(
            name="sponge_very_weak_top_right",
            description="Very weak top/right boundary sponge",
            points=[2, 2, 0, 0],
            amplitudes=[0.02, 0.02, 0.0, 0.0],
        ),
    ]


def run_scan(
    *,
    config_path: Path,
    stride_ksi: int,
    stride_eta: int,
    n_modes: int,
) -> list[CaseResult]:
    base_dir = config_path.parent
    cfg = load_jsonc(config_path)
    omega_target, _ = _resolve_frequency_target(cfg)
    omega_mat, omega_song = _load_reference_spectra(base_dir)

    results: list[CaseResult] = []
    for spec in _case_specs():
        out_path = base_dir / f"song_2023_scan_{spec.name}_s{stride_ksi}x{stride_eta}.npz"
        log_path = out_path.with_suffix(".log")
        for path in (out_path, log_path):
            if path.exists():
                path.unlink()

        case_cfg = copy.deepcopy(cfg)
        case_cfg["baseflow_preprocess"] = copy.deepcopy(spec.preprocess_cfg)
        try:
            run_case(
                case_dir=str(base_dir),
                Re=case_cfg["Re"],
                Ma=case_cfg["Ma"],
                Pr=case_cfg["Pr"],
                kappa=case_cfg["kappa"],
                T_0=case_cfg["T_0"],
                beta=case_cfg["beta"],
                St=float(case_cfg.get("St", 0.0)),
                omega_target=omega_target,
                n_modes_stability=n_modes,
                n_modes_resolvent=0,
                run_stability=True,
                run_resolvent=False,
                fd_order=case_cfg["fd_order"],
                stride_ksi=stride_ksi,
                stride_eta=stride_eta,
                penalty=case_cfg["penalty"],
                alpha_filter=case_cfg.get("alpha_filter", "none"),
                output_npz=str(out_path),
                bc_recipe=case_cfg.get("bc", {}).get("recipe"),
                solver_cfg=case_cfg.get("solver"),
                preprocess_cfg=case_cfg.get("baseflow_preprocess"),
            )
            metrics = _load_case_metrics(out_path, log_path, omega_target)
            omega = metrics.pop("omega")
            dmat = [float(np.min(np.abs(omega_mat - v))) for v in omega]
            dsong = [float(np.min(np.abs(omega_song - v))) for v in omega]
            results.append(
                CaseResult(
                    name=spec.name,
                    description=spec.description,
                    success=True,
                    preprocess_summary=str(metrics["preprocess_summary"]),
                    lead_omega=metrics["lead_omega"],
                    mean_nearest_matlab=float(sum(dmat) / len(dmat)),
                    mean_nearest_song=float(sum(dsong) / len(dsong)),
                    rough_x=metrics["rough_x"],
                    rough_y=metrics["rough_y"],
                    peak_memory_mib=metrics["peak_memory_mib"],
                    total_time_s=metrics["total_time_s"],
                    output_npz=str(out_path),
                    output_log=str(log_path),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic utility path
            results.append(
                CaseResult(
                    name=spec.name,
                    description=spec.description,
                    success=False,
                    preprocess_summary=spec.preprocess_cfg.get("enabled") and "failed during solve" or "disabled",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    output_npz=str(out_path),
                    output_log=str(log_path),
                )
            )
    return results


def _format_complex(z: complex | None) -> str:
    if z is None:
        return "-"
    return f"{z.real:+.3e}{z.imag:+.3e}j"


def _print_report(results: list[CaseResult]) -> None:
    raw = next((result for result in results if result.name == "raw"), None)
    ranked = sorted(results, key=lambda result: (not result.success, result.score(raw)))

    print("Song reduced preprocessing scan")
    print("name                     status   score   mean|d_mat|  mean|d_song|  rough_x  rough_y  peakMiB  t[s]    lead_omega")
    for result in ranked:
        status = "ok" if result.success else "fail"
        score = result.score(raw)
        score_txt = f"{score:6.3f}" if np.isfinite(score) else "  inf "
        print(
            f"{result.name:24s} {status:6s} {score_txt} "
            f"{(result.mean_nearest_matlab if result.mean_nearest_matlab is not None else float('nan')):11.4e} "
            f"{(result.mean_nearest_song if result.mean_nearest_song is not None else float('nan')):12.4e} "
            f"{(result.rough_x if result.rough_x is not None else float('nan')):8.4f} "
            f"{(result.rough_y if result.rough_y is not None else float('nan')):8.4f} "
            f"{(result.peak_memory_mib if result.peak_memory_mib is not None else float('nan')):8.1f} "
            f"{(result.total_time_s if result.total_time_s is not None else float('nan')):6.3f} "
            f"{_format_complex(result.lead_omega)}"
        )
    print()
    for result in ranked:
        print(f"[{result.name}] {result.description}")
        print(f"  success            : {result.success}")
        print(f"  preprocess_summary : {result.preprocess_summary}")
        if result.success:
            print(f"  output_npz         : {result.output_npz}")
            print(f"  output_log         : {result.output_log}")
        else:
            print(f"  error_type         : {result.error_type}")
            print(f"  error_message      : {result.error_message}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan reduced Song preprocessing branches.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Song JSONC configuration path.",
    )
    parser.add_argument("--stride-ksi", type=int, default=12, help="Reduced streamwise stride.")
    parser.add_argument("--stride-eta", type=int, default=12, help="Reduced wall-normal stride.")
    parser.add_argument("--n-modes", type=int, default=6, help="Number of temporal modes to extract.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_scan(
        config_path=args.config,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
        n_modes=args.n_modes,
    )
    _print_report(results)


if __name__ == "__main__":
    main()
