#!/usr/bin/env python3
"""Audit spectrum consistency for the Song/Ziming validation case.

This utility compares three spectrum sources that currently coexist in the
repository:

1. MATLAB reference temporal spectrum saved in ``matlab_reference/EigModes``
2. Local ``ref_results/eigenvalues_Ziming.dat`` spectrum
3. Optionally, the Python ``song_2023.npz`` spectrum if present

The main goal is to make normalization/scaling mismatches explicit.

Usage from repo root:

    PYTHONPATH=src python -m validation.song_2023.compare_spectra
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchReport:
    """Pairwise spectrum comparison summary."""

    label: str
    count_reference: int
    count_candidate: int
    mean_abs_error: float
    max_abs_error: float
    rms_abs_error: float
    mean_rel_error: float
    max_rel_error: float
    optimal_real_scale: float
    optimal_complex_scale: complex

    def as_lines(self) -> list[str]:
        return [
            f"[{self.label}]",
            f"  reference count       : {self.count_reference}",
            f"  candidate count       : {self.count_candidate}",
            f"  mean abs error        : {self.mean_abs_error:.6e}",
            f"  rms abs error         : {self.rms_abs_error:.6e}",
            f"  max abs error         : {self.max_abs_error:.6e}",
            f"  mean rel error        : {self.mean_rel_error:.6e}",
            f"  max rel error         : {self.max_rel_error:.6e}",
            f"  optimal real scale    : {self.optimal_real_scale:.12g}",
            (
                "  optimal complex scale : "
                f"{self.optimal_complex_scale.real:.12g}"
                f"{self.optimal_complex_scale.imag:+.12g}j"
            ),
        ]


def _load_complex_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected two-column real/imag file at {path}")
    return np.asarray(data[:, 0] + 1j * data[:, 1], dtype=complex)


def _load_matlab_spectrum(path: Path, key: str = "omega") -> np.ndarray:
    data = loadmat(path)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in {path}")
    arr = np.asarray(data[key])
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        offdiag = arr - np.diag(np.diag(arr))
        if np.allclose(offdiag, 0.0):
            return np.asarray(np.diag(arr), dtype=complex)
    return np.asarray(arr.reshape(-1), dtype=complex)


def _load_python_npz(path: Path, key: str = "omega") -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in {path}")
    return np.asarray(data[key]).reshape(-1).astype(complex)


def _format_sample(values: Iterable[complex], n: int = 5) -> str:
    arr = np.asarray(list(values), dtype=complex)
    if arr.size == 0:
        return "[]"
    sample = ", ".join(f"{z.real:.6g}{z.imag:+.6g}j" for z in arr[:n])
    suffix = ", ..." if arr.size > n else ""
    return f"[{sample}{suffix}]"


def _optimal_scales(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, complex]:
    denom = float(np.vdot(candidate, candidate).real)
    if denom <= 0.0:
        return float("nan"), complex(np.nan, np.nan)

    complex_scale = np.vdot(candidate, reference) / np.vdot(candidate, candidate)
    real_scale = float(np.real(np.vdot(candidate, reference)) / denom)
    return real_scale, complex_scale


def compare_spectra(reference: np.ndarray, candidate: np.ndarray, label: str) -> MatchReport:
    """Compare spectra with one-to-one global best matching."""
    reference = np.asarray(reference, dtype=complex).reshape(-1)
    candidate = np.asarray(candidate, dtype=complex).reshape(-1)
    if reference.size == 0 or candidate.size == 0:
        raise ValueError("Both spectra must contain at least one value")

    cost = np.abs(reference[:, None] - candidate[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    ref_sel = reference[row_ind]
    cand_sel = candidate[col_ind]
    abs_err = np.abs(ref_sel - cand_sel)
    rel_err = abs_err / np.maximum(np.abs(ref_sel), 1e-30)
    real_scale, complex_scale = _optimal_scales(ref_sel, cand_sel)

    return MatchReport(
        label=label,
        count_reference=int(reference.size),
        count_candidate=int(candidate.size),
        mean_abs_error=float(np.mean(abs_err)),
        max_abs_error=float(np.max(abs_err)),
        rms_abs_error=float(np.sqrt(np.mean(abs_err**2))),
        mean_rel_error=float(np.mean(rel_err)),
        max_rel_error=float(np.max(rel_err)),
        optimal_real_scale=real_scale,
        optimal_complex_scale=complex_scale,
    )


def build_audit_report(
    *,
    matlab_spectrum_path: Path,
    ziming_dat_path: Path,
    python_npz_path: Path | None = None,
    ziming_scale: float = 80.0,
) -> str:
    matlab = _load_matlab_spectrum(matlab_spectrum_path)
    ziming_raw = _load_complex_txt(ziming_dat_path)
    ziming_scaled = ziming_raw * ziming_scale

    lines = [
        "Song/Ziming spectrum audit",
        f"  MATLAB spectrum path  : {matlab_spectrum_path}",
        f"  Ziming dat path       : {ziming_dat_path}",
        f"  Ziming scale applied  : {ziming_scale}",
        f"  MATLAB sample         : {_format_sample(matlab)}",
        f"  Ziming raw sample     : {_format_sample(ziming_raw)}",
        f"  Ziming scaled sample  : {_format_sample(ziming_scaled)}",
        "",
    ]

    reports = [
        compare_spectra(matlab, ziming_raw, "MATLAB vs Ziming raw"),
        compare_spectra(matlab, ziming_scaled, f"MATLAB vs Ziming scaled x{ziming_scale:g}"),
    ]

    if python_npz_path is not None and python_npz_path.exists():
        python_omega = _load_python_npz(python_npz_path)
        lines.extend(
            [
                f"  Python npz path       : {python_npz_path}",
                f"  Python spectrum sample: {_format_sample(python_omega)}",
                "",
            ]
        )
        reports.extend(
            [
                compare_spectra(matlab, python_omega, "MATLAB vs Python npz"),
                compare_spectra(python_omega, ziming_raw, "Python npz vs Ziming raw"),
                compare_spectra(
                    python_omega,
                    ziming_scaled,
                    f"Python npz vs Ziming scaled x{ziming_scale:g}",
                ),
            ]
        )

    for report in reports:
        lines.extend(report.as_lines())
        lines.append("")

    if reports[1].mean_abs_error < reports[0].mean_abs_error:
        lines.append(
            "Conclusion: the scaled Ziming spectrum is closer to the MATLAB reference than the raw file,"
        )
        lines.append(
            "which strongly indicates that the local .dat reference uses a different normalization."
        )
    else:
        lines.append(
            "Conclusion: scaling the Ziming spectrum did not improve the match; inspect ordering and reference provenance."
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / "validation" / "song_2023"
    return argparse.ArgumentParser(description="Audit Song/Ziming spectrum conventions.").parse_args(
        []
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / "validation" / "song_2023"

    parser = argparse.ArgumentParser(description="Audit Song/Ziming spectrum conventions.")
    parser.add_argument(
        "--matlab-spectrum",
        type=Path,
        default=base_dir / "matlab_reference" / "EigModes" / "spectrum.mat",
        help="Path to MATLAB spectrum.mat file.",
    )
    parser.add_argument(
        "--ziming-dat",
        type=Path,
        default=base_dir / "ref_results" / "eigenvalues_Ziming.dat",
        help="Path to local Ziming two-column eigenvalue file.",
    )
    parser.add_argument(
        "--python-npz",
        type=Path,
        default=base_dir / "song_2023.npz",
        help="Optional Python output npz to compare when present.",
    )
    parser.add_argument(
        "--ziming-scale",
        type=float,
        default=80.0,
        help="Scale factor applied to the local Ziming dat values before comparison.",
    )
    args = parser.parse_args()

    report = build_audit_report(
        matlab_spectrum_path=args.matlab_spectrum,
        ziming_dat_path=args.ziming_dat,
        python_npz_path=args.python_npz,
        ziming_scale=args.ziming_scale,
    )
    print(report)


if __name__ == "__main__":
    main()
