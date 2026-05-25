#!/usr/bin/env python3
"""Plot currently available Song2023 spectra against the MATLAB reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
SCALING_RUNS = REPO / "scripts/validation/song_2023/scaling_runs"
MATLAB_SPECTRUM = (
    REPO / "scripts/validation/song_2023/reference/matlab/matlab_reference/EigModes/spectrum.mat"
)
OUTPUT_PNG = SCRIPT.with_name("song_spectrum_progress.png")


def _load_matlab_omega(path: Path) -> np.ndarray:
    data = loadmat(path)
    omega = np.asarray(data["omega"], dtype=complex)
    if omega.ndim == 2 and omega.shape[0] == omega.shape[1]:
        return np.diag(omega)
    return omega.reshape(-1)


def _load_total_time_s(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("total_time_s="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _available_runs() -> list[tuple[str, float | None, np.ndarray]]:
    runs: list[tuple[str, float | None, np.ndarray]] = []
    for npz_path in sorted(SCALING_RUNS.glob("*/song_2023.npz")):
        runtime = _load_total_time_s(npz_path.with_suffix(".log"))
        label = npz_path.parent.name
        with np.load(npz_path, allow_pickle=False) as data:
            if "omega" not in data:
                continue
            runs.append((label, runtime, np.asarray(data["omega"], dtype=complex).reshape(-1)))
    return runs


def _fd_from_label(label: str) -> int | None:
    head = label.split("_", 1)[0]
    if not head.startswith("fd"):
        return None
    try:
        return int(head[2:])
    except ValueError:
        return None


def main() -> None:
    matlab = _load_matlab_omega(MATLAB_SPECTRUM)
    runs = _available_runs()
    if not runs:
        raise SystemExit("No completed Song scaling spectra found yet.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(
        np.real(matlab), np.imag(matlab), c="k", marker="x", s=18, alpha=0.6, label="MATLAB ref"
    )

    fd_styles = {
        2: ("#1f77b4", "o"),
        4: ("#ff7f0e", "s"),
        6: ("#2ca02c", "^"),
        8: ("#d62728", "D"),
    }
    grouped: dict[int, list[tuple[str, float | None, np.ndarray]]] = {}
    for item in runs:
        fd = _fd_from_label(item[0])
        if fd is None:
            continue
        grouped.setdefault(fd, []).append(item)

    representatives: list[tuple[int, str, float | None, np.ndarray]] = []
    for fd, items in grouped.items():
        # Prefer the fastest completed run for the main label/marker.
        best = min(items, key=lambda item: float("inf") if item[1] is None else item[1])
        representatives.append((fd, best[0], best[1], best[2]))
    representatives.sort(key=lambda item: item[0])

    available_times = [runtime for _, _, runtime, _ in representatives if runtime is not None]
    slowest_time = max(available_times) if available_times else None

    for fd, items in sorted(grouped.items()):
        color, marker = fd_styles.get(fd, ("#7f7f7f", "o"))
        # Draw any alternate rank runs faintly to show rank invariance without
        # dominating the figure.
        for label, runtime, omega in items:
            ax.scatter(
                np.real(omega),
                np.imag(omega),
                c=color,
                marker=marker,
                s=18,
                alpha=0.18,
                linewidths=0.6,
            )

    for fd, label, runtime, omega in representatives:
        color, marker = fd_styles.get(fd, ("#7f7f7f", "o"))
        if runtime is not None and slowest_time is not None and slowest_time > 0.0:
            pct = 100.0 * (runtime - slowest_time) / slowest_time
            label = f"fd{fd} [{label}] ({runtime / 60.0:.1f} min, {pct:+.1f}%)"
        else:
            label = f"fd{fd} [{label}]"
        ax.scatter(
            np.real(omega),
            np.imag(omega),
            c=color,
            marker=marker,
            s=42,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel(r"$\Re(\omega)$")
    ax.set_ylabel(r"$\Im(\omega)$")
    ax.set_title("Song2023 Spectrum Progress by FD Order")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
