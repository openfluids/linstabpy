#!/usr/bin/env python3
"""Phase 1.5 companion — Lele 1992 2nd-order compact filter as Qfilt reference.

Diagnostic only. Does NOT add filter machinery to linstabpy/ core
(REFERENCE.md §3 explicitly excludes that). The goal is to:

1. render what Qfilt SHOULD look like in Fourier space, as a reference
   for the upstream-delivered Qfilt dataset;
2. quantify the reduction in above-M2-Nyquist energy and plateau level
   after filtering;
3. show the thickened shock structure visually.

Lele 1992 compact filter (Eq. 4.4.1a, tridiagonal form):

    α φ̂_{i-1} + φ̂_i + α φ̂_{i+1} =
        a_0 φ_i + (a_1/2)(φ_{i-1} + φ_{i+1}) + (a_2/2)(φ_{i-2} + φ_{i+2})

**F2** (2nd-order formal, N=1 support):

    a_0 = 1/2 + α,    a_1 = 1/2 + α,    a_2 = 0

Transfer: H(k) = (1/2 + α)(1 + cos k) / (1 + 2α cos k)
    H(π/2) = 1/2 + α      (50% attenuation at M2 Nyquist for α=0)

**F4** (4th-order formal, N=2 support):

    a_0 = (5 + 6α)/8,    a_1 = (1 + 2α)/2,    a_2 = -(1 - 2α)/8

    H(π/2) = (3 + 2α)/4    (25% attenuation at M2 Nyquist for α=0)

PLAN.md reports that upstream F2 (2nd-order) suppresses the EVP
contamination on this mean flow, but F4 (4th-order) does not. We
therefore use F2 as the default. F4 is also available as a diagnostic
to confirm PLAN's observation locally.

Reference: Lele, S. K. (1992). *J. Comput. Phys.* 103(1), 16–42, §4.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from linstab2d.baseflow.case1 import load_hamada_case  # noqa: E402

OUT_DIR = SCRIPT.parent / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def lele_compact_filter_matrix(n: int, alpha: float, order: int = 2) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Build LHS (A, tridiagonal) and RHS (B) matrices for the Lele 1992
    compact filter on a 1D line of length n.

    Parameters
    ----------
    n : number of grid points.
    alpha : filter free parameter in (-0.5, 0.5). Larger α = gentler
        filter. α=0 = explicit (maximal filtering consistent with the
        chosen order).
    order : 2 for F2 (default — this is PLAN's "2nd-order Lele"),
        4 for F4.

    Boundary handling: the first/last ⌊order/2⌋+1 points are
    passed through unchanged to keep wall / freestream values intact.
    """
    if not -0.5 < alpha < 0.5:
        raise ValueError("alpha must be in (-0.5, 0.5)")
    if order == 2:
        a0 = 0.5 + alpha
        a1 = 0.5 + alpha
        a2 = 0.0
        bw = 1
    elif order == 4:
        a0 = (5.0 + 6.0 * alpha) / 8.0
        a1 = (1.0 + 2.0 * alpha) / 2.0
        a2 = -(1.0 - 2.0 * alpha) / 8.0
        bw = 2
    else:
        raise ValueError("order must be 2 or 4")

    A = sparse.lil_matrix((n, n))
    B = sparse.lil_matrix((n, n))
    for i in range(n):
        if i < bw or i > n - 1 - bw:
            A[i, i] = 1.0
            B[i, i] = 1.0
        else:
            A[i, i - 1] = alpha
            A[i, i] = 1.0
            A[i, i + 1] = alpha
            B[i, i] = a0
            B[i, i - 1] = 0.5 * a1
            B[i, i + 1] = 0.5 * a1
            if order == 4:
                B[i, i - 2] = 0.5 * a2
                B[i, i + 2] = 0.5 * a2
    return A.tocsc(), B.tocsc()


class LeleFilter:
    """Apply a Lele compact filter to a 2D field along each axis in turn."""

    def __init__(self, shape: tuple[int, int], alpha: float, order: int = 2):
        ny, nx = shape
        A_y, B_y = lele_compact_filter_matrix(ny, alpha, order)
        A_x, B_x = lele_compact_filter_matrix(nx, alpha, order)
        self._solve_y = splu(A_y)
        self._solve_x = splu(A_x)
        self._B_y = B_y
        self._B_x = B_x
        self.alpha = alpha
        self.order = order

    def __call__(self, field: np.ndarray) -> np.ndarray:
        rhs = self._B_y @ field              # axis 0 = η
        filt = self._solve_y.solve(rhs)
        rhs = self._B_x @ filt.T             # axis 1 = ξ
        filt = self._solve_x.solve(rhs).T
        return filt


def energy_above_nyquist(profile: np.ndarray, stride: int) -> float:
    N = profile.size
    prof = profile - profile.mean()
    F = np.fft.rfft(prof)
    spec = (F.conj() * F).real
    k_nyq_dec = (N // stride) // 2
    total = spec.sum()
    above = spec[k_nyq_dec + 1 :].sum()
    return float(above / total) if total > 0 else 0.0


def plateau_level(profile: np.ndarray, tail_frac: float = 0.3) -> float:
    prof = profile - profile.mean()
    F = np.fft.rfft(prof)
    spec = (F.conj() * F).real / (F.conj() * F).real.sum()
    N = spec.size
    tail = spec[int((1 - tail_frac) * N) :]
    return float(np.median(tail))


def plot_2d_fft_comparison(raw: np.ndarray, filt: np.ndarray, stride: int,
                            name: str, path: Path) -> None:
    ny, nx = raw.shape
    kx_m2 = (nx // stride) // 2
    ky_m2 = (ny // stride) // 2
    kx_max = nx // 2
    ky_max = ny // 2
    extent = [-kx_max, kx_max, -ky_max, ky_max]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True,
                              sharex=True, sharey=True)
    vmin, vmax = -20, 4
    for ax, field, title in zip(axes, (raw, filt), ("Qraw", "Qfilt (Lele-2)")):
        F = field - field.mean()
        Fft = np.fft.fftshift(np.fft.fft2(F))
        power = np.log10(np.abs(Fft) ** 2 + 1e-30)
        im = ax.imshow(power, extent=extent, origin="lower", aspect="auto",
                        cmap="magma", vmin=vmin, vmax=vmax)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-kx_m2, -ky_m2), 2 * kx_m2, 2 * ky_m2,
                                fill=False, edgecolor="cyan", lw=1.5))
        ax.set_xlabel(r"$k_\xi$")
        ax.set_title(f"{title}: {name}")
    axes[0].set_ylabel(r"$k_\eta$")
    fig.colorbar(im, ax=axes, label=r"$\log_{10}|\hat{\phi}|^2$")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_wall_normal_comparison(
    raw: np.ndarray, filt: np.ndarray, stations: list[int], X: np.ndarray,
    stride: int, name: str, path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True,
                              sharey=True)
    N = raw.shape[0]
    k_nyq_dec = (N // stride) // 2
    ks = np.arange(N // 2 + 1)
    for ax, s in zip(axes, stations):
        for field, label, color in [(raw, "Qraw", "C3"), (filt, "Qfilt", "C0")]:
            p = field[:, s] - field[:, s].mean()
            F = np.fft.rfft(p)
            spec = (F.conj() * F).real
            ax.semilogy(ks, spec / spec.sum(), color=color, lw=1.2, label=label)
        ax.axvline(k_nyq_dec, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$k_\eta$")
        ax.set_title(f"S idx={s}  x≈{X[0, s]:.3f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(1e-16, 2)
    axes[0].set_ylabel(r"normalized |$\hat{\phi}$|²")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Qraw vs Qfilt wall-normal spectrum — {name}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shock_profile(raw: np.ndarray, filt: np.ndarray, X: np.ndarray,
                        Y: np.ndarray, y_idx: int, name: str, path: Path) -> None:
    """Compare along one wall-normal index line (freestream side) to
    show how shocks are thickened."""
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x_line = X[y_idx, :]
    ax.plot(x_line, raw[y_idx, :], color="C3", lw=1.0, label="Qraw")
    ax.plot(x_line, filt[y_idx, :], color="C0", lw=1.2, label="Qfilt (Lele-2, α=0.49)")
    ax.set_xlabel("x")
    ax.set_ylabel(name)
    ax.set_title(f"{name} along η-index {y_idx}  (y≈{Y[y_idx, 0]:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    X, Y, bf, _ = load_hamada_case(SCRIPT.parent / "baseflow")
    fields = {"rho": bf.RHO, "u": bf.U, "v": bf.V, "T": bf.T}
    ny, nx = bf.U.shape
    print(f"Qraw shape: (ny={ny}, nx={nx})")

    # Sweep Qraw vs F2 (at several α) vs F4 (canonical gentle α=0.49).
    # F2-α0 is the most aggressive Lele-2 setting.
    stations = [90, 209, 359, 508]
    stride = 2
    configs: list[tuple[str, int, float]] = [
        ("F2 α=0.00", 2, 0.0),
        ("F2 α=0.20", 2, 0.2),
        ("F2 α=0.40", 2, 0.4),
        ("F4 α=0.00", 4, 0.0),
        ("F4 α=0.49", 4, 0.49),
    ]
    print("\nwall-normal spectra, max over stations and variables:")
    print(f"{'config':>12s}   {'max frac_above_M2':>18s}   {'max plateau':>14s}")
    print("-" * 52)
    fracs_ref = {}
    plats_ref = {}
    for name, field in fields.items():
        fracs_ref[name] = max(energy_above_nyquist(field[:, s], stride) for s in stations)
        plats_ref[name] = max(plateau_level(field[:, s]) for s in stations)
    print(f"{'Qraw':>12s}   {max(fracs_ref.values()):18.3e}   "
          f"{max(plats_ref.values()):14.3e}")
    for label, order, alpha in configs:
        flt = LeleFilter((ny, nx), alpha=alpha, order=order)
        max_frac = 0.0
        max_plat = 0.0
        for name, field in fields.items():
            ff = flt(field)
            f = max(energy_above_nyquist(ff[:, s], stride) for s in stations)
            p = max(plateau_level(ff[:, s]) for s in stations)
            max_frac = max(max_frac, f)
            max_plat = max(max_plat, p)
        print(f"{label:>12s}   {max_frac:18.3e}   {max_plat:14.3e}")

    print("\nRendering comparison plots for F2 α=0.0 (PLAN's working config)...")
    flt = LeleFilter((ny, nx), alpha=0.0, order=2)
    for name, field in fields.items():
        filt_field = flt(field)
        if name in ("u", "T"):
            plot_2d_fft_comparison(
                field, filt_field, stride, name,
                OUT_DIR / f"qfilt_ref_{name}_2d_fft.png",
            )
        plot_wall_normal_comparison(
            field, filt_field, stations, X, stride, name,
            OUT_DIR / f"qfilt_ref_{name}_wall_normal.png",
        )
    y_idx_for_shock = ny // 2
    for name in ("rho", "u", "T"):
        filt_field = flt(fields[name])
        plot_shock_profile(
            fields[name], filt_field, X, Y, y_idx_for_shock, name,
            OUT_DIR / f"qfilt_ref_{name}_shock_profile_yidx{y_idx_for_shock}.png",
        )

    print(f"\nPlots written to {OUT_DIR}")
    print("\nThis is a *reference* Qfilt built inside this script. The "
          "production Qfilt must still come from the upstream delivery.")


if __name__ == "__main__":
    main()
