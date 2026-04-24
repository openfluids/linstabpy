#!/usr/bin/env python3
"""Phase 1.5 — baseflow spectrum diagnostic on Qraw.

Answers one question: does the raw Lui mean flow carry wall-normal content
above what the M2 decimated grid can resolve? If yes, Qfilt (Lele 1992
2nd-order compact filter, applied upstream) is the right axis of attack
for the next phase. If no, we look elsewhere for the source of the
contaminated EVP modes.

Convention (settled in worklog Entry 1):
- post-load arrays have shape (ny, nx) = (679, 599)
- axis 0 = wall-normal η (computational), wall at index 0
- axis 1 = streamwise ξ (computational)
- FFT is along the computational η index, where decimation operates
  and where spacing is uniform by construction. Physical y is clustered
  at the wall; a uniform FFT in physical y would be meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from linstab2d.baseflow.case1 import load_hamada_case  # noqa: E402

CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def load_qraw():
    """Load Qraw with default (stride=1) — native (679, 599) shape."""
    X, Y, bf, sponge = load_hamada_case(SCRIPT.parents[1] / "data" / "baseflow")
    return X, Y, bf, sponge


def pick_stations(nx: int) -> list[int]:
    """Four streamwise indices spanning inlet -> recovery.

    Rationale:
    - 15% in: upstream region, should be clean
    - 35% in: approach to interaction
    - 60% in: through / downstream of shock foot
    - 85% in: recovery / wake
    """
    fracs = [0.15, 0.35, 0.60, 0.85]
    return [int(round(f * (nx - 1))) for f in fracs]


def energy_above_nyquist(profile: np.ndarray, stride_eta_physical: int) -> tuple[np.ndarray, float]:
    """Compute 1D FFT energy spectrum along computational η and return
    (|FFT|^2 one-sided, fraction-above-decimated-Nyquist).

    Parameters
    ----------
    profile : 1D array of length N (wall-normal profile at one station).
    stride_eta_physical : integer decimation factor applied to the wall-normal
        axis. For M2 this is 2. Under the swapped naming in the loader this
        is numerically `stride_ksi`, but physically it decimates η.

    Returns
    -------
    spec : real array of length N//2+1, squared magnitude of the FFT.
    frac_above : fraction of total energy at wavenumbers above the
        decimated-grid Nyquist index N_dec/2 = N/(2*stride).
    """
    N = profile.size
    # Remove mean so the DC bin doesn't dominate the normalization.
    prof = profile - profile.mean()
    F = np.fft.rfft(prof)
    spec = (F.conj() * F).real
    N_dec = N // stride_eta_physical
    k_nyq_dec = N_dec // 2
    total = spec.sum()
    above = spec[k_nyq_dec + 1 :].sum()
    frac = float(above / total) if total > 0 else 0.0
    return spec, frac


def plot_field_with_stations(X: np.ndarray, Y: np.ndarray, U: np.ndarray,
                              stations: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    pc = ax.pcolormesh(X, Y, U, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, label=r"$\bar{u}/U_\infty$")
    for i, s in enumerate(stations):
        ax.plot(X[:, s], Y[:, s], color="white", lw=0.8, ls="--")
        ax.annotate(
            f"S{i+1}\nξ-idx={s}\nx≈{X[0, s]:.3f}",
            xy=(X[0, s], Y[0, s]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Qraw: u(x,y) and 4 wall-normal diagnostic stations")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra(specs: list[np.ndarray], stations: list[int],
                 X: np.ndarray, N: int, stride: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    k_nyq_dec = (N // stride) // 2
    ks = np.arange(specs[0].size)
    for i, (spec, s) in enumerate(zip(specs, stations)):
        ax.semilogy(ks, spec / spec.sum(), label=f"S{i+1} (x≈{X[0, s]:.3f})", lw=1.2)
    ax.axvline(k_nyq_dec, color="k", ls="--", lw=1, label=f"M2 η-Nyquist (k={k_nyq_dec})")
    ax.axvline(N // 2, color="gray", ls=":", lw=1, label=f"M1 η-Nyquist (k={N // 2})")
    ax.set_xlabel("wall-normal wavenumber index (η)")
    ax.set_ylabel(r"normalized |$\hat{u}$|²")
    ax.set_title("Wall-normal spectrum of Qraw u at 4 streamwise stations")
    ax.set_ylim(1e-16, 2)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra_multi(
    fields: dict[str, np.ndarray],
    stations: list[int],
    X: np.ndarray,
    N: int,
    stride: int,
    path: Path,
) -> None:
    """One subplot per field (rho, u, v, T), 4 station curves each."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True, sharey=True)
    k_nyq_dec = (N // stride) // 2
    ks = np.arange(N // 2 + 1)
    colors = ["C0", "C1", "C2", "C3"]
    for ax, (name, field) in zip(axes, fields.items()):
        for i, s in enumerate(stations):
            spec, _ = energy_above_nyquist(field[:, s], stride)
            ax.semilogy(
                ks,
                spec / spec.sum(),
                color=colors[i],
                lw=1.1,
                label=f"S{i+1}" if name == list(fields)[0] else None,
            )
        ax.axvline(k_nyq_dec, color="k", ls="--", lw=1)
        ax.axvline(N // 2, color="gray", ls=":", lw=1)
        ax.set_xlabel("wall-normal k (η)")
        ax.set_title(name)
        ax.set_ylim(1e-16, 2)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"normalized |$\hat{\phi}$|²")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Qraw wall-normal spectra, 4 variables × 4 stations (M2 η-Nyquist k={k_nyq_dec})"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_2d_fft(field: np.ndarray, stride: int, name: str, path: Path) -> None:
    """2D FFT of a field, with M2 Nyquist box overlaid."""
    F = field - field.mean()
    Fft = np.fft.fftshift(np.fft.fft2(F))
    power = np.log10(np.abs(Fft) ** 2 + 1e-30)
    ny, nx = field.shape
    ky_max = ny // 2
    kx_max = nx // 2
    ky_m2 = (ny // stride) // 2
    kx_m2 = (nx // stride) // 2
    extent = [-kx_max, kx_max, -ky_max, ky_max]
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    im = ax.imshow(power, extent=extent, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}|\hat{\phi}|^2$")
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (-kx_m2, -ky_m2),
            2 * kx_m2,
            2 * ky_m2,
            fill=False,
            edgecolor="cyan",
            lw=1.5,
            label="M2 Nyquist box",
        )
    )
    ax.add_patch(
        Rectangle(
            (-kx_max, -ky_max),
            2 * kx_max,
            2 * ky_max,
            fill=False,
            edgecolor="white",
            lw=1.0,
            ls="--",
            label="M1 (native) box",
        )
    )
    ax.set_xlabel(r"streamwise wavenumber $k_\xi$")
    ax.set_ylabel(r"wall-normal wavenumber $k_\eta$")
    ax.set_title(f"2D FFT of Qraw {name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plateau_level(profile: np.ndarray, tail_frac: float = 0.3) -> float:
    """Rough plateau level: median of normalized |FFT|^2 in the top `tail_frac` wavenumbers."""
    prof = profile - profile.mean()
    F = np.fft.rfft(prof)
    spec = (F.conj() * F).real
    spec = spec / spec.sum()
    N = spec.size
    tail = spec[int((1 - tail_frac) * N) :]
    return float(np.median(tail))


def main() -> None:
    X, Y, bf, _ = load_qraw()
    U = bf.U
    RHO = bf.RHO
    V = bf.V
    T = bf.T
    ny, nx = U.shape
    print(f"Loaded Qraw: U shape = {U.shape}  (ny=η={ny}, nx=ξ={nx})")
    print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Y range: [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"  U range: [{U.min():.4f}, {U.max():.4f}]")
    print(f"  RHO range: [{RHO.min():.4f}, {RHO.max():.4f}]")
    print(f"  V range: [{V.min():.4f}, {V.max():.4f}]")
    print(f"  T range: [{T.min():.4f}, {T.max():.4f}]")

    stations = pick_stations(nx)
    print(f"\nStation ξ-indices (out of {nx-1}): {stations}")
    for i, s in enumerate(stations):
        print(f"  S{i+1}  idx={s:3d}  x(wall)≈{X[0, s]:.4f}  "
              f"x(free)≈{X[-1, s]:.4f}  y-range=[{Y[0, s]:.4f},{Y[-1, s]:.4f}]")

    stride = 2  # wall-normal decimation factor for M2 (and M21 in code)
    fields = {"rho": RHO, "u": U, "v": V, "T": T}

    print(f"\nWall-normal FFT (N={ny}, decimated N={ny//stride}, "
          f"M2 η-Nyquist k={(ny//stride)//2}).")
    header = "  station | " + " | ".join(f"{n:>9s}" for n in fields) + " | "
    header += " | ".join(f"plat-{n}" for n in fields)
    print(header)
    fracs_by_field: dict[str, list[float]] = {n: [] for n in fields}
    plats_by_field: dict[str, list[float]] = {n: [] for n in fields}
    for i, s in enumerate(stations):
        row = f"   S{i+1}     |"
        plat_row = ""
        for name, field in fields.items():
            _, frac = energy_above_nyquist(field[:, s], stride)
            plat = plateau_level(field[:, s])
            fracs_by_field[name].append(frac)
            plats_by_field[name].append(plat)
            row += f" {frac:.2e} |"
            plat_row += f" {plat:.1e} "
        print(row + "   " + plat_row)

    print("\nplateau = median of normalized |FFT|^2 over top 30% of wavenumbers")
    print("(flat high-k noise floor; should be many decades below peak for a "
          "clean field)")

    verdict_frac = max(max(v) for v in fracs_by_field.values())
    verdict_plat = max(max(v) for v in plats_by_field.values())
    print("\n--- VERDICT ---")
    print(f"max frac above M2 Nyquist (any var, any station) = {verdict_frac:.3e}")
    print(f"max plateau level (any var, any station)        = {verdict_plat:.3e}")
    if verdict_frac > 1e-3 or verdict_plat > 1e-5:
        print("-> Qraw has broadband wall-normal content that the M2 grid "
              "cannot resolve AND/OR a noise-floor plateau well above "
              "machine-precision.")
        print("-> Qfilt (Lele 1992 2nd-order compact) is worth pursuing.")
    else:
        print("-> Qraw already effectively band-limited; Qfilt unlikely to "
              "move the EVP spectrum.")

    field_path = OUT_DIR / "qraw_u_with_stations.png"
    spec_path = OUT_DIR / "qraw_u_wall_normal_spectrum.png"
    multi_path = OUT_DIR / "qraw_multi_variable_spectrum.png"
    fft2_u_path = OUT_DIR / "qraw_u_2d_fft.png"
    fft2_T_path = OUT_DIR / "qraw_T_2d_fft.png"
    plot_field_with_stations(X, Y, U, stations, field_path)
    # Single-field legacy plot for u (keeps existing link in worklog valid).
    specs_u = [energy_above_nyquist(U[:, s], stride)[0] for s in stations]
    plot_spectra(specs_u, stations, X, ny, stride, spec_path)
    plot_spectra_multi(fields, stations, X, ny, stride, multi_path)
    plot_2d_fft(U, stride, "u", fft2_u_path)
    plot_2d_fft(T, stride, "T", fft2_T_path)

    summary_path = OUT_DIR / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Phase 1.5 — Qraw wall-normal spectrum diagnostic (extended)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Array shape: (ny=η={ny}, nx=ξ={nx})\n")
        f.write(f"Wall-normal decimation factor (M2 / M21 in code): {stride}\n")
        f.write(f"Decimated Nyquist k = {(ny//stride)//2} (native M1 k = {ny//2})\n\n")
        f.write("Fraction of wall-normal |FFT|^2 energy above M2 η-Nyquist:\n\n")
        f.write(f"{'station':>8s} {'x(wall)':>8s}  " +
                " ".join(f"{n:>10s}" for n in fields) + "\n")
        for i, s in enumerate(stations):
            row = f"     S{i+1} {X[0, s]:8.4f}  "
            for name in fields:
                row += f" {fracs_by_field[name][i]:10.3e}"
            f.write(row + "\n")
        f.write("\nPlateau level (median of normalized |FFT|^2 over top 30% of k):\n\n")
        f.write(f"{'station':>8s} {'x(wall)':>8s}  " +
                " ".join(f"{n:>10s}" for n in fields) + "\n")
        for i, s in enumerate(stations):
            row = f"     S{i+1} {X[0, s]:8.4f}  "
            for name in fields:
                row += f" {plats_by_field[name][i]:10.3e}"
            f.write(row + "\n")
        f.write(f"\nmax frac above Nyquist = {verdict_frac:.4e}\n")
        f.write(f"max plateau level      = {verdict_plat:.4e}\n")
        if verdict_frac > 1e-3 or verdict_plat > 1e-5:
            f.write("VERDICT: Qraw is not cleanly band-limited. Qfilt worth pursuing.\n")
        else:
            f.write("VERDICT: Qraw is effectively band-limited; Qfilt unlikely to help.\n")

    print(f"\nOutputs written to {OUT_DIR}")
    for p in (field_path, spec_path, multi_path, fft2_u_path, fft2_T_path, summary_path):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
