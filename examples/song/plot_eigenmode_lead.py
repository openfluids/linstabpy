#!/usr/bin/env python3
"""Most-unstable eigenmode: Song vs LinStabPy (supp Fig S3 layout, θ = 32°)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.figure_style import apply, check_plot_environment, save, signed_cmap
from examples.song.reference_paths import (
    FD4_NPZ,
    FIGURES_VALIDATION_DIR,
    LEADING_MODE_NPZ,
    LENGTH_SCALE,
    SPECTRUM_EIGENVALUES_DAT,
)
from examples.song.song_plot_utils import (
    _PRIMITIVES,
    component_primitive,
    load_song_leading_mode_npz,
    load_song_omega,
    max_omega_i_index,
    nearest_eigenvalue_index,
    normalized_real,
    per_primitive_phase_align,
    song_ref_on_mesh,
)

Y_MAX = 0.2
FIELD_LABELS = {
    "rho": r"$\Re(\rho')$",
    "u": r"$\Re(u')$",
    "v": r"$\Re(v')$",
    "w": r"$\Re(w')$",
    "t": r"$\Re(T')$",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--song-eigenvalues", type=Path, default=SPECTRUM_EIGENVALUES_DAT)
    p.add_argument("--song-mode", type=Path, default=LEADING_MODE_NPZ)
    p.add_argument("--npz", type=Path, default=FD4_NPZ)
    p.add_argument("--out-dir", type=Path, default=FIGURES_VALIDATION_DIR)
    p.add_argument("--song-scale", type=float, default=LENGTH_SCALE)
    p.add_argument("--ymax", type=float, default=Y_MAX)
    return p.parse_args()


def _plot_lead(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    prim_song: dict[str, np.ndarray],
    prim_py: dict[str, np.ndarray],
    out_path: Path,
    ymax: float,
) -> None:
    apply()
    fig, axes = plt.subplots(
        len(_PRIMITIVES),
        2,
        figsize=(6.6, 3.4),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.06, hspace=0.08, left=0.10, right=0.90, top=0.93, bottom=0.11)
    x1 = np.asarray(X[0, :], dtype=float)
    y1 = np.asarray(Y[:, 0], dtype=float)
    mesh = None
    for i, (_comp, key, _ylab) in enumerate(_PRIMITIVES):
        song_f = normalized_real(prim_song[key])
        py_f = normalized_real(prim_py[key])
        ax_s, ax_p = axes[i, 0], axes[i, 1]
        mesh = ax_s.pcolormesh(
            x1,
            y1,
            song_f,
            shading="gouraud",
            cmap=signed_cmap(),
            vmin=-1.0,
            vmax=1.0,
            rasterized=True,
        )
        ax_p.pcolormesh(
            x1,
            y1,
            py_f,
            shading="gouraud",
            cmap=signed_cmap(),
            vmin=-1.0,
            vmax=1.0,
            rasterized=True,
        )
        ax_s.set_ylabel(r"$y/X_{\mathrm{sh}}$")
        ax_s.text(
            0.02,
            0.82,
            FIELD_LABELS[key],
            transform=ax_s.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        if i == 0:
            ax_s.set_title("Song")
            ax_p.set_title("linstabpy")
    axes[-1, 0].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    axes[-1, 1].set_xlabel(r"$x/X_{\mathrm{sh}}$")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02)
    yticks = (0.0, 0.1, 0.2)
    yticklabels = ["0.0", "0.1", "0.2"]
    xspan = max(float(x1.max() - x1.min()), 1e-12)
    box_aspect = (ymax / xspan) * 2.0
    for ax in np.asarray(axes).ravel():
        ax.set_ylim(0.0, ymax)
        ax.set_yticks(yticks)
        ax.minorticks_off()
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(float(x1.min()), float(x1.max()))
        ax.set_box_aspect(box_aspect)
    for ax in axes[:, 0]:
        ax.set_yticklabels(yticklabels)
    save(fig, out_path)


def main() -> None:
    check_plot_environment(require_latex=True)
    args = _parse_args()
    npz = args.npz.expanduser().resolve()
    if not npz.is_file():
        raise FileNotFoundError(npz)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    omega_song = load_song_omega(args.song_eigenvalues.expanduser().resolve(), args.song_scale)
    X_s, Y_s, prim_ref = load_song_leading_mode_npz(
        args.song_mode.expanduser().resolve(), scale=args.song_scale
    )
    with np.load(npz, allow_pickle=False) as data:
        omega_py = np.asarray(data["omega"], dtype=complex).reshape(-1)
        dm = np.asarray(data["direct_modes"], dtype=complex)
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        eps_stab = (
            float(data["operator_dissipation_epsilon_stability"])
            if "operator_dissipation_epsilon_stability" in data.files
            else 0.0
        )
        dissip_on = (
            bool(data["operator_dissipation_enabled"])
            if "operator_dissipation_enabled" in data.files
            else False
        )

    shape = X.shape
    ngp = int(np.prod(shape))
    if dm.shape[0] != 5 * ngp:
        raise ValueError(f"{npz}: direct_modes rows {dm.shape[0]} != 5*ngp={5 * ngp}")

    k_ref = max_omega_i_index(omega_song)
    k_py = nearest_eigenvalue_index(omega_py, omega_song[k_ref])
    prim_ref_on: dict[str, np.ndarray] = {}
    py_raw: dict[str, np.ndarray] = {}
    for comp, key, _ in _PRIMITIVES:
        prim_ref_on[key] = prim_ref[key]
        py_on_song = song_ref_on_mesh(
            X_s, Y_s, X, Y, component_primitive(dm, ngp, comp, k_py, shape)
        )
        py_raw[key] = py_on_song
    prim_py, _ = per_primitive_phase_align(prim_ref_on, py_raw)
    for key in prim_py:
        ref_r = np.real(np.asarray(prim_ref_on[key]))
        py_r = np.real(np.asarray(prim_py[key]))
        if float(np.nansum(ref_r * py_r)) < 0.0:
            prim_py[key] = -prim_py[key]

    out_path = out_dir / "song2023_eigenmode_lead_theta32.png"
    _plot_lead(
        X=X_s,
        Y=Y_s,
        prim_song=prim_ref_on,
        prim_py=prim_py,
        out_path=out_path,
        ymax=args.ymax,
    )
    print(
        f"Song lead idx={k_ref} ω={omega_song[k_ref]:.6f}  "
        f"LinStabPy idx={k_py} ω={omega_py[k_py]:.6f}  "
        f"|Δω|={abs(omega_py[k_py] - omega_song[k_ref]):.4g}  "
        f"operator_dissipation_epsilon={eps_stab:g} enabled={dissip_on}"
    )
    print(f"npz {npz}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
