"""Shared helpers for Martini MATLAB vs linstabpy comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from examples.martini.compare_martini_examples import (
    _best_complex_scale,
    _best_spectrum_pairs,
    _load_npz_dict,
    _mode_matrix,
    _npz_get,
    load_matlab_bundle as load_matlab_fixture_bundle,
)
from examples.martini.matlab_hdf5 import compound_to_complex, sanitize_key


def ensure_matplotlib():
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.unicode_minus": False,
            "mathtext.fontset": "cm",
        }
    )
    if shutil.which("latex"):
        matplotlib.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
    return plt


REF_CODE_LABEL = "LinStab2D"
PY_CODE_LABEL = "linstabpy"


def _style_triptych_axes(
    axes,
    *,
    label_fs: float,
    tick_fs: float,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """x labels on every panel; y label/ticks only on the leftmost panel.

    Do not pair this with ``sharey=True``.
    Shared y-tick Text objects mean hiding labels on an inner panel also
    clears the left panel.
    """
    if ylim is None:
        ylim = axes[0].get_ylim()
    for i, ax in enumerate(axes):
        ax.set_xlabel(r"$x$", fontsize=label_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.set_aspect("auto")
        ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        if i == 0:
            ax.set_ylabel(r"$y$", fontsize=label_fs)
            ax.tick_params(axis="y", labelsize=tick_fs, left=True, labelleft=True)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)
            ax.set_yticklabels([])


def matlab_key(prefix: str, leaf: str) -> str:
    return sanitize_key(f"{prefix}/{leaf}")


def load_python_npz(path: Path) -> dict[str, np.ndarray]:
    return _load_npz_dict(path)


def load_matlab_bundle(path: Path) -> dict[str, np.ndarray]:
    stem = path.stem
    if stem.endswith("_values"):
        stem = stem[: -len("_values")]
    elif stem.endswith("_modes"):
        stem = stem[: -len("_modes")]
    return load_matlab_fixture_bundle(path.parent, stem)


def complex_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(compound_to_complex(values), dtype=complex).reshape(-1)


def to_grid(
    values: np.ndarray,
    used_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    flat = np.full(int(np.prod(shape)), np.nan, dtype=complex)
    flat[np.asarray(used_indices, dtype=int)] = np.asarray(values, dtype=complex).reshape(-1)
    return flat.reshape(shape, order="F")


def matlab_plot_grid_shape(X: np.ndarray) -> tuple[int, int]:
    """Return ``(ny, nx)`` for ``to_grid`` from a saved MATLAB ``plot_grid/X``.

    Some reference files keep raw MATLAB ``mesh.X`` transposed vs LinStabPy
    (``Xm.T == X_py``).
    ``plotFlow.m`` still indexes ``var(usedInd)`` on the untransposed
    ``(ny, nx)`` layout, so reshape with ``X.shape[::-1]``.

    Other fixtures (e.g. ``matlab_fixtures_npz``) already store ``X`` on the
    Python ``(ny, nx)`` layout — detect that and do **not** transpose again.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D plot_grid/X, got shape {X.shape}")
    # In (ny, nx) layout, streamwise x varies along axis 1 (columns).
    d0 = float(np.nanmax(np.abs(np.diff(X, axis=0)))) if X.shape[0] > 1 else 0.0
    d1 = float(np.nanmax(np.abs(np.diff(X, axis=1)))) if X.shape[1] > 1 else 0.0
    if d1 >= d0:
        return int(X.shape[0]), int(X.shape[1])
    return int(X.shape[1]), int(X.shape[0])


def matlab_plotflow_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X``, ``Y`` on the same ``(ny, nx)`` layout as ``matlab_plotflow_field``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if matlab_plot_grid_shape(X) == X.shape:
        return X, Y
    return X.T, Y.T


def match_modes_by_gains(
    gains_mat: np.ndarray,
    gains_py: np.ndarray,
    n_pairs: int,
) -> list[tuple[int, int]]:
    """Pair resolvent modes by singular-value (gain) closeness; order by MATLAB index."""
    gains_mat = np.asarray(gains_mat, dtype=float).reshape(-1)
    gains_py = np.asarray(gains_py, dtype=float).reshape(-1)
    n = min(int(n_pairs), gains_mat.size, gains_py.size)
    if n <= 0:
        return []
    cost = np.abs(gains_mat[:, None] - gains_py[None, :])
    ridx, cidx = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(ridx.tolist(), cidx.tolist())]
    pairs.sort(key=lambda ij: -float(gains_mat[ij[0]]))
    return pairs[:n]


def extract_component(
    mode_vec: np.ndarray,
    idx_comp: np.ndarray,
    *,
    one_based: bool,
) -> np.ndarray:
    idx = np.asarray(idx_comp, dtype=int).reshape(-1)
    if one_based:
        idx = idx - 1
    return np.asarray(mode_vec, dtype=complex).reshape(-1)[idx]


def align_grid_to_reference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Transpose the candidate field when MATLAB/Python store (ny, nx) differently."""
    if reference.shape == candidate.shape:
        return candidate
    if reference.shape == candidate.shape[::-1]:
        return candidate.T
    raise ValueError(
        f"Grid shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
    )


def align_matlab_field_to_python(
    matlab_field: np.ndarray,
    python_field: np.ndarray,
) -> np.ndarray:
    """Return the MATLAB field array on the same layout as the Python field."""
    matlab_field = np.asarray(matlab_field, dtype=float)
    python_field = np.asarray(python_field, dtype=float)
    if matlab_field.shape == python_field.shape:
        if np.allclose(matlab_field, python_field, rtol=0, atol=1e-9, equal_nan=True):
            return matlab_field
        if np.allclose(matlab_field.T, python_field, rtol=0, atol=1e-9, equal_nan=True):
            return matlab_field.T
        return matlab_field
    return align_grid_to_reference(python_field, matlab_field)


def symmetric_vmax(
    *arrays: np.ndarray,
    percentile: float | None = None,
) -> float:
    """Symmetric color limit for diverging mode plots, from a percentile of |field|."""
    chunks: list[np.ndarray] = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(np.abs(finite).reshape(-1))
    if not chunks:
        return 1.0
    merged = np.concatenate(chunks)
    if percentile is not None:
        return max(float(np.percentile(merged, percentile)), 1e-30)
    return max(float(np.max(merged)), 1e-30)


def mask_error_quiet_region(
    matlab_grid: np.ndarray,
    python_grid: np.ndarray,
    err: np.ndarray,
    *,
    peak_fraction: float = 0.05,
) -> np.ndarray:
    """Hide |error| where both fields are below ``peak_fraction`` of their peak."""
    matlab_grid = np.asarray(matlab_grid, dtype=float)
    python_grid = np.asarray(python_grid, dtype=float)
    err = np.asarray(err, dtype=float)
    peak = float(
        np.nanmax(
            np.abs(
                np.stack(
                    [
                        matlab_grid[np.isfinite(matlab_grid)],
                        python_grid[np.isfinite(python_grid)],
                    ]
                )
            )
        )
    )
    if not np.isfinite(peak) or peak <= 0.0:
        return err
    active = np.maximum(np.abs(matlab_grid), np.abs(python_grid)) >= peak_fraction * peak
    masked = err.copy()
    masked[~active] = np.nan
    return masked


def suggest_mode_ylim(
    Y: np.ndarray,
    grid: np.ndarray,
    *,
    energy_fraction: float = 0.02,
    pad_fraction: float = 0.08,
) -> tuple[float, float] | None:
    """Pick a wall-normal y window where the plotted field has most of its energy."""
    Y = np.asarray(Y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if Y.ndim != 2 or grid.ndim != 2 or Y.shape != grid.shape:
        return None
    row_energy = np.nanmax(np.abs(grid), axis=1)
    peak = float(np.nanmax(row_energy))
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    mask = row_energy >= energy_fraction * peak
    if not np.any(mask):
        return None
    y_line = Y[:, 0]
    y_lo = float(np.min(y_line[mask]))
    y_hi = float(np.max(y_line[mask]))
    pad = max((y_hi - y_lo) * pad_fraction, 1e-12)
    return y_lo - pad, y_hi + pad


def resolve_mode_ylim(
    case: dict[str, Any],
    matlab: dict[str, np.ndarray] | None,
    Y: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """y limits for mode panels: registry, MATLAB bundle, or auto zoom."""
    if "mode_ylim" in case:
        values = [float(v) for v in np.asarray(case["mode_ylim"], dtype=float).reshape(-1)]
        if len(values) >= 2:
            return values[0], values[1]
    if matlab is not None:
        for key, value in matlab.items():
            if "mode_plot_ylim" in key:
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.reshape(-1)
                values = [float(v) for v in arr.reshape(-1)]
                if len(values) >= 2:
                    return values[0], values[1]
    if grid is not None:
        return suggest_mode_ylim(Y, grid)
    return None


def _resolve_complex_part(*, part: str = "real", magnitude: bool = False) -> str:
    """Normalize field extraction to ``real``, ``imag``, or ``abs``."""
    if magnitude and part != "real":
        raise ValueError("Pass either magnitude=True or part='abs', not both styles")
    if magnitude:
        return "abs"
    key = str(part).strip().lower()
    if key in {"re", "real"}:
        return "real"
    if key in {"im", "imag", "imaginary"}:
        return "imag"
    if key in {"abs", "mag", "magnitude"}:
        return "abs"
    raise ValueError(f"Unsupported complex part {part!r} (use real, imag, abs)")


def _complex_field_values(values: np.ndarray, part: str) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if part == "abs":
        return np.abs(values)
    if part == "imag":
        return np.imag(values)
    return np.real(values)


def python_component_grid(
    modes: np.ndarray,
    mode_index: int,
    *,
    X: np.ndarray,
    used_ind: np.ndarray,
    idx_comp: np.ndarray,
    one_based_idx: bool = False,
    magnitude: bool = False,
    part: str = "real",
) -> np.ndarray:
    ny, nx = X.shape
    vec = modes[:, mode_index]
    component = extract_component(vec, idx_comp, one_based=one_based_idx)
    used = np.asarray(used_ind, dtype=int).reshape(-1)
    if used.size and int(used.min()) >= 1:
        used = used - 1
    grid = to_grid(component, used, (ny, nx))
    resolved = _resolve_complex_part(part=part, magnitude=magnitude)
    return np.asarray(_complex_field_values(grid, resolved), dtype=float)


def matlab_plotflow_field(
    mode_column: np.ndarray,
    *,
    idx_dof: np.ndarray,
    X: np.ndarray,
    used_ind: np.ndarray,
    magnitude: bool = False,
    part: str = "real",
) -> np.ndarray:
    """Replicate LinStab2D ``plotFlow.m`` grid layout for real/imag/abs(mode)."""
    X = np.asarray(X, dtype=float)
    used = np.asarray(used_ind, dtype=int).reshape(-1)
    idx = np.asarray(idx_dof, dtype=int).reshape(-1)
    if used.min() >= 1:
        used0 = used - 1
    else:
        used0 = used
    if idx.min() >= 1:
        idx0 = idx - 1
    else:
        idx0 = idx
    values = np.asarray(mode_column, dtype=complex).reshape(-1)[idx0]
    resolved = _resolve_complex_part(part=part, magnitude=magnitude)
    values = _complex_field_values(values, resolved)
    grid = to_grid(values, used0, matlab_plot_grid_shape(X))
    return np.asarray(np.real(grid), dtype=float)


def matlab_component_grid(
    modes: np.ndarray,
    mode_index: int,
    matlab: dict[str, np.ndarray],
    *,
    idx_prefix: str,
    plot_grid_prefix: str,
    component: str,
    magnitude: bool = False,
    part: str = "real",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_raw = np.asarray(_npz_get(matlab, f"{plot_grid_prefix}/X"), dtype=float)
    Y_raw = np.asarray(_npz_get(matlab, f"{plot_grid_prefix}/Y"), dtype=float)
    used = np.asarray(_npz_get(matlab, f"{plot_grid_prefix}/usedInd"), dtype=int).reshape(-1)
    vec = _mode_matrix(modes)[:, mode_index]
    idx_dof = _npz_get(matlab, f"{idx_prefix}/{component}_j")
    grid = matlab_plotflow_field(
        vec,
        idx_dof=idx_dof,
        X=X_raw,
        used_ind=used,
        magnitude=magnitude,
        part=part,
    )
    X, Y = matlab_plotflow_coordinates(X_raw, Y_raw)
    return X, Y, grid


def jet_sponge_zero_mask(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    mat_path: Path | None = None,
    used_ind: np.ndarray | None = None,
) -> np.ndarray:
    """Return length-``ngp`` sponge-open mask ``(sponge==0)`` on ``used_ind`` points.

    Matches ``example_2_jet.py`` / LinStab2D: forcing/observation where sponge is zero.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import loadmat

    from examples.martini.matlab_baseflow import _BASEFLOW_DATA_DIR

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    path = Path(mat_path) if mat_path is not None else (_BASEFLOW_DATA_DIR / "M09jet.mat")
    mat = loadmat(path)
    r_1D = np.asarray(mat["r_1D"], dtype=float).reshape(-1)
    z_1D = np.asarray(mat["z_1D"], dtype=float).reshape(-1)
    source_shape = (int(r_1D.size), int(z_1D.size))
    sponge = np.asarray(mat["spongeFun"], dtype=float).ravel(order="F").reshape(
        source_shape, order="F"
    )
    if tuple(X.shape) != source_shape:
        interpolator = RegularGridInterpolator(
            (r_1D.astype(float), z_1D.astype(float)),
            sponge,
            bounds_error=False,
            fill_value=np.nan,
        )
        query = np.column_stack(
            (
                Y.ravel(order="F").astype(float),
                X.ravel(order="F").astype(float),
            )
        )
        sponge = interpolator(query).reshape(X.shape, order="F")
    sponge0 = (sponge.ravel(order="F") == 0.0).astype(float)
    if used_ind is None:
        return sponge0
    used = np.asarray(used_ind, dtype=int).reshape(-1)
    if used.min() >= 1:
        used = used - 1
    return sponge0[used]


def jet_bc_mask_diag(
    sponge0: np.ndarray,
    *,
    layout: str = "kron",
    n_fields: int = 5,
) -> np.ndarray:
    """Diagonal of Ex.~2 ``B=C`` for published ``kron`` or block-correct ``tile``.

    - ``kron`` / ``repeat``: ``np.repeat(sponge0, n_fields)`` (MATLAB ``kron``)
    - ``tile`` / ``repmat``: ``np.tile(sponge0, n_fields)`` (MATLAB ``repmat``)
    """
    sponge0 = np.asarray(sponge0, dtype=float).reshape(-1)
    key = layout.strip().lower()
    if key in {"kron", "repeat", "published"}:
        return np.repeat(sponge0, int(n_fields))
    if key in {"tile", "repmat", "corrected", "block"}:
        return np.tile(sponge0, int(n_fields))
    raise ValueError(
        f"Unsupported jet B/C layout {layout!r} (use kron/repeat or tile/repmat)"
    )


def jet_resolvent_response_c_diag(
    sponge0: np.ndarray,
    *,
    mode_dofs: int,
    u_j: np.ndarray,
    v_j: np.ndarray,
    layout: str = "kron",
) -> np.ndarray:
    """Diagonal ``C`` for Ex.~2 MATLAB responses before plotting.

    Full ``5×ngp`` stacks use the published ``kron(B, I_5)`` layout.
    Slim ``2×ngp`` u-then-v reference files must mask each row with the
    matching entry from that full diagonal (via ``u_j``/``v_j``).
    Do not use ``repeat(sponge0, 2)``.
    """
    sponge0 = np.asarray(sponge0, dtype=float).reshape(-1)
    n_gp = int(sponge0.size)
    u = np.asarray(u_j, dtype=int).reshape(-1)
    v = np.asarray(v_j, dtype=int).reshape(-1)
    c_full = jet_bc_mask_diag(sponge0, layout=layout, n_fields=5)
    if mode_dofs == 5 * n_gp:
        return c_full
    if mode_dofs == 2 * n_gp:
        if u.size != n_gp or v.size != n_gp:
            raise ValueError(
                f"Compact jet idx length mismatch: u_j={u.size}, v_j={v.size}, ngp={n_gp}"
            )
        one_based = int(u.min()) >= 1
        u0 = u - 1 if one_based else u
        v0 = v - 1 if one_based else v
        return np.concatenate([c_full[u0], c_full[v0]])
    raise ValueError(
        f"Jet response DOFs {mode_dofs} do not match 2 or 5 x ngp={n_gp}"
    )


def apply_diagonal_c(
    modes: np.ndarray,
    c_diag: np.ndarray,
) -> np.ndarray:
    """Apply diagonal observation ``C`` column-wise: ``C @ modes``."""
    out = _mode_matrix(modes).copy()
    c = np.asarray(c_diag, dtype=float).reshape(-1)
    if c.size != out.shape[0]:
        raise ValueError(
            f"C diag length {c.size} does not match mode DOFs {out.shape[0]}"
        )
    out *= c[:, None]
    return out


def compact_resolvent_modes_uv(
    modes: np.ndarray,
    u_j: np.ndarray,
    v_j: np.ndarray,
    *,
    one_based_idx: bool | None = None,
) -> np.ndarray:
    """Stack u then v DOFs to match slim u+v resolvent fixtures."""
    mat = _mode_matrix(modes)
    u = np.asarray(u_j, dtype=int).reshape(-1)
    v = np.asarray(v_j, dtype=int).reshape(-1)
    if one_based_idx is None:
        one_based_idx = int(u.min()) >= 1
    if one_based_idx:
        u = u - 1
        v = v - 1
    return np.concatenate([mat[u, :], mat[v, :]], axis=0)


def compact_uv_component_idx(
    n_gp: int,
    component: str,
    *,
    one_based: bool = True,
) -> np.ndarray:
    """Index vector for a component inside a compact u-then-v mode matrix."""
    component = component.strip().lower()
    if component not in {"u", "v"}:
        raise ValueError(f"Unsupported compact component {component!r}")
    start = 1 if one_based else 0
    if component == "v":
        start += int(n_gp)
    return np.arange(start, start + int(n_gp), dtype=int)


def patch_matlab_compact_uv_idx(
    matlab: dict[str, np.ndarray],
    *,
    idx_prefix: str,
    n_gp: int,
) -> dict[str, np.ndarray]:
    """Map global u/v DOF labels to compact u-then-v columns for gridding."""
    patched = dict(matlab)
    for comp in ("u", "v"):
        ref = np.asarray(_npz_get(matlab, f"{idx_prefix}/{comp}_j"))
        key = sanitize_key(f"{idx_prefix}/{comp}_j")
        if key not in patched:
            alt = [k for k in patched if k.endswith(f"idx__{comp}_j")]
            if not alt:
                raise KeyError(f"Missing MATLAB idx for component {comp!r}")
            key = alt[0]
        patched[key] = compact_uv_component_idx(n_gp, comp, one_based=True).reshape(ref.shape)
    return patched


def patch_matlab_compact_uvw_idx(
    matlab: dict[str, np.ndarray],
    *,
    idx_prefix: str,
    n_gp: int,
) -> dict[str, np.ndarray]:
    """Map global u/v/w DOF labels to compact u-then-v-then-w columns for gridding."""
    patched = dict(matlab)
    for comp in ("u", "v", "w"):
        ref = np.asarray(_npz_get(matlab, f"{idx_prefix}/{comp}_j"))
        key = sanitize_key(f"{idx_prefix}/{comp}_j")
        if key not in patched:
            alt = [k for k in patched if k.endswith(f"idx__{comp}_j")]
            if not alt:
                raise KeyError(f"Missing MATLAB idx for component {comp!r}")
            key = alt[0]
        patched[key] = compact_uvw_component_idx(n_gp, comp, one_based=True).reshape(ref.shape)
    return patched


def compact_resolvent_modes_uvw(
    modes: np.ndarray,
    u_j: np.ndarray,
    v_j: np.ndarray,
    w_j: np.ndarray,
    *,
    one_based_idx: bool | None = None,
) -> np.ndarray:
    """Stack u, v, then w DOFs to match slim u+v+w resolvent fixtures."""
    mat = _mode_matrix(modes)
    u = np.asarray(u_j, dtype=int).reshape(-1)
    v = np.asarray(v_j, dtype=int).reshape(-1)
    w = np.asarray(w_j, dtype=int).reshape(-1)
    if one_based_idx is None:
        one_based_idx = int(u.min()) >= 1
    if one_based_idx:
        u = u - 1
        v = v - 1
        w = w - 1
    return np.concatenate([mat[u, :], mat[v, :], mat[w, :]], axis=0)


def compact_uvw_component_idx(
    n_gp: int,
    component: str,
    *,
    one_based: bool = True,
) -> np.ndarray:
    """Index vector for a component inside a compact u-then-v-then-w mode matrix."""
    component = component.strip().lower()
    if component not in {"u", "v", "w"}:
        raise ValueError(f"Unsupported compact component {component!r}")
    offsets = {"u": 0, "v": 1, "w": 2}
    start = offsets[component] * int(n_gp)
    if one_based:
        start += 1
    return np.arange(start, start + int(n_gp), dtype=int)


def maybe_compact_python_resolvent_uvw(
    modes_py: np.ndarray,
    modes_mat: np.ndarray,
    *,
    u_j: np.ndarray,
    v_j: np.ndarray,
    w_j: np.ndarray,
) -> np.ndarray:
    """Compact full-stack Python modes when MATLAB fixtures store u+v+w only."""
    py = _mode_matrix(modes_py)
    mat = _mode_matrix(modes_mat)
    if py.shape[0] == mat.shape[0]:
        return modes_py
    uvw_dofs = int(u_j.size + v_j.size + w_j.size)
    if mat.shape[0] == uvw_dofs:
        return compact_resolvent_modes_uvw(modes_py, u_j, v_j, w_j)
    raise ValueError(
        f"Python mode DOFs {py.shape[0]} cannot be matched to MATLAB {mat.shape[0]}"
    )


def maybe_compact_python_resolvent_uv(
    modes_py: np.ndarray,
    modes_mat: np.ndarray,
    *,
    u_j: np.ndarray,
    v_j: np.ndarray,
) -> np.ndarray:
    """Compact full-stack Python modes when MATLAB fixtures store u+v only."""
    py = _mode_matrix(modes_py)
    mat = _mode_matrix(modes_mat)
    if py.shape[0] == mat.shape[0]:
        return modes_py
    uv_dofs = int(u_j.size + v_j.size)
    if mat.shape[0] == uv_dofs:
        return compact_resolvent_modes_uv(modes_py, u_j, v_j)
    raise ValueError(
        f"Python mode DOFs {py.shape[0]} cannot be matched to MATLAB {mat.shape[0]}"
    )


def compact_resolvent_modes_component(
    modes: np.ndarray,
    idx_comp: np.ndarray,
    *,
    one_based_idx: bool | None = None,
) -> np.ndarray:
    """Extract one component block from a full-stack mode matrix."""
    mat = _mode_matrix(modes)
    idx = np.asarray(idx_comp, dtype=int).reshape(-1)
    if one_based_idx is None:
        one_based_idx = int(idx.min()) >= 1
    if one_based_idx:
        idx = idx - 1
    return mat[idx, :]


def maybe_compact_python_modes_to_matlab(
    modes_py: np.ndarray,
    modes_mat: np.ndarray,
    idx_comp: np.ndarray,
    *,
    one_based_idx: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Align Python mode storage with slim MATLAB fixtures when needed."""
    py = _mode_matrix(modes_py)
    mat = _mode_matrix(modes_mat)
    if py.shape[0] == mat.shape[0]:
        return modes_py, idx_comp, one_based_idx
    compact = compact_resolvent_modes_component(
        modes_py,
        idx_comp,
        one_based_idx=one_based_idx,
    )
    if compact.shape[0] != mat.shape[0]:
        raise ValueError(
            f"Compact Python mode DOFs {compact.shape[0]} "
            f"do not match MATLAB {mat.shape[0]}"
        )
    n_gp = int(mat.shape[0])
    return compact, np.arange(1, n_gp + 1, dtype=int), True


def plot_spectrum_overlay(
    *,
    out_path: Path,
    title: str,
    matlab_eigs: np.ndarray,
    python_eigs: np.ndarray,
    xlabel: str,
    ylabel: str,
    n_annotate: int = 0,
    matlab_zoom_inset: bool = False,
    marker_size: float = 7.0,
    marker_edgewidth: float = 1.2,
    dpi: int = 200,
    title_fs: float | None = None,
    label_fs: float | None = None,
    tick_fs: float | None = None,
    legend_fs: float | None = None,
    figsize: tuple[float, float] = (7.0, 5.5),
) -> list[tuple[int, int]]:
    plt = ensure_matplotlib()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    matlab_eigs = complex_vector(matlab_eigs)
    python_eigs = complex_vector(python_eigs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        np.real(matlab_eigs),
        np.imag(matlab_eigs),
        linestyle="none",
        marker="o",
        mfc="none",
        mec="#b94f4f",
        mew=marker_edgewidth,
        ms=marker_size,
        label=REF_CODE_LABEL,
        zorder=3,
    )
    ax.plot(
        np.real(python_eigs),
        np.imag(python_eigs),
        linestyle="none",
        marker="x",
        color="#2a5caa",
        ms=marker_size,
        mew=marker_edgewidth,
        label=PY_CODE_LABEL,
        zorder=4,
    )

    pairs = _best_spectrum_pairs(matlab_eigs, python_eigs, max(len(matlab_eigs), len(python_eigs)))
    if n_annotate > 0:
        for mat_i, py_i, _, _ in pairs[:n_annotate]:
            ax.plot(
                [np.real(matlab_eigs[mat_i]), np.real(python_eigs[py_i])],
                [np.imag(matlab_eigs[mat_i]), np.imag(python_eigs[py_i])],
                color="0.5",
                linewidth=0.8,
                alpha=0.45,
                zorder=1,
            )

    if matlab_zoom_inset and matlab_eigs.size > 0:
        axins = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=1.0)
        axins.plot(
            np.real(matlab_eigs),
            np.imag(matlab_eigs),
            linestyle="none",
            marker="o",
            mfc="none",
            mec="#b94f4f",
            mew=1.0,
            ms=5,
            zorder=3,
        )
        re = np.real(matlab_eigs)
        im = np.imag(matlab_eigs)
        pad = 0.2
        dx = max(float(np.ptp(re)), 1e-3) * (1.0 + pad)
        dy = max(float(np.ptp(im)), 1e-3) * (1.0 + pad)
        cx = float(np.mean(re))
        cy = float(np.mean(im))
        axins.set_xlim(cx - dx / 2.0, cx + dx / 2.0)
        axins.set_ylim(cy - dy / 2.0, cy + dy / 2.0)
        axins.set_title(REF_CODE_LABEL, fontsize=8, pad=2.0)
        axins.grid(True, alpha=0.25)
        axins.tick_params(labelsize=7)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    if str(title).strip():
        ax.set_title(title, fontsize=title_fs)
    if tick_fs is not None:
        ax.tick_params(labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=legend_fs)
    if matlab_zoom_inset:
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.9)
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [(int(a), int(b)) for a, b, _, _ in pairs]


def plot_mode_triptych(
    *,
    out_path: Path,
    title: str,
    X: np.ndarray,
    Y: np.ndarray,
    matlab_grid: np.ndarray,
    python_grid: np.ndarray,
    X_py: np.ndarray | None = None,
    Y_py: np.ndarray | None = None,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    omega_label: str = "",
    ref_gain: float | None = None,
    py_gain: float | None = None,
    color_percentile: float | None = None,
    error_mask_peak_fraction: float | None = None,
    err_color_percentile: float | None = None,
    magnitude: bool = False,
    field_cbar_label: str | None = None,
    field_vmax: float | None = None,
    normalize_by_peak: bool = False,
) -> None:
    """Plot LinStab2D vs linstabpy modes with peak-normalized % error (Ex0 style)."""
    plt = ensure_matplotlib()
    # Thesis-readable sizes (full-width figure in Appendix I shrinks fonts).
    title_fs = 22
    label_fs = 26
    tick_fs = 22
    cbar_fs = 20
    panel_title_fs = 18 if (ref_gain is not None or py_gain is not None) else title_fs
    X_py = np.asarray(X if X_py is None else X_py, dtype=float)
    Y_py = np.asarray(Y if Y_py is None else Y_py, dtype=float)
    matlab_grid = np.asarray(matlab_grid, dtype=float)
    python_native = np.asarray(python_grid, dtype=float)
    python_aligned = (
        align_grid_to_reference(matlab_grid, python_native)
        if matlab_grid.shape != python_native.shape
        else python_native
    )
    # Shared peak so both panels use the same scale (and error stays consistent).
    peak = float(
        np.nanmax(
            np.abs(
                np.stack(
                    [
                        np.asarray(matlab_grid, dtype=float),
                        np.asarray(python_native, dtype=float),
                        np.asarray(python_aligned, dtype=float),
                    ]
                )
            )
        )
    )
    peak = max(peak, 1e-30)
    if normalize_by_peak:
        matlab_grid = matlab_grid / peak
        python_native = python_native / peak
        python_aligned = python_aligned / peak
    err = np.abs(matlab_grid - python_aligned)
    if error_mask_peak_fraction is not None:
        err_plot = mask_error_quiet_region(
            matlab_grid,
            python_aligned,
            err,
            peak_fraction=error_mask_peak_fraction,
        )
    else:
        err_plot = err
    if normalize_by_peak and field_vmax is None:
        vmax = 1.0
    else:
        vmax = (
            float(field_vmax)
            if field_vmax is not None
            else symmetric_vmax(matlab_grid, python_native, python_aligned, percentile=color_percentile)
        )
    vmax = max(float(vmax), 1e-30)
    # Peak-normalized percentage error (same convention as Example 0).
    err_pct = 100.0 * err_plot / vmax
    err_finite = err_pct[np.isfinite(err_pct)]
    if err_finite.size:
        if err_color_percentile is not None:
            err_vmax = symmetric_vmax(err_finite, percentile=err_color_percentile)
        else:
            err_vmax = float(np.nanmax(err_finite))
    else:
        err_vmax = 100.0 * symmetric_vmax(err, percentile=err_color_percentile) / vmax

    fig_h = 5.5 if ylim is not None else 4.6
    # Explicit columns: LinStab2D | linstabpy | field cbar | error | error cbar
    # (mpl 3.6 ignores colorbar(location=...); divider+tight_layout overlaps axes.)
    fig = plt.figure(figsize=(15.0, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.07, 1.0, 0.07])
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3)]
    cax_f = fig.add_subplot(gs[0, 2])
    cax_e = fig.add_subplot(gs[0, 4])
    if magnitude:
        kwargs = dict(cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
        cbar_label = field_cbar_label or (r"$|q|/\max|q|$" if normalize_by_peak else r"$|q|$")
    else:
        kwargs = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        cbar_label = field_cbar_label or (r"$\Re(q)/\max|\Re(q)|$" if normalize_by_peak else r"$\Re(q)$")
    im0 = axes[0].pcolormesh(X, Y, matlab_grid, **kwargs)
    if ref_gain is not None:
        axes[0].set_title(
            rf"{REF_CODE_LABEL}" + "\n" + rf"$\sigma={float(ref_gain):.4g}$",
            fontsize=panel_title_fs,
        )
    else:
        axes[0].set_title(REF_CODE_LABEL, fontsize=title_fs)
    axes[1].pcolormesh(X_py, Y_py, python_native, **kwargs)
    if py_gain is not None:
        axes[1].set_title(
            rf"{PY_CODE_LABEL}" + "\n" + rf"$\sigma={float(py_gain):.4g}$",
            fontsize=panel_title_fs,
        )
    else:
        axes[1].set_title(PY_CODE_LABEL, fontsize=title_fs)
    im2 = axes[2].pcolormesh(
        X,
        Y,
        err_pct,
        cmap="magma",
        vmin=0.0,
        vmax=err_vmax,
        shading="auto",
    )
    axes[2].set_title(r"$|\mathrm{error}|$ (\%)", fontsize=title_fs)
    if omega_label:
        fig.suptitle(f"{title}\n{omega_label}", fontsize=title_fs)
    else:
        fig.suptitle(title, fontsize=title_fs)

    cbar0 = fig.colorbar(im0, cax=cax_f)
    cbar0.ax.tick_params(labelsize=cbar_fs)
    cbar0.set_label(cbar_label, fontsize=cbar_fs)

    cbar2 = fig.colorbar(im2, cax=cax_e)
    cbar2.ax.tick_params(labelsize=cbar_fs)
    cbar2.set_label("")
    cbar2.ax.set_ylabel("")

    _style_triptych_axes(axes, label_fs=label_fs, tick_fs=tick_fs, ylim=ylim, xlim=xlim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gain_comparison(
    *,
    out_path: Path,
    title: str,
    matlab_gains: np.ndarray,
    python_gains: np.ndarray,
) -> None:
    plt = ensure_matplotlib()
    matlab_gains = np.asarray(matlab_gains, dtype=float).reshape(-1)
    python_gains = np.asarray(python_gains, dtype=float).reshape(-1)
    n = min(matlab_gains.size, python_gains.size)
    x = np.arange(1, n + 1)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(x - width / 2, matlab_gains[:n], width=width, label=REF_CODE_LABEL, color="#b94f4f", alpha=0.75)
    ax.bar(x + width / 2, python_gains[:n], width=width, label=PY_CODE_LABEL, color="#2a5caa", alpha=0.75)
    ax.set_xlabel(r"Mode rank")
    ax.set_ylabel(r"Gain")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_field_triptych(
    *,
    out_path: Path,
    title: str,
    X_mat: np.ndarray,
    Y_mat: np.ndarray,
    matlab_field: np.ndarray,
    X_py: np.ndarray,
    Y_py: np.ndarray,
    python_field: np.ndarray,
    field_label: str | None = None,
) -> None:
    """LinStab2D vs linstabpy field triptych — same packing template as Ex0 ``_validation``."""
    plt = ensure_matplotlib()
    title_fs = 22
    label_fs = 26
    tick_fs = 22
    cbar_fs = 20
    X_mat = np.asarray(X_mat, dtype=float)
    Y_mat = np.asarray(Y_mat, dtype=float)
    X_py = np.asarray(X_py, dtype=float)
    Y_py = np.asarray(Y_py, dtype=float)
    matlab_field = np.asarray(matlab_field, dtype=float)
    python_field = np.asarray(python_field, dtype=float)

    # Keep (X, Y, field) in one layout. Example 0 Python export is the transpose
    # of the MATLAB reference; transpose the whole MATLAB triple together.
    if matlab_field.shape != python_field.shape and matlab_field.T.shape == python_field.shape:
        matlab_field = matlab_field.T
        X_mat = X_mat.T
        Y_mat = Y_mat.T
    elif (
        matlab_field.shape == python_field.shape
        and not np.allclose(matlab_field, python_field, rtol=0, atol=1e-9, equal_nan=True)
        and np.allclose(matlab_field.T, python_field, rtol=0, atol=1e-9, equal_nan=True)
    ):
        matlab_field = matlab_field.T
        X_mat = X_mat.T
        Y_mat = Y_mat.T

    err = np.abs(matlab_field - python_field)
    vmax = float(np.nanmax(np.abs(np.stack([matlab_field, python_field]))))
    vmax = max(vmax, 1e-30)
    err_pct = 100.0 * err / vmax
    err_vmax = float(np.nanmax(err_pct[np.isfinite(err_pct)])) if np.any(np.isfinite(err_pct)) else 1.0
    err_vmax = max(err_vmax, 1e-30)

    # Same domain / packing as example_0_curvilinear_validation.png
    phys_xlim = (-0.5, float(np.nanmax(np.concatenate([X_mat.ravel(), X_py.ravel()]))))
    phys_ylim = (0.0, 1.4)
    gap_12 = 0.01
    gap_23 = 0.075
    gap_cbar = 0.005
    cbar_w = 0.02
    left, right, bottom, top = 0.08, 0.98, 0.14, 0.84

    fig = plt.figure(figsize=(14.5, 4.6))
    ax0 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax1 = fig.add_axes([left, bottom, 0.22, top - bottom])
    ax2 = fig.add_axes([left, bottom, 0.22, top - bottom])
    cax_f = fig.add_axes([left, bottom, cbar_w, top - bottom])
    cax_e = fig.add_axes([left, bottom, cbar_w, top - bottom])

    kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    im0 = ax0.pcolormesh(X_mat, Y_mat, matlab_field, **kw)
    ax1.pcolormesh(X_py, Y_py, python_field, **kw)
    im2 = ax2.pcolormesh(X_py, Y_py, err_pct, cmap="magma", shading="auto", vmin=0.0, vmax=err_vmax)

    if field_label:
        ax0.set_title(rf"{REF_CODE_LABEL}: {field_label}", fontsize=title_fs)
        ax1.set_title(rf"{PY_CODE_LABEL}: {field_label}", fontsize=title_fs)
        ax2.set_title(rf"$|\mathrm{{error}}|$ (\%), {field_label}", fontsize=title_fs)
    else:
        ax0.set_title(REF_CODE_LABEL, fontsize=title_fs)
        ax1.set_title(PY_CODE_LABEL, fontsize=title_fs)
        ax2.set_title(r"$|\mathrm{error}|$ (\%)", fontsize=title_fs)

    for ax in (ax0, ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(phys_xlim)
        ax.set_ylim(phys_ylim)

    ax0.set_ylabel(r"$y$", fontsize=label_fs)
    ax0.tick_params(axis="y", left=True, labelleft=True)
    for ax in (ax1, ax2):
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=True, labelleft=False)

    cbar0 = fig.colorbar(im0, cax=cax_f)
    cbar0.ax.tick_params(labelsize=cbar_fs)
    cbar2 = fig.colorbar(im2, cax=cax_e)
    cbar2.ax.tick_params(labelsize=cbar_fs)

    fig.canvas.draw()

    def _size(ax):
        b = ax.get_position()
        return b.width, b.height

    w0, h0 = _size(ax0)
    w1, h1 = _size(ax1)
    w2, h2 = w1, h1
    y0 = bottom + 0.5 * ((top - bottom) - h0)
    y1 = bottom + 0.5 * ((top - bottom) - h1)
    y2 = bottom + 0.5 * ((top - bottom) - h2)

    x0 = left
    x1 = x0 + w0 + gap_12
    xf = x1 + w1 + gap_cbar
    x2 = x1 + w1 + gap_23
    xe = x2 + w2 + gap_cbar

    ax0.set_position([x0, y0, w0, h0])
    ax1.set_position([x1, y1, w1, h1])
    cax_f.set_position([xf, y1, cbar_w, h1])
    ax2.set_position([x2, y2, w2, h2])
    cax_e.set_position([xe, y2, cbar_w, h2])

    # Centre the three panels (not colourbars) in the figure.
    panel_mid = 0.5 * (x0 + x2 + w2)
    shift = 0.5 - panel_mid
    for ax in (ax0, ax1, ax2, cax_f, cax_e):
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
    if title:
        fig.suptitle(title, fontsize=title_fs, x=0.5, y=0.97, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_mode_triptych_mosaic(
    *,
    out_path: Path,
    title: str,
    rows: list[dict[str, Any]],
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    dpi: int = 400,
    row_height: float = 1.9,
) -> None:
    """Stack of Ex0-style mode triptychs (LinStab2D | linstabpy | |error|%).

    Each ``rows`` entry needs these keys.

        row_label, X_mat, Y_mat, matlab_grid, X_py, Y_py, python_grid.

    Optional ``ref_gain`` / ``py_gain`` (forcing rows): σ titles on the matching
    columns only.
    Optional ``banner_ref`` / ``banner_py`` (mode rows): ω (or similar) above
    the LinStab2D and linstabpy panels.
    A single ``banner`` is still accepted and centred over both field
    columns.
    All panel rows share the same height.
    A thin σ strip sits above forcing rows only.
    Left ylabel is ``$y-\\Re(\\ldots)$``.
    Thesis captions carry the outer figure titles.
    Pass a non-empty ``title`` only for slide dumps.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    plt = ensure_matplotlib()
    n_rows = len(rows)
    title_fs = 28
    label_fs = 22
    tick_fs = 18
    cbar_fs = 16
    side_fs = 18
    col_fs = 24
    sigma_fs = 18

    gap_12 = 0.015
    gap_23 = 0.075
    gap_cbar = 0.005
    cbar_w = 0.016
    # Extra strip above a forcing row for σ titles, as a figure fraction of that row.
    sigma_strip = 0.032
    gap_after_forcing = 0.020  # after a forcing row
    gap_after_response = 0.002  # after a response row (Ex1 stack)
    # Mode-only mosaics (Ex4): visible white gutter between ranks (0.002 looked glued).
    gap_mode_rows = 0.045
    # ω banner strip sized in *inches* (unused when rows omit banner keys).
    mode_banner_in = 0.36
    # Column-header band in *inches* (large fonts). Outer figure title is
    # omitted unless ``title`` is non-empty (thesis captions replace it).
    show_title = bool(str(title).strip())
    title_h_in = 0.42 if show_title else 0.0
    header_h_in = 0.36
    gap_title_header_in = 0.05 if show_title else 0.0
    gap_header_content_in = 0.06
    top_pad_in = title_h_in + gap_title_header_in + header_h_in + gap_header_content_in
    bottom_pad_in = 0.35
    left, right = 0.10, 0.985

    has_sigma = [
        row.get("ref_gain") is not None or row.get("py_gain") is not None for row in rows
    ]
    has_banner = [
        bool(row.get("banner") or row.get("banner_ref") or row.get("banner_py"))
        for row in rows
    ]
    # Same panel height for forcing and response (response size preferred).
    body_w = np.ones(n_rows, dtype=float)

    mode_only = not any(has_sigma)
    gaps = []
    for r in range(n_rows - 1):
        if has_sigma[r]:
            gaps.append(gap_after_forcing)
        elif mode_only:
            gaps.append(gap_mode_rows)
        else:
            gaps.append(gap_after_response)
    gaps.append(0.0)

    n_sigma = int(sum(has_sigma))
    n_banner = int(sum(has_banner)) if mode_only else 0
    banner_pad_in = (n_banner * mode_banner_in) if mode_only else 0.0
    # ``row_height`` sets vertical panel size in inches (domain/ylim unchanged).
    fig_h = float(row_height) * n_rows + top_pad_in + bottom_pad_in + banner_pad_in
    mode_banner_strip = (mode_banner_in / fig_h) if mode_only else 0.0
    reserved = (
        float(sum(gaps))
        + n_sigma * sigma_strip
        + n_banner * mode_banner_strip
    )
    # Place each label at the *centre* of its reserved band (va="center").
    fig_margin_top = 0.01
    fig_top = 1.0 - fig_margin_top
    title_y = fig_top - 0.5 * title_h_in / fig_h
    header_y = fig_top - (title_h_in + gap_title_header_in + 0.5 * header_h_in) / fig_h
    top = fig_top - top_pad_in / fig_h
    bottom = bottom_pad_in / fig_h
    if top <= bottom + 0.2:
        raise ValueError(f"Title band leaves too little room for panels: top={top:.3f}")

    usable = top - bottom - reserved
    body_h = usable * (body_w / body_w.sum())

    panel_w = (right - left - gap_12 - gap_23 - 2.0 * gap_cbar - 2.0 * cbar_w) / 3.0
    fig = plt.figure(figsize=(15.5, fig_h))

    row_axes: list[tuple[Any, Any, Any, Any, Any]] = []
    sigma_y: list[float | None] = []
    banner_y: list[float | None] = []
    y_cursor = top
    for r in range(n_rows):
        if has_sigma[r]:
            sigma_y.append(y_cursor - 0.5 * sigma_strip)
            y_cursor -= sigma_strip
        else:
            sigma_y.append(None)
        if mode_only and has_banner[r]:
            banner_y.append(y_cursor - 0.5 * mode_banner_strip)
            y_cursor -= mode_banner_strip
        else:
            banner_y.append(None)
        h = float(body_h[r])
        y = y_cursor - h
        x0 = left
        x1 = x0 + panel_w + gap_12
        xf = x1 + panel_w + gap_cbar
        x2 = x1 + panel_w + gap_23
        xe = x2 + panel_w + gap_cbar
        ax0 = fig.add_axes([x0, y, panel_w, h])
        ax1 = fig.add_axes([x1, y, panel_w, h])
        cax_f = fig.add_axes([xf, y, cbar_w, h])
        ax2 = fig.add_axes([x2, y, panel_w, h])
        cax_e = fig.add_axes([xe, y, cbar_w, h])
        row_axes.append((ax0, ax1, ax2, cax_f, cax_e))
        y_cursor = y - gaps[r]

    # Column headers above the first row; optional outer title above that.
    if show_title:
        fig.text(
            0.5,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
        )
    for ax, col_title in zip(
        (row_axes[0][0], row_axes[0][1], row_axes[0][2]),
        (REF_CODE_LABEL, PY_CODE_LABEL, r"$|\mathrm{error}|$ (\%)"),
    ):
        pos = ax.get_position()
        fig.text(
            0.5 * (pos.x0 + pos.x1),
            header_y,
            col_title,
            ha="center",
            va="center",
            fontsize=col_fs,
        )

    for r, (row, (ax0, ax1, ax2, cax_f, cax_e)) in enumerate(zip(rows, row_axes)):
        X_mat = np.asarray(row["X_mat"], dtype=float)
        Y_mat = np.asarray(row["Y_mat"], dtype=float)
        X_py = np.asarray(row["X_py"], dtype=float)
        Y_py = np.asarray(row["Y_py"], dtype=float)
        matlab_grid = np.asarray(row["matlab_grid"], dtype=float)
        python_native = np.asarray(row["python_grid"], dtype=float)
        python_aligned = (
            align_grid_to_reference(matlab_grid, python_native)
            if matlab_grid.shape != python_native.shape
            else python_native
        )
        peak = float(
            np.nanmax(
                np.abs(np.stack([matlab_grid, python_native, python_aligned]))
            )
        )
        peak = max(peak, 1e-30)
        matlab_n = matlab_grid / peak
        python_n = python_native / peak
        python_a = python_aligned / peak
        err_pct = 100.0 * np.abs(matlab_n - python_a)
        err_finite = err_pct[np.isfinite(err_pct)]
        err_vmax = float(np.nanmax(err_finite)) if err_finite.size else 1.0
        err_vmax = max(err_vmax, 1e-30)

        kw = dict(cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
        im0 = ax0.pcolormesh(X_mat, Y_mat, matlab_n, **kw)
        ax1.pcolormesh(X_py, Y_py, python_n, **kw)
        im2 = ax2.pcolormesh(
            X_mat, Y_mat, err_pct, cmap="magma", vmin=0.0, vmax=err_vmax, shading="auto"
        )

        # σ above forcing panels only (dedicated strip — never clips into axes).
        if sigma_y[r] is not None:
            ref_gain = row.get("ref_gain")
            py_gain = row.get("py_gain")
            if ref_gain is not None:
                p0 = ax0.get_position()
                fig.text(
                    0.5 * (p0.x0 + p0.x1),
                    float(sigma_y[r]),
                    rf"$\sigma={float(ref_gain):.4g}$",
                    ha="center",
                    va="center",
                    fontsize=sigma_fs,
                )
            if py_gain is not None:
                p1 = ax1.get_position()
                fig.text(
                    0.5 * (p1.x0 + p1.x1),
                    float(sigma_y[r]),
                    rf"$\sigma={float(py_gain):.4g}$",
                    ha="center",
                    va="center",
                    fontsize=sigma_fs,
                )

        # Optional mode banners (e.g. ω): per-column if banner_ref / banner_py,
        # else a single ``banner`` centred over the two field panels.
        if banner_y[r] is not None:
            yb = float(banner_y[r])
            p0 = ax0.get_position()
            p1 = ax1.get_position()
            b_ref = row.get("banner_ref")
            b_py = row.get("banner_py")
            if b_ref or b_py:
                if b_ref:
                    fig.text(
                        0.5 * (p0.x0 + p0.x1),
                        yb,
                        str(b_ref),
                        ha="center",
                        va="center",
                        fontsize=sigma_fs,
                    )
                if b_py:
                    fig.text(
                        0.5 * (p1.x0 + p1.x1),
                        yb,
                        str(b_py),
                        ha="center",
                        va="center",
                        fontsize=sigma_fs,
                    )
            elif row.get("banner"):
                fig.text(
                    0.5 * (p0.x0 + p1.x1),
                    yb,
                    str(row["banner"]),
                    ha="center",
                    va="center",
                    fontsize=sigma_fs,
                )

        show_xlabel = r == n_rows - 1
        raw = str(row.get("row_label", f"row {r + 1}")).strip()
        if raw.startswith("$") and raw.endswith("$"):
            side = rf"$y-{raw[1:-1]}$"
        else:
            side = rf"$y$-{raw}"

        for i, ax in enumerate((ax0, ax1, ax2)):
            ax.set_aspect("auto")
            if ylim is not None:
                ax.set_ylim(float(ylim[0]), float(ylim[1]))
            if xlim is not None:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))
            ax.tick_params(labelsize=tick_fs)
            if show_xlabel:
                ax.set_xlabel(r"$x$", fontsize=label_fs)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if i == 0:
                ax.set_ylabel(side, fontsize=side_fs, labelpad=8)
                ax.tick_params(axis="y", left=True, labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=True, labelleft=False)

        cbar0 = fig.colorbar(im0, cax=cax_f)
        cbar0.ax.tick_params(labelsize=cbar_fs)
        cbar2 = fig.colorbar(im2, cax=cax_e)
        cbar2.ax.tick_params(labelsize=cbar_fs)

        # Colorbar construction can nudge shared autoscaling; lock limits again.
        for ax in (ax0, ax1, ax2):
            if ylim is not None:
                y0, y1 = float(ylim[0]), float(ylim[1])
                ax.set_ylim(y0, y1)
                # Few ticks so stacked mosaic rows do not collide at shared edges.
                mid = 0.5 * (y0 + y1)
                ax.set_yticks([y0, mid, y1])
            if xlim is not None:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pad_inches keeps title/header breathing room that tight crop otherwise eats.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def match_mode_pairs(
    modes_mat: np.ndarray,
    modes_py: np.ndarray,
    n_pairs: int,
) -> list[tuple[int, int]]:
    """Match modes by gain vector correlation proxy using column norms + hungarian on shape overlap."""
    mat = _mode_matrix(modes_mat)
    py = _mode_matrix(modes_py)
    n = min(n_pairs, mat.shape[1], py.shape[1])
    if mat.shape[0] != py.shape[0]:
        return []
    # Use correlation magnitude between columns as cost
    cost = np.zeros((mat.shape[1], py.shape[1]))
    for i in range(mat.shape[1]):
        for j in range(py.shape[1]):
            a = mat[:, i]
            b = py[:, j]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                cost[i, j] = 1.0
            else:
                cost[i, j] = 1.0 - abs(np.vdot(a, b)) / denom
    ridx, cidx = linear_sum_assignment(cost)
    pairs = list(zip(ridx.tolist(), cidx.tolist()))
    pairs.sort(key=lambda rc: cost[rc[0], rc[1]])
    return [(int(i), int(j)) for i, j in pairs[:n]]


def normalize_grid_by_max(grid: np.ndarray) -> np.ndarray:
    """Return ``grid / max(|grid|)`` for shape comparison on ``[0, 1]``."""
    grid = np.asarray(grid, dtype=float)
    peak = float(np.nanmax(np.abs(grid)))
    if not np.isfinite(peak) or peak <= 0.0:
        return grid
    return grid / peak


def scaled_python_mode(
    modes_py: np.ndarray,
    modes_mat: np.ndarray,
    py_index: int,
    mat_index: int,
) -> np.ndarray:
    ref = _mode_matrix(modes_mat)[:, mat_index]
    cand = _mode_matrix(modes_py)[:, py_index]
    return cand * _best_complex_scale(ref, cand)
