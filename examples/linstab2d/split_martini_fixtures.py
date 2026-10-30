#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Split extracted Martini NPZ bundles into values + modes files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODE_LEAVES = frozenset({"V", "VAdj", "responses", "forces"})

# Match linstabpy resolvent/EVP storage (``dtype=complex`` → complex128) and full
# campaign MATLAB references; do not downcast slim bundles to complex64.
MODE_DTYPE = np.complex128

# Resolvent references that ship compact mode DOFs for presentation plots.
RESOLVENT_UV_STEMS = (
    "example_1_Blasius_results",
    "example_2_Jet_results",
)
EX5_STEM = "example_5_Paraboloid_results"

# Stems that ship as *_values.npz + *_modes.npz in LinStab2DResults/.
SPLIT_STEM_SUFFIXES: tuple[str, ...] = ()

EX3_TEMPORAL_STEM_MARK = "example_3_paper"
EX3_N_PLOT_MODES = 3

EX4_DIRECT_STEM = "example_4_Cavity_results_direct"
EX4_ADJOINT_STEM = "example_4_Cavity_results_adjoint"
EX4_N_PLOT_MODES = 3
EX4_UVW_COMPONENTS = ("u", "v", "w")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _mode_leaf(key: str) -> str:
    if key.startswith("__meta"):
        return key
    if "__" in key:
        return key.rsplit("__", 1)[-1]
    if "/" in key:
        return key.rsplit("/", 1)[-1]
    return key


def is_mode_key(key: str) -> bool:
    return _mode_leaf(key) in MODE_LEAVES


def should_split_stem(stem: str) -> bool:
    return any(stem.endswith(suffix) for suffix in SPLIT_STEM_SUFFIXES)


def _resolvent_struct_prefix(payload: dict[str, np.ndarray]) -> str | None:
    """Return the struct key prefix for a resolvent u+v reference, if present."""
    force_keys = [
        key for key in payload if not key.startswith("__meta") and key.endswith("__forces")
    ]
    for forces_key in sorted(force_keys, key=len):
        prefix = forces_key[: -len("forces")]
        if f"{prefix}idx__u_j" in payload and f"{prefix}idx__v_j" in payload:
            return prefix
    return None


def _mode_matrix(values: np.ndarray) -> np.ndarray:
    from examples.linstab2d.compare_martini_examples import _mode_matrix as mode_matrix

    return mode_matrix(values)


def slice_resolvent_uv(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop rho/w/T DOFs from resolvent modes; keep compact u+v layout for plots.

    Mode columns stack as u-then-v.
    ``idx__u_j`` / ``idx__v_j`` keep the original global DOF labels from the
    full-stack export (not renumbered to 1..2*n_gp).
    Plot code maps them to compact columns when gridding.
    """
    prefix = _resolvent_struct_prefix(payload)
    if prefix is None:
        return payload

    forces_key = f"{prefix}forces"
    responses_key = f"{prefix}responses"
    uj_key = f"{prefix}idx__u_j"
    vj_key = f"{prefix}idx__v_j"

    uj = np.asarray(payload[uj_key], dtype=int).reshape(-1)
    vj = np.asarray(payload[vj_key], dtype=int).reshape(-1)
    one_based = int(uj.min()) >= 1
    u_idx = uj - 1 if one_based else uj
    v_idx = vj - 1 if one_based else vj
    n_gp = int(u_idx.size)
    if int(v_idx.size) != n_gp:
        raise ValueError(f"{uj_key} and {vj_key} length mismatch")

    forces_m = _mode_matrix(payload[forces_key])
    responses_m = _mode_matrix(payload[responses_key])

    out = dict(payload)
    if forces_m.shape[0] != 2 * n_gp:
        compact_f = np.concatenate([forces_m[u_idx, :], forces_m[v_idx, :]], axis=0)
        compact_r = np.concatenate([responses_m[u_idx, :], responses_m[v_idx, :]], axis=0)
        out[forces_key] = compact_f.T.astype(MODE_DTYPE, copy=False)
        out[responses_key] = compact_r.T.astype(MODE_DTYPE, copy=False)

    drop_suffixes = (
        "idx__T_j",
        "idx__rho_j",
        "idx__w_j",
        "St",
        "omega",
        "nEig",
        "mode_plot_indices",
    )
    for key in list(out):
        if key.startswith("__meta"):
            continue
        if not key.startswith(prefix):
            continue
        leaf = key[len(prefix) :]
        if leaf in drop_suffixes:
            del out[key]
    return out


def maybe_slice_resolvent_uv(stem: str, payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if stem not in RESOLVENT_UV_STEMS:
        return payload
    return slice_resolvent_uv(payload)


def _paraboloid_prefix(payload: dict[str, np.ndarray]) -> str | None:
    if "results_Paraboloid__forces" in payload:
        return "results_Paraboloid__"
    return None


def slice_resolvent_uvw(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop rho/T DOFs from Ex.~5 resolvent modes; keep compact u+v+w layout for plots.

    Mode columns stack as u-then-v-then-w.
    Global ``idx__*_j`` labels stay as exported (not renumbered).
    Plot code maps them to compact columns when gridding.
    """
    try:
        forces_key = _payload_key(payload, "forces")
    except KeyError:
        return payload
    if "Paraboloid" not in forces_key:
        return payload

    responses_key = _payload_key(payload, "responses")
    gains_key = _payload_key(payload, "gains")
    uj_key = _payload_key(payload, "idx__u_j")
    vj_key = _payload_key(payload, "idx__v_j")
    wj_key = _payload_key(payload, "idx__w_j")

    uj = np.asarray(payload[uj_key], dtype=int).reshape(-1)
    vj = np.asarray(payload[vj_key], dtype=int).reshape(-1)
    wj = np.asarray(payload[wj_key], dtype=int).reshape(-1)
    one_based = int(uj.min()) >= 1
    u_idx = uj - 1 if one_based else uj
    v_idx = vj - 1 if one_based else vj
    w_idx = wj - 1 if one_based else wj
    n_gp = int(u_idx.size)
    if int(v_idx.size) != n_gp or int(w_idx.size) != n_gp:
        raise ValueError("Paraboloid u/v/w idx length mismatch")

    forces_m = _mode_matrix(payload[forces_key])
    responses_m = _mode_matrix(payload[responses_key])

    out = dict(payload)
    if forces_m.shape[0] != 3 * n_gp:
        compact_f = np.concatenate(
            [forces_m[u_idx, :], forces_m[v_idx, :], forces_m[w_idx, :]], axis=0
        )
        compact_r = np.concatenate(
            [responses_m[u_idx, :], responses_m[v_idx, :], responses_m[w_idx, :]], axis=0
        )
        out[forces_key] = compact_f.T.astype(MODE_DTYPE, copy=False)
        out[responses_key] = compact_r.T.astype(MODE_DTYPE, copy=False)

    drop_suffixes = ("idx__T_j", "idx__rho_j", "St", "omega", "nEig", "mode_plot_indices")
    prefix = forces_key[: -len("forces")]
    for key in list(out):
        if key.startswith("__meta"):
            continue
        if not key.startswith(prefix):
            continue
        leaf = key[len(prefix) :]
        if leaf in drop_suffixes:
            del out[key]
    return out


def maybe_slice_resolvent_uvw(stem: str, payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if stem != EX5_STEM:
        return payload
    return slice_resolvent_uvw(payload)


def _temporal_struct_prefix(payload: dict[str, np.ndarray]) -> str | None:
    for prefix in ("results__results_StreakyBL_temporal__", "results_StreakyBL_temporal__"):
        if f"{prefix}omega" in payload and f"{prefix}V" in payload:
            return prefix
    return None


def _spatial_struct_prefix(payload: dict[str, np.ndarray]) -> str | None:
    for prefix in ("results__results_StreakyBL_spatial__", "results_StreakyBL_spatial__"):
        if f"{prefix}alphas" in payload:
            return prefix
    return None


def _complex_vector(values: np.ndarray) -> np.ndarray:
    from examples.linstab2d.plot_compare_utils import complex_vector

    return complex_vector(values)


def slim_ex3_temporal_plot_payload(
    payload: dict[str, np.ndarray],
    *,
    n_modes: int = EX3_N_PLOT_MODES,
) -> dict[str, np.ndarray]:
    """Keep spectrum omega, top-N unstable V columns on w-DOF only, and plot grid."""
    prefix = _temporal_struct_prefix(payload)
    if prefix is None:
        return payload

    omega_key = f"{prefix}omega"
    v_key = f"{prefix}V"
    wj_key = f"{prefix}idx__w_j"
    grid_keys = (
        f"{prefix}plot_grid__X",
        f"{prefix}plot_grid__Y",
        f"{prefix}plot_grid__usedInd",
    )
    for key in (omega_key, v_key, wj_key, *grid_keys):
        if key not in payload:
            raise KeyError(f"Missing {key} in Example 3 temporal payload")

    omega = _complex_vector(payload[omega_key]).reshape(-1)
    modes = _mode_matrix(payload[v_key])
    wj = np.asarray(payload[wj_key], dtype=int).reshape(-1)
    one_based = int(wj.min()) >= 1
    w_idx = wj - 1 if one_based else wj

    eig_order = np.argsort(-np.imag(omega))[:n_modes]
    saved_1based = (eig_order + 1).astype(np.float64)
    compact_v = modes[w_idx, :][:, eig_order].T.astype(MODE_DTYPE, copy=False)

    wj_shape = payload[wj_key].shape
    if one_based:
        new_wj = np.arange(1, w_idx.size + 1, dtype=np.float64).reshape(wj_shape)
    else:
        new_wj = np.arange(0, w_idx.size, dtype=np.float64).reshape(wj_shape)

    out = {
        omega_key: np.asarray(payload[omega_key]),
        v_key: compact_v,
        f"{prefix}saved_mode_indices": saved_1based.reshape(1, -1),
        wj_key: new_wj,
    }
    for key in grid_keys:
        out[key] = np.asarray(payload[key])
    return out


def slim_ex3_spatial_plot_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Spatial MATLAB reference is spectrum-only for plots."""
    prefix = _spatial_struct_prefix(payload)
    if prefix is None:
        return payload
    alphas_key = f"{prefix}alphas"
    if alphas_key not in payload:
        raise KeyError(f"Missing {alphas_key} in Example 3 spatial payload")
    return {alphas_key: np.asarray(payload[alphas_key])}


def maybe_slim_ex3_plot_bundle(stem: str, payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if EX3_TEMPORAL_STEM_MARK in stem and stem.endswith("_results_temporal"):
        return slim_ex3_temporal_plot_payload(payload)
    if EX3_TEMPORAL_STEM_MARK in stem and stem.endswith("_results_spatial"):
        return slim_ex3_spatial_plot_payload(payload)
    return payload


def _payload_key(payload: dict[str, np.ndarray], leaf: str) -> str:
    hits = [
        k
        for k in payload
        if not k.startswith("__meta") and (k.endswith(f"__{leaf}") or k.rsplit("__", 1)[-1] == leaf)
    ]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits.sort(key=len)
        return hits[0]
    raise KeyError(f"Missing payload key with leaf {leaf!r}")


def _cavity_uvw_index_ranges(n_dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n_gp = n_dof // 5
    u_idx = np.arange(n_gp, 2 * n_gp, dtype=int)
    v_idx = np.arange(2 * n_gp, 3 * n_gp, dtype=int)
    w_idx = np.arange(3 * n_gp, 4 * n_gp, dtype=int)
    return u_idx, v_idx, w_idx, n_gp


def _cavity_uvw_indices_from_payload(
    payload: dict[str, np.ndarray],
    *,
    n_dof: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    try:
        uj = np.asarray(payload[_payload_key(payload, "idx__u_j")], dtype=int).reshape(-1)
        vj = np.asarray(payload[_payload_key(payload, "idx__v_j")], dtype=int).reshape(-1)
        wj = np.asarray(payload[_payload_key(payload, "idx__w_j")], dtype=int).reshape(-1)
        one_based = int(uj.min()) >= 1
        u_idx = uj - 1 if one_based else uj
        v_idx = vj - 1 if one_based else vj
        w_idx = wj - 1 if one_based else wj
        n_gp = int(u_idx.size)
        return u_idx, v_idx, w_idx, n_gp, one_based
    except KeyError:
        u_idx, v_idx, w_idx, n_gp = _cavity_uvw_index_ranges(n_dof)
        return u_idx, v_idx, w_idx, n_gp, False


def _compact_cavity_uvw_modes(
    modes: np.ndarray,
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], int, bool]:
    u_idx, v_idx, w_idx, n_gp, one_based = _cavity_uvw_indices_from_payload(
        payload, n_dof=int(modes.shape[0])
    )
    compact = np.concatenate([modes[u_idx, :], modes[v_idx, :], modes[w_idx, :]], axis=0)
    idx_updates: dict[str, np.ndarray] = {}
    for comp, offset in zip(EX4_UVW_COMPONENTS, (0, n_gp, 2 * n_gp)):
        try:
            key = _payload_key(payload, f"idx__{comp}_j")
            shape = np.asarray(payload[key]).shape
        except KeyError:
            key = f"idx__{comp}_j"
            shape = (1, n_gp)
        if one_based:
            values = np.arange(offset + 1, offset + n_gp + 1, dtype=np.float64)
        else:
            values = np.arange(offset, offset + n_gp, dtype=np.float64)
        idx_updates[key] = values.reshape(shape)
    return compact, idx_updates, n_gp, one_based


def _adjoint_omega_direct_convention(values: np.ndarray) -> np.ndarray:
    omega = _complex_vector(values).reshape(-1)
    return np.conjugate(omega)


def slim_ex4_cavity_direct_plot_payload(
    payload: dict[str, np.ndarray],
    *,
    n_modes: int = EX4_N_PLOT_MODES,
) -> dict[str, np.ndarray]:
    omega_key = _payload_key(payload, "omega")
    mode_key = _payload_key(payload, "V")
    grid_keys = tuple(
        _payload_key(payload, leaf)
        for leaf in ("plot_grid__X", "plot_grid__Y", "plot_grid__usedInd")
    )

    omega = _complex_vector(payload[omega_key]).reshape(-1)
    modes = _mode_matrix(payload[mode_key])
    eig_order = np.argsort(-np.imag(omega))[:n_modes]
    prefix = mode_key.rsplit("__", 1)[0] + "__"
    saved_key = f"{prefix}saved_mode_indices"

    compact, idx_updates, _, _ = _compact_cavity_uvw_modes(modes[:, eig_order], payload)
    out = {
        omega_key: np.asarray(payload[omega_key]),
        mode_key: compact.T.astype(MODE_DTYPE, copy=False),
        saved_key: (eig_order + 1).astype(np.float64).reshape(1, -1),
    }
    for key in grid_keys:
        out[key] = np.asarray(payload[key])
    out.update(idx_updates)
    return out


def slim_ex4_cavity_adjoint_plot_payload(
    payload: dict[str, np.ndarray],
    *,
    n_modes: int = EX4_N_PLOT_MODES,
    idx_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    omega_key = _payload_key(payload, "omegaAdj")
    mode_key = _payload_key(payload, "VAdj")
    idx_source = idx_payload if idx_payload is not None else payload

    omega = _adjoint_omega_direct_convention(payload[omega_key])
    modes = _mode_matrix(payload[mode_key])
    eig_order = np.argsort(-np.imag(omega))[:n_modes]
    prefix = mode_key.rsplit("__", 1)[0] + "__"
    saved_key = f"{prefix}saved_mode_indices"

    compact, _, _, _ = _compact_cavity_uvw_modes(modes[:, eig_order], idx_source)
    return {
        omega_key: np.asarray(payload[omega_key]),
        mode_key: compact.T.astype(MODE_DTYPE, copy=False),
        saved_key: (eig_order + 1).astype(np.float64).reshape(1, -1),
    }


def maybe_slim_ex4_plot_bundle(stem: str, payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if stem == EX4_DIRECT_STEM:
        return slim_ex4_cavity_direct_plot_payload(payload)
    if stem == EX4_ADJOINT_STEM:
        return slim_ex4_cavity_adjoint_plot_payload(payload)
    return payload


def split_payload(
    payload: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Partition one extracted NPZ dict into values vs mode arrays."""
    meta = {k: v for k, v in payload.items() if k.startswith("__meta")}
    values: dict[str, np.ndarray] = dict(meta)
    modes: dict[str, np.ndarray] = {}

    for key, value in payload.items():
        if key.startswith("__meta"):
            continue
        if is_mode_key(key):
            modes[key] = np.asarray(value)
        else:
            values[key] = np.asarray(value)

    return values, modes


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def process_file(src: Path, output_dir: Path) -> list[dict[str, str | int]]:
    with np.load(src, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    stem = src.stem
    records: list[dict[str, str | int]] = []

    if should_split_stem(stem) and any(is_mode_key(k) for k in payload):
        values, modes = split_payload(payload)
        if modes:
            values_path = output_dir / f"{stem}_values.npz"
            modes_path = output_dir / f"{stem}_modes.npz"
            write_npz(values_path, values)
            write_npz(modes_path, modes)
            records.append(
                {
                    "stem": stem,
                    "kind": "split",
                    "values_file": _display_path(values_path),
                    "modes_file": _display_path(modes_path),
                    "n_value_keys": sum(1 for k in values if not k.startswith("__meta")),
                    "n_mode_keys": len(modes),
                }
            )
            return records

    dst = output_dir / src.name
    payload = maybe_slice_resolvent_uv(stem, payload)
    payload = maybe_slice_resolvent_uvw(stem, payload)
    payload = maybe_slim_ex3_plot_bundle(stem, payload)
    payload = maybe_slim_ex4_plot_bundle(stem, payload)
    write_npz(dst, payload)
    records.append(
        {
            "stem": stem,
            "kind": "copy",
            "npz_file": _display_path(dst),
            "n_keys": sum(1 for k in payload if not k.startswith("__meta")),
        }
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "martini_examples",
        help="Directory with full extracted stem.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples" / "linstab2d" / "LinStab2DResults",
        help="Destination for split/copy fixtures.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "examples" / "linstab2d" / "martini_manifest.json",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for src in sorted(input_dir.glob("*.npz")):
        entries.extend(process_file(src, output_dir))

    manifest = {
        "generated_at_utc": _now_utc(),
        "input_dir": _display_path(input_dir),
        "output_dir": _display_path(output_dir),
        "files": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "Martini fixture split:",
        f"files={len(entries)}",
        f"output_dir={_display_path(output_dir)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
