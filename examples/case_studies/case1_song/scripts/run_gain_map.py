#!/usr/bin/env python3
"""Compute a Case 1 Song/Hamada SWBLI resolvent gain map."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
ROOT = SCRIPT.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstabpy.baseflow import load_hamada_case, sutherland_air
from linstabpy.mesh import Mesh
from linstabpy.solvers.router.resolvent_map import resolvent_map
from linstabpy.utils.jsonc import load_jsonc
from linstabpy.utils.provenance import collect_provenance

DEFAULT_CASE_DIR = CASE_DIR / "data" / "baseflow"
DEFAULT_OUTPUT_DIR = CASE_DIR / "output" / "gain_map"
X_SH_M = 0.08
GAMMA_AIR = 1.4
R_AIR = 287.058


def _cfg_get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _case_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else CASE_DIR / raw


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_axes(
    n_freq: int, n_beta: int, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_freq < 1:
        raise ValueError("--n-freq must be >= 1")
    if n_beta < 1:
        raise ValueError("--n-beta must be >= 1")

    f_nondim = np.logspace(-4.0, 1.0, int(n_freq))
    beta_nondim = np.linspace(0.0, 200.0, int(n_beta))

    ratio_l = float(_cfg_get(cfg, "analysis.ratio_l", 0.1))
    ratio_uinf = float(_cfg_get(cfg, "analysis.ratio_uinf", 1.0))
    if ratio_l <= 0.0 or ratio_uinf <= 0.0:
        raise ValueError("analysis.ratio_l and analysis.ratio_uinf must be positive")

    # Operator-native nondim (NOT SI). The solver's omega/beta live in the same
    # nondimensionalization as the base flow loaded by load_hamada_case (mesh is
    # nondimensional: x/L ~ O(1)). Mirror the canonical St->omega map used by the
    # production driver (driver/run.py): omega = 2*pi*St*ratio_uinf/ratio_l, with the
    # bead's dimensionless frequency f*X_sh/U_inf playing the role of St.
    omega = 2.0 * np.pi * f_nondim * ratio_uinf / ratio_l
    # Song's spanwise nondim is beta*L (the unstable stripe is beta*L ~ 7.85..32,
    # lambda/L ~ 0.2..0.8). With L ~ X_sh (shock slice at x/L ~ 1, shock at 80 mm),
    # the operator beta is taken directly as the bead's beta*X_sh axis. Axis
    # calibration (L vs X_sh) is an inference; the qualitative ridge structure is not.
    beta = beta_nondim.copy()
    return f_nondim, beta_nondim, omega, beta


def _load_case_and_mesh(case_dir: Path, cfg: dict[str, Any]):
    physics = cfg.get("physics", {}) if isinstance(cfg.get("physics"), dict) else {}
    numerics = cfg.get("numerics", {}) if isinstance(cfg.get("numerics"), dict) else {}
    sponge_cfg = physics.get("sponge") if isinstance(physics.get("sponge"), dict) else None
    native_sponge_cfg = sponge_cfg if bool((sponge_cfg or {}).get("enabled", False)) else None
    sponge_mat = _case_path(
        _cfg_get(
            cfg,
            "paths.case_inputs_mat",
            "reference/matlab/MATLAB_exports_resolvent/case_inputs_used.mat",
        )
    )
    use_matlab_sponge_aux = bool(physics.get("use_matlab_sponge_aux", False))

    X, Y, base_flow, sponge = load_hamada_case(
        case_dir,
        Re=float(physics.get("Re", 395000.0)),
        Ma=float(physics.get("Ma", 2.0)),
        Pr=float(physics.get("Pr", 0.747)),
        kappa=float(physics.get("kappa", 1.4)),
        T_0=float(physics.get("T_0", 1540.0)),
        stride_ksi=int(numerics.get("stride_ksi", 1)),
        stride_eta=int(numerics.get("stride_eta", 1)),
        transpose_input=True,
        zero_spanwise=bool(physics.get("zero_spanwise", False)),
        sponge_mat_path=sponge_mat,
        prefer_matlab_sponge_aux=use_matlab_sponge_aux and sponge_mat.exists(),
        native_sponge_cfg=native_sponge_cfg,
    )
    sutherland_air(base_flow)
    if bool(physics.get("zero_mu_t_derivs", False)):
        base_flow.dmudT[:] = 0.0
        base_flow.d2mudT2[:] = 0.0

    ny, nx = X.shape
    mesh = Mesh.create(
        (0.0, 1.0),
        (0.0, 0.5),
        nx=nx,
        ny=ny,
        fd_order=int(numerics.get("fd_order", 4)),
        alpha_filter=numerics.get("alpha_filter", "none"),
        numerics_operator_dissipation=numerics.get("operator_dissipation"),
    ).deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge
    return base_flow, mesh


def _router_cfg(cfg: dict[str, Any], mesh: Mesh) -> dict[str, Any]:
    run_cfg = deepcopy(cfg)
    run_cfg["mesh"] = mesh
    run_cfg["bc_penalty"] = float(_cfg_get(cfg, "bc.penalty", 1.0e8))
    run_cfg["fd_order"] = int(_cfg_get(cfg, "numerics.fd_order", 4))
    run_cfg["alpha_filter"] = _cfg_get(cfg, "numerics.alpha_filter", "none")
    return run_cfg


def run(args: argparse.Namespace) -> Path:
    config_path = _case_path(args.config)
    cfg = load_jsonc(config_path)
    case_dir = _case_path(args.case_dir)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    f_nondim, beta_nondim, omega_range, beta_range = _build_axes(args.n_freq, args.n_beta, cfg)
    base_flow, mesh = _load_case_and_mesh(case_dir, cfg)
    result = resolvent_map(base_flow, beta_range, omega_range, _router_cfg(cfg, mesh))

    provenance = collect_provenance(config_path)
    metadata = {
        "case_dir": str(case_dir),
        "config": str(config_path),
        "x_sh_m": X_SH_M,
        "angle_deg": "<TODO: verify>",
        "angle_source": "not found in local base-flow metadata, MATLAB exports, or config comments",
        "f_axis": "f*X_sh/U_inf",
        "beta_axis": "beta*X_sh",
    }
    out = output_dir / f"case1_gain_map_{_now_tag()}.npz"
    np.savez_compressed(
        out,
        beta=result.beta,
        f=result.f,
        gain=result.gain,
        regime_label=result.regime_label,
        reason_code=result.reason_code,
        cost_metrics=np.array(json.dumps(_jsonable(result.cost_metrics), sort_keys=True)),
        f_nondim=f_nondim,
        beta_nondim=beta_nondim,
        provenance=np.array(json.dumps(_jsonable(provenance), sort_keys=True)),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"wrote {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-freq", type=int, default=5)
    parser.add_argument("--n-beta", type=int, default=10)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def help() -> None:
    build_parser().print_help()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
