"""Generate the Case 2 50-point (beta, St) resolvent grid config.

Phase 3 setup deliverable (bead resolvent-gzn).

Emits a single multi-case jsonc consumed by perf_memory.py at:
    studies/case2/configs/resolvent/resolvent_grid_50pt_fd6_stride4.jsonc

with 50 cases, one per (beta, St) point. perf_memory.py provides
per-case failure isolation (status=failed -> continue), so this single
file is equivalent to 50 individual configs but is git-friendlier and
inherits _defaults.jsonc + parity_config.jsonc through $include
(unavailable to the single-case run_case_once.sh path).

Grid:
    beta in {0, 1, 2, 4, 8, 12, 16, 24}     # 8 values
    St   in {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}  # 7 values
    Union: 56 cells; drop 6 high-cost high-(beta, St) cells -> 50.

Drops (deliberately documented; no scientific peak expected there):
    (beta=24, St=5.0)  (beta=24, St=2.0)
    (beta=16, St=5.0)  (beta=16, St=2.0)
    (beta=12, St=5.0)
    (beta=8,  St=5.0)

omega <-> St:
    omega = 2*pi * St    (per case2 PLAN.md and existing phase_r scripts)

Per-case overrides only:
    numerics.beta                  = beta (rad / non-dim spanwise wavenumber)
    analysis.omega_target.real     = 2 * pi * St
    analysis.omega_target.imag     = 0.0  (real-omega resolvent)

Per-default overrides (set once at top-level):
    numerics.{stride_ksi, stride_eta} = 4
    numerics.fd_order                 = 6
    numerics.alpha_filter             = "none"
    analysis.{run_stability, run_resolvent} = (false, true)
    analysis.n_modes_resolvent        = 10
    solver.resolvent.eigen.eps_tol    = 1e-8
    solver.linsolve.type              = "mumps"

Run as:
    cd linstabpy
    PYTHONPATH=src .venv/bin/python studies/case2/scripts/gen_resolvent_grid_50pt.py

The generator is deterministic; re-running produces the same file
byte-for-byte (modulo the embedded generator-version comment). Commit
the generated jsonc alongside the script so the c8 run can git pull a
self-contained snapshot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# --- grid definition ----------------------------------------------

BETA_VALUES: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
ST_VALUES:   tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

# Deliberately dropped high-(beta, St) cells (least scientific interest;
# OM-1 lives at moderate St, LU-S at low St / high beta, neither expects
# a peak at (beta>=8, St>=5) or (beta>=16, St>=2)). Six total -> 50 cells.
DROPS: frozenset[tuple[float, float]] = frozenset({
    (24.0, 5.0),
    (24.0, 2.0),
    (16.0, 5.0),
    (16.0, 2.0),
    (12.0, 5.0),
    (8.0,  5.0),
})

# --- output path --------------------------------------------------

# __file__ = linstabpy/studies/case2/scripts/gen_resolvent_grid_50pt.py
# parents[0]=scripts, [1]=case2, [2]=studies, [3]=linstabpy (repo root).
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "studies" / "case2" / "configs" / "resolvent"
OUT_JSONC = OUT_DIR / "resolvent_grid_50pt_fd6_stride4.jsonc"


def _format_omega(st_value: float) -> dict[str, float]:
    """Convert St to {real, imag} omega for resolvent target."""
    return {"real": 2.0 * math.pi * st_value, "imag": 0.0}


def _case_id(beta: float, st_value: float) -> str:
    """Short stable identifier per (beta, St) point."""
    beta_token = f"{beta:g}".replace(".", "p")
    st_token = f"{st_value:g}".replace(".", "p")
    return f"b{beta_token}_st{st_token}_fd6_stride4"


def _label(beta: float, st_value: float) -> str:
    omega = 2.0 * math.pi * st_value
    return f"beta={beta:g}, St={st_value:g} (omega={omega:.4f})"


def build_cases() -> list[dict]:
    """Materialise the 50-point case list."""
    cases: list[dict] = []
    for beta in BETA_VALUES:
        for st_value in ST_VALUES:
            if (beta, st_value) in DROPS:
                continue
            cases.append(
                {
                    "case_id": _case_id(beta, st_value),
                    "label": _label(beta, st_value),
                    "enabled": True,
                    "config_overrides": {
                        "numerics": {"beta": beta},
                        "analysis": {"omega_target": _format_omega(st_value)},
                    },
                }
            )
    return cases


def build_config(cases: list[dict]) -> dict:
    """Assemble the perf_memory.py multi-case schema."""
    return {
        "$include": "../_defaults.jsonc",
        "outputs": {
            "report_json": "output/resolvent_grid_50pt_fd6_stride4/report_latest.json",
            "report_csv":  "output/resolvent_grid_50pt_fd6_stride4/report_table.csv",
            "log_dir":     "output/resolvent_grid_50pt_fd6_stride4/logs",
            "run_dir":     "output/resolvent_grid_50pt_fd6_stride4/runs",
        },
        "defaults": {
            "case_defaults": {
                "config_overrides": {
                    "numerics": {
                        "stride_ksi": 4,
                        "stride_eta": 4,
                        "fd_order": 6,
                        "alpha_filter": "none",
                    },
                    "analysis": {
                        "run_stability": False,
                        "run_resolvent": True,
                        "n_modes_resolvent": 10,
                    },
                    "solver": {
                        "petsc_matrix_mode": "distributed",
                        "resolvent": {
                            "type": "mumps",
                            "eigen": {
                                "eps_tol": 1e-8,
                                "ncv": 40,
                                "conv_test": "rel",
                                "true_residual": True,
                                "eps_max_it": 1000,
                            },
                        },
                        "linsolve": {"type": "mumps"},
                    },
                },
            },
        },
        "cases": cases,
    }


def render_jsonc(config: dict, n_cases: int, n_dropped: int) -> str:
    """Render config to JSONC with a generator-provenance header."""
    header = (
        "// Phase 3: Case 2 resolvent gain map at fd6 stride-4.\n"
        "//\n"
        "// 50 cases spanning (beta, St) for paper section 6 (Lui resolvent).\n"
        "// Generated by studies/case2/scripts/gen_resolvent_grid_50pt.py;\n"
        "// regenerate via that script after any grid change. Do not\n"
        "// hand-edit (changes will be lost on next regen).\n"
        "//\n"
        f"// {n_cases} cases ({len(BETA_VALUES)}x{len(ST_VALUES)} - {n_dropped} dropped).\n"
        f"// Dropped (beta, St): {sorted(DROPS)}\n"
        "//\n"
        "// Per-point overrides: numerics.beta, analysis.omega_target.real\n"
        "// (omega = 2*pi*St). All other knobs inherit from _defaults.jsonc\n"
        "// (parity_config -> _defaults -> this file).\n"
        "//\n"
        "// Tracked by bead resolvent-gzn (Phase 3 setup) and resolvent-25o\n"
        "// (Phase 3 launch). Expected wall ~47 h serial; ~15 GiB peak RSS per\n"
        "// point at fd6 stride-4 distributed-10 ranks.\n"
        "\n"
    )
    body = json.dumps(config, indent=2, sort_keys=False)
    return header + body + "\n"


def main() -> int:
    cases = build_cases()
    expected = len(BETA_VALUES) * len(ST_VALUES) - len(DROPS)
    if len(cases) != expected:
        raise SystemExit(
            f"Generated {len(cases)} cases, expected {expected} "
            f"(grid {len(BETA_VALUES)}x{len(ST_VALUES)} - {len(DROPS)} drops)"
        )
    config = build_config(cases)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = render_jsonc(config, len(cases), len(DROPS))
    OUT_JSONC.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_JSONC} ({len(cases)} cases, {len(DROPS)} drops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
