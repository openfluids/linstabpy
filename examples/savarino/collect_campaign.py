#!/usr/bin/env python3
"""Collects per-point outputs into the run-record CSV.

Produces a CSV in the same layout as ``savarino_map2_200pt.csv``,
the 200-point resolvent gain map (supplement figure S6). Requires the
Slurm output logs (``slurm-*.out``/``.err``) written into each point
directory by a campaign generated with ``gen_campaign.py``.

Run:  python collect_campaign.py results/campaign [machine_label] > results/campaign.csv

Every row carries its own provenance (commit SHA, machine, ranks, backend,
the forcing restriction P, epsilon, DOF, nnz) so it stays meaningful when
copied out of the repository.

Points that have not finished, or that failed, are reported with status !=
"ok" rather than silently omitted: a map with holes in it must say where
the holes are.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/campaign")
MACHINE = sys.argv[2] if len(sys.argv) > 2 else ""

PAT = {
    "gain": re.compile(r"leading gain\s*:\s*([0-9eE.+-]+)"),
    "resid": re.compile(r"resolvent residual \(fwd, adj\)\s*:\s*([0-9eE.+-]+),\s*([0-9eE.+-]+)"),
    "wall": re.compile(r"compute finished in ([0-9.]+)s"),
    "rss": re.compile(r"peak RSS \[bytes\]\s*:\s*([0-9]+)"),
    "ndof": re.compile(r"ndof\s*:\s*([0-9]+)"),
    # Anchored to the assembly line: a bare `nnz=` also matches `sponge_nnz=`.
    "nnz": re.compile(r"operator assembled with BCs:.*?nnz=([0-9]+)"),
    "hdr": re.compile(r"SHA=(\S+)\s+BETA=(\S+)\s+F=(\S+)"),
    "eps": re.compile(r"epsilon -> resolvent \(effective\)\s*:\s*([0-9eE.+-]+)"),
}

DIRPAT = re.compile(r"sav_map2?_b(?P<b>[0-9p]+)_f(?P<f>[0-9pm]+)(?P<suffix>_[A-Za-z]+)?$")


def backend_of(point_dir: Path) -> tuple[str, int]:
    """(backend, ranks) for a point directory, read from the job's own files.

    ``<root>/<stem>.sbatch`` carries ``#SBATCH --ntasks=N`` and
    ``<root>/configs/<stem>.yaml`` the resolvent solver block ("mumps", or
    "iterative/strumpack" for FGMRES around a STRUMPACK LU -- the GPU arm), so
    the row says what actually ran. Without those files fall back to the
    directory name: a '_GPU' suffix marks a STRUMPACK/GPU variant at 1 rank,
    anything else MUMPS at the CPU default of 4. A row that claimed MUMPS on
    the CPU rank count for a GPU point would be false provenance in a file
    whose whole purpose is that each row describes its own run.
    """
    point_dir = Path(point_dir)
    name = point_dir.name
    backend, ranks = ("strumpack", 1) if name.endswith("_GPU") else ("mumps", 4)
    sbatch = point_dir.parent / f"{name}.sbatch"
    if sbatch.is_file():
        m = re.search(r"^#SBATCH --ntasks=(\d+)", sbatch.read_text(errors="replace"), re.M)
        if m:
            ranks = int(m.group(1))
    config = point_dir.parent / "configs" / f"{name}.yaml"
    if config.is_file():
        try:
            import yaml

            solver = (yaml.safe_load(config.read_text()) or {}).get("solver") or {}
        except Exception:  # noqa: BLE001 - provenance falls back to the name rule
            solver = {}
        res = solver.get("resolvent") or {}
        solver_type = str(res.get("type") or "mumps").lower()
        factor = str(res.get("factor_backend") or solver.get("factor_backend") or "").lower()
        if solver_type in ("mumps", "direct", "lu"):
            backend = factor or solver_type
        else:
            backend = f"{solver_type}/{factor}" if factor else solver_type
    return backend, ranks


def gpu_peak_mib(point_dir: Path) -> str:
    """Peak device memory (MiB) from the job's ``nvsmi.<job>.log`` sampler, or ''."""
    peak = None
    for log in Path(point_dir).glob("nvsmi.*.log"):
        for line in log.read_text(errors="replace").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    used = int(float(parts[1]))
                except ValueError:
                    continue
                peak = used if peak is None else max(peak, used)
    return "" if peak is None else str(peak)


def point_from_dirname(name: str) -> tuple[float | None, float | None]:
    """Recover (beta, f) from the directory name so an unfinished point is still
    identifiable in the CSV. The generator encodes '.' as 'p' and '-' as 'm'."""
    m = DIRPAT.match(name)
    if m is None:
        return None, None

    def un(s: str) -> float | None:
        try:
            return float(s.replace("p", ".").replace("m", "-"))
        except ValueError:
            return None

    return un(m.group("b")), un(m.group("f"))


HEADER = (
    "beta_Xsh,f_Xsh_Uinf,gain,rho_fwd,rho_adj,status,wall_s,peak_rss_bytes,ndof,nnz,"
    "epsilon,stride,fd_order,sponge,forcing_norm,response_norm,P_vars,P_x_min,P_x_max,"
    "P_y_max,mask_bc_forcing,ranks,backend,git_sha,slurm_job,machine,gpu_peak_mib"
)


def main() -> int:
    print(HEADER)
    rows = []
    # Directories only: the .sbatch files live alongside them and share the prefix.
    for d in sorted(p for p in ROOT.glob("sav_map*_b*") if p.is_dir()):
        outs = sorted(d.glob("slurm-*.out"))
        if not outs:
            rows.append((None, None, "not_started", d.name, None))
            continue
        out = outs[-1]
        job = out.stem.split("-")[-1]
        txt = out.read_text(errors="replace")

        h = PAT["hdr"].search(txt)
        if h is None:
            rows.append((None, None, "no_header", d.name, job))
            continue
        sha, beta, f = h.group(1)[:7], float(h.group(2)), float(h.group(3))

        g = PAT["gain"].search(txt)
        w = PAT["wall"].search(txt)
        if g is None or w is None:
            err = (d / out.name.replace(".out", ".err")).read_text(errors="replace")[-200:]
            status = "failed" if "Error" in err or "Traceback" in err else "running"
            rows.append((beta, f, status, d.name, job))
            continue

        r = PAT["resid"].search(txt)
        rss = PAT["rss"].search(txt)
        ndof = PAT["ndof"].search(txt)
        nnz = PAT["nnz"].search(txt)
        eps = PAT["eps"].search(txt)
        # x_max follows the measured frequency split (see gen_campaign.py).
        x_max = 1.9470833 if (f == 0 or f >= 2) else 1.7954167
        backend, ranks = backend_of(d)
        print(
            f"{beta},{f},{g.group(1)},"
            f"{r.group(1) if r else ''},{r.group(2) if r else ''},ok,"
            f"{w.group(1)},{rss.group(1) if rss else ''},"
            f"{ndof.group(1) if ndof else ''},{nnz.group(1) if nnz else ''},"
            f"{eps.group(1) if eps else ''},1,4,false,l2,chu,"
            f"u|v|w,0.20291667,{x_max},0.29627645,true,{ranks},{backend},{sha},{job},{MACHINE},"
            f"{gpu_peak_mib(d)}"
        )

    for beta, f, status, name, job in rows:
        if beta is None or f is None:
            beta, f = point_from_dirname(name)
        backend, ranks = backend_of(ROOT / name)
        print(
            f"{beta if beta is not None else ''},{f if f is not None else ''},,,,"
            f"{status},,,,,,1,4,false,l2,chu,u|v|w,0.20291667,,0.29627645,true,{ranks},"
            f"{backend},,{job or ''},{MACHINE},"
        )
        print(f"# incomplete: {name} -> {status}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
