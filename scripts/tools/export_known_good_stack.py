#!/usr/bin/env python3
"""Export a machine/install manifest for a known-good linstabpy Linux stack."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"ok": False, "error": f"command not found: {args[0]}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def probe_mpiexec_launchers() -> dict[str, Any]:
    candidates = ["mpiexec.mpich", "mpiexec.hydra", "mpiexec.openmpi", "mpirun.openmpi", "mpiexec"]
    results: list[dict[str, Any]] = []
    best = None
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is None:
            continue
        try:
            proc = subprocess.run(
                [resolved, "-n", "1", "/bin/true"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            ok = proc.returncode == 0
            result = {
                "candidate": candidate,
                "path": resolved,
                "ok": ok,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            ok = False
            result = {
                "candidate": candidate,
                "path": resolved,
                "ok": False,
                "timeout": True,
            }
        results.append(result)
        if ok and best is None:
            best = resolved
    return {
        "best_working_mpiexec": best,
        "candidates": results,
    }


def first_nonempty_line(*chunks: str) -> str | None:
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                return line
    return None


def read_os_release() -> dict[str, str]:
    data: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def detect_mpi_vendor(mpiexec_text: str | None, mpicc_text: str | None) -> str:
    blob = " ".join(filter(None, [mpiexec_text, mpicc_text])).lower()
    if any(token in blob for token in ("open mpi", "openrte", "orte")):
        return "openmpi"
    if any(token in blob for token in ("mpich", "hydra")):
        return "mpich"
    if any(token in blob for token in ("intel mpi", "intel(r) mpi", "impi")):
        return "intelmpi"
    return "unknown"


def sanitize_string(value: str, repo_root: Path, hostname: str) -> str:
    text = value
    home = str(Path.home())
    if home:
        text = text.replace(home, "~")
    text = text.replace(str(repo_root), "<repo-root>")
    if hostname:
        text = text.replace(hostname, "<redacted-host>")
    return text


def sanitize_manifest(value: Any, repo_root: Path, hostname: str) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_manifest(subvalue, repo_root, hostname) for key, subvalue in value.items()
        }
    if isinstance(value, list):
        return [sanitize_manifest(item, repo_root, hostname) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, repo_root, hostname)
    return value


def collect_python_packages() -> dict[str, str | None]:
    pkgs = ["numpy", "scipy", "matplotlib", "mpi4py", "petsc4py", "slepc4py", "h5py"]
    return {name: package_version(name) for name in pkgs}


def collect_petsc_probe() -> dict[str, Any]:
    probe: dict[str, Any] = {}
    try:
        import mpi4py.MPI  # noqa: F401
        import numpy as np
        from petsc4py import PETSc
        from slepc4py import SLEPc

        probe["petsc_version"] = ".".join(map(str, PETSc.Sys.getVersion()))
        probe["slepc_version"] = ".".join(map(str, SLEPc.Sys.getVersion()))
        probe["petsc_scalar_type"] = (
            "complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "real"
        )
    except Exception as exc:  # pragma: no cover - best-effort environment probe
        probe["probe_error"] = repr(exc)
    return probe


def collect_git_info(repo_root: Path) -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False}
    commit = run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    branch = run_command(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_command(["git", "-C", str(repo_root), "status", "--short"])
    return {
        "available": True,
        "commit": commit["stdout"] if commit["ok"] else None,
        "branch": branch["stdout"] if branch["ok"] else None,
        "dirty": bool(status.get("stdout")),
    }


def latest_benchmark_summary(repo_root: Path) -> Path | None:
    bench_dir = repo_root / "benchmark_runs"
    if not bench_dir.is_dir():
        return None
    summaries = sorted(
        bench_dir.glob("*/benchmark_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def parse_benchmark_summary(path: Path, repo_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key in ("ranks", "exit_code"):
                parsed[key] = int(parsed[key])
            for key in ("wall_seconds", "peak_rss_kb"):
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return {
        "source": str(path.relative_to(repo_root)),
        "rows": rows,
    }


def build_manifest(
    repo_root: Path, label: str | None, benchmark_summary: Path | None
) -> dict[str, Any]:
    os_release = read_os_release()
    mpiexec_path = shutil.which("mpiexec")
    mpicc_path = shutil.which("mpicc")
    mpiexec_version = run_command(["mpiexec", "--version"]) if mpiexec_path else {"ok": False}
    mpicc_show = run_command(["mpicc", "--show"]) if mpicc_path else {"ok": False}
    mpiexec_probe = probe_mpiexec_launchers()
    gcc_version = run_command(["gcc", "--version"])
    gfortran_version = run_command(["gfortran", "--version"])
    cmake_version = run_command(["cmake", "--version"])
    uv_version = run_command(["uv", "--version"]) if shutil.which("uv") else {"ok": False}
    lscpu = run_command(["lscpu"]) if shutil.which("lscpu") else {"ok": False}

    benchmark = None
    if benchmark_summary is not None and benchmark_summary.is_file():
        benchmark = parse_benchmark_summary(benchmark_summary, repo_root)

    manifest = {
        "schema_version": 1,
        "kind": "linstabpy_known_good_linux_stack",
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This is a tested reference environment, not a mandatory pin for every Linux machine.",
            "Match the MPI vendor/runtime policy first; exact distro package versions matter less than ABI/toolchain consistency.",
        ],
        "repo": {
            "root": str(repo_root),
            "git": collect_git_info(repo_root),
            "uv_lock_present": (repo_root / "uv.lock").is_file(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "distro": os_release.get("PRETTY_NAME"),
            "distribution_id": os_release.get("ID"),
            "distribution_version": os_release.get("VERSION_ID"),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "toolchain": {
            "gcc": {
                "path": shutil.which("gcc"),
                "version_line": first_nonempty_line(gcc_version.get("stdout", "")),
            },
            "gfortran": {
                "path": shutil.which("gfortran"),
                "version_line": first_nonempty_line(gfortran_version.get("stdout", "")),
            },
            "cmake": {
                "path": shutil.which("cmake"),
                "version_line": first_nonempty_line(cmake_version.get("stdout", "")),
            },
            "uv": {
                "path": shutil.which("uv"),
                "version_line": first_nonempty_line(
                    uv_version.get("stdout", ""), uv_version.get("stderr", "")
                ),
            },
            "mpi": {
                "vendor": detect_mpi_vendor(
                    mpiexec_version.get("stdout"), mpicc_show.get("stdout")
                ),
                "mpiexec_path": mpiexec_path,
                "mpiexec_version_line": first_nonempty_line(
                    mpiexec_version.get("stdout", ""), mpiexec_version.get("stderr", "")
                ),
                "mpicc_path": mpicc_path,
                "mpicc_show": mpicc_show.get("stdout"),
                "best_working_mpiexec": mpiexec_probe["best_working_mpiexec"],
                "launcher_probe": mpiexec_probe["candidates"],
            },
        },
        "linstabpy_runtime_policy": {
            "use_one_mpi_stack_per_environment": True,
            "prefer_example_configs_with_assembly_backend": "scipy",
            "mpi_example_wrapper": "bash run_all_examples.sh --mpi",
            "environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
            "rank_guidance": "Start with half of the physical cores or one socket, then compare against all physical cores.",
        },
        "python_packages": collect_python_packages(),
        "petsc_probe": collect_petsc_probe(),
        "hardware_probe": {
            "lscpu_summary": [
                line
                for line in lscpu.get("stdout", "").splitlines()
                if line.startswith(
                    (
                        "Architecture:",
                        "CPU(s):",
                        "Socket(s):",
                        "Core(s) per socket:",
                        "Thread(s) per core:",
                    )
                )
            ],
        },
        "benchmark_reference": benchmark,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON manifest to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        help="Optional benchmark_summary.csv to embed. Defaults to the latest benchmark_runs/*/benchmark_summary.csv if present.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human-readable label for this stack.",
    )
    parser.add_argument(
        "--redact-local-details",
        action="store_true",
        help="Replace hostname and local absolute paths with portable placeholders. Use this when committing manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    benchmark_summary = args.benchmark_summary
    if benchmark_summary is None:
        benchmark_summary = latest_benchmark_summary(repo_root)
    elif not benchmark_summary.is_absolute():
        benchmark_summary = (Path.cwd() / benchmark_summary).resolve()

    manifest = build_manifest(repo_root, args.label, benchmark_summary)
    if args.redact_local_details:
        manifest = sanitize_manifest(manifest, repo_root, socket.gethostname())
    rendered = json.dumps(manifest, indent=2, sort_keys=False)

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
