#!/usr/bin/env python3
"""Run or list validation gates for the architecture refactor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess  # nosec B404 - validation runner executes fixed local commands.
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts"
    / "architecture_refactor"
    / "architecture_refactor_phase0"
    / "matrix_run_latest.json"
)
GIT_TIMEOUT_SECONDS = 10.0
DEFAULT_CHECK_TIMEOUT_SECONDS = 900.0
LONG_CHECK_TIMEOUT_SECONDS = 7200.0
ARTIFACT_INVENTORY_TIMEOUT_SECONDS = 60.0
GROUP_ALIASES = {
    "phase0-golden-local": (
        "phase0-local",
        "phase0-expanded-local",
        "documentation-gates",
    ),
    "phase0-golden-all": (
        "phase0-local",
        "phase0-expanded-local",
        "phase0-external-assets",
        "documentation-gates",
    ),
    "safety-infrastructure": (
        "equivalence-gates",
        "documentation-gates",
    ),
}


@dataclass(frozen=True)
class Check:
    id: str
    group: str
    description: str
    command: tuple[str, ...]
    artifact: Path | None = None
    long_running: bool = False
    timeout_seconds: float | None = None


def _python() -> str:
    override = os.environ.get("LINSTABPY_PYTHON") or os.environ.get("LINSTABPY_PYTHON_BIN")
    if override:
        return override
    return str(ROOT / ".venv" / "bin" / "python")


def _checks() -> list[Check]:
    py = _python()
    phase0_dir = ROOT / "artifacts/architecture_refactor/architecture_refactor_phase0"
    docs_dir = ROOT / "artifacts/architecture_refactor/documentation"
    invariant_helpers = (
        "linstabpy.operators.assembly._prepare_coefficients",
        "linstabpy.operators.assembly._assemble_from_prepared",
        "linstabpy.operators.assembly._apply_inv_B",
        "linstabpy.boundary_conditions.recipe._apply_bc_recipe_impl",
        "linstabpy.results_runtime._restrict_operator_for_stability",
        "linstabpy.utils.petsc_init._sanitize_petsc_argv",
        "linstabpy.solvers.eigenvalue._configure_st_for_eigenvalue",
    )
    invariant_args = tuple(
        arg for helper in invariant_helpers for arg in ("--require-private", helper)
    )
    return [
        Check(
            id="pytest-full",
            group="phase0-local",
            description="Full pytest suite.",
            command=(py, "-m", "pytest", "tests/", "-q"),
            long_running=True,
        ),
        Check(
            id="pytest-focused",
            group="phase0-local",
            description="Focused tests for BCs, operators, solvers, and acceptance logic.",
            command=(
                py,
                "-m",
                "pytest",
                "tests/test_boundary_conditions.py",
                "tests/test_operators.py",
                "tests/test_solvers.py",
                "tests/test_stability_acceptance.py",
                "-q",
            ),
        ),
        Check(
            id="example-blasius-smoke",
            group="phase0-local",
            description="Fresh reduced Blasius example run without MATLAB parity.",
            command=("bash", "run_all_examples.sh", "--example", "1s", "--skip-parity", "--fresh"),
            artifact=ROOT / "examples/example_1_blasius_smoke.npz",
        ),
        Check(
            id="parity-matrix-status",
            group="phase0-local",
            description="Current Martini operator/branch parity matrix status.",
            command=(
                py,
                "scripts/validation/martini2024/tools/check_parity_matrices.py",
                "--allow-incomplete",
                "--artifacts-dir",
                str(phase0_dir.relative_to(ROOT)),
            ),
            artifact=phase0_dir / "parity_gate_summary.json",
        ),
        Check(
            id="operator-perf-memory",
            group="phase0-local",
            description="Small PETSc/SLEPc operator performance and memory baseline.",
            command=(
                py,
                "scripts/validation/martini2024/tools/perf_memory_operator_parity.py",
                "--meshes",
                "24x16,32x24",
                "--output",
                str((phase0_dir / "operator_perf_memory.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "operator_perf_memory.json",
        ),
        Check(
            id="phase0-characterization",
            group="phase0-expanded-local",
            description="BC index sets and coefficient tensor hashes for Phase 0.",
            command=(
                py,
                "scripts/validation/phase0_characterization.py",
                "--output",
                str((phase0_dir / "phase0_characterization.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "phase0_characterization.json",
        ),
        Check(
            id="operator-construction-parity",
            group="phase0-expanded-local",
            description="Numeric operator construction rebuild and PETSc matvec parity.",
            command=(
                py,
                "scripts/validation/martini2024/tools/compare_operator_construction.py",
                "--strict",
                "--output",
                str((phase0_dir / "operator_numeric_parity.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "operator_numeric_parity.json",
        ),
        Check(
            id="assembly-pair-2d",
            group="phase0-expanded-local",
            description="SciPy/PETSc assembly parity on deterministic 2-D case.",
            command=(
                py,
                "scripts/validation/martini2024/tools/assembly_parity.py",
                "pair",
                "--model",
                "2D",
                "--backend-before",
                "scipy",
                "--backend-after",
                "petsc",
                "--skip-eigs",
                "--artifact-before",
                str((phase0_dir / "assembly_pair_2d_scipy.npz").relative_to(ROOT)),
                "--artifact-after",
                str((phase0_dir / "assembly_pair_2d_petsc.npz").relative_to(ROOT)),
                "--output",
                str((phase0_dir / "assembly_pair_2d.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "assembly_pair_2d.json",
        ),
        Check(
            id="assembly-pair-axy",
            group="phase0-expanded-local",
            description="SciPy/PETSc assembly parity on deterministic axisymmetric case.",
            command=(
                py,
                "scripts/validation/martini2024/tools/assembly_parity.py",
                "pair",
                "--model",
                "axy",
                "--backend-before",
                "scipy",
                "--backend-after",
                "petsc",
                "--skip-eigs",
                "--artifact-before",
                str((phase0_dir / "assembly_pair_axy_scipy.npz").relative_to(ROOT)),
                "--artifact-after",
                str((phase0_dir / "assembly_pair_axy_petsc.npz").relative_to(ROOT)),
                "--output",
                str((phase0_dir / "assembly_pair_axy.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "assembly_pair_axy.json",
        ),
        Check(
            id="curvilinear-example0",
            group="phase0-expanded-local",
            description="Fresh Example 0 curvilinear derivative validation.",
            command=("bash", "run_all_examples.sh", "--example", "0", "--skip-parity", "--fresh"),
            artifact=ROOT / "examples/example_0_curvilinear_validation.npz",
        ),
        Check(
            id="high-mode-local-baseline",
            group="phase0-expanded-local",
            description="100-mode local stability baseline for interior and boundary-influenced targets.",
            command=(
                py,
                "scripts/validation/phase0_characterization.py",
                "--output",
                str((phase0_dir / "phase0_characterization.json").relative_to(ROOT)),
                "--high-mode-output",
                str((phase0_dir / "phase0_high_mode_baseline.json").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "phase0_high_mode_baseline.json",
        ),
        Check(
            id="coefficient-tensor-snapshot",
            group="phase0-expanded-local",
            description="Exact selected coefficient tensor snapshots for P1 equivalence checks.",
            command=(
                py,
                "scripts/validation/phase0_characterization.py",
                "--output",
                str((phase0_dir / "phase0_characterization.json").relative_to(ROOT)),
                "--coefficient-snapshot-output",
                str((phase0_dir / "phase0_coefficient_tensors.npz").relative_to(ROOT)),
            ),
            artifact=phase0_dir / "phase0_coefficient_tensors.npz",
        ),
        Check(
            id="case2-local-assembly-baseline",
            group="phase0-expanded-local",
            description="Case 2 assembly-only timing and memory baseline for P1/P2/P8.",
            command=(
                "bash",
                "-lc",
                "LINSTABPY_CASE2_PERF_CONFIG="
                "examples/case_studies/case2_lui/configs/perf/"
                "phase0_local_assembly_baseline.jsonc "
                "LINSTABPY_STAGE_HEARTBEAT_SECONDS=60 "
                f"{shlex.quote(py)} examples/case_studies/case2_lui/perf_memory.py",
            ),
            artifact=ROOT
            / "examples/case_studies/case2_lui/output/phase0_local_assembly_baseline"
            / "case2_phase0_local_assembly_baseline_latest.json",
            timeout_seconds=600.0,
        ),
        Check(
            id="martini-fixture-preflight",
            group="phase0-external-assets",
            description="Check that raw MATLAB exports and extracted Martini NPZ fixtures are present.",
            command=(
                py,
                "scripts/validation/martini2024/tools/extract_martini_to_npz.py",
                "--check-fixtures",
            ),
            artifact=ROOT / "scripts/validation/martini2024/fixtures",
        ),
        Check(
            id="martini-full-examples",
            group="phase0-external-assets",
            description="Regenerate all Martini example artifacts and require MATLAB parity to be available.",
            command=("bash", "run_all_examples.sh", "--fresh"),
            artifact=ROOT / "scripts/validation/martini2024/reports/martini_parity_latest.json",
            long_running=True,
        ),
        Check(
            id="docstring-inventory",
            group="documentation-gates",
            description="AST docstring inventory for public symbols and selected invariant helpers.",
            command=(
                py,
                "scripts/validation/check_docstrings.py",
                "--root",
                "src/linstabpy",
                *invariant_args,
                "--output",
                str((docs_dir / "docstring_inventory.json").relative_to(ROOT)),
            ),
            artifact=docs_dir / "docstring_inventory.json",
        ),
        Check(
            id="lnse-equivalence-harness",
            group="equivalence-gates",
            description="Before/after LNSE assembly equivalence contract for P1/P2/P3 work.",
            command=(
                py,
                "scripts/validation/lnse_equivalence_harness.py",
                "--case",
                "2D",
                "--case",
                "axy",
                "--output",
                str(
                    (
                        ROOT / "artifacts/architecture_refactor/safety_infrastructure/"
                        "lnse_equivalence_harness.json"
                    ).relative_to(ROOT)
                ),
            ),
            artifact=ROOT / "artifacts/architecture_refactor/safety_infrastructure/"
            "lnse_equivalence_harness.json",
        ),
        Check(
            id="lnse-dual-path-env",
            group="equivalence-gates",
            description="Environment-enabled dual-path LNSE migration safety gate.",
            command=(
                "bash",
                "-lc",
                "LINSTABPY_DUAL_PATH=1 "
                "LINSTABPY_DUAL_PATH_REPORT="
                "artifacts/architecture_refactor/safety_infrastructure/"
                "lnse_dual_path_env.json "
                f"{shlex.quote(py)} scripts/validation/lnse_equivalence_harness.py "
                "--case 2D --case axy",
            ),
            artifact=ROOT / "artifacts/architecture_refactor/safety_infrastructure/"
            "lnse_dual_path_env.json",
        ),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(  # nosec B603, B607 - fixed git command, bounded timeout.
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _timeout_seconds(check: Check) -> float:
    if check.timeout_seconds is not None:
        return check.timeout_seconds
    if check.long_running:
        return LONG_CHECK_TIMEOUT_SECONDS
    return DEFAULT_CHECK_TIMEOUT_SECONDS


def _path_for_report(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _check_inventory_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("artifact inventory timed out")


def _hash_file(path: Path, deadline: float | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            _check_inventory_deadline(deadline)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _artifact_inventory(
    path: Path | None, timeout_seconds: float = ARTIFACT_INVENTORY_TIMEOUT_SECONDS
) -> dict[str, object] | None:
    if path is None:
        return None
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    try:
        if not _path_exists(path):
            return {"kind": "missing", "path": _path_for_report(path), "exists": False}
        if path.is_file():
            sha256, size = _hash_file(path, deadline)
            return {
                "kind": "file",
                "path": _path_for_report(path),
                "exists": True,
                "size_bytes": size,
                "sha256": sha256,
            }
        if path.is_dir():
            digest = hashlib.sha256()
            file_count = 0
            total_size = 0
            for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
                _check_inventory_deadline(deadline)
                relative = file_path.relative_to(path).as_posix()
                file_sha256, size = _hash_file(file_path, deadline)
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(size).encode("ascii"))
                digest.update(b"\0")
                digest.update(file_sha256.encode("ascii"))
                digest.update(b"\n")
                file_count += 1
                total_size += size
            return {
                "kind": "directory",
                "path": _path_for_report(path),
                "exists": True,
                "file_count": file_count,
                "total_size_bytes": total_size,
                "sha256": digest.hexdigest(),
            }
        return {
            "kind": "other",
            "path": _path_for_report(path),
            "exists": True,
        }
    except (OSError, TimeoutError) as exc:
        return {
            "kind": "error",
            "path": _path_for_report(path),
            "exists": _path_exists(path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _selected_groups(group: str | None) -> set[str] | None:
    if group is None:
        return None
    return set(GROUP_ALIASES.get(group, (group,)))


def _select(checks: Iterable[Check], *, group: str | None, ids: set[str]) -> list[Check]:
    groups = _selected_groups(group)
    selected = []
    for check in checks:
        if ids and check.id not in ids:
            continue
        if groups is not None and check.group not in groups:
            continue
        selected.append(check)
    return selected


def _run_check(check: Check, *, dry_run: bool) -> dict[str, object]:
    timeout_seconds = _timeout_seconds(check)
    payload: dict[str, object] = {
        "id": check.id,
        "group": check.group,
        "description": check.description,
        "command": list(check.command),
        "artifact": _path_for_report(check.artifact) if check.artifact else None,
        "timeout_seconds": timeout_seconds,
        "started_at_utc": _now(),
        "dry_run": dry_run,
    }
    if dry_run:
        payload.update({"returncode": None, "status": "dry-run"})
        return payload
    try:
        result = subprocess.run(  # nosec B603 - commands come from checked-in matrix definitions.
            check.command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        lines = output.splitlines()
        lines.append(f"Timed out after {timeout_seconds:g} seconds.")
        payload.update(
            {
                "finished_at_utc": _now(),
                "returncode": None,
                "status": "timeout",
                "output_tail": lines[-80:],
                "artifact_exists": bool(check.artifact and _path_exists(check.artifact)),
                "artifact_inventory": _artifact_inventory(check.artifact),
            }
        )
        return payload
    payload.update(
        {
            "finished_at_utc": _now(),
            "returncode": result.returncode,
            "status": "pass" if result.returncode == 0 else "fail",
            "output_tail": result.stdout.splitlines()[-80:],
            "artifact_exists": bool(check.artifact and _path_exists(check.artifact)),
            "artifact_inventory": _artifact_inventory(check.artifact),
        }
    )
    return payload


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List checks and exit.")
    parser.add_argument("--group", help="Run only checks in this group.")
    parser.add_argument(
        "--check", action="append", default=[], help="Run one check ID. Repeatable."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the report without running commands."
    )
    parser.add_argument("--skip-long", action="store_true", help="Skip long-running checks.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    checks = _checks()
    if args.list:
        for check in checks:
            marker = " long" if check.long_running else ""
            print(f"{check.id:24s} {check.group:14s}{marker:6s} {' '.join(check.command)}")
        print("\nVirtual groups:")
        for name, groups in GROUP_ALIASES.items():
            print(f"{name:24s} {'+'.join(groups)}")
        return 0

    selected = _select(checks, group=args.group, ids=set(args.check))
    if not selected:
        raise SystemExit("No checks selected.")
    if args.skip_long:
        selected = [check for check in selected if not check.long_running]
    if not selected:
        raise SystemExit("Only long-running checks were selected; omit --skip-long to run them.")

    results = [_run_check(check, dry_run=args.dry_run) for check in selected]
    report = {
        "generated_at_utc": _now(),
        "git_sha": _git_sha(),
        "root": str(ROOT),
        "selected_group": args.group,
        "selected_ids": args.check,
        "skip_long": args.skip_long,
        "status": "pass"
        if all(item["status"] in {"pass", "dry-run"} for item in results)
        else "fail",
        "checks": results,
    }
    _write_report(args.report, report)
    print(f"Architecture refactor matrix: status={report['status']} report={args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
