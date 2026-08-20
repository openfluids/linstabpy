# SPDX-License-Identifier: GPL-3.0-only
"""Check the Savarino line-campaign kit against the 200-point GRID."""

from __future__ import annotations

import csv
from pathlib import Path

from examples.savarino.compare_map_timings import main as compare_main
from examples.savarino.gen_campaign import GRID, x_max


def test_line_csv_fields_name_every_stage_clock() -> None:
    src = (Path(__file__).resolve().parents[1] / "examples/savarino/savarino_line.py").read_text(
        encoding="utf-8"
    )
    needed = (
        "load_s",
        "assemble_s",
        "window_s",
        "factor_s",
        "gmres_s",
        "rsvd_s",
        "recycle_setup_s",
        "wall_s",
        "line_wall_s",
        "n_anchors",
        "recycle_rank",
    )
    for name in needed:
        assert f'"{name}"' in src


def test_line_repo_root_contains_the_package() -> None:
    from examples.savarino.gen_line_campaign import REPO

    assert (REPO / "src" / "linstabpy").is_dir()
    assert (REPO / "examples" / "savarino" / "savarino_line.py").is_file()


def test_grid_has_two_hundred_points() -> None:
    n = sum(len(freqs) for freqs in GRID.values())
    assert n == 200
    assert len(GRID) == 14


def test_x_max_follows_the_measured_window() -> None:
    assert x_max(0.0) == 1.9470833
    assert x_max(2.0) == 1.9470833
    assert x_max(1.6) == 1.7954167


def test_gen_campaign_beta_filter_writes_only_those_lines(tmp_path, monkeypatch) -> None:
    """A farm arm for two lines must not queue the other 153 points.

    The job script must carry its own placement/interpreter and the ncv it ran.
    """
    import examples.savarino.gen_campaign as gen

    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_campaign.py",
            "--out",
            str(tmp_path),
            "--dry-run",
            "--beta",
            "40",
            "45",
            "--ncv",
            "5",
            "--env",
            "I_MPI_PIN=off",
        ],
    )
    assert gen.main() == 0
    scripts = sorted(tmp_path.glob("sav_map_b*.sbatch"))
    assert len(scripts) == len(GRID[40.0]) + len(GRID[45.0]) == 47
    text = (tmp_path / "sav_map_b40p0_f1.sbatch").read_text(encoding="utf-8")
    assert "export I_MPI_PIN=off" in text and "export I_MPI_DEBUG=4" in text
    assert "|| true; fi" in text
    cfg = (tmp_path / "configs" / "sav_map_b40p0_f1.yaml").read_text(encoding="utf-8")
    assert "ncv: 5" in cfg and "forcing_norm: l2" in cfg
    # The job script does `cd <repo root>`; a case_dir left relative to
    # examples/savarino/ made every point job fail on load (c9 jobs 1868-1914).
    import yaml

    case_dir = yaml.safe_load(cfg)["paths"]["case_dir"]
    assert Path(case_dir).is_absolute()
    assert case_dir.endswith("examples/savarino/output/baseflow")


def test_gen_line_campaign_dry_run_writes_one_script_per_beta(tmp_path, monkeypatch) -> None:
    import examples.savarino.gen_line_campaign as gen

    monkeypatch.setattr(
        "sys.argv",
        ["gen_line_campaign.py", "--out", str(tmp_path), "--dry-run"],
    )
    assert gen.main() == 0
    scripts = list(tmp_path.glob("savarino_line_b*.sbatch"))
    assert len(scripts) == 14
    text = (tmp_path / "savarino_line_b45p0.sbatch").read_text(encoding="utf-8")
    assert "method=recycle" in text
    assert "--method recycle" in text
    assert "--beta 45.0" in text or "--beta 45" in text


def test_compare_map_timings_matches_shared_points(tmp_path) -> None:
    point = tmp_path / "point.csv"
    line_dir = tmp_path / "lines"
    line_dir.mkdir()
    point.write_text(
        "beta_Xsh,f_Xsh_Uinf,gain,wall_s,status,backend\n"
        "45.0,2.0,179.0,900.0,ok,mumps\n"
        "45.0,1.0,100.0,880.0,ok,mumps\n",
        encoding="utf-8",
    )
    (line_dir / "savarino_line_b45p0.csv").write_text(
        "beta_Xsh,f_Xsh_Uinf,gain,wall_s,assemble_s,line_wall_s,status,method\n"
        "45.0,2.0,179.0,820.0,70.0,1700.0,ok,line_mumps\n",
        encoding="utf-8",
    )
    import sys

    sys.argv = [
        "compare_map_timings.py",
        str(point),
        str(line_dir),
        "-o",
        str(tmp_path / "timing.csv"),
    ]
    assert compare_main() == 0
    with (tmp_path / "timing.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["gain_rel"]) == 0.0


def test_env_lines_quote_a_value_with_spaces() -> None:
    """``PETSC_OPTIONS=-use_gpu_aware_mpi 0`` must reach the shell as one word."""
    from examples.savarino.gen_campaign import env_lines

    assert env_lines(["I_MPI_PIN=off"]) == "export I_MPI_PIN=off\n"
    assert env_lines(["PETSC_OPTIONS=-use_gpu_aware_mpi 0"]) == (
        "export PETSC_OPTIONS='-use_gpu_aware_mpi 0'\n"
    )


def test_gpu_block_gives_the_job_one_device_by_lock(tmp_path) -> None:
    from examples.savarino.gen_campaign import gpu_block

    assert gpu_block(0, tmp_path) == ""
    block = gpu_block(2, tmp_path)
    assert "for _g in $(seq 0 1); do" in block
    assert "flock -n 9" in block and 'export CUDA_VISIBLE_DEVICES="$_g"' in block
    assert "no free GPU among 2" in block
    assert f'"{tmp_path}/nvsmi.${{SLURM_JOB_ID:-local}}.log"' in block
    assert "nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu" in block
    # the sampler must not inherit the lock: "( exec 9>&-; while true; do nvidia-smi"
    assert "( exec 9>&-; while true; do nvidia-smi" in block


def test_gen_campaign_gpu_arm_writes_config_cores_and_device_pick(tmp_path, monkeypatch) -> None:
    """The GPU farm arm: STRUMPACK+FGMRES config, one rank, ten cores, a device
    by lock.
    All of that is visible in the generated files.
    The generated files inherit none of it from the shell.
    """
    import yaml

    import examples.savarino.gen_campaign as gen

    gpu_yaml = Path(gen.__file__).with_name("savarino_point_gpu.yaml")
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_campaign.py",
            "--out",
            str(tmp_path),
            "--dry-run",
            "--beta",
            "40",
            "--nranks",
            "1",
            "--cpus-per-task",
            "10",
            "--gpus",
            "2",
            "--config",
            str(gpu_yaml),
            "--env",
            "PETSC_OPTIONS=-use_gpu_aware_mpi 0",
            "--env",
            "OMP_NUM_THREADS=10",
        ],
    )
    assert gen.main() == 0
    text = (tmp_path / "sav_map_b40p0_f1.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in text and "#SBATCH --cpus-per-task=10" in text
    assert "flock -n 9" in text and "mpiexec -n 1 " in text
    assert "export PETSC_OPTIONS='-use_gpu_aware_mpi 0'" in text
    cfg = yaml.safe_load((tmp_path / "configs" / "sav_map_b40p0_f1.yaml").read_text())
    assert cfg["solver"]["resolvent"]["type"] == "iterative"
    assert cfg["solver"]["resolvent"]["factor_backend"] == "strumpack"
    assert cfg["solver"]["resolvent"]["ksp_type"] == "fgmres"
    assert Path(cfg["paths"]["case_dir"]).is_absolute()
    assert cfg["paths"]["case_dir"].endswith("examples/savarino/output/baseflow")
    # The CPU default stays as-is: one core per rank, no device pick.
    monkeypatch.setattr(
        "sys.argv", ["gen_campaign.py", "--out", str(tmp_path / "cpu"), "--dry-run", "--beta", "40"]
    )
    assert gen.main() == 0
    cpu = (tmp_path / "cpu" / "sav_map_b40p0_f1.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=1" in cpu and "flock" not in cpu


def test_collect_campaign_reads_backend_and_ranks_from_the_job_files(tmp_path) -> None:
    """A GPU point must not be recorded as MUMPS on four ranks."""
    from examples.savarino.collect_campaign import backend_of, gpu_peak_mib

    d = tmp_path / "sav_map_b40p0_f1"
    d.mkdir()
    (tmp_path / "sav_map_b40p0_f1.sbatch").write_text(
        "#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=10\n"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "sav_map_b40p0_f1.yaml").write_text(
        "solver:\n  factor_backend: mumps\n  resolvent:\n    type: iterative\n"
        "    ksp_type: fgmres\n    factor_backend: strumpack\n"
    )
    assert backend_of(d) == ("iterative/strumpack", 1)
    (d / "nvsmi.4321.log").write_text(
        "2026/08/18 20:00:00.000, 15, 0\n2026/08/18 20:00:05.000, 30801, 97\n"
        "2026/08/18 20:00:10.000, 12000, 40\n"
    )
    assert gpu_peak_mib(d) == "30801"
    # No job files: the directory-name rule of the historical campaigns.
    assert backend_of(tmp_path / "sav_map_b45p0_f2_GPU") == ("strumpack", 1)
    assert backend_of(tmp_path / "sav_map_b45p0_f2") == ("mumps", 4)
    assert gpu_peak_mib(tmp_path / "sav_map_b45p0_f2") == ""
