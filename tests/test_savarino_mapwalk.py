# SPDX-License-Identifier: GPL-3.0-only
"""Check the Savarino map-walk driver against the farm CSV."""

from __future__ import annotations

from pathlib import Path

from examples.savarino.gen_campaign import x_max
from examples.savarino.savarino_mapwalk import FARM_CSV, FIELDS, load_farm, relative_error

DRIVER = Path(__file__).resolve().parents[1] / "examples/savarino/savarino_mapwalk.py"


def test_farm_beta40_comes_from_committed_csv() -> None:
    """Pin the file by its gains, which a rerun must reproduce.

    The gains are a property of the operator and agree across backends and
    execution modes to about 1e-10, so they identify the file and survive a
    recomputation. An earlier version pinned the sum of the wall times, which
    is a property of the machine on the day and changed with every campaign.
    """
    rows = [row for row in load_farm(FARM_CSV) if abs(row["beta"] - 40.0) < 1.0e-12]
    assert [row["f"] for row in rows] == [1.0, 1.4, 1.6, 1.8, 2.0, 2.1, 2.15, 2.2, 2.5]
    expected_gains = [
        92.8918,
        156.685,
        176.266,
        184.929,
        175.243,
        159.180,
        155.069,
        153.896,
        119.913,
    ]
    for row, expected in zip(rows, expected_gains, strict=True):
        assert abs(row["gain"] - expected) <= 5.0e-4 * expected
    assert all(row["wall_s"] > 0.0 for row in rows)
    assert FARM_CSV.name == "savarino_map3_200pt.csv"


def test_farm_forcing_window_switches_at_f2() -> None:
    assert x_max(1.0) == 1.7954167
    assert x_max(2.0) == 1.9470833


def test_driver_uses_linewalker_not_rsvd() -> None:
    src = DRIVER.read_text(encoding="utf-8")
    assert "LineWalker" in src
    assert "append_row" in src
    assert "certified_rsvd" not in src
    assert "sweep_stable_regime" not in src
    assert relative_error(10.0, 10.0) == 0.0


def test_every_row_states_its_build_and_box() -> None:
    assert FIELDS[-3:] == ("git_sha", "slurm_job", "machine")


def test_every_row_states_what_made_its_cost() -> None:
    """Same commit, same box.
    Reference BLAS vs MKL was 5.6x on the factor, rank placement 1.6x, ncv
    2.5x on the solves (jobs 1850/1859/1861).
    A row without these four cannot sit next to another row in a comparison.
    """
    for column in ("blas", "ncv", "n_anchor", "pin_domain"):
        assert column in FIELDS


def test_pin_domain_names_the_placement_the_job_asked_for() -> None:
    from examples.savarino.savarino_mapwalk import _pin_domain

    assert _pin_domain({}) == "default"
    assert _pin_domain({"I_MPI_PIN_DOMAIN": "auto"}) == "auto"
    assert _pin_domain({"I_MPI_PIN_DOMAIN": "5"}) == "5"
    assert _pin_domain({"I_MPI_PIN": "off"}) == "off"
    assert _pin_domain({"I_MPI_PIN": "off", "I_MPI_PIN_DOMAIN": "socket"}) == "socket"


def test_walk_rows_carry_the_farm_cost_columns() -> None:
    """A cost comparison needs the same measurements on both sides.

    A gain needs the same certification (rho_fwd, rho_adj) on both sides.
    """
    farm_cols = FARM_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in (
        "wall_s",
        "peak_rss_bytes",
        "ndof",
        "ranks",
        "git_sha",
        "machine",
        "rho_fwd",
        "rho_adj",
    ):
        assert column in farm_cols
        assert column in FIELDS


def test_walk_peak_rss_means_the_same_as_the_farm_column() -> None:
    """Farm rows sum ru_maxrss over ranks (driver/run.py). A same-named
    column with a rank-local value would read 4x too small on 4 ranks and
    no reader could tell. The reduction is collective, so it must sit
    outside the rank-0 block; a serial test can only inspect the source."""
    src = DRIVER.read_text(encoding="utf-8")
    assert "comm.allreduce(int(_peak_rss_bytes()), op=MPI.SUM)" in src
    assert "peak_rss_total," in src
    reduce_at = src.index("comm.allreduce(int(_peak_rss_bytes())")
    guard_at = src.index("if rank == 0:", reduce_at)
    append_at = src.index("append_row(", reduce_at)
    assert reduce_at < guard_at < append_at


def test_gen_mapwalk_campaign_writes_the_arm_into_the_job_script(tmp_path, monkeypatch) -> None:
    """The job script must carry its own placement, interpreter, ncv and
    n_anchor.
    A run that inherits them from the submitting shell is not reproducible
    from the script.
    """
    import examples.savarino.gen_mapwalk_campaign as gen

    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_mapwalk_campaign.py",
            "--out",
            str(tmp_path),
            "--beta",
            "40",
            "--dry-run",
            "--nranks",
            "20",
            "--env",
            "I_MPI_PIN_DOMAIN=auto",
            "--env",
            "LINSTABPY_PYTHON=/venv/bin/python",
            "--n-anchor",
            "99",
            "--ncv",
            "5",
            "--driver-out",
            str(tmp_path / "direct20"),
        ],
    )
    assert gen.main() == 0
    text = (tmp_path / "savarino_mapwalk_b40.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=20" in text
    assert "export I_MPI_DEBUG=4" in text
    assert "export I_MPI_PIN_DOMAIN=auto" in text
    assert "export LINSTABPY_PYTHON=/venv/bin/python" in text
    assert "--n-anchor 99 --ncv 5" in text
    assert f"--out {tmp_path / 'direct20'}" in text
    assert "|| true; fi" in text  # setvars.sh --force must not kill the job under set -e
    assert (tmp_path / "direct20").is_dir()


def test_every_row_names_the_solver_that_factored_it() -> None:
    """CPU (MUMPS) and GPU (FGMRES around a STRUMPACK LU) rows share one CSV
    layout in the 2x2 cost matrix.
    The row must say which solver factored it.
    """
    import yaml

    from examples.savarino.savarino_mapwalk import _backend_label

    assert "backend" in FIELDS
    here = Path(__file__).resolve().parents[1] / "examples" / "savarino"
    assert (
        _backend_label(yaml.safe_load((here / "configs" / "mack_f2_b45.yaml").read_text()))
        == "mumps"
    )
    assert (
        _backend_label(yaml.safe_load((here / "savarino_point_gpu.yaml").read_text()))
        == "iterative/strumpack"
    )


def test_gen_mapwalk_campaign_gpu_arm_takes_its_config_cores_and_device(
    tmp_path, monkeypatch
) -> None:
    import examples.savarino.gen_mapwalk_campaign as gen

    gpu_yaml = Path(gen.__file__).with_name("savarino_point_gpu.yaml")
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_mapwalk_campaign.py",
            "--out",
            str(tmp_path),
            "--beta",
            "40",
            "--dry-run",
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
            "--n-anchor",
            "99",
            "--ncv",
            "5",
            "--driver-out",
            str(tmp_path / "gpu_lines"),
        ],
    )
    assert gen.main() == 0
    text = (tmp_path / "savarino_mapwalk_b40.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in text and "#SBATCH --cpus-per-task=10" in text
    assert f"--config {gpu_yaml.resolve()}" in text
    assert "flock -n 9" in text and "export PETSC_OPTIONS='-use_gpu_aware_mpi 0'" in text
    assert f'"{tmp_path / "gpu_lines"}/nvsmi.${{SLURM_JOB_ID:-local}}.log"' in text
    assert "mpiexec -n 1 " in text


def test_line_record_never_writes_the_real_host_name() -> None:
    """The row is committed and travels with the paper.

    The package policy is opt-in through LINSTABPY_HOSTNAME with the default
    "unknown". An earlier version of the walker called socket.gethostname()
    and put a private host name in three shipped run records.
    """
    src = DRIVER.read_text(encoding="utf-8")
    assert "gethostname" not in src
    assert "LINSTABPY_HOSTNAME" in src
