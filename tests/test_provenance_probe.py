# SPDX-License-Identifier: GPL-3.0-only
"""Provenance probes drop the NUL padding some MPI libraries add."""

from linstabpy.utils.provenance import _probe_string, _without_builder


def test_probe_string_drops_nul_padding():
    assert (
        _probe_string(lambda: "Open MPI v5.0.11, Sep 16, 2026\x00")
        == "Open MPI v5.0.11, Sep 16, 2026"
    )


def test_probe_string_reports_unavailable_for_empty_or_failing_probes():
    assert _probe_string(lambda: "\x00") == "unavailable"

    def fail():
        raise RuntimeError

    assert _probe_string(fail) == "unavailable"


def test_mpi_version_drops_the_build_host():
    line = (
        "Open MPI v4.1.2, package: Open MPI root@build-node.example.org Distribution, "
        "ident: 4.1.2, repo rev: v4.1.2, Nov 24, 2021"
    )
    assert _without_builder(line) == "Open MPI v4.1.2"
    assert _without_builder("MPICH Version: 4.3.2") == "MPICH Version: 4.3.2"
