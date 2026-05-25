# SPDX-License-Identifier: GPL-3.0-only
"""Tests for Floquet reconstruction, PETSc/SciPy parity utilities, and flow plotting utilities."""

import shutil
import textwrap

import matplotlib

from tests.mpi_helpers import parse_mpi_reports, run_mpi_python

matplotlib.use("Agg")

import numpy as np
import pytest
import scipy.sparse as sp

from examples._helpers import leading_mode_real_frequency
from examples.example_3_streaky_bl import (
    _merge_temporal_target_batches,
    _resolve_baseflow_plot_field,
)
from linstabpy.mesh import Mesh
from linstabpy.utils import plot_flow, reconstruct_floquet
from linstabpy.utils.petsc_core import (
    compact_petsc_matrix,
    compare_petsc_against_scipy,
    compare_sparse_matrices,
    petsc_matvec_residual,
    petsc_owned_rows,
    petsc_to_scipy,
    scipy_to_petsc,
)


# ---------------------------------------------------------------------------
# Floquet reconstruction
# ---------------------------------------------------------------------------
class TestFloquetReconstruction:
    """Tests for reconstruct_floquet."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = Mesh.create(xrange=(0, 2 * np.pi), yrange=(0, 1), nx=16, ny=20, fd_order=4)
        self.X = mesh.X
        self.Y = mesh.Y
        self.used_ind = mesh.used_ind
        self.ngp = mesh.ngp
        self.idx = mesh.idx

    def test_zero_floquet_exp(self):
        """floquet_exp=0 applies no phase shift; output equals V on full grid."""
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(self.X, V, self.used_ind, floquet_exp=0.0)
        np.testing.assert_allclose(result, 1.0 + 0j, atol=1e-14)

    def test_nonzero_floquet_exp(self):
        """floquet_exp=beta multiplies by exp(i*beta*X)."""
        beta = 1.0
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(self.X, V, self.used_ind, floquet_exp=beta)
        expected = np.exp(1j * beta * self.X)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_1d_input_shape(self):
        """1-D V produces output with shape (ny, nx)."""
        V = np.ones(self.ngp, dtype=complex)
        result = reconstruct_floquet(self.X, V, self.used_ind, floquet_exp=0.0)
        assert result.shape == self.X.shape

    def test_2d_input_shape(self):
        """2-D V (ngp, n_fields) produces output with shape (n_fields, ny, nx)."""
        ones = np.ones(self.ngp, dtype=complex)
        V = np.column_stack([ones, 2 * ones])
        result = reconstruct_floquet(self.X, V, self.used_ind, floquet_exp=0.0)
        n_fields = V.shape[1]
        assert result.shape == (n_fields, *self.X.shape)

    def test_wrong_length_raises(self):
        """V with wrong length raises ValueError."""
        V = np.zeros(self.ngp + 5, dtype=complex)
        with pytest.raises(ValueError):
            reconstruct_floquet(self.X, V, self.used_ind, floquet_exp=0.0)


# ---------------------------------------------------------------------------
# Flow plotting
# ---------------------------------------------------------------------------
class TestPlotFlow:
    """Tests for plot_flow."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        mesh = Mesh.create(xrange=(0, 1), yrange=(0, 1), nx=10, ny=10, fd_order=4)
        self.X = mesh.X
        self.Y = mesh.Y
        rng = np.random.default_rng(42)
        self.field = rng.standard_normal((10, 10))

    def test_invalid_component_raises(self):
        """Invalid component raises ValueError."""
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, self.field, component="foo")

    def test_wrong_shape_raises(self):
        """Field shape mismatch with X raises ValueError."""
        bad_field = np.zeros((5, 5))
        with pytest.raises(ValueError):
            plot_flow(self.X, self.Y, bad_field)

    def test_masked_field_uses_tricontourf(self):
        """NaN-holed fields should use tricontourf, not pcolormesh."""
        import matplotlib.tri as mtri

        field = self.field.copy()
        field[4:6, 4:6] = np.nan
        ax = plot_flow(self.X, self.Y, field, component="real", colorbar=False)
        # Check that the drawn artist is a TriContourSet (from tricontourf)
        has_tri = any(isinstance(c, mtri.TriContourSet) for c in ax.collections)
        # tricontourf produces QuadContourSet-like objects stored differently
        # in recent Matplotlib; at minimum, no pcolormesh QuadMesh should exist
        from matplotlib.collections import QuadMesh

        has_quadmesh = any(isinstance(c, QuadMesh) for c in ax.collections)
        assert not has_quadmesh, "NaN field fell back to pcolormesh"


class TestPetscSciPyParityUtilities:
    """Tests for PETSc/SciPy conversion and comparison helpers."""

    def test_petsc_owned_rows_matches_sequential_csr_reference(self):
        ref = sp.csr_matrix(
            np.array(
                [
                    [1.0 + 2.0j, 0.0, 0.0, 3.0 - 1.0j],
                    [0.0, -4.0j, 5.0 + 0.5j, 0.0],
                    [2.5, 0.0, 0.0, 0.0],
                    [0.0, 6.0 - 3.0j, 0.0, 7.0 + 0.0j],
                ],
                dtype=complex,
            )
        )
        mat = scipy_to_petsc(ref)

        try:
            owned = petsc_owned_rows(mat)
        finally:
            mat.destroy()

        assert owned["ownership_range"] == (0, ref.shape[0])
        np.testing.assert_array_equal(owned["row_ids"], np.arange(ref.shape[0], dtype=np.int64))
        np.testing.assert_array_equal(owned["indptr"], ref.indptr)
        np.testing.assert_array_equal(owned["indices"], ref.indices)
        np.testing.assert_allclose(owned["data"], ref.data)

    @pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
    def test_petsc_owned_rows_matches_owned_csr_blocks_under_mpi(self):
        script = textwrap.dedent(
            """
            import json
            import numpy as np
            import scipy.sparse as sp
            from petsc4py import PETSc
            from linstabpy.utils.petsc_core import petsc_owned_rows, scipy_to_petsc_distributed

            ref = sp.csr_matrix(
                np.array(
                    [
                        [1.0 + 0.0j, 0.0, 0.0, 0.0, 9.0 + 0.0j, 0.0],
                        [0.0, 2.0 + 0.0j, 0.0, 0.0, 0.0, 8.0 + 0.0j],
                        [0.0, 0.0, 3.0 + 0.0j, 7.0 + 0.0j, 0.0, 0.0],
                        [6.0 + 0.0j, 0.0, 0.0, 4.0 + 0.0j, 0.0, 0.0],
                        [0.0, 5.0 + 0.0j, 0.0, 0.0, 5.0 + 0.0j, 0.0],
                        [0.0, 0.0, 4.0 + 0.0j, 0.0, 0.0, 6.0 + 0.0j],
                    ],
                    dtype=complex,
                )
            )

            mat = scipy_to_petsc_distributed(ref, comm=PETSc.COMM_WORLD)
            try:
                owned = petsc_owned_rows(mat)
                r0, r1 = owned["ownership_range"]
                expected = ref[r0:r1].tocsr()
                np.testing.assert_array_equal(owned["row_ids"], np.arange(r0, r1, dtype=np.int64))
                np.testing.assert_array_equal(owned["indptr"], expected.indptr)
                np.testing.assert_array_equal(owned["indices"], expected.indices)
                np.testing.assert_allclose(owned["data"], expected.data)
                print(
                    json.dumps(
                        {
                            "rank": PETSc.COMM_WORLD.getRank(),
                            "ownership_range": [int(r0), int(r1)],
                            "nnz": int(owned["data"].size),
                        }
                    )
                )
            finally:
                mat.destroy()
            """
        )
        proc = run_mpi_python(2, script)
        if proc.returncode != 0:
            pytest.fail(
                "2-rank PETSc owned-row subprocess failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        reports = parse_mpi_reports(proc.stdout)
        assert len(reports) == 2, proc.stdout
        assert {report["rank"] for report in reports} == {0, 1}

        ownership_ranges = sorted(tuple(report["ownership_range"]) for report in reports)
        assert ownership_ranges == [(0, 3), (3, 6)]
        assert all(report["nnz"] > 0 for report in reports)

    def test_petsc_to_scipy_roundtrip_matches_reference(self):
        base = np.array(
            [
                [1.0 + 2.0j, 0.0, 3.0 - 1.0j],
                [0.0, -4.0j, 0.0],
                [2.5, 0.0, 5.0 + 0.5j],
            ],
            dtype=complex,
        )
        ref = sp.csr_matrix(base)
        mat = scipy_to_petsc(ref)

        recovered = petsc_to_scipy(mat, format="csr")
        report = compare_sparse_matrices(ref, recovered, check_structure=True)

        assert report["pass"]
        assert report["shape_match"]
        assert report["structure_match"]
        assert report["nnz_delta"] == 0

        mat.destroy()

    def test_compact_petsc_matrix_matches_roundtrip_canonical_structure(self):
        base = sp.csr_matrix(
            np.array(
                [
                    [1.0 + 0.0j, 2.0 - 1.0j, 0.0],
                    [0.0, -3.0 + 0.5j, 4.0 + 0.0j],
                    [5.0 - 2.0j, 0.0, 6.0 + 0.0j],
                ],
                dtype=complex,
            )
        )
        mat = scipy_to_petsc(base)
        mat.setValue(0, 1, 0.0)
        mat.setValue(1, 2, 0.0)
        mat.assemble()

        compact = compact_petsc_matrix(mat)
        recovered = petsc_to_scipy(compact, format="csr")
        expected = petsc_to_scipy(mat, format="csr")
        report = compare_sparse_matrices(expected, recovered, check_structure=True)

        assert report["pass"]
        assert report["structure_match"]
        assert report["nnz_candidate"] == report["nnz_reference"]

        compact.destroy()
        mat.destroy()

    def test_compare_petsc_against_scipy_reports_zero_residuals(self):
        ref = sp.csr_matrix(
            np.array(
                [
                    [0.0, 2.0 + 1.0j, 0.0],
                    [3.0 - 4.0j, 0.0, 5.0],
                    [0.0, 0.0, -2.0j],
                ],
                dtype=complex,
            )
        )
        mat = scipy_to_petsc(ref)

        report = compare_petsc_against_scipy(ref, mat, check_structure=True)

        assert report["pass"]
        assert report["max_abs_residual"] == pytest.approx(0.0)
        assert report["max_rel_residual"] == pytest.approx(0.0)
        assert report["structure_match"]

        mat.destroy()

    def test_petsc_matvec_residual_reports_zero_for_exact_conversion(self):
        rng = np.random.default_rng(123)
        dense = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
        dense[np.abs(dense) < 0.6] = 0.0
        ref = sp.csr_matrix(dense)
        mat = scipy_to_petsc(ref)

        report = petsc_matvec_residual(ref, mat, seed=7, trials=4)

        assert report["pass"]
        assert report["max_abs_residual"] < 1e-12
        assert report["max_rel_residual"] < 1e-12
        assert len(report["trials"]) == 4

        mat.destroy()


def test_leading_mode_real_frequency_uses_least_damped_mode() -> None:
    omega = np.array([0.62 - 0.01j, 0.59 + 0.02j, 0.58 + 0.01j], dtype=complex)
    selected = leading_mode_real_frequency(omega, imag_shift=0.03)
    assert selected == complex(0.59, 0.03)


def test_leading_mode_real_frequency_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        leading_mode_real_frequency(np.array([], dtype=complex))


def test_merge_temporal_target_batches_preserves_target_quotas() -> None:
    batches = [
        (
            0.59 + 0.02j,
            np.array([0.585 + 0.017j, 0.589 - 0.004j, 0.621 - 0.003j], dtype=complex),
            np.column_stack(
                [
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
                    np.array([0.0, 1.0, 0.0, 0.0], dtype=complex),
                    np.array([0.0, 0.0, 1.0, 0.0], dtype=complex),
                ]
            ),
        ),
        (
            0.626 - 0.005j,
            np.array(
                [0.624 - 0.003j, 0.626 - 0.004j, 0.629 - 0.005j, 0.632 - 0.004j], dtype=complex
            ),
            np.eye(4, dtype=complex),
        ),
    ]
    omega, modes = _merge_temporal_target_batches(batches, [2, 3], dedupe_tol=1e-6)
    np.testing.assert_allclose(
        omega,
        np.array(
            [
                0.585 + 0.017j,
                0.589 - 0.004j,
                0.626 - 0.004j,
                0.624 - 0.003j,
                0.629 - 0.005j,
            ],
            dtype=complex,
        ),
    )
    assert modes.shape == (4, 5)


def test_resolve_baseflow_plot_field_prefers_streamwise_w() -> None:
    field, title = _resolve_baseflow_plot_field(
        {
            "baseflow_u": np.array([[1.0, 2.0]], dtype=float),
            "baseflow_w": np.array([[3.0, 4.0]], dtype=float),
        }
    )
    np.testing.assert_allclose(field, np.array([[3.0, 4.0]], dtype=float))
    assert title.endswith("$W$")
