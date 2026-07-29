from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricClass:
    # grid parameters
    x: np.ndarray
    y: np.ndarray

    ### derivative scheme
    scheme_metrics: Callable

    phys_to_comp: np.ndarray = field(init=False)
    comp_to_phys: np.ndarray = field(init=False)
    det_jacobian: np.ndarray = field(init=False)
    grid_orthogonality: np.ndarray = field(init=False)

    def __post_init__(self):
        shape = np.shape(self.x)
        self.nx = shape[0]
        self.ny = shape[1]

        self.x_comp = np.linspace(1, self.nx, self.nx)
        self.y_comp = np.linspace(1, self.ny, self.ny)

    def eval_metric_terms(self) -> None:
        """########################################################################
        function to evaluate the metric terms.
        -----------------
        Input parameters:
        -----------------
        -----------------
        Output parameters:
        -----------------
            self: dataclass with the metric terms
        ########################################################################"""

        self.eval_phys_to_comp()
        self.eval_det_jacobian()
        self.eval_comp_to_phys()

    def eval_phys_to_comp(self) -> None:
        """########################################################################
        function to evaluate the relation between the physical (curvilinear) mesh
        to the computational (cartesian) mesh
        -----------------
        Input parameters:
        -----------------
        -----------------
        Output parameters:
        -----------------
            self: dataclass with phys_to_comp variable
        ########################################################################"""

        self.phys_to_comp = np.zeros((self.nx, self.ny, 2, 2))

        if self.nx > 1:
            for j in range(self.ny):
                self.phys_to_comp[:, j, 0, 0] = self.scheme_metrics(self.x[:, j], self.x_comp)
                self.phys_to_comp[:, j, 1, 0] = self.scheme_metrics(self.y[:, j], self.x_comp)
        else:
            self.phys_to_comp[:, :, 0, 0] = 1.0
            self.phys_to_comp[:, :, 1, 0] = 0.0

        if self.ny > 1:
            for i in range(self.nx):
                self.phys_to_comp[i, :, 0, 1] = self.scheme_metrics(self.x[i, :], self.y_comp)
                self.phys_to_comp[i, :, 1, 1] = self.scheme_metrics(self.y[i, :], self.y_comp)
        else:
            self.phys_to_comp[:, :, 0, 1] = 0.0
            self.phys_to_comp[:, :, 1, 1] = 1.0

    def eval_det_jacobian(self) -> None:
        """########################################################################
        function to evaluate the determinant of the Jacobian matrix between the physical (curvilinear) mesh
        to the computational (cartesian) mesh

        If self.phys_to_comp dos not exists it call the function eval_phys_to_comp
        -----------------
        Input parameters:
        -----------------
        -----------------
        Output parameters:
        -----------------
            self: dataclass with the Jacobian determinant variable
        ########################################################################"""

        if not hasattr(self, "phys_to_comp"):
            self.eval_phys_to_comp()

        self.det_jacobian = (
            self.phys_to_comp[:, :, 1, 1] * self.phys_to_comp[:, :, 0, 0]
            - self.phys_to_comp[:, :, 0, 1] * self.phys_to_comp[:, :, 1, 0]
        )

    def eval_comp_to_phys(self) -> None:
        """########################################################################
        function to evaluate the relation between the computational (cartesian) mesh
        to the physical (curvilinear) mesh

        If self.phys_to_comp dos not exists it call the function eval_phys_to_comp
        If self.det_jacobian dos not exists it call the function eval_det_jacobian
        -----------------
        Input parameters:
        -----------------
        -----------------
        Output parameters:
        -----------------
            self: dataclass with comp_to_phys variable
        ########################################################################"""

        if not hasattr(self, "phys_to_comp"):
            self.eval_phys_to_comp()

        if not hasattr(self, "det_jacobian"):
            self.eval_det_jacobian()

        self.comp_to_phys = np.zeros((self.nx, self.ny, 2, 2))

        self.comp_to_phys[:, :, 0, 0] = (self.phys_to_comp[:, :, 1, 1]) / self.det_jacobian
        self.comp_to_phys[:, :, 0, 1] = -(self.phys_to_comp[:, :, 0, 1]) / self.det_jacobian

        self.comp_to_phys[:, :, 1, 0] = -(self.phys_to_comp[:, :, 1, 0]) / self.det_jacobian
        self.comp_to_phys[:, :, 1, 1] = (self.phys_to_comp[:, :, 0, 0]) / self.det_jacobian

    def check_grid_orthogonality(self) -> None:
        """########################################################################
        function to evaluate the grid orthogonality

        If self.phys_to_comp dos not exists it call the function eval_phys_to_comp
        -----------------
        Input parameters:
        -----------------
        -----------------
        Output parameters:
        -----------------
            self: dataclass with grid_orthogonality variable
        ########################################################################"""
        if "phys_to_comp" not in locals():
            self.eval_phys_to_comp()

        g11 = (
            self.phys_to_comp[:, :, 0, 0] * self.phys_to_comp[:, :, 0, 0]
            + self.phys_to_comp[:, :, 1, 0] * self.phys_to_comp[:, :, 1, 0]
        )
        g22 = (
            self.phys_to_comp[:, :, 0, 1] * self.phys_to_comp[:, :, 0, 1]
            + self.phys_to_comp[:, :, 1, 1] * self.phys_to_comp[:, :, 1, 1]
        )
        g12 = (
            self.phys_to_comp[:, :, 0, 0] * self.phys_to_comp[:, :, 0, 1]
            + self.phys_to_comp[:, :, 1, 0] * self.phys_to_comp[:, :, 1, 1]
        )

        self.grid_orthogonality = g12 / np.sqrt(g11 * g22)

    def write_comp_to_phys(self, path: str, filename: str) -> None:
        """########################################################################
        function to write the computational to physical grid metrics matrix into a file
        -----------------
        Input parameters:
        -----------------
            path: path to save the matrix
            filename: file name
        -----------------
        Output parameters:
        -----------------
        ########################################################################"""
        np.save(path + filename + ".npy", self.comp_to_phys)

    def write_phys_to_comp(self, path: str, filename: str) -> None:
        """########################################################################
        function to write the physical to computational grid metrics matrix into a file
        -----------------
        Input parameters:
        -----------------
            path: path to save the matrix
            filename: file name
        -----------------
        Output parameters:
        -----------------
        ########################################################################"""
        np.save(path + filename + ".npy", self.phys_to_comp)

    def write_det_jacobian(self, path: str, filename: str) -> None:
        """########################################################################
        function to write the jacobian determinant matrix into a file
        -----------------
        Input parameters:
        -----------------
            path: path to save the matrix
            filename: file name
        -----------------
        Output parameters:
        -----------------
        ########################################################################"""
        np.save(path + filename + ".npy", self.det_jacobian)

    def write_grid_orthogonality(self, path: str, filename: str) -> None:
        """########################################################################
        function to write the grid orthogonality matrix into a file
        -----------------
        Input parameters:
        -----------------
            path: path to save the matrix
            filename: file name
        -----------------
        Output parameters:
        -----------------
        ########################################################################"""
        np.save(path + filename + ".npy", self.grid_orthogonality)
