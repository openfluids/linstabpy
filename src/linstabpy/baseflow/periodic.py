# SPDX-License-Identifier: GPL-3.0-only
"""Periodic base flow container for harmonic resolvent analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .baseflow import BaseFlow
from .viscosity import sutherland_air


@dataclass(frozen=True)
class PeriodicBaseFlow:
    """Fourier modes of a T-periodic base flow Q(t) = sum_k Qhat_k exp(+i k omega0 t).

    The flow is real, so negatives are conjugates and are NOT stored.
    Harmonics k=0..N_h are kept; higher harmonics alias the discarded tail
    and must be checked by doubling N_t.

    Parameters
    ----------
    Qhat : np.ndarray
        Complex array of shape (N_h+1, 5, ny, nx) holding harmonics k=0..N_h
        of Q(t). Variables in order: rho, u, v, w, T.
    T_period : float
        Period of oscillation, in case nondimensional time units (t·U_inf/L).
    x, y : np.ndarray
        Mesh coordinate arrays, in case nondimensional length units (L is ref length).
    Re : float
        Reynolds number.
    Ma : float
        Mach number.
    Pr : float
        Prandtl number.
    kappa : float
        Heat capacity ratio (gamma).
    T_0 : float
        Reference temperature [K].
    extras : dict | None, optional
        Optional metadata: commit, source, n_snapshots, n_cycles.
    """

    Qhat: np.ndarray
    T_period: float
    x: np.ndarray
    y: np.ndarray
    Re: float
    Ma: float
    Pr: float
    kappa: float
    T_0: float
    extras: dict | None = None

    @property
    def N_h(self) -> int:
        """Highest harmonic index k stored."""
        return self.Qhat.shape[0] - 1

    @property
    def omega0(self) -> float:
        """Fundamental frequency omega0 = 2*pi/T_period."""
        return 2.0 * np.pi / self.T_period

    @property
    def shape(self) -> tuple[int, int]:
        """Mesh shape (ny, nx)."""
        return self.Qhat.shape[2:4]

    def __post_init__(self) -> None:
        """Validate the periodic base flow data."""
        if self.Qhat.ndim != 4:
            raise ValueError(f"Qhat must have shape (N_h+1, 5, ny, nx), got ndim={self.Qhat.ndim}")
        if self.Qhat.shape[1] != 5:
            raise ValueError(f"Qhat must have 5 variables (rho,u,v,w,T), got {self.Qhat.shape[1]}")
        if not np.isfinite(self.Qhat).all():
            raise ValueError("Qhat contains non-finite entries")
        if self.T_period <= 0:
            raise ValueError(f"T_period must be > 0, got {self.T_period}")

        ny, nx = self.shape
        if self.x.shape != (nx,):
            raise ValueError(f"x has shape {self.x.shape}, expected ({nx},) for nx={nx}")
        if self.y.shape != (ny,):
            raise ValueError(f"y has shape {self.y.shape}, expected ({ny},) for ny={ny}")

        # Check that Qhat[0] (the mean) has negligible imaginary part
        qhat0_imag_norm = np.max(np.abs(np.imag(self.Qhat[0])))
        qhat0_real_norm = np.max(np.abs(np.real(self.Qhat[0])))
        if qhat0_real_norm > 0 and qhat0_imag_norm / qhat0_real_norm > 1e-10:
            raise ValueError(
                f"Qhat[0] has imaginary part {qhat0_imag_norm:.2e} relative to "
                f"real {qhat0_real_norm:.2e}, exceeds tolerance 1e-10"
            )

    def reconstruct(self, t: float) -> BaseFlow:
        """Reconstruct Q(t) at instant t.

        Q(t) = Qhat[0] + 2 * sum_{k=1..N_h} Re(Qhat[k] * exp(+i*k*omega0*t))

        Parameters
        ----------
        t : float
            Time instant, in case nondimensional time units.

        Returns
        -------
        BaseFlow
            Reconstructed base flow at instant t.
        """
        omega0 = self.omega0
        # Mean component
        q_real = np.real(self.Qhat[0])

        # Harmonic sum: 2*Re(Qhat_k * exp(+i*k*omega0*t))
        for k in range(1, self.N_h + 1):
            phase = 1j * k * omega0 * t
            q_real = q_real + 2.0 * np.real(self.Qhat[k] * np.exp(phase))

        # Unpack variables in order: rho, u, v, w, T
        rho = q_real[0]
        u = q_real[1]
        v = q_real[2]
        w = q_real[3]
        temp = q_real[4]

        baseflow = BaseFlow(
            Re=self.Re,
            Ma=self.Ma,
            Pr=self.Pr,
            kappa=self.kappa,
            T_0=self.T_0,
            RHO=rho,
            U=u,
            V=v,
            W=w,
            T=temp,
        )
        # Viscosity must be filled at each reconstructed instant because
        # viscosity follows T(t) through Sutherland's law; filling once at
        # load time would freeze mu at a single phase of the cycle.
        sutherland_air(baseflow)
        return baseflow

    def sample_times(self, N_t: int) -> np.ndarray:
        """Sample times over one period.

        Parameters
        ----------
        N_t : int
            Number of sample points per period.

        Returns
        -------
        np.ndarray
            Array of shape (N_t,) with times t_n = n * T_period / N_t,
            n = 0..N_t-1, in case nondimensional time units.
        """
        return np.arange(N_t) * self.T_period / N_t


def load_periodic_baseflow(
    path: str,
    *,
    mesh_shape: tuple[int, int] | None = None,
    freestream_rtol: float | None = 0.05,
) -> PeriodicBaseFlow:
    """Load periodic base flow from npz file.

    File contract (required keys):
      - Qhat: complex, shape (N_h+1, 5, ny, nx), harmonics k=0..N_h.
      - T_period: float, period in case nondimensional time units.
      - x, y: coordinate arrays in case nondimensional length units.
      - Re, Ma, Pr, kappa, T_0: scalar parameters.

    Optional keys: commit, source, n_snapshots, n_cycles.

    Parameters
    ----------
    path : str
        Path to npz file.
    mesh_shape : tuple(ny, nx), optional
        If given, check Qhat shape matches (N_h+1, 5, ny, nx).
    freestream_rtol : float or None, optional
        Relative tolerance for freestream check on top boundary.
        Default 0.05 (5%). None skips the check, for a flow whose last
        y row is not a free stream (a jet, a cavity under a wall, a
        descending y grid).

    Returns
    -------
    PeriodicBaseFlow

    Raises
    ------
    ValueError
        Missing required key, non-finite entries, T_period <= 0,
        Qhat mismatched shape/ndim, mesh_shape mismatch, imaginary Qhat[0],
        freestream check failure.
    """
    with np.load(path) as data:
        # Check required keys
        required = {"Qhat", "T_period", "x", "y", "Re", "Ma", "Pr", "kappa", "T_0"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        qhat = np.array(data["Qhat"], dtype=complex)
        t_period = float(data["T_period"])
        x_arr = np.array(data["x"])
        y_arr = np.array(data["y"])
        re = float(data["Re"])
        ma = float(data["Ma"])
        pr = float(data["Pr"])
        kappa_param = float(data["kappa"])
        t_0 = float(data["T_0"])

        # Optional fields
        extras: dict[str, Any] = {}
        for key in ["commit", "source", "n_snapshots", "n_cycles"]:
            if key in data.files:
                value = data[key]
                extras[key] = value.item() if np.size(value) == 1 else value

    # Shape check
    if qhat.ndim != 4:
        raise ValueError(f"Qhat must have ndim=4, got {qhat.ndim}")
    if qhat.shape[1] != 5:
        raise ValueError(f"Qhat must have 5 variables, got {qhat.shape[1]}")

    ny_file, nx_file = qhat.shape[2:4]
    if mesh_shape is not None:
        ny_req, nx_req = mesh_shape
        if (ny_file, nx_file) != (ny_req, nx_req):
            raise ValueError(
                f"Mesh shape mismatch: file (ny,nx)=({ny_file},{nx_file}), "
                f"expected ({ny_req},{nx_req})"
            )

    # Freestream check: rho, u, T at top boundary (last y row) must be ~1.0
    # (canonical normalization: rho_inf = u_inf = T_inf = 1). A mis-scaled
    # base flow voided a whole session once; this check prevents recurrence.
    # The signed mean is used, so a flow with u reversed fails the check.
    qhat0_real = np.real(qhat[0])
    rho_top = qhat0_real[0, -1, :]  # shape (nx,)
    u_top = qhat0_real[1, -1, :]
    t_top = qhat0_real[4, -1, :]

    rho_mean = np.mean(rho_top)
    u_mean = np.mean(u_top)
    t_mean = np.mean(t_top)

    checks = [] if freestream_rtol is None else [("rho", rho_mean), ("u", u_mean), ("T", t_mean)]
    for name, val in checks:
        if abs(val - 1.0) > freestream_rtol:
            raise ValueError(
                f"Freestream {name} mean on top boundary = {val:.4f}, "
                f"not within {freestream_rtol} of 1.0 (canonical normalization)"
            )

    # Check and drop imaginary part of Qhat[0] if below tolerance
    qhat0_imag_norm = np.max(np.abs(np.imag(qhat[0])))
    qhat0_real_norm = np.max(np.abs(np.real(qhat[0])))
    if qhat0_real_norm > 0 and qhat0_imag_norm / qhat0_real_norm > 1e-10:
        raise ValueError(
            f"Qhat[0] has imaginary part {qhat0_imag_norm:.2e} relative to "
            f"real {qhat0_real_norm:.2e}, exceeds tolerance 1e-10"
        )
    # Cast to real (this is why the imaginary magnitude check must come before)
    qhat[0] = np.real(qhat[0])

    return PeriodicBaseFlow(
        Qhat=qhat,
        T_period=t_period,
        x=x_arr,
        y=y_arr,
        Re=re,
        Ma=ma,
        Pr=pr,
        kappa=kappa_param,
        T_0=t_0,
        extras=extras if extras else None,
    )
