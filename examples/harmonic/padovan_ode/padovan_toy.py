# SPDX-License-Identifier: GPL-3.0-only
"""Padovan ODE toy problem for harmonic resolvent validation (JFM 2020 §3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp


def evaluate_dynamics(t: float, q: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    """Padovan ODE right side: 3-state system from Padovan, Otto & Rowley JFM 2020.

    dx/dt = mu*x - y - alpha*x*z - beta*x*y
    dy/dt = x + mu*y - alpha*y*z + beta*x^2
    dz/dt = -alpha*z + alpha*(x^2 + y^2)

    Parameters
    ----------
    t : float
        Time (not used in the autonomous system).
    q : np.ndarray
        State vector [x, y, z].
    params : tuple[float, ...]
        Parameters (mu, alpha, beta).

    Returns
    -------
    np.ndarray
        Time derivative [dx/dt, dy/dt, dz/dt].
    """
    mu, alpha, beta = params
    x, y, z = q
    rhs = np.zeros(3)
    rhs[0] = mu * x - y - alpha * x * z - beta * x * y
    rhs[1] = x + mu * y - alpha * y * z + beta * x**2
    rhs[2] = -alpha * z + alpha * (x**2 + y**2)
    return rhs


def evaluate_jacobian(t: float, q: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    """Jacobian of Padovan ODE via central differences.

    Parameters
    ----------
    t : float
        Time (not used).
    q : np.ndarray
        State vector [x, y, z].
    params : tuple[float, ...]
        Parameters (mu, alpha, beta).

    Returns
    -------
    np.ndarray
        Jacobian matrix A = dF/dq, shape (3, 3).
    """
    eps = 1e-5
    A = np.zeros((3, 3))
    for i in range(3):
        veci = np.zeros(3)
        veci[i] = 1.0
        A[:, i] = (
            evaluate_dynamics(t, q + eps * veci, params)
            - evaluate_dynamics(t, q - eps * veci, params)
        ) / (2 * eps)
    return A


def main(
    N_h: int = 10,
    out_dir: str = ".",
) -> None:
    """Integrate Padovan ODE, FFT to harmonics, export for harmonic resolvent.

    Parameters
    ----------
    N_h : int
        Highest harmonic to keep. Default 10.
    out_dir : str
        Output directory for .npz, .log, and COO files. Default current dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = (1 / 5, 1 / 5, 1 / 5)
    mu, alpha, beta = params
    omega = np.sqrt(1 - beta**2 * mu / alpha)
    T = 2 * np.pi / omega

    dt = T / 300
    t_final = 200 * T
    t_eval = np.arange(0, t_final + dt, dt)

    sol = solve_ivp(
        evaluate_dynamics,
        [0, t_final],
        0.1 * np.ones(3),
        method="RK45",
        t_eval=t_eval,
        args=(params,),
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )

    idx0 = np.argmin(np.abs(sol.t - 198 * T))
    idxf = np.argmin(np.abs(sol.t - 199 * T))
    t_period = sol.t[idx0:idxf] - sol.t[idx0]
    Q = sol.y[:, idx0:idxf]

    Qhat = (1.0 / len(t_period)) * np.fft.rfft(Q, axis=-1)[:, : (N_h + 1)]

    As = np.array(
        [
            evaluate_jacobian(t_period[i], Q[:, i], params).reshape(-1, 1)
            for i in range(len(t_period))
        ]
    )
    As = np.concatenate(As, axis=-1)
    Ashat = (1.0 / len(t_period)) * np.fft.rfft(As, axis=-1)[:, : (N_h + 1)]

    omega0 = 2 * np.pi / (t_period[-1] - t_period[0])
    T_period = t_period[-1] - t_period[0]

    data_to_save = {
        "Qhat": Qhat,
        "T_period": T_period,
        "omega0": omega0,
        "N_h": N_h,
        "N_t": len(t_period),
        "params": np.array(params),
    }
    np.save(str(out_dir / "padovan_toy.npz"), data_to_save)

    log_text = f"""{Path(__file__).name} — Padovan ODE Harmonic Resolvent Validation

Parameters
----------
mu={mu:.4f}, alpha={alpha:.4f}, beta={beta:.4f}
Period T = 2π/ω, ω = √(1 - β²μ/α) = {omega:.6f}
T = {T:.6f} s

Integration
-----------
Method: RK45, rtol=1e-12, atol=1e-12
Time sampling: dt = T/300 = {dt:.6e} s
Transient: 200 periods (198T to 199T)
FFT window: 1 period (99th cycle) at t ∈ [{t_period[0]:.6e}, {t_period[-1]:.6e}]
N_t = {len(t_period)} samples

Fourier Analysis
----------------
FFT convention: forward (1/N_t), numpy sign exp(-2πijn/N_t)
Q(t) = Σ_k Qhat_k exp(+i*k*ω₀*t), k=0..{N_h}
A(t) = Σ_k Ahat_k exp(+i*k*ω₀*t), k=0..{N_h}

ω₀ = 2π/T_period = {omega0:.6f} rad/s

Harmonic Content
----------------
"""

    for k in range(N_h + 1):
        norm_q = np.linalg.norm(Qhat[:, k])
        norm_a = np.linalg.norm(Ashat[:, k])
        log_text += f"k={k:2d}: ||Qhat_k||_2={norm_q:.4e}, ||Ahat_k||_2={norm_a:.4e}\n"

    log_text += "\nCOO Export\n"
    log_text += "----------\n"
    log_text += f"Exported harmonics k=0..{N_h} to COO format (rows, cols, vals).\n"
    log_text += f"Two-sided storage: k=-{N_h}..{N_h} (via conjugation at kx=0).\n"

    log_text += "\nPETSc/resolvent4py Stage\n"
    log_text += "------------------------\n"
    log_text += "Run on c9 (complex PETSc + MUMPS) to compute harmonic resolvent gains.\n"

    (out_dir / "padovan_toy.log").write_text(log_text)
    print(f"Wrote {out_dir / 'padovan_toy.npz'} and {out_dir / 'padovan_toy.log'}")

    try:
        from linstabpy.operators.harmonic_blocks import HarmonicBlocks, write_resolvent4py_coo
    except ImportError:
        print("linstabpy not in PYTHONPATH; skipping COO export")
        return

    ahat_csr_list = []
    for k in range(N_h + 1):
        ahat_k_dense = Ashat[:, k].reshape((3, 3), order="C")
        ahat_csr = sp.csr_matrix(ahat_k_dense)
        ahat_csr_list.append(ahat_csr)

    blocks = HarmonicBlocks(
        Ahat=ahat_csr_list,
        N_h=N_h,
        N_t=len(t_period),
    )

    try:
        coo_fnames = write_resolvent4py_coo(blocks, str(out_dir / "coo"))
        print(f"COO files written to {out_dir / 'coo'}")
    except ImportError:
        print("resolvent4py not available; COO export skipped")
        print("To run harmonic resolvent on c9, install resolvent4py and run:")
        print(f"  cd {out_dir}")
        print(
            '  python -c "from linstabpy.solvers.harmonic_resolvent import harmonic_resolvent_gains"'
        )


if __name__ == "__main__":
    main()
