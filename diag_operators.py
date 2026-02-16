#!/usr/bin/env python3
"""Diagnostic: check operator accuracy on the Blasius mesh.

Verifies Dx@x, Dy@y, Dx@(x^2), Dy@(y^2) on both undeformed and deformed meshes.
"""
import numpy as np
from linstab2d.mesh import Mesh


def check_ops(label, mesh):
    """Print derivative accuracy diagnostics."""
    p = mesh.used_ind
    x = mesh.X.ravel(order="F")[p]
    y = mesh.Y.ravel(order="F")[p]

    # Interior mask: exclude boundary stencils (10% margin)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx_range = xmax - xmin
    dy_range = ymax - ymin
    interior = (
        (x > xmin + 0.1 * dx_range) & (x < xmax - 0.1 * dx_range)
        & (y > ymin + 0.1 * dy_range) & (y < ymax - 0.1 * dy_range)
    )

    Dx = mesh.Dx
    D2x = mesh.D2x
    Dy = mesh.Dy
    D2y = mesh.D2y

    # First derivatives of linear functions
    Dx_x = Dx @ x
    Dy_y = Dy @ y
    Dx_y = Dx @ y
    Dy_x = Dy @ x

    print(f"\n{label}:")
    print(f"  x: [{x.min():.6f}, {x.max():.6f}], y: [{y.min():.6f}, {y.max():.6f}]")
    print(f"  Interior points: {interior.sum()} / {p.size}")
    print(f"  Dx@x:  max_err={np.max(np.abs(Dx_x[interior] - 1.0)):.2e}")
    print(f"  Dy@y:  max_err={np.max(np.abs(Dy_y[interior] - 1.0)):.2e}")
    print(f"  Dx@y:  max_err={np.max(np.abs(Dx_y[interior])):.2e}  (should be 0)")
    print(f"  Dy@x:  max_err={np.max(np.abs(Dy_x[interior])):.2e}  (should be 0)")

    # Second derivatives of quadratics
    D2x_x2 = D2x @ (x**2)
    D2y_y2 = D2y @ (y**2)
    print(f"  D2x@(x^2): max_err={np.max(np.abs(D2x_x2[interior] - 2.0)):.2e}")
    print(f"  D2y@(y^2): max_err={np.max(np.abs(D2y_y2[interior] - 2.0)):.2e}")

    # Check Dx @ sin(2*pi*x/(xmax-xmin)) for accuracy
    k = 2 * np.pi / (xmax - xmin)
    f = np.sin(k * x)
    Dx_f = Dx @ f
    exact = k * np.cos(k * x)
    err = np.max(np.abs(Dx_f[interior] - exact[interior]))
    print(f"  Dx@sin(kx): max_interior_err={err:.2e}")

    k_y = 2 * np.pi / (ymax - ymin)
    g = np.sin(k_y * y)
    Dy_g = Dy @ g
    exact_g = k_y * np.cos(k_y * y)
    err_g = np.max(np.abs(Dy_g[interior] - exact_g[interior]))
    print(f"  Dy@sin(ky): max_interior_err={err_g:.2e}")

    # Operator sparsity
    print(f"  Dx: nnz/row={Dx.nnz/Dx.shape[0]:.1f}")
    print(f"  Dy: nnz/row={Dy.nnz/Dy.shape[0]:.1f}")

    return Dx_x, Dy_y


def main():
    nx, ny = 300, 100
    dx = 0.01
    x_min, x_max = 5 * dx, 1.25 + dx
    y_i, y_max_phys = 0.01, 0.05

    # Undeformed mesh
    mesh_comp = Mesh.create((x_min, x_max), (-1, 1), nx, ny, fd_order=4)
    check_ops("UNDEFORMED (computational)", mesh_comp)

    # Deformed mesh (Hanifi)
    Y = mesh_comp.Y.copy()
    a = y_i * y_max_phys / (y_max_phys - 2 * y_i)
    b = 1 + 2 * a / y_max_phys
    Y = a * (1 + Y) / (b - Y)
    mesh_phys = mesh_comp.deform(Y_new=Y)
    check_ops("DEFORMED (Hanifi physical)", mesh_phys)

    # Also check integration weights
    p = mesh_phys.used_ind
    W = mesh_phys.W.ravel(order="F")[p]
    print(f"\n  Integration weights: min={W.min():.2e}, max={W.max():.2e}")
    print(f"  sum(W)={W.sum():.6f}, domain area={(x_max-x_min)*y_max_phys:.6f}")


if __name__ == "__main__":
    main()
