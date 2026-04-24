"""Verify Dxy_new coefficient convention: Python (Ji11*Ji22) vs MATLAB-swap (Ji12*Ji21).

Claim under test
----------------
Python's _transform_second_derivatives uses coefficient Ji11*Ji22 (= a*d) on the
computational Dxy operator when building physical Dxy_new, while MATLAB's
DeformMesh.m reportedly uses Ji12*Ji21 (= b*c).  For smooth interior points,
Dxy ≡ Dyx continuously so the difference vanishes.  The claim is that the gap
shows up at boundary-closure rows where the discrete Dxy and Dyx matrices differ.

Test function: f(x,y) = x^2 * y^3  =>  f_xy = 6*x*y^2  (analytic)

Shear mapping on [0.1,1.1] x [0.2,1.2]  (domain avoids zero to prevent
division-by-zero in relative error; analytic value 6*x*y^2 >= 0.024 there):
  X_new = xi + 0.3*eta
  Y_new = eta + 0.1*xi
This makes Ji genuinely non-diagonal: Ji11*Ji22 = 1.063, Ji12*Ji21 = 0.032.

Key algebraic fact
------------------
On a full rectangular tensor-product mesh (Mesh.create with no masking),
Dxy_comp = Dyx_comp EXACTLY, by the Kronecker identity:
  (D1x ⊗ I_ny)(I_nx ⊗ D1y) = D1x ⊗ D1y = (I_nx ⊗ D1y)(D1x ⊗ I_ny)
SBP boundary closure affects D1x and D1y individually but does NOT break this.
Therefore the swap (c12 <-> c21) inside _transform_second_block is
ALGEBRAICALLY INVISIBLE: c12*Dxy + c21*Dyx = (c12+c21)*Dxy when Dxy=Dyx.

This script:
  1. Confirms the algebraic invisibility on full grids.
  2. Reports absolute/relative errors for Python Dxy_new vs analytic.
  3. Reports the *coefficient difference* (a*d vs b*c) that would matter
     on irregular/masked meshes where Dxy != Dyx.
  4. Prints the summary table requested.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from linstab2d.mesh.mesh import Mesh, _transform_second_block


def _transform_second_derivatives_matlab_swap(
    a, b, c, d, D2x, D2y, Dxy, Dyx, Dx, Dy, x, y
):
    """Same as mesh.py but with c12/c21 swapped in Dxy_new's _transform_second_block call.

    Python Dxy_new: c11=a*c, c12=a*d, c21=b*c, c22=b*d
    MATLAB-swap:    c11=a*c, c12=b*c, c21=a*d, c22=b*d
    """
    D2x_x = D2x @ x;  D2x_y = D2x @ y
    Dxy_x = Dx @ (Dy @ x);  Dxy_y = Dx @ (Dy @ y)
    Dyx_x = Dy @ (Dx @ x);  Dyx_y = Dy @ (Dx @ y)
    D2y_x = D2y @ x;  D2y_y = D2y @ y

    dJ11_xi  = D2x_x;  dJ12_xi  = D2x_y
    dJ21_xi  = Dxy_x;  dJ22_xi  = Dxy_y
    dJ11_eta = Dyx_x;  dJ12_eta = Dyx_y
    dJ21_eta = D2y_x;  dJ22_eta = D2y_y

    p11_xi  = dJ11_xi*a  + dJ12_xi*c;   p12_xi  = dJ11_xi*b  + dJ12_xi*d
    p21_xi  = dJ21_xi*a  + dJ22_xi*c;   p22_xi  = dJ21_xi*b  + dJ22_xi*d
    p11_eta = dJ11_eta*a + dJ12_eta*c;  p12_eta = dJ11_eta*b + dJ12_eta*d
    p21_eta = dJ21_eta*a + dJ22_eta*c;  p22_eta = dJ21_eta*b + dJ22_eta*d

    dJi11_xi  = -(a*p11_xi  + b*p21_xi);   dJi12_xi  = -(a*p12_xi  + b*p22_xi)
    dJi21_xi  = -(c*p11_xi  + d*p21_xi);   dJi22_xi  = -(c*p12_xi  + d*p22_xi)
    dJi11_eta = -(a*p11_eta + b*p21_eta);  dJi12_eta = -(a*p12_eta + b*p22_eta)
    dJi21_eta = -(c*p11_eta + d*p21_eta);  dJi22_eta = -(c*p12_eta + d*p22_eta)

    q11_x = a*dJi11_xi + b*dJi11_eta;  q11_y = a*dJi12_xi + b*dJi12_eta
    q12_x = c*dJi11_xi + d*dJi11_eta;  q12_y = c*dJi12_xi + d*dJi12_eta
    q21_x = a*dJi21_xi + b*dJi21_eta;  q21_y = a*dJi22_xi + b*dJi22_eta
    q22_x = c*dJi21_xi + d*dJi21_eta;  q22_y = c*dJi22_xi + d*dJi22_eta

    D2x_new = _transform_second_block(
        a*a, a*b, b*a, b*b, q11_x, q11_y,
        D2x, D2y, Dxy, Dyx, Dx, Dy,
    )
    # MATLAB-swap: b*c on Dxy (c12), a*d on Dyx (c21)
    Dxy_new = _transform_second_block(
        a*c, b*c, a*d, b*d,  # c12=b*c, c21=a*d  [swapped vs python: a*d, b*c]
        q12_x, q12_y,
        D2x, D2y, Dxy, Dyx, Dx, Dy,
    )
    Dyx_new = _transform_second_block(
        c*a, c*b, d*a, d*b, q21_x, q21_y,
        D2x, D2y, Dxy, Dyx, Dx, Dy,
    )
    D2y_new = _transform_second_block(
        c*c, c*d, d*c, d*d, q22_x, q22_y,
        D2x, D2y, Dxy, Dyx, Dx, Dy,
    )
    return D2x_new, Dxy_new, Dyx_new, D2y_new


def run_for_order(fd_order, nx=40, ny=40):
    # Domain avoids zero so analytic 6*x*y^2 >= 0.024 everywhere
    xrange = (0.1, 1.1)
    yrange = (0.2, 1.2)

    mesh = Mesh.create(xrange, yrange, nx, ny, fd_order)
    xi_2d = mesh.X;  eta_2d = mesh.Y

    # Shear mapping
    X_new = xi_2d + 0.3 * eta_2d
    Y_new = eta_2d + 0.1 * xi_2d

    # Check computational Dxy == Dyx (should always be true on full tensor-product mesh)
    dxy_dyx_diff = abs(mesh.Dxy - mesh.Dyx).max()

    # Deformed mesh (Python convention)
    m = mesh.deform(X_new=X_new, Y_new=Y_new)
    x_flat = m.X.ravel(order="F")
    y_flat = m.Y.ravel(order="F")

    f = x_flat**2 * y_flat**3
    f_xy_analytic = 6.0 * x_flat * y_flat**2

    f_xy_py  = m.Dxy @ f
    f_yx_py  = m.Dyx @ f

    # Boundary and interior masks
    bi = m.idx["bi"]; ti = m.idx["ti"]; li = m.idx["li"]; ri = m.idx["ri"]
    bnd = np.zeros(m.ngp, dtype=bool)
    for idx_arr in [bi, ti, li, ri]:
        bnd[idx_arr] = True
    interior = ~bnd

    err_py_int = np.abs(f_xy_py[interior] - f_xy_analytic[interior]).max()
    err_py_bnd = np.abs(f_xy_py[bnd]      - f_xy_analytic[bnd]).max()
    err_dyx_bnd = np.abs(f_yx_py[bnd]     - f_xy_analytic[bnd]).max()

    # Build MATLAB-swap Dxy manually from computational operators
    p = mesh.used_ind
    Dx = mesh.Dx; D2x = mesh.D2x
    Dy = mesh.Dy; D2y = mesh.D2y
    Dxy_comp = mesh.Dxy; Dyx_comp = mesh.Dyx

    x_comp = X_new.ravel(order="F")[p]
    y_comp = Y_new.ravel(order="F")[p]

    J11 = Dx @ x_comp; J12 = Dx @ y_comp
    J21 = Dy @ x_comp; J22 = Dy @ y_comp
    detJ = J11*J22 - J12*J21; inv_det = 1.0/detJ
    a = J22*inv_det; b = -J12*inv_det; c = -J21*inv_det; d = J11*inv_det

    _, Dxy_ml, _, _ = _transform_second_derivatives_matlab_swap(
        a, b, c, d, D2x, D2y, Dxy_comp, Dyx_comp, Dx, Dy, x_comp, y_comp
    )

    f_xy_ml = Dxy_ml @ f
    err_ml_int = np.abs(f_xy_ml[interior] - f_xy_analytic[interior]).max()
    err_ml_bnd = np.abs(f_xy_ml[bnd]      - f_xy_analytic[bnd]).max()

    # Are the two matrices identical (expected: yes, because Dxy_comp = Dyx_comp)
    matrices_identical = np.allclose(m.Dxy.toarray(), Dxy_ml.toarray(), atol=1e-14)

    # Coefficient ratio (how different would they be on an irregular mesh)
    ad = float(np.mean(a * d))
    bc = float(np.mean(b * c))

    return {
        "order": fd_order,
        "dxy_dyx_comp_diff": dxy_dyx_diff,
        "matrices_identical": matrices_identical,
        "ad": ad, "bc": bc,
        "err_py_int": err_py_int,
        "err_py_bnd": err_py_bnd,
        "err_ml_bnd": err_ml_bnd,
        "err_dyx_bnd": err_dyx_bnd,
    }


if __name__ == "__main__":
    results = []
    for order in [4, 6, 8]:
        r = run_for_order(order)
        results.append(r)

    print()
    print("Verification: Dxy_new Python vs MATLAB-swap convention")
    print("Domain: [0.1,1.1] x [0.2,1.2], shear X=xi+0.3*eta, Y=eta+0.1*xi, nx=ny=40")
    print()

    print(f"{'order':>5}  {'py Dxy @ interior':>18}  {'py Dxy @ boundary':>18}  {'ml-swap Dxy @ bnd':>18}")
    print("-" * 68)
    for r in results:
        print(
            f"{r['order']:>5}  "
            f"{r['err_py_int']:18.3e}  "
            f"{r['err_py_bnd']:18.3e}  "
            f"{r['err_ml_bnd']:18.3e}"
        )

    print()
    print("Algebraic checks:")
    print(f"{'order':>5}  {'||Dxy-Dyx||_comp':>18}  {'matrices_equal':>14}  {'a*d (Python coef)':>18}  {'b*c (MATLAB coef)':>18}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['order']:>5}  "
            f"{r['dxy_dyx_comp_diff']:18.2e}  "
            f"{str(r['matrices_identical']):>14}  "
            f"{r['ad']:18.6f}  "
            f"{r['bc']:18.6f}"
        )

    print()
    print("=== VERDICT ===")
    r4 = results[0]
    print(f"  a*d (Python Ji11*Ji22) = {r4['ad']:.6f}")
    print(f"  b*c (MATLAB Ji12*Ji21) = {r4['bc']:.6f}")
    print(f"  Ratio a*d / b*c        = {r4['ad']/r4['bc']:.2f}x")
    print()
    print("  On full rectangular tensor-product meshes, Dxy_comp = Dyx_comp EXACTLY")
    print("  (Kronecker product identity; SBP closure does not break this).")
    print("  Therefore both conventions produce IDENTICAL Dxy_new matrices.")
    print("  The Python/MATLAB coefficient swap is numerically invisible here.")
    print()
    print("  The claim 'gap shows up at boundary-closure rows' is INCORRECT")
    print("  for full grids: boundary closure is absorbed into D1x and D1y")
    print("  but Dxy = D1x (kron) D1y = Dyx holds regardless of closure scheme.")
    print()
    all_tied = all(abs(r["err_py_bnd"] - r["err_ml_bnd"]) / (r["err_py_bnd"] + 1e-30) < 1e-10
                   for r in results)
    if all_tied:
        print("  CONCLUSION: Python code is CORRECT for its use case (full rectangular")
        print("  grids). The MATLAB/Python coefficient difference (Ji11*Ji22 vs Ji12*Ji21)")
        print("  is real but has zero numerical effect on any tensor-product mesh.")
        print("  No test on a Mesh.create() grid (without masking) can distinguish them.")
