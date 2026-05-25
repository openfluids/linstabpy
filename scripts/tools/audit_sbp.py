"""SBP operator audit script.

Checks stencil fidelity against MATLAB Dmats_SBP.m, polynomial exactness,
SBP property Q+Q^T=B, consistency, and anti-symmetry for orders 4, 6, 8.

Run with:
    .venv/bin/python scripts/tools/audit_sbp.py
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from linstabpy.mesh.sbp import build_sbp_matrix, sbp_stencils
from linstabpy.mesh.sbp_generator import _sbp_norm_weights

# ---------------------------------------------------------------------------
# MATLAB reference stencils — transcribed literally from Dmats_SBP.m
# ---------------------------------------------------------------------------


def matlab_stencils(order: int) -> dict[str, np.ndarray]:
    """Return stencil coefficients transcribed exactly from Dmats_SBP.m."""
    if order == 4:
        a_i1 = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        a_i2 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
        a_b1 = np.array(
            [
                [-24 / 17, 59 / 34, -4 / 17, -3 / 34, 0, 0],
                [-1 / 2, 0, 1 / 2, 0, 0, 0],
                [4 / 43, -59 / 86, 0, 59 / 86, -4 / 43, 0],
                [3 / 98, 0, -59 / 98, 0, 32 / 49, -4 / 49],
            ]
        )
        a_b2 = np.array(
            [
                [2, -5, 4, -1, 0, 0],
                [1, -2, 1, 0, 0, 0],
                [-4 / 43, 59 / 43, -110 / 43, 59 / 43, -4 / 43, 0],
                [-1 / 49, 0, 59 / 49, -118 / 49, 64 / 49, -4 / 49],
            ]
        )

    elif order == 6:
        a_i1 = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
        a_i2 = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
        a_b1 = np.array(
            [
                [
                    -21600 / 13649,
                    104009 / 54596,
                    30443 / 81894,
                    -33311 / 27298,
                    16863 / 27298,
                    -15025 / 163788,
                    0,
                    0,
                    0,
                ],
                [
                    -104009 / 240260,
                    0,
                    -311 / 72078,
                    20229 / 24026,
                    -24337 / 48052,
                    36661 / 360390,
                    0,
                    0,
                    0,
                ],
                [
                    -30443 / 162660,
                    311 / 32532,
                    0,
                    -11155 / 16266,
                    41287 / 32532,
                    -21999 / 54220,
                    0,
                    0,
                    0,
                ],
                [
                    33311 / 107180,
                    -20229 / 21436,
                    485 / 1398,
                    0,
                    4147 / 21436,
                    25427 / 321540,
                    72 / 5359,
                    0,
                    0,
                ],
                [
                    -16863 / 78770,
                    24337 / 31508,
                    -41287 / 47262,
                    -4147 / 15754,
                    0,
                    342523 / 472620,
                    -1296 / 7877,
                    144 / 7877,
                    0,
                ],
                [
                    15025 / 525612,
                    -36661 / 262806,
                    21999 / 87602,
                    -25427 / 262806,
                    -342523 / 525612,
                    0,
                    32400 / 43801,
                    -6480 / 43801,
                    720 / 43801,
                ],
            ]
        )
        a_b2 = np.array(
            [
                [
                    114170 / 40947,
                    -438107 / 54596,
                    336409 / 40947,
                    -276997 / 81894,
                    3747 / 13649,
                    21035 / 163788,
                    0,
                    0,
                    0,
                ],
                [
                    6173 / 5860,
                    -2066 / 879,
                    3283 / 1758,
                    -303 / 293,
                    2111 / 3516,
                    -601 / 4395,
                    0,
                    0,
                    0,
                ],
                [
                    -52391 / 81330,
                    134603 / 32532,
                    -21982 / 2711,
                    112915 / 16266,
                    -46969 / 16266,
                    30409 / 54220,
                    0,
                    0,
                    0,
                ],
                [
                    68603 / 321540,
                    -12423 / 10718,
                    112915 / 32154,
                    -75934 / 16077,
                    53369 / 21436,
                    -54899 / 160770,
                    48 / 5359,
                    0,
                    0,
                ],
                [
                    -7053 / 39385,
                    86551 / 94524,
                    -46969 / 23631,
                    53369 / 15754,
                    -87904 / 23631,
                    820271 / 472620,
                    -1296 / 7877,
                    96 / 7877,
                    0,
                ],
                [
                    21035 / 525612,
                    -24641 / 131403,
                    30409 / 87602,
                    -54899 / 131403,
                    820271 / 525612,
                    -117600 / 43801,
                    64800 / 43801,
                    -6480 / 43801,
                    480 / 43801,
                ],
            ]
        )

    elif order == 8:
        a_i1 = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
        a_i2 = np.array(
            [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
        )
        a_b1 = np.array(
            [
                [
                    -2540160 / 1498139,
                    5544277 / 5992556,
                    198794991 / 29962780,
                    -256916579 / 17977668,
                    20708767 / 1498139,
                    -41004357 / 5992556,
                    27390659 / 17977668,
                    -2323531 / 29962780,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -5544277 / 31004596,
                    0,
                    -85002381 / 22146140,
                    49607267 / 4429228,
                    -165990199 / 13287684,
                    7655859 / 1107307,
                    -7568311 / 4429228,
                    48319961 / 465068940,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -66264997 / 8719620,
                    9444709 / 415220,
                    0,
                    -20335981 / 249132,
                    32320879 / 249132,
                    -35518713 / 415220,
                    2502774 / 103805,
                    -3177073 / 1743924,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    256916579 / 109619916,
                    -49607267 / 5219996,
                    61007943 / 5219996,
                    0,
                    -68748371 / 5219996,
                    65088123 / 5219996,
                    -66558305 / 15659988,
                    3870214 / 9134993,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -20708767 / 2096689,
                    165990199 / 3594324,
                    -96962637 / 1198108,
                    68748371 / 1198108,
                    0,
                    -27294549 / 1198108,
                    14054993 / 1198108,
                    -42678199 / 25160268,
                    -2592 / 299527,
                    0,
                    0,
                    0,
                ],
                [
                    13668119 / 8660148,
                    -850651 / 103097,
                    35518713 / 2061940,
                    -21696041 / 1237164,
                    9098183 / 1237164,
                    0,
                    -231661 / 412388,
                    7120007 / 43300740,
                    3072 / 103097,
                    -288 / 103097,
                    0,
                    0,
                ],
                [
                    -27390659 / 56287644,
                    7568311 / 2680364,
                    -22524966 / 3350455,
                    66558305 / 8041092,
                    -14054993 / 2680364,
                    2084949 / 2680364,
                    0,
                    70710683 / 93812740,
                    -145152 / 670091,
                    27648 / 670091,
                    -2592 / 670091,
                    0,
                ],
                [
                    2323531 / 102554780,
                    -48319961 / 307664340,
                    9531219 / 20510956,
                    -3870214 / 5127739,
                    2246221 / 3238572,
                    -21360021 / 102554780,
                    -70710683 / 102554780,
                    0,
                    4064256 / 5127739,
                    -1016064 / 5127739,
                    193536 / 5127739,
                    -18144 / 5127739,
                ],
            ]
        )
        a_b2 = np.array(
            [
                [
                    4870382994799 / 1358976868290,
                    -893640087518 / 75498714905,
                    926594825119 / 60398971924,
                    -1315109406200 / 135897686829,
                    39126983272 / 15099742981,
                    12344491342 / 75498714905,
                    -451560522577 / 2717953736580,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    333806012194 / 390619153855,
                    -154646272029 / 111605472530,
                    1168338040 / 33481641759,
                    82699112501 / 133926567036,
                    -171562838 / 11160547253,
                    -28244698346 / 167408208795,
                    11904122576 / 167408208795,
                    -2598164715 / 312495323084,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    7838984095 / 52731029988,
                    1168338040 / 5649753213,
                    -88747895 / 144865467,
                    423587231 / 627750357,
                    -43205598281 / 22599012852,
                    4876378562 / 1883251071,
                    -5124426509 / 3766502142,
                    10496900965 / 39548272491,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -94978241528 / 828644350023,
                    82699112501 / 157837019052,
                    1270761693 / 13153084921,
                    -167389605005 / 118377764289,
                    48242560214 / 39459254763,
                    -31673996013 / 52612339684,
                    43556319241 / 118377764289,
                    -44430275135 / 552429566682,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1455067816 / 21132528431,
                    -171562838 / 3018932633,
                    -43205598281 / 36227191596,
                    48242560214 / 9056797899,
                    -52276055645 / 6037865266,
                    57521587238 / 9056797899,
                    -80321706377 / 36227191596,
                    8078087158 / 21132528431,
                    -1296 / 299527,
                    0,
                    0,
                    0,
                ],
                [
                    10881504334 / 327321118845,
                    -28244698346 / 140280479505,
                    4876378562 / 9352031967,
                    -10557998671 / 12469375956,
                    57521587238 / 28056095901,
                    -278531401019 / 93520319670,
                    73790130002 / 46760159835,
                    -137529995233 / 785570685228,
                    2048 / 103097,
                    -144 / 103097,
                    0,
                    0,
                ],
                [
                    -135555328849 / 8509847458140,
                    11904122576 / 101307707835,
                    -5124426509 / 13507694378,
                    43556319241 / 60784624701,
                    -80321706377 / 81046166268,
                    73790130002 / 33769235945,
                    -950494905688 / 303923123505,
                    239073018673 / 141830790969,
                    -145152 / 670091,
                    18432 / 670091,
                    -1296 / 670091,
                    0,
                ],
                [
                    0,
                    -2598164715 / 206729925524,
                    10496900965 / 155047444143,
                    -44430275135 / 310094888286,
                    425162482 / 2720130599,
                    -137529995233 / 620189776572,
                    239073018673 / 155047444143,
                    -144648000000 / 51682481381,
                    8128512 / 5127739,
                    -1016064 / 5127739,
                    129024 / 5127739,
                    -9072 / 5127739,
                ],
            ]
        )
    else:
        raise ValueError(f"No MATLAB reference for order {order}")

    return {"a_i1": a_i1, "a_i2": a_i2, "a_b1": a_b1, "a_b2": a_b2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
ORDERS = [4, 6, 8]

results: dict[int, dict[str, str]] = {o: {} for o in ORDERS}


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# (A) Stencil equality
# ---------------------------------------------------------------------------

section("(A) Stencil equality vs MATLAB Dmats_SBP.m — tol=1e-13")

for order in ORDERS:
    py = sbp_stencils(order)
    ml = matlab_stencils(order)
    mismatches = []
    for key in ["a_i1", "a_i2", "a_b1", "a_b2"]:
        pa = py[key]
        ma = ml[key]
        if pa.shape != ma.shape:
            mismatches.append(f"  {key}: shape mismatch Python={pa.shape} MATLAB={ma.shape}")
            continue
        diff = np.abs(pa - ma)
        if not np.allclose(pa, ma, atol=1e-13, rtol=0):
            idxs = np.argwhere(diff > 1e-13)
            for idx in idxs:
                idx_t = tuple(idx)
                mismatches.append(
                    f"  {key}[{idx_t}]: python={pa[idx_t]:.16g}  "
                    f"matlab={ma[idx_t]:.16g}  |diff|={diff[idx_t]:.3e}"
                )
    if mismatches:
        print(f"order={order}: FAIL")
        for m in mismatches:
            print(m)
        results[order]["stencils"] = FAIL
    else:
        print(f"order={order}: PASS")
        results[order]["stencils"] = PASS


# ---------------------------------------------------------------------------
# (B) Polynomial exactness
# ---------------------------------------------------------------------------

section("(B) Polynomial exactness — D1 and D2")

for order in ORDERS:
    n = 6 * order
    dz = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    D1, D2 = build_sbp_matrix(n, dz, order)
    D1d = D1.toarray()
    D2d = D2.toarray()

    stencils = sbp_stencils(order)
    r = stencils["a_b1"].shape[0]
    w = (len(stencils["a_i1"]) - 1) // 2

    int_start = w
    int_end = n - r  # exclusive

    print(f"\n--- order={order}  n={n}  r={r}  w={w} ---")
    print(
        f"  {'k':>3}  {'D1_int':>12}  {'D1_bm':>12}  {'D1_bp':>12}  "
        f"{'D2_int':>12}  {'D2_bm':>12}  {'D2_bp':>12}"
    )

    any_fail = False
    for k in range(order + 3):
        f = x**k
        fp_exact = k * x ** (k - 1) if k >= 1 else np.zeros(n)
        fpp_exact = k * (k - 1) * x ** (k - 2) if k >= 2 else np.zeros(n)

        d1f = D1d @ f
        d2f = D2d @ f

        e1 = np.abs(d1f - fp_exact)
        e2 = np.abs(d2f - fpp_exact)

        e1_int = np.max(e1[int_start:int_end]) if int_end > int_start else 0.0
        e1_bm = np.max(e1[:r])
        e1_bp = np.max(e1[n - r :])
        e2_int = np.max(e2[int_start:int_end]) if int_end > int_start else 0.0
        e2_bm = np.max(e2[:r])
        e2_bp = np.max(e2[n - r :])

        # Flag unexpected failures
        # Interior: should be exact for k <= order
        # Boundary: should be exact for k <= order//2 (half-order)
        flags = []
        if k <= order and e1_int > 1e-8:
            flags.append("D1_int!")
        if k <= order and e2_int > 1e-8:
            flags.append("D2_int!")
        if k <= order // 2 and e1_bm > 1e-8:
            flags.append("D1_bm!")
        if k <= order // 2 and e1_bp > 1e-8:
            flags.append("D1_bp!")
        if k <= order // 2 and e2_bm > 1e-8:
            flags.append("D2_bm!")
        if k <= order // 2 and e2_bp > 1e-8:
            flags.append("D2_bp!")

        flag_str = "  <-- " + " ".join(flags) if flags else ""
        if flags:
            any_fail = True

        print(
            f"  {k:>3d}  {e1_int:12.3e}  {e1_bm:12.3e}  {e1_bp:12.3e}  "
            f"{e2_int:12.3e}  {e2_bm:12.3e}  {e2_bp:12.3e}{flag_str}"
        )

    status = FAIL if any_fail else PASS
    results[order]["poly"] = status
    print(f"  => {status}")


# ---------------------------------------------------------------------------
# (C) SBP property Q + Q^T = B
# ---------------------------------------------------------------------------

section("(C) SBP property Q + Q^T = B — tol=1e-10")

for order in ORDERS:
    n = 6 * order
    dz = 1.0 / (n - 1)
    D1, _ = build_sbp_matrix(n, dz, order)
    D1d = D1.toarray()

    h_bdy = _sbp_norm_weights(order)
    r = len(h_bdy)
    h_vec = np.ones(n)
    h_vec[:r] = h_bdy
    h_vec[n - r :] = h_bdy[::-1]
    H = np.diag(h_vec * dz)

    Q = H @ D1d
    S = Q + Q.T

    # B should be: S[0,0] = -1, S[n-1,n-1] = +1, rest = 0
    B_ref = np.zeros((n, n))
    B_ref[0, 0] = -1.0
    B_ref[n - 1, n - 1] = +1.0

    err = S - B_ref
    # Mask the two diagonal corners
    mask = np.ones((n, n), dtype=bool)
    mask[0, 0] = False
    mask[n - 1, n - 1] = False

    worst_off = np.max(np.abs(err[mask]))
    worst_diag_0 = abs(err[0, 0])
    worst_diag_n = abs(err[n - 1, n - 1])
    worst_idx = np.unravel_index(np.argmax(np.abs(err[mask].reshape(-1))), err[mask].shape)

    # Locate the actual worst off-diagonal
    err_masked = np.abs(err.copy())
    err_masked[0, 0] = 0.0
    err_masked[n - 1, n - 1] = 0.0
    wi, wj = np.unravel_index(np.argmax(err_masked), err_masked.shape)

    tol = 1e-10
    ok = (worst_off < tol) and (worst_diag_0 < tol) and (worst_diag_n < tol)
    status = PASS if ok else FAIL
    results[order]["SBP"] = status
    print(
        f"order={order}: {status}  "
        f"worst_off={worst_off:.3e} at ({wi},{wj})  "
        f"|B[0,0]+1|={worst_diag_0:.3e}  |B[n-1,n-1]-1|={worst_diag_n:.3e}"
    )


# ---------------------------------------------------------------------------
# (D) Consistency checks
# ---------------------------------------------------------------------------

section("(D) Consistency: D1@1=0, D2@1=0, D2@x=0, D1@x=1")

for order in ORDERS:
    n = 6 * order
    dz = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    D1, D2 = build_sbp_matrix(n, dz, order)
    D1d = D1.toarray()
    D2d = D2.toarray()

    ones = np.ones(n)
    e_D1_1 = np.max(np.abs(D1d @ ones))
    e_D2_1 = np.max(np.abs(D2d @ ones))
    e_D2_x = np.max(np.abs(D2d @ x))
    e_D1_x = np.max(np.abs(D1d @ x - ones))

    tol = 1e-10
    ok = all(e < tol for e in [e_D1_1, e_D2_1, e_D2_x, e_D1_x])
    status = PASS if ok else FAIL
    results[order]["consistency"] = status
    print(
        f"order={order}: {status}  "
        f"|D1@1|={e_D1_1:.3e}  |D2@1|={e_D2_1:.3e}  "
        f"|D2@x|={e_D2_x:.3e}  |D1@x-1|={e_D1_x:.3e}"
    )


# ---------------------------------------------------------------------------
# (E) Anti-symmetry under reversal: D1[i,j] == -D1[n-1-i, n-1-j]
# ---------------------------------------------------------------------------

section("(E) Anti-symmetry under reversal — n=40, tol=1e-13")

for order in ORDERS:
    n = 40
    dz = 1.0 / (n - 1)
    D1, _ = build_sbp_matrix(n, dz, order)
    D1d = D1.toarray()
    D1_rev = D1d[::-1, ::-1]
    diff = D1d + D1_rev  # should be zero
    worst = np.max(np.abs(diff))
    wi, wj = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    ok = worst < 1e-13
    status = PASS if ok else FAIL
    results[order]["reversal"] = status
    print(
        f"order={order}: {status}  "
        f"max|D1+D1_rev|={worst:.3e} at ({wi},{wj})  "
        f"D1[{wi},{wj}]={D1d[wi, wj]:.6g}  D1_rev[{wi},{wj}]={D1_rev[wi, wj]:.6g}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

section("SUMMARY TABLE")
checks = ["stencils", "poly", "SBP", "consistency", "reversal"]
header = f"{'order':>6}" + "".join(f"  {c:>13}" for c in checks)
print(header)
print("-" * len(header))
for o in ORDERS:
    row = f"{o:>6}"
    for c in checks:
        val = results[o].get(c, "?")
        sym = "OK" if val == PASS else "FAIL"
        row += f"  {sym:>13}"
    print(row)

any_failure = any(results[o].get(c, FAIL) == FAIL for o in ORDERS for c in checks)
print()
print("OVERALL:", "FAIL — see details above" if any_failure else "PASS — all checks green")
