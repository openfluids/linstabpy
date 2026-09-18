#!/usr/bin/env python3
"""Compare the leading response modes of two runs of the same point.

Load two runtime ``.npz`` files from one ``(f, beta)`` point.
Print the Chu-weighted correlation of their leading response modes.
This shows that a second backend returns the same mode, not only
the same gain. The metric matches ``compare_modes.py``: unit Chu
norm for both modes, then correlation ``|<q_a, W q_b>|``. The
conjugate of the second mode is also tried. This covers an
opposite temporal Fourier convention.

Usage::

    python compare_backend_modes.py reference_run.npz other_run.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


def _chu_ip(
    q_a: NDArray[np.complex128],
    w_diag: NDArray[np.float64],
    q_b: NDArray[np.complex128],
) -> complex:
    """Compute the Chu inner product of two modes.

    Returns
    -------
    complex
        The inner product ``<q_a, W q_b>``.
    """
    w_complex = w_diag.astype(np.complex128)
    # np.vdot has no typed return in the numpy stubs; the value is a scalar.
    return complex(cast("np.complex128", np.vdot(q_a, w_complex * q_b)))


def _correlate(
    q_a: NDArray[np.complex128],
    q_b: NDArray[np.complex128],
    w_diag: NDArray[np.float64],
) -> tuple[float, bool]:
    """Correlate two modes at unit Chu norm.

    Returns
    -------
    tuple[float, bool]
        The correlation and a flag. The flag is true when the
        conjugate of ``q_b`` correlated better than ``q_b``.
    """
    q_a_unit = q_a / float(np.sqrt(_chu_ip(q_a, w_diag, q_a).real))
    q_b_unit = q_b / float(np.sqrt(_chu_ip(q_b, w_diag, q_b).real))
    c_plain = abs(_chu_ip(q_a_unit, w_diag, q_b_unit))
    c_conj = abs(_chu_ip(q_a_unit, w_diag, np.conj(q_b_unit)))
    return max(c_plain, c_conj), c_conj > c_plain


def _load(path: Path) -> tuple[NDArray[np.complex128], NDArray[np.float64], float]:
    """Load one runtime ``.npz`` file.

    Returns
    -------
    tuple
        The leading response mode, the Chu weight, and the gain.
    """
    data = np.load(path, allow_pickle=True)
    mode = np.asarray(data["responses"][:, 0], dtype=np.complex128)
    w_diag = np.asarray(data["W_diag"], dtype=np.float64)
    return mode, w_diag, float(data["gains"][0])


def main() -> int:
    """Run the command-line comparison.

    Returns
    -------
    int
        Zero on success.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference_npz", type=Path, help="runtime .npz of the reference run")
    parser.add_argument("other_npz", type=Path, help="runtime .npz of the run to compare")
    args = parser.parse_args()

    q_ref, w_ref, gain_ref = _load(args.reference_npz)
    q_other, w_other, gain_other = _load(args.other_npz)
    if q_ref.shape != q_other.shape:
        parser.error(f"mode sizes differ: {q_ref.shape} vs {q_other.shape}")

    align, conjugated = _correlate(q_ref, q_other, w_ref)

    # This is a command-line tool; the printed lines are its product.
    if not np.allclose(w_ref, w_other):
        print("note: the runs carry different Chu weights; the reference weight is used")
    print(f"gain      {gain_ref:.10f}  vs  {gain_other:.10f}")
    print(f"Chu-weighted mode correlation = {align:.16f}  (1 - {1.0 - align:.3e})")
    if conjugated:
        print("(the second mode was conjugated: opposite temporal Fourier convention)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
