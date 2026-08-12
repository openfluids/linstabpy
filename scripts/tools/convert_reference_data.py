#!/usr/bin/env python3
"""Record of the Song and Martini reference pack.

This script wrote examples/song/*.npz and examples/martini/*.npz from the
source trees that now live in the private paper archive. Re-run is not
needed. The 32 deg field uses velocity / c_inf and temperature / T_inf
(u *= Ma, T left at one). Martini mode arrays are complex64.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "The reference pack already sits under examples/song and examples/martini. "
        "The source trees are in the private archive."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
