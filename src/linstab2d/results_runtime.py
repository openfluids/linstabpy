"""Compatibility alias for the study runtime implementation.

The case-style orchestration lives under ``studies.common``. Importing this
module keeps existing ``linstab2d.results_runtime`` users on the same module
object so monkeypatches and direct attribute imports continue to work.
"""

from __future__ import annotations

import sys

from studies.common import results_runtime as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
