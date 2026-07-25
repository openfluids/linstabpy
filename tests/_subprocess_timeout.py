# SPDX-License-Identifier: GPL-3.0-only
"""Load-aware timeouts for tests that shell out to a subprocess.

Why these timeouts exist, and what they are NOT
-----------------------------------------------
Several tests run a checked-in script or a re-entrant copy of themselves in a
subprocess. The ``timeout=`` on those calls is a **guard against a hang** -- it
stops a wedged solve from blocking the suite forever. It is *not* a performance
assertion: no correctness property of linstabpy depends on a subprocess finishing
within N seconds, and the real assertions (eigenvalue parity, residuals) run after
the subprocess returns.

That distinction matters, because a tight timeout does not fail safe. It fails as a
*phantom* failure that looks like a numerical disagreement, on any machine that
happens to be busy -- a contributor's laptop with a browser open, CI on a shared
runner, or a workstation with another job on it. The failure mode is a new user
concluding the package is broken when it is merely sharing a CPU.

So the bases here are deliberately generous. Slower detection of a genuine hang is
a much cheaper failure than a false accusation of numerical error.

Known limitation
----------------
The scaling reads ``os.getloadavg()``, which counts *runnable processes*. It does
not see memory-bandwidth starvation. A sparse direct factorization on 10 ranks of a
20-core node registers as load ~10 -- an oversubscription of 0.5, i.e. no scaling at
all -- while consuming far more than half the machine's effective throughput.

Measured instance (2026-07-25): a 10-rank factorization job on a 20-core node slowed
a co-resident 2-CPU test run by roughly 3x, while this heuristic computed a
multiplier of only 1.17x. Four tests timed out spuriously as a result.

The bases below absorb that gap. If a co-resident memory-bound job is present, do
not rely on the multiplier -- run the suite when the node is free, and treat any
timing-derived number from a shared node as contaminated.
"""

from __future__ import annotations

import os

#: Upper bound on the load multiplier, so a runaway load average cannot make the
#: suite hang effectively forever.
LOAD_MULTIPLIER_CAP = 8.0


def load_aware_timeout(base_seconds: float, *, cap: float = LOAD_MULTIPLIER_CAP) -> float:
    """Scale ``base_seconds`` up by CPU oversubscription, bounded by ``cap``.

    Returns ``base_seconds`` unchanged on an idle machine, or on any platform
    without load-average support (notably Windows).
    """
    try:
        loadavg1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        return float(base_seconds)

    ncpu = os.cpu_count() or 1
    oversubscription = loadavg1 / ncpu
    multiplier = min(max(oversubscription, 1.0), cap)
    return float(base_seconds * multiplier)
