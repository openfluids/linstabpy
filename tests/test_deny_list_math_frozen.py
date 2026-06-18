# SPDX-License-Identifier: GPL-3.0-only
"""DENY-LIST tripwire for the four frozen validated-math files (Invariant A).

``coefficients.py`` (LNSE coefficient algebra -- the file that carried the
Front-E A0_55 bug), ``mesh.py`` (SBP / curvilinear metric math), ``sbp.py``
(the SBP generator), and ``energy_norm.py`` (the Chu energy weight W) are frozen
during the clean-code refactor: the refactor may move their IMPORT SEAMS but must
not rewrite their math. This guard records an AST-structural signature of each
file's executable math body -- excluding comments, docstrings, and import
statements -- and FAILS if any math-body statement changes.

It is a REGRESSION GUARD, not a correctness gate. The seven Appendix-A oracles are
the correctness gate. A trip here means "a frozen-math line changed -- prove
<1e-10 oracle parity (not just unit-green) or revert", NEVER "re-baseline this
hash to make the test pass" (the Front-E lesson: snapshotting a change makes it
the false ground truth).

Allowed without tripping: comment edits, docstring edits, import-path/seam moves,
whitespace/formatting. Trips on: any change to an executable statement in the math
body. The AST approach drops comments inherently and we strip docstrings + imports
explicitly, so signatures are invariant to formatting and seam relocation.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "linstabpy"
_FROZEN_FILES = {
    "coefficients": _SRC / "operators" / "coefficients.py",
    "energy_norm": _SRC / "operators" / "energy_norm.py",
    "mesh": _SRC / "mesh" / "mesh.py",
    "sbp": _SRC / "mesh" / "sbp.py",
    # sbp.py delegates the actual stencil/norm generation to sbp_generator.py
    # (e.g. generate_sbp_stencils, _sbp_norm_weights); that file IS the SBP
    # generator the deny-list means to freeze, so it must be guarded too,
    # otherwise the math could change in sbp_generator.py without tripping.
    "sbp_generator": _SRC / "mesh" / "sbp_generator.py",
}

# Known math that the frozen files delegate to: any NEW frozen file added below
# must also be covered here. Math reachable outside this set (a future extraction
# into an unguarded module) is a coverage gap the guard cannot see -- a reviewer
# must keep delegated-math files in _FROZEN_FILES.


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


class _MathBodyNormalizer(ast.NodeTransformer):
    """Drop imports (at every nesting level) and docstrings; keep executable math.

    Import statements are removed wherever they appear -- module/function/class
    bodies AND inside ``try``/``if``/``else``/``with``/``for`` blocks -- so a seam
    relocation (an import path changing because a module moved) never trips the
    guard, including the ``try/except ImportError`` SciPy import in mesh.py and the
    ``else``-branch ``sbp_generator`` import in sbp.py. Docstrings (the leading
    string statement at module/function/class scope) are stripped so the release
    commentary/docstring beads can improve documentation without tripping. Comments
    never reach the AST, so they are excluded automatically.

    Not covered (documented limitations): math DELEGATED to a module outside
    _FROZEN_FILES can change without tripping (keep delegated-math files in the set
    -- sbp_generator.py is included for exactly this reason); and renaming a local
    binding or changing a type annotation inside a frozen file trips, since those
    are part of the executable AST.
    """

    def visit_Import(self, node: ast.Import):  # noqa: N802 - ast visitor name
        return None  # remove import anywhere (seam moves must not trip)

    def visit_ImportFrom(self, node: ast.ImportFrom):  # noqa: N802
        return None

    def _scope(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)  # recurse first: removes imports at all depths
        node.body = _strip_leading_docstring(node.body)  # type: ignore[attr-defined]
        return node

    visit_Module = _scope
    visit_FunctionDef = _scope
    visit_AsyncFunctionDef = _scope
    visit_ClassDef = _scope


def math_signature(path: Path) -> str:
    """SHA-256 of the AST-normalized math body (no comments/docstrings/imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = _MathBodyNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, include_attributes=False)  # excludes lineno/col -> format-invariant
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


# Frozen math-body signatures recorded at the revamp start (linstabpy ccdbeeb).
# A trip means a frozen-math body changed. Prove <1e-10 oracle parity and pass the
# seven Appendix-A oracles, then update the signature WITH that evidence in the
# commit message -- do NOT edit these to silence a failure.
_FROZEN_SIGNATURES = {
    "coefficients": "d1c819cd693ff221c5e36298f3b6076fe61691271d8cb75b92d18e4c54271722",
    "energy_norm": "cdbba7b6ff26b91b5647a6753b2efdd0837e8d2918465c1bc156318c6358a4bf",
    # mesh re-baselined for bead r3uu.3.2: Mesh.create gained an
    # argument-validation preamble (two new module-level validators + one call;
    # zero math statements changed). Evidence: all 13 oracle/harness files green
    # (154 passed) at the re-baseline commit; user-approved 2026-06-12.
    "mesh": "8ab8227af5b90ebe7981caaa553ee3009c140e6cde4dd59c81b1e9236a2c9185",
    "sbp": "1ff8cfe68c6a3e464c2b76a7ebe5c77175a06f2868020d0a06f5b60d2036efc1",
    "sbp_generator": "42827492ac4806916ee246aa7046c706fc312c8521187c74fb280f73acd45beb",
}


@pytest.mark.parametrize("name", sorted(_FROZEN_FILES))
def test_deny_list_math_body_frozen(name: str) -> None:
    path = _FROZEN_FILES[name]
    assert path.exists(), f"frozen-math file missing: {path}"
    current = math_signature(path)
    assert current == _FROZEN_SIGNATURES[name], (
        f"DENY-LIST tripwire: the math body of {path.name} changed.\n"
        f"  expected {_FROZEN_SIGNATURES[name]}\n"
        f"  got      {current}\n"
        "These four files are frozen (Invariant A). If this is a real math edit it "
        "MUST show <1e-10 parity against the seven independent Appendix-A oracles "
        "(not merely unit-green); only then update the frozen signature WITH that "
        "evidence in the commit. If it is an import-seam / comment / docstring move "
        "the signature should not have changed -- investigate the guard, do not "
        "re-baseline. NEVER edit these hashes just to make the test pass."
    )
