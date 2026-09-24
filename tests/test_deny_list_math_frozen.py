# SPDX-License-Identifier: GPL-3.0-only
"""Check that the four frozen validated-math files stay unchanged (Invariant A).

``coefficients.py`` is LNSE coefficient algebra (the file that carried the
Front-E A0_55 error).
``mesh.py`` is SBP / curvilinear metric math.
``sbp.py`` is the SBP generator.
``energy_norm.py`` is the Chu energy weight W.
All four stay frozen during the clean-code rewrite.
The rewrite may move their IMPORT SEAMS but must not rewrite their math.
This guard records an AST-structural signature of each file's executable
math body, excluding comments, docstrings, and import statements.
It FAILS if any math-body statement changes.

It is a REGRESSION GUARD, not a correctness gate.
The seven Appendix-A closed-form checks are the correctness gate.
A trip here means "a frozen-math line changed -- prove <1e-10 closed-form
parity (not just unit-green) or revert".
Never re-baseline this hash to make the check pass.
The Front-E lesson: snapshotting a change makes it the false ground truth.

Allowed without tripping: comment edits, docstring edits, import-path/seam
moves, whitespace/formatting.
Trips on: any change to an executable statement in the math body.
The AST approach drops comments inherently.
We strip docstrings and imports explicitly, so signatures stay invariant
to formatting and seam relocation.
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
    # generator the deny-list means to freeze, so the guard must include it too,
    # otherwise the math could change in sbp_generator.py without tripping.
    "sbp_generator": _SRC / "mesh" / "sbp_generator.py",
}

# Known math that the frozen files delegate to: any NEW frozen file added below
# must also appear here. Math reachable outside this set (a future extraction
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

    Drop import statements wherever they appear -- module/function/class
    bodies AND inside ``try``/``if``/``else``/``with``/``for`` blocks.
    Then a seam relocation (an import path changing because a module
    moved) never trips the guard.
    That includes the ``try/except ImportError`` SciPy import in mesh.py
    and the ``else``-branch ``sbp_generator`` import in sbp.py.
    Strip docstrings (the leading string statement at module/function/class
    scope) so documentation edits do not trip.
    Comments never reach the AST, so they drop automatically.

    Not covered (documented limitations): math DELEGATED to a module outside
    _FROZEN_FILES can change without tripping.
    Keep delegated-math files in the set -- sbp_generator.py is in the set
    for exactly this reason.
    Renaming a local binding or changing a type annotation inside a frozen
    file trips.
    Those names are part of the executable AST.
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
# A trip means a frozen-math body changed. Prove <1e-10 closed-form parity and
# pass the seven Appendix-A checks, then update the signature WITH that evidence in the
# commit message -- do NOT edit these to silence a failure.
_FROZEN_SIGNATURES = {
    # coefficients re-baselined 2026-08-12: removed the axis R-floor
    # (R = max(R, min positive radial spacing)) that silently rewrote the 1/R
    # coefficients at real near-axis points, and replaced it with a hard
    # ValueError for grids containing r = 0. Evidence: full 250k-DOF jet
    # (example 2) operator now matches the upstream MATLAB LinStab2D L0
    # ENTRYWISE to max|delta| = 4.4e-12 (4e-16 relative), axis rows included;
    # before the fix the axis rows differed by O(1) and the example-2 gains
    # missed the MATLAB reference.
    "coefficients": "592595e2b4de0821343b9baeb711e4eb9aa23aada8c47dc1dd78e0f3b8024d84",
    "energy_norm": "cdbba7b6ff26b91b5647a6753b2efdd0837e8d2918465c1bc156318c6358a4bf",
    # mesh re-baselined 2026-06-12: Mesh.create gained an
    # argument-validation preamble; zero math statements changed. Evidence: all 13
    # closed-form and timing files green (154 passed); user-approved.
    # mesh re-baselined 2026-08-19 (1.0.0rc6): a real math edit. In
    # _deform_full_2d the dJ/Q metric terms of the symm/asymm family operators
    # use the full Dy/D2y (the coordinate map has no eta
    # parity), as LinStab2D DeformMesh.m does; the family stencils still enter the
    # operator combination unchanged. Evidence: (1) deformed y-symmetric mesh,
    # FD4, two non-separable analytic maps, odd/even test functions with sympy
    # second derivatives: before, max axis-row rel. error of D2y_asymm grew like
    # 1/h (7.7 -> 61 for n = 16 -> 128; Z^2 map 3.5 -> 29) and D2y_symm sat at
    # 0.54; after, 2.2e-4 -> 3.9e-7 (order 3) and 2.1e-3 -> 3.1e-5 (order 2).
    # (2) Only y_symmetry=True meshes on the non-separable full-2D path are
    # touched; the separable and y-only paths and every non-symmetric mesh are
    # byte-identical in behaviour (Martini Ex. 0 curvilinear parity vs the
    # LinStab2D bundle: x, y, f, df at 1e-16..1e-14). (3) Martini Ex. 5
    # paraboloid D2y_asymm vs LinStab2D, measured in G. Hamada's audit: rel.
    # Frobenius 4e-5 -> 2.6e-10, L0 gap 6e-6 -> parity
    # (examples/linstab2d/example_5_D2y_asymm_fix.md).
    # (4) 18 closed-form and timing files green at the re-baseline (367 passed, 4
    # skipped); full suite 1432 passed. Regression guard:
    # test_mesh.py::test_deform_full_2d_family_metric_terms_use_full_operators.
    "mesh": "bfad6c846c80005bcb1a152b3a92ce15978e8e45ee681ea19c2ac883be549ec1",
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
