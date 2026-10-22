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
    # coefficients re-baselined 2026-10-22: the axisymmetric operator now keeps the
    # mu'(T) T' div(S_bar) terms (momentum rows) and the matching mu', k' terms
    # (energy row) that LinStab2D getCoeffsLaminar.m omits. The flag
    # matlab_parity_dmudt=True reproduces LinStab2D bit for bit. Evidence:
    # production == full sympy oracle to <1e-12 (default) and == oracle with the
    # LinStab2D omission (flag on); measured effect: operator ~1e-4/Re, leading
    # eigenvalue 1e-4 at Re 500 to 1e-6 at Re 5e4, resolvent gains up to 1.2e-2 at
    # Re 500 (docs/plans/axy_dmudt/); user-approved 2026-10-22.
    "coefficients": "a187d42a47dd9b53969dae8be081d51ccb1b39344f3cf95094b09832ebc1e1f1",
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
    # mesh and sbp re-baselined 2026-10-22, user-approved ("fix all"). Two real math
    # edits. (a) sbp.min_points is max(m, 2r), not m: below 2r the two boundary
    # closures overlap. Measured SBP defect max|H D1 + (H D1)^T - B| = 0.16 (fd4, n=6),
    # 1.4 (fd6, n=9), 48 (fd8, n=12) against 1e-16 at n = 2r; D1 @ 1 was nonzero at
    # fd4 n=7, fd6 n=10-11, fd8 n=13-15 (up to 28). SummationByPartsOperators.jl uses
    # the same rule (N > 2r). Short masked segments now take the highest order whose
    # closures fit, instead of order 2. (b) _line_breaks compares each gap with its
    # smaller neighbour (split above 1.5x) instead of max(1.5, 3*median): a one-point
    # hole was spanned by the stencil (max|Dx x - 1| = 0.5). Evidence: operator hashes
    # on 44 meshes (Mesh.create fd2-8 at n = 16-31, y_symmetry on/off; masked cavity,
    # block, short-segment cases) are bitwise identical except masked meshes with a
    # one-point hole or a segment shorter than 2r; on those, Dx x = 1, Dy y = 1,
    # D2 x^2 = 2 hold to <3e-12. Full suite green at the re-baseline.
    # mesh re-baselined again 2026-10-22, user-approved: _deform_x_only and
    # _deform_y_only returned a reset metric (J = identity) for the other direction, so
    # chained one-direction deforms lost J (Neumann curvilinear-normal rows read J). They now
    # compose it: J11, J21 scale by dX_new/dX_old in an x-only step, J12, J22 by dY_new/dY_old
    # in a y-only step. No derivative operator or weight statement changed. Evidence: 84
    # meshes x 23 fields (Mesh.create fd2-6, y_symmetry on/off, n = 16, 24, with and without
    # dissipation; x-only, y-only, full-2D, separable-2D deforms; masked cavity; all-True
    # mask) hash identically before and after, bit for bit; chains x-y, y-x, x-x, y-y now
    # match J of the final map computed on the base grid to 1e-10 (test_mesh.py::
    # test_chained_one_direction_deformations_compose_the_metric, failing before the fix).
    # mesh re-baselined again 2026-10-22, user-approved: Mesh.mask rebuilds the SBP
    # dissipation Q_diss_x, Q_diss_y on the used points (new _build_masked_dissipation),
    # one 1-D operator per line run cut by the same rule as the masked derivatives; before,
    # it carried the full-grid (nx, nx) and (ny, ny) matrices and the assembler failed with a
    # shape error on any masked mesh. The line-run split of _build_operators moved into the
    # shared helper _line_segments, statements unchanged. Evidence: the same 84-mesh x 23-field
    # hash set as above is bit-for-bit identical before and after except Q_diss_x and Q_diss_y
    # of the 16 masked and all-True-mask meshes with dissipation (32 hashes); every derivative
    # matrix, weight and J, masked or not, is unchanged. Tests: test_mesh_dissipation.py::
    # TestMaskedMeshDissipation and test_assembly_dissipation.py::
    # test_masked_mesh_assembles_with_dissipation (failing before the fix).
    "mesh": "9ba71e1b1d22be51ecc795cc6f637d1f43578480fb8c295020760199717a4b1b",
    # sbp re-hashed again 2026-10-22: build_sbp_pole_matrices and
    # build_sbp_dissipation_matrices refuse n < min_points like build_sbp_matrix.
    # Guard statements only; no operator value changes (full suite green).
    "sbp": "777a896cd3fb5a214301f96de3a6e264ecd80b5dd72b235d686c77d1706a82a8",
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
