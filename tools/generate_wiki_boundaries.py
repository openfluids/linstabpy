#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Inline generated contract tables into wiki/boundaries.md.

Sources of truth (parsed with stdlib ``ast`` so this script needs no
dependencies and never imports the package):

- ``CLI_ENV_VARS`` in ``src/linstabpy/driver/cli.py``
- ``TOP_LEVEL_ALLOWED`` / ``_ALLOWED_FD_ORDERS`` in ``src/linstabpy/config/schema.py``
- ``_LAZY_EXPORTS`` in ``src/linstabpy/__init__.py``

Each generated region in ``wiki/boundaries.md`` sits between
``<!-- GENERATED:<KEY>-START -->`` and ``<!-- GENERATED:<KEY>-END -->``
markers. Rewrites only those regions; everything else is left untouched.

Usage:
    python3 tools/generate_wiki_boundaries.py          # rewrite in place
    python3 tools/generate_wiki_boundaries.py --check  # CI: fail if stale
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOUNDARIES = REPO / "wiki" / "boundaries.md"


def _eval(node: ast.AST):
    """Evaluate simple literal AST (constants, tuples, lists, sets, dicts, frozenset)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Set):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
    ):
        return _eval(node.args[0])
    raise ValueError(f"unsupported AST node: {ast.dump(node)[:120]}")


def _load_assign(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return _eval(node.value)
    raise SystemExit(f"{name} not found in {path}")


def _cli_env() -> str:
    cli = REPO / "src" / "linstabpy" / "driver" / "cli.py"
    entries = _load_assign(cli, "CLI_ENV_VARS")
    lines = []
    for name, aliases, default, purpose in entries:
        alias_text = ""
        if aliases:
            quoted = ", ".join(f"`{a}`" for a in aliases)
            alias_text = f" Aliases {quoted}."
        lines.append(f"- `{name}`: {purpose}{alias_text} Default `{default}`.")
    return "\n".join(lines) + "\n"


def _yaml_keys() -> str:
    schema = REPO / "src" / "linstabpy" / "config" / "schema.py"
    allowed = sorted(_load_assign(schema, "TOP_LEVEL_ALLOWED"))
    fd_orders = sorted(_load_assign(schema, "_ALLOWED_FD_ORDERS"))
    keys = ", ".join(f"`{key}`" for key in allowed)
    orders = ", ".join(str(o) for o in fd_orders)
    return (
        f"**Top-level keys** (`TOP_LEVEL_ALLOWED`, generated \u2014 {len(allowed)} keys):\n"
        f"\n{keys}\n"
        f"\nSBP `fd_order` values allowed: {orders}.\n"
    )


def _python_api() -> str:
    init = REPO / "src" / "linstabpy" / "__init__.py"
    exports = _load_assign(init, "_LAZY_EXPORTS")
    lines = [
        f"**Public names** (`_LAZY_EXPORTS`, generated \u2014 {len(exports)} names):",
        "",
        "| Public name | Module | Attribute |",
        "|---|---|---|",
    ]
    for name in sorted(exports):
        module, attr = exports[name]
        lines.append(f"| `{name}` | `{module}` | `{attr}` |")
    return "\n".join(lines) + "\n"


def _replace_block(text: str, key: str, content: str) -> str:
    start = f"<!-- GENERATED:{key}-START -->\n"
    end = f"<!-- GENERATED:{key}-END -->"
    before, sep, rest = text.partition(start)
    if not sep:
        raise SystemExit(f"missing {key} START marker in {BOUNDARIES}")
    _, sep2, after = rest.partition(end)
    if not sep2:
        raise SystemExit(f"missing {key} END marker in {BOUNDARIES}")
    return before + start + content.rstrip("\n") + "\n" + end + after


def build() -> str:
    text = BOUNDARIES.read_text(encoding="utf-8")
    text = _replace_block(text, "CLI-ENV", _cli_env())
    text = _replace_block(text, "YAML-KEYS", _yaml_keys())
    text = _replace_block(text, "PYTHON-API", _python_api())
    return text


def main(argv: list[str]) -> int:
    check = "--check" in argv
    current = BOUNDARIES.read_text(encoding="utf-8")
    fresh = build()
    if check:
        if current != fresh:
            print(f"{BOUNDARIES} is stale; run tools/generate_wiki_boundaries.py")
            return 1
        print("wiki boundaries are in sync.")
        return 0
    if current != fresh:
        BOUNDARIES.write_text(fresh, encoding="utf-8")
        print(f"wrote {BOUNDARIES}")
    else:
        print(f"{BOUNDARIES} already in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
