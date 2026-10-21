"""The engine imports only numpy, soundfile and the standard library."""

import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "grblr"
CORE_DEPS = {"numpy", "soundfile", "grblr"}


def _top_level_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)


def test_engine_core_imports_only_numpy_and_soundfile():
    allowed = CORE_DEPS | set(sys.stdlib_module_names)
    bad = {}
    for f in PKG.rglob("*.py"):
        hits = (
            set(_top_level_imports(ast.parse(f.read_text(encoding="utf-8")))) - allowed
        )
        if hits:
            bad[str(f.relative_to(PKG))] = sorted(hits)
    assert bad == {}


def test_engine_core_ships_no_ui():
    assert not (PKG / "ui").exists()
