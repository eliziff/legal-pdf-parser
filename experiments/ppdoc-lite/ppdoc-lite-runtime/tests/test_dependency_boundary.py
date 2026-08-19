from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_wheel_has_no_training_or_compiler_imports() -> None:
    root = Path(__file__).parents[1] / "src" / "ppdoc_lite"
    forbidden = {
        "onnx",
        "paddle",
        "paddlex",
        "ppdet",
        "pycocotools",
        "torch",
        "torchvision",
    }
    found: dict[str, set[str]] = {}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        bad = imports & forbidden
        if bad:
            found[path.name] = bad
    assert not found


def test_build_and_benchmark_tools_are_not_in_runtime_package() -> None:
    package = Path(__file__).parents[1] / "src" / "ppdoc_lite"
    assert not (package / "compiler.py").exists()
    assert not (package / "benchmark.py").exists()
