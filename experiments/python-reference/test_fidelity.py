from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("fidelity.py")
SPEC = importlib.util.spec_from_file_location("legalpdf_fidelity", PATH)
assert SPEC and SPEC.loader
FIDELITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIDELITY)


def test_manifest_is_strict_and_resolves_relative_pdfs(tmp_path: Path) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"pdf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": FIDELITY.MANIFEST_SCHEMA,
                "cases": [{"id": "one", "pdf": "case.pdf", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    assert FIDELITY.load_manifest(manifest)[0]["pdf"] == str(source.resolve())
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["legacy"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        FIDELITY.load_manifest(manifest)


def test_differences_are_bounded_and_descriptive() -> None:
    found = FIDELITY.differences(
        {"items": [{"value": index} for index in range(10)]},
        {"items": [{"value": -index} for index in range(10)]},
        limit=3,
    )
    assert len(found) == 3
    assert found[0].startswith("/items/1/value:")


def test_selected_ignores_non_contract_fields() -> None:
    assert FIDELITY.selected({"source_sha256": "a", "extra": True}, ["source_sha256"]) == {
        "source_sha256": "a"
    }


def test_atomic_json_replaces_complete_value(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    FIDELITY.atomic_json(output, {"one": 1})
    FIDELITY.atomic_json(output, {"two": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"two": 2}
