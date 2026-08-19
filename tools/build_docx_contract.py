from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def docx_fixture(path: Path, footnote: str) -> None:
    document = f'''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>Proposition</w:t></w:r>
<w:r><w:footnoteReference w:id="2"/></w:r></w:p></w:body></w:document>'''
    notes = f'''<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="{W_NS}" xmlns:r="{R_NS}">
<w:footnote w:id="-1" w:type="separator"><w:p/></w:footnote>
<w:footnote w:id="2"><w:p><w:r><w:footnoteRef/></w:r>
<w:r><w:rPr><w:i/></w:rPr><w:t>{footnote}</w:t></w:r></w:p></w:footnote>
</w:footnotes>'''
    content_types = '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>
</Types>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/footnotes.xml", notes)
        archive.writestr("[Content_Types].xml", content_types)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--operation",
        choices=("docx_batch", "docx_extract", "docx_apply_fixture"),
        default="docx_batch",
    )
    arguments = parser.parse_args()
    cases = []
    seen: set[Path] = set()
    with arguments.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source = row.get("docx") or row.get("source_docx")
            if not source:
                continue
            path = Path(source).resolve()
            if path.is_file() and path not in seen:
                cases.append({"docx": str(path)})
                seen.add(path)
            if arguments.limit is not None and len(cases) >= arguments.limit:
                break
    payload = {
        "schema_version": "legalpdf.contract-input.v1",
        "operation": arguments.operation,
    }
    if arguments.operation == "docx_apply_fixture":
        fixture_root = arguments.output.resolve().parent / "docx-apply-fixture"
        source = fixture_root / "source.docx"
        linked = fixture_root / "linked.docx"
        text = (
            "Criminal Code, RSC 1985, c C-46, s 7; "
            "R v Example, 2024 SCC 1"
        )
        docx_fixture(source, text)
        payload.update(
            {
                "operation": "docx_apply",
                "docx": str(source),
                "output": str(linked),
                "links": {
                    "2:1": "https://laws.example.test/code#sec7",
                    "2:2": "https://cases.example.test/example",
                },
            }
        )
    elif arguments.operation == "docx_extract":
        if not cases:
            raise ValueError("manifest contains no available DOCX")
        payload["docx"] = cases[0]["docx"]
    else:
        payload["cases"] = cases
    atomic_json(arguments.output.resolve(), payload)
    print(f"wrote {len(cases)} DOCX cases to {arguments.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
