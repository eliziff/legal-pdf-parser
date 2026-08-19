#!/usr/bin/env python3
"""Exhaustive, resumable fidelity gate for the public source-PDF contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REQUEST_SCHEMA = "legalpdf.document-request.v1"
RESULT_SCHEMA = "legalpdf.document-result.v1"
RECORD_SCHEMA = "legalpdf.cache-contract-record.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode())
    os.replace(temporary, path)


def normalized_envelope(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result["source"].pop("cache_hit", None)
    return result


def utf16_slice(text: str, start: int, end: int) -> str:
    raw = text.encode("utf-16-le")
    return raw[start * 2 : end * 2].decode("utf-16-le")


def safe_token(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")[:80]


class Gate:
    def __init__(self, binary: Path, manifest: Path, raw: Path, selected: set[str]) -> None:
        self.binary = binary.resolve()
        self.manifest_path = manifest.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.binary_sha256 = file_sha256(self.binary)
        self.run_id = digest(
            {
                "binary": self.binary_sha256,
                "manifest": digest(self.manifest),
                "harness": file_sha256(Path(__file__)),
            }
        )[:20]
        self.raw = raw.resolve() / self.run_id
        self.records = self.raw / "records"
        self.selected = selected
        self.calls = 0
        self.skipped = 0
        self.lookup_digest_rows: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []
        self.started = time.monotonic()

    def request(self, operation: str, pdf: Path, cache: Path | None = None, **fields: Any) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": REQUEST_SCHEMA,
            "operation": operation,
            "source_pdf": str(pdf.resolve()),
        }
        if cache is not None:
            value["cache_dir"] = str(cache.resolve())
        value.update(fields)
        return value

    def call(self, label: str, request: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        key = digest({"label": label, "request": request})
        record_path = self.records / f"{key}.json"
        if not force and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("schema_version") == RECORD_SCHEMA and record.get("binary_sha256") == self.binary_sha256:
                self.skipped += 1
                return record["response"]
        request_path = self.raw / "requests" / f"{key}.json"
        atomic_json(request_path, request)
        started = time.monotonic()
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        completed = subprocess.run(
            [str(self.binary), "contract", str(request_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            creationflags=flags,
        )
        if completed.returncode:
            raise AssertionError(f"{label}: legalpdf failed ({completed.returncode}): {completed.stderr.strip()}")
        response = json.loads(completed.stdout)
        assert response["schema_version"] == RESULT_SCHEMA, (label, response.get("schema_version"))
        assert response["operation"] == request["operation"], (label, response.get("operation"))
        atomic_json(
            record_path,
            {
                "schema_version": RECORD_SCHEMA,
                "binary_sha256": self.binary_sha256,
                "label": label,
                "request": request,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "response": response,
            },
        )
        self.calls += 1
        return response

    def assert_source(self, label: str, response: dict[str, Any], expected_sha: str, pages: int) -> None:
        source = response["source"]
        assert source["sha256"] == expected_sha, (label, source["sha256"], expected_sha)
        assert source["page_count"] == pages, (label, source["page_count"], pages)
        assert isinstance(source["parser_version"], str) and source["parser_version"], label

    def corrupt_cache(self, token: str, pdf: Path, expected_sha: str, pages: int) -> dict[str, Any]:
        output = self.raw / "corruption" / f"{token}.json"
        if output.is_file():
            return json.loads(output.read_text(encoding="utf-8"))
        cache = self.raw / "cache-corrupt" / token
        if cache.is_dir():
            shutil.rmtree(cache)
        request = self.request("source_doc", pdf, cache)
        cold = self.call(f"{token}:corrupt:cold", request, force=True)
        self.assert_source("corrupt cold", cold, expected_sha, pages)
        assert cold["source"]["cache_hit"] is False
        key = cold["source"]["cache_key"]
        document_cache = cache / "parse-v1" / "documents" / f"{key}.json.gz"
        extraction_caches = sorted((cache / "parse-v1" / "extractions").glob("*.json.gz"))
        assert document_cache.is_file() and extraction_caches, token
        document_cache.write_bytes(b"corrupt document cache")
        document_rebuilt = self.call(f"{token}:corrupt:document", request, force=True)
        assert document_rebuilt["source"]["cache_hit"] is False
        assert normalized_envelope(document_rebuilt) == normalized_envelope(cold)
        document_cache.write_bytes(b"corrupt document cache again")
        for path in extraction_caches:
            path.write_bytes(b"corrupt extraction cache")
        both_rebuilt = self.call(f"{token}:corrupt:both", request, force=True)
        assert both_rebuilt["source"]["cache_hit"] is False
        assert normalized_envelope(both_rebuilt) == normalized_envelope(cold)
        warm = self.call(f"{token}:corrupt:warm", request, force=True)
        assert warm["source"]["cache_hit"] is True
        assert normalized_envelope(warm) == normalized_envelope(cold)
        result = {
            "cache_key": key,
            "document_rebuild_identical": True,
            "document_and_extraction_rebuild_identical": True,
            "warm_identical": True,
        }
        atomic_json(output, result)
        return result

    def source_separation(self, first: dict[str, Any], pdf: Path) -> dict[str, Any]:
        output = self.raw / "source-separation.json"
        if output.is_file():
            return json.loads(output.read_text(encoding="utf-8"))
        mutation = self.raw / "source-separation" / f"{pdf.stem}-trailing-comment.pdf"
        mutation.parent.mkdir(parents=True, exist_ok=True)
        mutation.write_bytes(pdf.read_bytes() + b"\n% cache identity probe\n")
        cache = self.raw / "cache-source-separation"
        fixed = {"id": "cache-contract-source-separation"}
        original = self.call("source-separation:original", self.request("source_doc", pdf, cache / "original", **fixed))
        changed = self.call("source-separation:changed", self.request("source_doc", mutation, cache / "changed", **fixed))
        assert original["source"]["sha256"] == first["sha256"]
        assert changed["source"]["sha256"] != original["source"]["sha256"]
        assert changed["source"]["cache_key"] != original["source"]["cache_key"]
        assert original["result"] == changed["result"]
        result = {
            "original_sha256": original["source"]["sha256"],
            "changed_sha256": changed["source"]["sha256"],
            "original_cache_key": original["source"]["cache_key"],
            "changed_cache_key": changed["source"]["cache_key"],
            "source_doc_identical": True,
        }
        atomic_json(output, result)
        return result

    def structural_queries(
        self,
        token: str,
        pdf: Path,
        cache: Path,
        envelope: dict[str, Any],
    ) -> dict[str, int]:
        source_doc = envelope["result"]["source_doc"]
        text = source_doc["text"]
        blocks = source_doc["blocks"]
        counts = {"paragraph": 0, "section": 0, "footnote": 0, "queries": 0, "bounded_aliases": 0}

        def lookup(kind: str, locator: str, suffix: str, **extra: Any) -> dict[str, Any]:
            query = {"locator_kind": kind, "locator": locator, **extra}
            response = self.call(
                f"{token}:structure:{kind}:{suffix}",
                self.request("structure_lookup", pdf, cache, query=query),
            )
            assert response["source"]["cache_hit"] is True
            assert response["source"]["cache_key"] == envelope["source"]["cache_key"]
            counts["queries"] += 1
            self.lookup_digest_rows.append({"document": token, "query": query, "result": response["result"]})
            return response["result"]

        for index, block in enumerate(item for item in blocks if item["kind"] == "paragraph"):
            result = lookup("paragraph", block["label"], f"paragraph-{index + 1}")
            assert result["status"] == "found" and result["exact"] is True, (token, block, result)
            assert len(result["units"]) == 1
            unit = result["units"][0]
            assert unit["id"] == block["anchor"]
            assert unit["text"] == utf16_slice(text, block["start"], block["end"])
            assert unit["page_numbers"] == sorted(set(unit["page_numbers"]))
            counts["paragraph"] += 1

        seen_section_queries: set[tuple[str, str]] = set()
        for index, block in enumerate(item for item in blocks if item["kind"] == "section"):
            candidates = [block["label"], *block.get("aliases", [])]
            candidates = [value for value in candidates if value]
            assert candidates, (token, "section has no locator", block)
            resolved = False
            for candidate in dict.fromkeys(candidates):
                pair = (candidate, block["anchor"])
                if pair in seen_section_queries:
                    continue
                seen_section_queries.add(pair)
                result = lookup("section", candidate, f"section-{index + 1}-{digest(candidate)[:10]}")
                if len(candidate.encode("utf-16-le")) // 2 > 200:
                    assert result["status"] == "invalid", (token, block, candidate, result)
                    counts["bounded_aliases"] += 1
                    continue
                assert result["status"] in {"found", "ambiguous"}, (token, block, candidate, result)
                assert block["anchor"] in result["matches"], (token, block, candidate, result)
                resolved = True
                if result["status"] == "found":
                    assert result["exact"] is True and result["units"][0]["id"] == block["anchor"]
                counts["queries"] += 0
            assert resolved, (token, "section has no resolvable locator", block)
            counts["section"] += 1

        for index, block in enumerate(item for item in blocks if item["kind"] == "footnote"):
            anchor = block.get("anchor")
            assert anchor, (token, "footnote has no anchor", block)
            by_anchor = lookup("footnote", anchor, f"footnote-{index + 1}-anchor")
            assert by_anchor["status"] == "found" and len(by_anchor["units"]) == 1, (token, block, by_anchor)
            unit = by_anchor["units"][0]
            assert unit["id"] == anchor
            assert unit["text"] == utf16_slice(text, block["start"], block["end"])
            occurrence = unit["note"]["occurrence"]
            locators = [block["label"], *block.get("aliases", [])]
            for candidate in dict.fromkeys(value for value in locators if value):
                result = lookup(
                    "footnote",
                    candidate,
                    f"footnote-{index + 1}-{occurrence}-{digest(candidate)[:10]}",
                    occurrence=occurrence,
                )
                assert result["status"] == "found" and result["units"] == by_anchor["units"], (
                    token,
                    block,
                    candidate,
                    occurrence,
                    result,
                )
            counts["footnote"] += 1
        return counts

    def document(self, row: dict[str, Any], position: int, total: int) -> dict[str, Any]:
        relative = row["path"]
        if self.selected and relative not in self.selected and Path(relative).name not in self.selected:
            return {}
        pdf = ROOT / relative
        token = f"{position:02d}-{safe_token(Path(relative).stem)}"
        expected_sha = row["sha256"]
        pages = row["pages"]
        assert pdf.is_file(), pdf
        assert file_sha256(pdf) == expected_sha, (relative, file_sha256(pdf), expected_sha)
        print(f"DOCUMENT {position}/{total} {relative} ({pages} pages)", flush=True)

        inspect = self.call(f"{token}:inspect", self.request("inspect", pdf))
        self.assert_source("inspect", inspect, expected_sha, pages)
        assert inspect["source"]["cache_key"] is None and inspect["source"]["cache_hit"] is False

        direct_cache = self.raw / "cache-direct" / token
        direct_request = self.request("source_doc", pdf, direct_cache)
        direct_cold = self.call(f"{token}:source-doc:cold", direct_request)
        direct_warm = self.call(f"{token}:source-doc:warm", direct_request)
        self.assert_source("direct cold", direct_cold, expected_sha, pages)
        assert direct_cold["source"]["cache_hit"] is False
        assert direct_warm["source"]["cache_hit"] is True
        assert normalized_envelope(direct_cold) == normalized_envelope(direct_warm)

        prepared_cache = self.raw / "cache-prepared" / token
        prepare = self.call(f"{token}:prepare", self.request("prepare", pdf, prepared_cache))
        prepared = self.call(f"{token}:prepared-source-doc", self.request("source_doc", pdf, prepared_cache))
        assert prepare["source"]["cache_hit"] is False
        assert prepared["source"]["cache_hit"] is True
        assert prepare["source"]["cache_key"] == prepared["source"]["cache_key"]
        assert normalized_envelope(direct_cold) == normalized_envelope(prepared)

        targeted = 0
        profile_keys: dict[tuple[int, ...], str] = {}
        for page in range(1, pages + 1):
            for context in range(3):
                selected = tuple(range(max(1, page - context), min(pages, page + context) + 1))
                query = {"locator_kind": "page", "locator": str(page), "context_blocks": context}
                full = self.call(
                    f"{token}:page:{page}:context:{context}:full",
                    self.request("structure_lookup", pdf, prepared_cache, query=query),
                )
                target_cache = self.raw / "cache-targeted" / token / f"page-{page}-context-{context}"
                target_request = self.request(
                    "structure_lookup", pdf, target_cache, pages=list(selected), query=query
                )
                cold = self.call(f"{token}:page:{page}:context:{context}:target-cold", target_request)
                warm = self.call(f"{token}:page:{page}:context:{context}:target-warm", target_request)
                assert cold["source"]["cache_hit"] is False, (token, page, context)
                assert warm["source"]["cache_hit"] is True, (token, page, context)
                assert normalized_envelope(cold) == normalized_envelope(warm)
                assert cold["result"] == full["result"], (token, page, context)
                assert cold["source"]["sha256"] == expected_sha
                assert cold["source"]["cache_key"] != prepared["source"]["cache_key"]
                prior = profile_keys.setdefault(selected, cold["source"]["cache_key"])
                assert prior == cold["source"]["cache_key"], (token, selected, prior, cold["source"]["cache_key"])
                self.lookup_digest_rows.append({"document": token, "query": query, "result": full["result"]})
                targeted += 1
            if page % 10 == 0 or page == pages:
                elapsed = time.monotonic() - self.started
                print(
                    f"  pages {page}/{pages}; target comparisons={targeted}; calls={self.calls}; "
                    f"resumed={self.skipped}; elapsed={elapsed:.1f}s",
                    flush=True,
                )

        structure = self.structural_queries(token, pdf, prepared_cache, prepared)
        corruption = self.corrupt_cache(token, pdf, expected_sha, pages)
        result = {
            "path": relative,
            "sha256": expected_sha,
            "pages": pages,
            "parser_version": prepared["source"]["parser_version"],
            "full_cache_key": prepared["source"]["cache_key"],
            "target_profile_count": len(profile_keys),
            "targeted_page_context_comparisons": targeted,
            "source_doc_sha256": digest(prepared["result"]),
            "paragraph_blocks": structure["paragraph"],
            "section_blocks": structure["section"],
            "footnote_blocks": structure["footnote"],
            "structure_queries": structure["queries"],
            "bounded_section_aliases": structure["bounded_aliases"],
            "corruption": corruption,
        }
        atomic_json(self.raw / "documents" / f"{token}.json", result)
        print(
            f"PASS {relative}: target={targeted}, paragraphs={structure['paragraph']}, "
            f"sections={structure['section']}, footnotes={structure['footnote']}, "
            f"structure queries={structure['queries']}",
            flush=True,
        )
        return result

    def run(self) -> dict[str, Any]:
        assert self.manifest["schema_version"] == "legalpdf.cache-contract-corpus.v1"
        documents = self.manifest["documents"]
        self.raw.mkdir(parents=True, exist_ok=True)
        selected_rows = [
            row
            for row in documents
            if not self.selected or row["path"] in self.selected or Path(row["path"]).name in self.selected
        ]
        assert selected_rows, "no documents selected"
        for position, row in enumerate(documents, 1):
            report = self.document(row, position, len(documents))
            if report:
                self.reports.append(report)
        full_run = len(selected_rows) == len(documents)
        source_keys = {(row["sha256"], row["full_cache_key"]) for row in self.reports}
        assert len({item[0] for item in source_keys}) == len(source_keys)
        assert len({item[1] for item in source_keys}) == len(source_keys)
        separation = self.source_separation(self.reports[0], ROOT / self.reports[0]["path"])
        report = {
            "schema_version": "legalpdf.cache-contract-fidelity.v1",
            "passed": True,
            "full_corpus": full_run,
            "binary": str(self.binary),
            "binary_sha256": self.binary_sha256,
            "run_id": self.run_id,
            "documents": len(self.reports),
            "pages": sum(row["pages"] for row in self.reports),
            "targeted_page_context_comparisons": sum(
                row["targeted_page_context_comparisons"] for row in self.reports
            ),
            "paragraph_blocks": sum(row["paragraph_blocks"] for row in self.reports),
            "section_blocks": sum(row["section_blocks"] for row in self.reports),
            "footnote_blocks": sum(row["footnote_blocks"] for row in self.reports),
            "structure_queries": sum(row["structure_queries"] for row in self.reports),
            "bounded_section_aliases": sum(row["bounded_section_aliases"] for row in self.reports),
            "contract_calls_executed": self.calls,
            "contract_calls_resumed": self.skipped,
            "corrupt_cache_rebuilds": len(self.reports) * 2,
            "corpus_sha256": digest([{"path": row["path"], "sha256": row["sha256"]} for row in self.reports]),
            "source_docs_sha256": digest(
                [{"path": row["path"], "sha256": row["source_doc_sha256"]} for row in self.reports]
            ),
            "lookups_sha256": digest(self.lookup_digest_rows),
            "source_separation": separation,
            "documents_detail": self.reports,
        }
        atomic_json(self.raw / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report


def self_test() -> None:
    sample = "A\U0001f600B"
    assert utf16_slice(sample, 1, 3) == "\U0001f600"
    value = {"source": {"cache_hit": False, "sha256": "x"}, "result": {"x": 1}}
    assert normalized_envelope(value) == {"source": {"sha256": "x"}, "result": {"x": 1}}
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    print("self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "legal-pdf-parser" / "target" / "release" / "legalpdf.exe",
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--raw", type=Path, default=HERE / "raw")
    parser.add_argument("--document", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return 0
    Gate(arguments.binary, arguments.manifest, arguments.raw, set(arguments.document)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
