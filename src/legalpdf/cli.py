from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .core import add_pdf_geometry, improve, lookup_artifact_footnote, parse_pdf
from .docx_linking import apply_docx_links, plan_docx_links
from .model import load_artifacts, write_artifacts
from .ocr import TesseractOCRProvider


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legalpdf",
        description="Parse legal PDFs into stable local structural artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse", help="Parse a PDF")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--mode", choices=("local", "codex"), default="local")
    parse.add_argument("--cache-dir", type=Path)
    parse.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Bypass the deterministic full-document artifact cache",
    )
    parse.add_argument("--model")
    parse.add_argument("--effort")
    parse.add_argument("--ocr-provider", choices=("tesseract",))
    parse.add_argument("--ocr-language", default="eng")
    parse.add_argument("--ocr-dpi", type=int, default=180)
    parse.add_argument("--ocr-psm", type=int, default=3)
    parse.add_argument("--expected-ocr-identity")
    parse.add_argument(
        "--compact-pages",
        action="store_true",
        help="Publish only the stable page and line fields used by compact consumers",
    )

    geometry = commands.add_parser(
        "add-geometry",
        help="Add geometry pages to a matching compact parse",
    )
    geometry.add_argument("pdf", type=Path)
    geometry.add_argument("--document", type=Path, required=True)
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--ocr-provider", choices=("tesseract",))
    geometry.add_argument("--ocr-language", default="eng")
    geometry.add_argument("--ocr-dpi", type=int, default=180)
    geometry.add_argument("--ocr-psm", type=int, default=3)
    geometry.add_argument("--expected-ocr-identity")

    page_count = commands.add_parser(
        "page-count", help="Report the PDF's physical page count"
    )
    page_count.add_argument("pdf", type=Path)

    ocr_identity = commands.add_parser(
        "ocr-identity", help="Report the detected OCR executable identity"
    )
    ocr_identity.add_argument("--provider", choices=("tesseract",), required=True)
    ocr_identity.add_argument("--ocr-language", default="eng")
    ocr_identity.add_argument("--ocr-dpi", type=int, default=180)
    ocr_identity.add_argument("--ocr-psm", type=int, default=3)

    commands.add_parser(
        "repair-identity",
        help="Report the bounded Codex structural-repair contract",
    )

    improve_command = commands.add_parser(
        "improve", help="Apply selective Codex structural repair"
    )
    improve_command.add_argument("pdf", type=Path)
    improve_command.add_argument("--document", type=Path, required=True)
    improve_command.add_argument("--output", type=Path, required=True)
    improve_command.add_argument("--cache-dir", type=Path)
    improve_command.add_argument("--model", required=True)
    improve_command.add_argument("--effort", required=True)

    footnote = commands.add_parser(
        "footnote", help="Look up one persisted footnote"
    )
    footnote.add_argument("document", type=Path)
    footnote.add_argument("label_or_pair_id")
    footnote.add_argument("--page", type=int)
    footnote.add_argument("--occurrence", type=int)
    footnote.add_argument(
        "--proposition",
        choices=("sentence", "passage_since_prior_note"),
        default="sentence",
    )

    link_plan = commands.add_parser(
        "docx-link-plan",
        help="Build bounded citation intents for DOCX footnotes",
    )
    link_plan.add_argument("docx", type=Path)
    link_plan.add_argument("--output", type=Path, required=True)
    link_plan.add_argument(
        "--strategy", choices=("auto", "direct", "hybrid"), default="auto"
    )
    link_plan.add_argument("--model", default="gpt-5.6-sol")
    link_plan.add_argument("--effort", default="none")
    link_plan.add_argument("--cache-dir", type=Path)
    link_plan.add_argument("--timeout-seconds", type=int, default=600)

    apply_links = commands.add_parser(
        "docx-apply-links",
        help="Apply provider-verified URLs to a DOCX link plan",
    )
    apply_links.add_argument("docx", type=Path)
    apply_links.add_argument("--plan", type=Path, required=True)
    apply_links.add_argument("--links", type=Path, required=True)
    apply_links.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    arguments = _parser().parse_args(argv)
    if arguments.command == "parse":
        ocr_provider = (
            TesseractOCRProvider(
                language=arguments.ocr_language,
                dpi=arguments.ocr_dpi,
                psm=arguments.ocr_psm,
            )
            if arguments.ocr_provider == "tesseract"
            else None
        )
        if (
            arguments.expected_ocr_identity
            and (
                ocr_provider is None
                or ocr_provider.identity != arguments.expected_ocr_identity
            )
        ):
            raise RuntimeError("Tesseract identity changed before OCR began.")
        document = parse_pdf(
            arguments.pdf,
            mode=arguments.mode,
            cache_dir=arguments.cache_dir,
            model=arguments.model,
            effort=arguments.effort,
            ocr_provider=ocr_provider,
            use_cache=arguments.use_cache,
        )
        manifest = write_artifacts(
            document,
            arguments.output,
            compact_pages=arguments.compact_pages,
        )
        print(
            json.dumps(
                {
                    "document": str(manifest),
                    "status": document.status,
                    "pages": document.page_count,
                    "footnotes": len(document.footnotes),
                    "cache_hit": document.provenance.get("cache_hit", False),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.command == "add-geometry":
        ocr_provider = (
            TesseractOCRProvider(
                language=arguments.ocr_language,
                dpi=arguments.ocr_dpi,
                psm=arguments.ocr_psm,
            )
            if arguments.ocr_provider == "tesseract"
            else None
        )
        if (
            arguments.expected_ocr_identity
            and (
                ocr_provider is None
                or ocr_provider.identity != arguments.expected_ocr_identity
            )
        ):
            raise RuntimeError("Tesseract identity changed before OCR began.")
        manifest = add_pdf_geometry(
            arguments.pdf,
            document=arguments.document,
            output=arguments.output,
            ocr_provider=ocr_provider,
        )
        print(json.dumps({"geometry": str(manifest)}, ensure_ascii=False))
        return 0
    if arguments.command == "page-count":
        import fitz

        with fitz.open(arguments.pdf) as document:
            print(json.dumps({"pages": len(document)}))
        return 0
    if arguments.command == "ocr-identity":
        provider = TesseractOCRProvider(
            language=arguments.ocr_language,
            dpi=arguments.ocr_dpi,
            psm=arguments.ocr_psm,
        )
        print(
            json.dumps(
                {"provider": arguments.provider, "identity": provider.identity},
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.command == "repair-identity":
        from .codex_repair import repair_identity

        print(json.dumps(repair_identity(), ensure_ascii=False))
        return 0
    if arguments.command == "improve":
        source_document = load_artifacts(arguments.document)
        document = improve(
            source_document,
            arguments.pdf,
            model=arguments.model,
            effort=arguments.effort,
            cache_dir=arguments.cache_dir,
        )
        manifest = write_artifacts(document, arguments.output)
        print(
            json.dumps(
                {
                    "document": str(manifest),
                    "status": document.status,
                    "repairs": [asdict(repair) for repair in document.repairs],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.command == "footnote":
        result = lookup_artifact_footnote(
            arguments.document,
            arguments.label_or_pair_id,
            page=arguments.page,
            occurrence=arguments.occurrence,
            proposition_mode=arguments.proposition,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "found" else 2
    if arguments.command == "docx-link-plan":
        plan = plan_docx_links(
            arguments.docx,
            strategy=arguments.strategy,
            model=arguments.model,
            effort=arguments.effort,
            cache_dir=arguments.cache_dir,
            timeout_seconds=arguments.timeout_seconds,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(arguments.output.resolve())
        return 0
    links_payload = json.loads(arguments.links.read_text(encoding="utf-8"))
    links = links_payload.get("links", links_payload)
    if not isinstance(links, dict):
        raise ValueError("links JSON must be an object or contain a links object")
    result = apply_docx_links(
        arguments.docx,
        arguments.plan,
        links,
        arguments.output,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
