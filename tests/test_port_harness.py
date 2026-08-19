from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "port_harness.py"
SPEC = importlib.util.spec_from_file_location("port_harness", MODULE_PATH)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def test_read_jsonl_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"case_id":"one"}\n', encoding="utf-8-sig")

    assert HARNESS.read_jsonl(path) == [{"case_id": "one"}]


def test_case_pdf_accepts_frozen_and_lightweight_manifests() -> None:
    assert HARNESS.case_pdf({"pdf": "frozen.pdf"}) == Path("frozen.pdf")
    assert HARNESS.case_pdf({"pdf_path": "external.pdf"}) == Path("external.pdf")


def test_phase_commands_use_hidden_extract_and_replay_contracts(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    rust = tmp_path / "legalpdf.exe"
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"

    oracle_command, oracle_cwd, _ = HARNESS.parity_command(
        "extract", "oracle", source, output, oracle, rust
    )
    rust_command, rust_cwd, _ = HARNESS.parity_command(
        "replay", "rust", source, output, oracle, rust
    )

    assert oracle_command[-4:] == ["extract", str(source), "--output", str(output)]
    assert oracle_cwd == oracle
    assert rust_command == [
        str(rust),
        "_parity-replay",
        str(source),
        "--output",
        str(output),
    ]
    assert rust_cwd == rust.parent


def test_sequence_error_is_true_levenshtein_distance() -> None:
    assert HARNESS.sequence_error("aa", "ba") == 1
    assert HARNESS.sequence_error(["old", "text"], ["new", "text", "here"]) == 2


def test_source_metrics_do_not_match_text_across_page_boundaries() -> None:
    pages = [
        {"lines": [{"text": "second", "reading_order": 1}]},
        {"lines": [{"text": "first", "reading_order": 1}]},
    ]

    metrics = HARNESS.page_aligned_text_metrics({1: "first", 2: "second"}, pages, "source")

    assert metrics["source.cer"] > 0
    assert metrics["source.wer"] == 1.0


def test_gate_rejects_any_per_document_regression() -> None:
    oracle = {
        "contract.invalid_count": 0,
        "body.cer": 0.01,
        "notes.labels.f1": 0.98,
        "document.page_count": 4,
    }
    assert HARNESS.metric_regressions(oracle, dict(oracle)) == []
    failures = HARNESS.metric_regressions(
        oracle,
        {
            "contract.invalid_count": 0,
            "body.cer": 0.02,
            "notes.labels.f1": 0.97,
            "document.page_count": 4,
        },
    )
    assert any("body.cer" in failure for failure in failures)
    assert any("notes.labels.f1" in failure for failure in failures)


def test_journal_reference_keeps_only_complete_pairs(tmp_path: Path) -> None:
    pages = tmp_path / "pages.jsonl"
    rows = [
        {
            "pdf_page": 9,
            "text": "Body 1 Note",
            "annotations": [
                {
                    "taxonomy_name": "fn_label",
                    "pair_id": "one",
                    "pair_status": "paired",
                    "selected_text": "1",
                    "start_line_order": 8,
                    "start_offset": 0,
                },
                {
                    "taxonomy_name": "fn_ref",
                    "pair_id": "one",
                    "pair_status": "paired",
                    "selected_text": "1",
                    "start_line_order": 2,
                    "start_offset": 4,
                },
                {
                    "taxonomy_name": "fn_label",
                    "pair_id": "incomplete",
                    "pair_status": "paired",
                    "selected_text": "2",
                    "start_line_order": 9,
                    "start_offset": 0,
                },
            ],
        }
    ]
    pages.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    reference = HARNESS.journal_reference(pages)
    assert reference["page_count"] == 1
    assert reference["pdf_pages"] == [9]
    assert reference["page_lines"] == {9: []}
    assert [(pair["label"], pair["reference_pages"]) for pair in reference["pairs"]] == [
        ("1", [9])
    ]


def test_journal_sampling_is_deterministic_and_honors_dataset_minimums() -> None:
    rows = [
        {"dataset": dataset, "article_id": article_id}
        for dataset in ("A", "B")
        for article_id in range(10)
    ]
    first = HARNESS._sample_journal_rows(rows, 2, {"B": 4}, "frozen")
    second = HARNESS._sample_journal_rows(list(reversed(rows)), 2, {"B": 4}, "frozen")
    assert first == second
    assert sum(row["dataset"] == "A" for row in first) == 2
    assert sum(row["dataset"] == "B" for row in first) == 4


def test_diverse_journal_sampling_round_robins_metadata_strata() -> None:
    rows = [
        {
            "dataset": "A",
            "article_id": article_id,
            "date": f"{decade + article_id % 2}-01-01",
            "page_count": pages,
            "language": language,
            "doc_type": "article",
            "pdf_type": pdf_type,
        }
        for article_id, (decade, pages, language, pdf_type) in enumerate(
            [
                (1970, 8, "en", "scan"),
                (1990, 18, "fr", "scan"),
                (2010, 35, "en", "digital"),
                (2020, 70, "fr", "digital"),
            ]
            * 2,
            start=1,
        )
    ]

    selected = HARNESS._sample_journal_rows(rows, 4, {}, "held-out", diverse=True)

    assert len(selected) == 4
    assert len({row["page_count"] for row in selected}) == 4


def test_journal_sampling_fills_an_exact_total_after_dataset_quotas() -> None:
    rows = [
        {"dataset": dataset, "article_id": article_id}
        for dataset in ("A", "B")
        for article_id in range(10 * (dataset == "B"), 10 * (dataset == "B") + 4)
    ]

    selected = HARNESS._sample_journal_rows(rows, 1, {}, "held-out", total=3)

    assert len(selected) == 3
    assert {row["dataset"] for row in selected} == {"A", "B"}


def test_journal_sampling_honors_total_without_dataset_quota() -> None:
    rows = [
        {"dataset": dataset, "article_id": article_id}
        for dataset in ("A", "B")
        for article_id in range(4)
    ]

    selected = HARNESS._sample_journal_rows(rows, None, {}, "held-out", total=3)

    assert len(selected) == 3
    assert {row["dataset"] for row in selected} == {"A", "B"}


def test_common_input_case_filter_is_exact_and_order_preserving() -> None:
    cases = [{"case_id": "one"}, {"case_id": "two"}, {"case_id": "three"}]
    assert HARNESS.selected_case_ids(cases, ["three", "one"]) == [cases[0], cases[2]]

    try:
        HARNESS.selected_case_ids(cases, ["missing"])
    except ValueError as error:
        assert str(error) == "unknown case IDs: missing"
    else:
        raise AssertionError("missing case ID was accepted")


def test_journal_metrics_compare_only_reference_pdf_pages() -> None:
    reference = {
        "text": "Article",
        "page_texts": {2: "Article"},
        "pairs": [],
        "pdf_pages": [2],
    }
    document = {
        "pages": [
            {"lines": [{"text": "Cover", "reading_order": 1}]},
            {"lines": [{"text": "Article", "reading_order": 1}]},
        ],
        "footnotes": [],
    }
    assert HARNESS.journal_metrics(reference, document)["source.cer"] == 0.0


def test_page_reading_order_metrics_catch_column_interleaving() -> None:
    expected = {
        1: [
            {"text": "left first"},
            {"text": "left second"},
            {"text": "right first"},
            {"text": "right second"},
        ]
    }
    candidate = [
        {
            "lines": [
                {"text": "left first", "reading_order": 1},
                {"text": "right first", "reading_order": 2},
                {"text": "left second", "reading_order": 3},
                {"text": "right second", "reading_order": 4},
            ]
        }
    ]

    metrics = HARNESS.page_reading_order_metrics(expected, candidate)

    assert metrics["reading_order.anchor_recall"] == 1.0
    assert metrics["reading_order.pairwise"] == 5 / 6
    assert metrics["reading_order.adjacent"] == 2 / 3


def test_visual_metrics_match_relative_geometry_across_page_scales() -> None:
    expected = [{"page": 2, "bbox": (0.1, 0.2, 0.6, 0.7)}]
    actual = [{"page_number": 2, "bbox": [60, 160, 360, 560]}]
    pages = [{"number": 1, "width": 600, "height": 800}, {"number": 2, "width": 600, "height": 800}]

    metrics, matches = HARNESS.visual_metrics(expected, actual, pages, "images.detection")

    assert metrics["images.detection.f1"] == 1.0
    assert len(matches) == 1


def test_common_input_gate_rejects_missing_stage_trace() -> None:
    shared = {
        "prepared_pages": [],
        "derived_pages": [],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [],
        "marker_summary": {},
        "pairing_summary": {},
    }
    assert HARNESS.common_input_regressions(shared, dict(shared)) == []
    candidate = dict(shared)
    candidate["markers"] = None
    assert HARNESS.common_input_regressions(shared, candidate) == [
        "/markers: Rust marker-stage trace is missing"
    ]


def test_common_input_gate_accepts_only_non_product_candidate_pruning() -> None:
    shared = {
        "prepared_pages": [],
        "derived_pages": [],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [],
        "marker_summary": {"label_candidate_count": 4},
        "pairing_summary": {"label_candidate_count": 4},
    }
    candidate = json.loads(json.dumps(shared))
    candidate["marker_summary"]["label_candidate_count"] = 3
    candidate["pairing_summary"]["label_candidate_count"] = 3

    failures, improvements = HARNESS.common_input_qualification(
        {"evidence": {}}, shared, candidate
    )

    assert failures == []
    assert improvements == ["stricter-unused-label-candidate-pruning"]


def test_common_input_gate_ignores_unused_candidate_count_growth() -> None:
    oracle = {
        "prepared_pages": [],
        "derived_pages": [],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [],
        "marker_summary": {"label_candidate_count": 2},
        "pairing_summary": {"label_candidate_count": 2},
    }
    candidate = json.loads(json.dumps(oracle))
    candidate["marker_summary"]["label_candidate_count"] = 3
    candidate["pairing_summary"]["label_candidate_count"] = 3

    failures, improvements = HARNESS.common_input_qualification({}, oracle, candidate)

    assert failures == []
    assert improvements == ["non-product-unused-label-candidate-accounting"]


def test_common_input_gate_names_richer_native_superscript_evidence() -> None:
    oracle = {
        "prepared_pages": [],
        "derived_pages": [],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [{"candidate_reason": "attached_symbol_marker"}],
        "marker_summary": {},
        "pairing_summary": {},
    }
    candidate = json.loads(json.dumps(oracle))
    candidate["markers"][0]["candidate_reason"] = "native_superscript_span"

    failures, improvements = HARNESS.common_input_qualification(
        {"evidence": {}}, oracle, candidate
    )

    assert failures == []
    assert improvements == ["native-superscript-evidence-over-geometric-fallback"]


def test_common_input_gate_accepts_rejection_of_malformed_label_prefix() -> None:
    line = {
        "id": "line",
        "text": "0,& malformed",
        "suppress_footnote_label": True,
    }
    page = {"lines": [line]}
    oracle = {
        "prepared_pages": [page],
        "derived_pages": [json.loads(json.dumps(page))],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [],
        "marker_summary": {},
        "pairing_summary": {},
    }
    candidate = json.loads(json.dumps(oracle))
    candidate["prepared_pages"][0]["lines"][0]["suppress_footnote_label"] = False
    candidate["derived_pages"][0]["lines"][0]["suppress_footnote_label"] = False

    failures, improvements = HARNESS.common_input_qualification(
        {"evidence": {}}, oracle, candidate
    )

    assert failures == []
    assert improvements == ["rejected-malformed-label-prefixes"]


def test_common_input_gate_composes_independent_trace_improvements() -> None:
    line = {
        "id": "line",
        "text": "0,& malformed",
        "suppress_footnote_label": True,
    }
    page = {"lines": [line]}
    oracle = {
        "prepared_pages": [page],
        "derived_pages": [json.loads(json.dumps(page))],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [{"candidate_reason": "attached_symbol_marker"}],
        "marker_summary": {"label_candidate_count": 2},
        "pairing_summary": {"label_candidate_count": 2},
    }
    candidate = json.loads(json.dumps(oracle))
    for field in ("prepared_pages", "derived_pages"):
        candidate[field][0]["lines"][0]["suppress_footnote_label"] = False
    candidate["markers"][0]["candidate_reason"] = "native_superscript_span"
    candidate["marker_summary"]["label_candidate_count"] = 1
    candidate["pairing_summary"]["label_candidate_count"] = 1

    failures, improvements = HARNESS.common_input_qualification(
        {"evidence": {}}, oracle, candidate
    )

    assert failures == []
    assert improvements == [
        "stricter-unused-label-candidate-pruning",
        "native-superscript-evidence-over-geometric-fallback",
        "rejected-malformed-label-prefixes",
    ]


def test_detached_note_label_improvement_rejects_ordinary_same_row_text() -> None:
    body = {
        "id": "body",
        "text": "Lecompte, supra note 12.",
        "bbox": [89.0, 100.0, 220.0, 111.0],
        "reading_order": 1,
        "region_type": "footnote",
        "spans": [
            {
                "text": "Lecompte, supra note 12.",
                "size": 10.0,
                "superscript": False,
            }
        ],
    }
    label = {
        "id": "label",
        "text": "17",
        "bbox": [72.0, 100.1, 78.0, 107.0],
        "reading_order": 2,
        "region_type": "footnote",
        "spans": [{"text": "17", "size": 7.0, "superscript": True}],
    }
    oracle = {"prepared_pages": [{"lines": [body, label]}]}
    candidate = json.loads(json.dumps(oracle))
    candidate["prepared_pages"][0]["lines"].reverse()
    candidate["prepared_pages"][0]["lines"][0]["reading_order"] = 1
    candidate["prepared_pages"][0]["lines"][1]["reading_order"] = 2

    assert HARNESS._only_attaches_detached_note_labels(oracle, candidate)

    oracle["prepared_pages"][0]["lines"][1]["text"] = "5th"
    candidate["prepared_pages"][0]["lines"][0]["text"] = "5th"
    assert not HARNESS._only_attaches_detached_note_labels(oracle, candidate)


def test_column_layout_improvement_requires_source_gain_without_text_changes(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "pages.jsonl"
    expected = ["left 1", "left 2", "left 3", "right 1", "right 2", "right 3"]
    reference.write_text(
        json.dumps({"pdf_page": 1, "text": "\n".join(expected), "annotations": []})
        + "\n",
        encoding="utf-8",
    )
    lines = {
        text: {
            "id": text,
            "text": text,
            "bbox": [
                60.0 if text.startswith("left") else 360.0,
                100.0 + (int(text[-1]) - 1) * 12.0,
                240.0 if text.startswith("left") else 540.0,
                110.0 + (int(text[-1]) - 1) * 12.0,
            ],
            "reading_order": 0,
            "region_id": "region",
            "region_type": "body",
            "note_region_mode": "",
            "suppress_footnote_label": False,
        }
        for text in expected
    }

    def result(order: list[str]) -> dict:
        ordered = []
        for index, text in enumerate(order, start=1):
            line = dict(lines[text])
            line["reading_order"] = index
            ordered.append(line)
        page = {
            "id": "p0001",
            "index": 0,
            "number": 1,
            "width": 600.0,
            "height": 800.0,
            "lines": ordered,
            "regions": [],
        }
        return {
            "prepared_pages": [json.loads(json.dumps(page))],
            "derived_pages": [json.loads(json.dumps(page))],
            "paragraphs": [],
            "sections": [],
            "footnotes": [],
            "diagnostics": [],
            "status": "ready",
            "validation": "ok",
            "markers": [],
            "marker_summary": {},
            "pairing_summary": {},
        }

    oracle = result(["left 1", "right 1", "left 2", "right 2", "left 3", "right 3"])
    candidate = result(expected)
    case = {
        "evidence": {
            "kind": "canonical-derived",
            "path": str(reference),
            "sha256": HARNESS.sha256(reference),
        }
    }

    failures, improvements = HARNESS.common_input_qualification(case, oracle, candidate)

    assert failures == []
    assert improvements == ["source-supported-column-reading-order-and-notes"]

    candidate["prepared_pages"][0]["lines"][0]["text"] = "changed"
    candidate["derived_pages"][0]["lines"][0]["text"] = "changed"
    failures, improvements = HARNESS.common_input_qualification(case, oracle, candidate)
    assert failures
    assert improvements == []


def test_source_supported_product_change_rejects_metric_regression(tmp_path: Path) -> None:
    reference = tmp_path / "pages.jsonl"
    reference.write_text(
        json.dumps({"pdf_page": 1, "text": "correct text", "annotations": []}) + "\n",
        encoding="utf-8",
    )
    line = {
        "id": "line",
        "text": "correct text",
        "bbox": [60.0, 100.0, 200.0, 110.0],
        "reading_order": 1,
        "region_id": "region",
        "region_type": "body",
        "source": "native",
        "source_index": 1,
        "spans": [],
        "words": [],
    }
    oracle = {
        "prepared_pages": [{"lines": [line]}],
        "derived_pages": [{"lines": [dict(line, text="wrong text")]}],
        "paragraphs": [],
        "sections": [],
        "footnotes": [],
        "diagnostics": [],
        "status": "ready",
        "validation": "ok",
        "markers": [],
        "marker_summary": {},
        "pairing_summary": {},
    }
    candidate = json.loads(json.dumps(oracle))
    candidate["derived_pages"][0]["lines"][0]["text"] = "correct text"
    case = {
        "evidence": {
            "kind": "canonical-derived",
            "path": str(reference),
            "sha256": HARNESS.sha256(reference),
        }
    }

    assert HARNESS._source_supported_product_change(case, oracle, candidate)

    candidate["derived_pages"][0]["lines"][0]["text"] = "even worse"
    assert not HARNESS._source_supported_product_change(case, oracle, candidate)


def test_source_supported_product_change_accepts_only_reference_recovery(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "pages.jsonl"
    reference.write_text(
        json.dumps(
            {
                "pdf_page": 1,
                "text": "Author *\n* Biography",
                "annotations": [
                    {
                        "taxonomy_name": "fn_ref",
                        "pair_status": "paired",
                        "pair_id": "author",
                        "selected_text": "*",
                        "start_line_order": 1,
                        "start_offset": 7,
                    },
                    {
                        "taxonomy_name": "fn_label",
                        "pair_status": "paired",
                        "pair_id": "author",
                        "selected_text": "*",
                        "start_line_order": 2,
                        "start_offset": 0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        {"id": "ref", "text": "Author *", "reading_order": 1},
        {"id": "label", "text": "* Biography", "reading_order": 2},
    ]
    note = {
        "label": "*",
        "occurrence": 1,
        "body": "Biography",
        "body_line_ids": ["label"],
        "body_pages": [1],
        "reference_line_id": None,
        "reference_page": None,
        "sentence_proposition": "",
        "passage_since_prior_note": "",
        "warnings": ["label_only"],
    }
    oracle = {
        "prepared_pages": [{"lines": lines}],
        "derived_pages": [{"lines": json.loads(json.dumps(lines))}],
        "paragraphs": [],
        "sections": [],
        "footnotes": [note],
        "diagnostics": [],
        "validation": "ok",
    }
    candidate = json.loads(json.dumps(oracle))
    candidate["footnotes"][0].update(
        {
            "reference_line_id": "ref",
            "reference_page": 1,
            "sentence_proposition": "Author",
            "passage_since_prior_note": "Author",
            "warnings": [],
        }
    )
    case = {
        "evidence": {
            "kind": "canonical-derived",
            "path": str(reference),
            "sha256": HARNESS.sha256(reference),
        }
    }

    assert HARNESS._source_supported_product_change(case, oracle, candidate)

    candidate["footnotes"][0]["body"] = "Damaged"
    assert not HARNESS._source_supported_product_change(case, oracle, candidate)


def test_note_partition_quality_detects_a_swallowed_later_note() -> None:
    lines = [
        {"id": "one", "text": "1 First note", "reading_order": 1},
        {"id": "two", "text": "2 Second note", "reading_order": 2},
    ]
    broken = {
        "prepared_pages": [{"lines": lines}],
        "footnotes": [
            {"label": "1", "body_line_ids": ["one", "two"]},
            {"label": "2", "body_line_ids": []},
        ],
    }
    repaired = {
        "prepared_pages": [{"lines": lines}],
        "footnotes": [
            {"label": "1", "body_line_ids": ["one"]},
            {"label": "2", "body_line_ids": ["two"]},
        ],
    }

    assert HARNESS._note_partition_quality(broken) == (0.5, 1)
    assert HARNESS._note_partition_quality(repaired) == (1.0, 0)

    citation = json.loads(json.dumps(repaired))
    citation["prepared_pages"][0]["lines"].insert(
        1, {"id": "citation", "text": "2 Canadian Intellectual Reporter", "reading_order": 2}
    )
    citation["footnotes"][0]["body_line_ids"].append("citation")
    assert HARNESS._note_partition_quality(citation) == (1.0, 0)


def test_source_note_support_accepts_an_attached_author_symbol() -> None:
    lines = {
        "ref": {"id": "ref", "text": "Author**", "spans": []},
        "label": {"id": "label", "text": "** Biography", "spans": []},
    }
    note = {
        "reference_line_id": "ref",
        "body": "Biography",
        "body_line_ids": ["label"],
    }

    assert HARNESS._source_note_supported(("**", 1), note, lines)

    lines["ref"]["text"] = "Author"
    assert not HARNESS._source_note_supported(("**", 1), note, lines)


def test_credible_furniture_requires_repetition_at_the_page_edge() -> None:
    pages = []
    for index in range(2):
        pages.append(
            {
                "index": index,
                "height": 800.0,
                "lines": [
                    {
                        "id": f"header-{index}",
                        "text": "Article title",
                        "bbox": [60.0, 70.0, 180.0, 80.0],
                        "region_type": "header",
                    },
                    {
                        "id": f"body-{index}",
                        "text": "Article title",
                        "bbox": [60.0, 300.0, 180.0, 310.0],
                        "region_type": "body",
                    },
                ],
            }
        )

    assert HARNESS._credible_furniture_line_ids({"prepared_pages": pages}) == {
        "header-0",
        "header-1",
    }


def test_margin_geometry_prefers_visual_order_within_one_side_lane() -> None:
    old = {
        "width": 600.0,
        "lines": [
            {"id": "note", "bbox": [500.0, 500.0, 570.0, 510.0], "reading_order": 1},
            {"id": "quote", "bbox": [500.0, 100.0, 570.0, 110.0], "reading_order": 2},
        ],
    }
    new = json.loads(json.dumps(old))
    new["lines"].reverse()
    for reading_order, line in enumerate(new["lines"], start=1):
        line["reading_order"] = reading_order

    assert HARNESS._margin_geometry_inversions(old, {"note", "quote"}) == 1
    assert HARNESS._margin_geometry_inversions(new, {"note", "quote"}) == 0


def test_table_band_requires_repeated_geometric_rows() -> None:
    lines = [
        {"id": "caption", "text": "Table 1. Results", "bbox": [60, 70, 180, 80]}
    ]
    for row in range(6):
        for column, x in enumerate((60.0, 180.0, 300.0)):
            lines.append(
                {
                    "id": f"r{row}c{column}",
                    "text": str(row * 3 + column),
                    "bbox": [x, 100.0 + row * 12, x + 30, 109.0 + row * 12],
                }
            )
    lines.append({"id": "prose", "text": "Following prose", "bbox": [60, 300, 360, 310]})

    table = HARNESS._table_band_line_ids({"lines": lines})

    assert {f"r{row}c{column}" for row in range(6) for column in range(3)} <= table
    assert "prose" not in table
    assert HARNESS._table_band_line_ids({"lines": [lines[0], lines[-1]]}) == set()


def _extraction_line(text: str, index: int, block: int = 1) -> dict:
    return {
        "id": f"p0001-l{index:04d}",
        "text": text,
        "bbox": [0.0, float(index), 100.0, float(index + 1)],
        "block_index": block,
        "source_index": index,
        "reading_order": index,
        "source": "native",
        "spans": [
            {
                "text": text,
                "start": 0,
                "end": len(text),
                "bbox": [0.0, float(index), 100.0, float(index + 1)],
                "font": "Times",
                "size": 10.0,
                "flags": 0,
                "superscript": False,
            }
        ],
        "words": [
            {
                "text": text,
                "start": 0,
                "end": len(text),
                "bbox": [0.0, float(index), 100.0, float(index + 1)],
            }
        ],
    }


def _extraction_value(lines: list[dict], separator: float | None = 500.0) -> dict:
    return {
        "pages": [
            {
                "width": 612.0,
                "height": 792.0,
                "text_quality": 1.0,
                "lines": lines,
            }
        ],
        "separators": [separator],
        "metadata": {"author": "Author"},
    }


def test_extraction_autopsy_classifies_line_merges_without_downstream_scoring() -> None:
    oracle = _extraction_value(
        [_extraction_line("33", 1), _extraction_line("Ibid.", 2)]
    )
    candidate = _extraction_value([_extraction_line("33 Ibid.", 1)])

    result = HARNESS.extraction_contract_diagnostics(oracle, candidate)

    assert result["issues"]["line.merge"] == 1
    assert "line.content_and_segmentation_change" not in result["issues"]


def test_extraction_autopsy_has_an_exhaustive_oracle_projection_gate() -> None:
    oracle = _extraction_value([_extraction_line("Same", 1)])
    candidate = json.loads(json.dumps(oracle))
    candidate["debug"] = {"candidate_only": True}

    exact = HARNESS.extraction_contract_diagnostics(oracle, candidate)

    assert exact["passed"] is True
    assert exact["issue_count"] == 0
    assert exact["observations"]["contract.oracle_projection_differences"] == 0

    candidate["pages"][0]["lines"][0]["unclassified_legacy_field"] = "changed"
    oracle["pages"][0]["lines"][0]["unclassified_legacy_field"] = "original"
    changed = HARNESS.extraction_contract_diagnostics(oracle, candidate)

    assert changed["passed"] is False
    assert changed["issues"]["contract.oracle_projection_mismatch"] == 1
    assert changed["observations"]["contract.oracle_projection_differences"] == 1


def test_extraction_autopsy_reports_all_independent_field_gaps() -> None:
    oracle = _extraction_value(
        [_extraction_line("First", 1, 1), _extraction_line("Second", 2, 2)]
    )
    candidate = _extraction_value(
        [_extraction_line("First", 1, 1), _extraction_line("Second", 2, 1)],
        separator=None,
    )
    candidate["metadata"] = {}
    candidate["pages"][0]["lines"][1]["spans"][0]["font"] = "F1"
    candidate["pages"][0]["lines"][1]["spans"][0]["flags"] = 16
    candidate["pages"][0]["lines"][1]["words"][0]["bbox"][2] = 90.0

    result = HARNESS.extraction_contract_diagnostics(oracle, candidate)

    for issue in (
        "line.block_index_mismatch",
        "block.boundary_missing",
        "spans.font_mismatch",
        "spans.flags_mismatch",
        "separator.missing",
        "metadata.source_key_missing",
    ):
        assert result["issues"][issue] >= 1
    assert result["measurements"]["words.bbox.x1.absolute_error"]["max"] == 10.0


def test_score_summary_keeps_distribution_and_ignores_missing_values() -> None:
    summary = HARNESS.summarize_metric_rows(
        [
            {"metrics": {"source.cer": 0.0, "reading_order.pairwise": None}},
            {"metrics": {"source.cer": 0.2, "reading_order.pairwise": 1.0}},
        ]
    )

    assert summary["source.cer"] == {
        "count": 2,
        "mean": 0.1,
        "median": 0.1,
        "p05": 0.0,
        "p95": 0.2,
        "min": 0.0,
        "max": 0.2,
    }
    assert summary["reading_order.pairwise"]["count"] == 1
