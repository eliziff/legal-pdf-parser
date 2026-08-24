use super::*;
use legal_pdf_core::model::Word;

#[test]
fn pdf_text_index_preserves_exact_lines_and_scalar_offsets() {
    let pages: Vec<Page> = serde_json::from_value(json!([
        {
            "id": "p2",
            "index": 1,
            "number": 2,
            "width": 100.0,
            "height": 100.0,
            "lines": [{
                "id": "excluded",
                "page_index": 1,
                "page_number": 2,
                "source_index": 0,
                "reading_order": 0,
                "block_index": 0,
                "text": "EXCLUDED",
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "exclude_from_body": true
            }],
            "regions": []
        },
        {
            "id": "p1",
            "index": 0,
            "number": 1,
            "width": 100.0,
            "height": 100.0,
            "lines": [
                {
                    "id": "unicode",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 1,
                    "reading_order": 1,
                    "block_index": 0,
                    "text": "\u{1f600}e\u{301}",
                    "bbox": [0.0, 10.0, 10.0, 20.0]
                },
                {
                    "id": "alpha",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 0,
                    "reading_order": 0,
                    "block_index": 0,
                    "text": "\u{3b1}",
                    "bbox": [0.0, 0.0, 10.0, 10.0]
                }
            ],
            "regions": []
        }
    ]))
    .unwrap();

    let index = PdfTextIndex::from_pages(&pages);

    assert_eq!(index.text(), "\u{3b1}\n\u{1f600}e\u{301}\nEXCLUDED");
    assert_eq!(
        index.line("alpha").unwrap().range,
        ScalarRange { start: 0, end: 1 }
    );
    assert_eq!(
        index.line("unicode").unwrap().range,
        ScalarRange { start: 2, end: 5 }
    );
    assert_eq!(
        index.global_range("unicode", 0, 1),
        Some(ScalarRange { start: 2, end: 3 })
    );
    assert_eq!(
        index.line_ids(ScalarRange { start: 0, end: 5 }),
        ["alpha", "unicode"]
    );
    assert_eq!(index.page_range(0), Some(ScalarRange { start: 0, end: 5 }));
    assert_eq!(index.page_range(1), Some(ScalarRange { start: 6, end: 14 }));
}

#[test]
fn pdf_adapter_maps_raw_numeric_candidates_to_exact_line_ids_once() {
    let pages: Vec<Page> = serde_json::from_value(json!([{
        "id": "page-1",
        "index": 0,
        "number": 1,
        "width": 600.0,
        "height": 800.0,
        "lines": [
            {
                "id": "line-1",
                "page_index": 0,
                "page_number": 1,
                "source_index": 0,
                "reading_order": 0,
                "block_index": 0,
                "text": "1. First paragraph has prose.",
                "bbox": [72.0, 100.0, 400.0, 112.0],
                "region_type": "body"
            },
            {
                "id": "line-2",
                "page_index": 0,
                "page_number": 1,
                "source_index": 1,
                "reading_order": 1,
                "block_index": 1,
                "text": "2. Second paragraph has prose.",
                "bbox": [72.0, 120.0, 410.0, 132.0],
                "region_type": "body"
            }
        ],
        "regions": []
    }]))
    .unwrap();

    let adapter = PdfResolutionInput::from_pages(&pages, &PdfPrimitiveEvidence::default());
    let run = adapter
        .runs
        .iter()
        .find(|run| run.grammar == CandidateGrammar::Numeric)
        .expect("numeric candidate run");
    let mapped = run
        .markers
        .iter()
        .map(|candidate| {
            adapter
                .evidence
                .iter()
                .find(|evidence| evidence.candidate_id == candidate.id)
                .expect("mapped candidate")
        })
        .collect::<Vec<_>>();

    assert_eq!(mapped[0].line_ids, ["line-1"]);
    assert_eq!(mapped[1].line_ids, ["line-2"]);
    assert!(mapped.iter().all(|evidence| evidence
        .observations
        .contains(&CandidateObservationV2::BodyProseFlow)));
}

#[test]
fn pdf_adapter_abstains_on_contents_rows_and_transcript_line_columns() {
    assert!(contents_row("1.1 Background ........ 3"));
    assert!(!contents_row("1.1 Background and application"));

    let mut contents = test_page(
        (1..=3)
            .map(|number| {
                let mut line = test_line(
                    &format!("{number}. Topic heading ........ {}", number + 4),
                    [72.0, 80.0 + f64::from(number) * 20.0, 500.0, 94.0],
                    vec![],
                );
                line.region_type = "body".to_owned();
                line
            })
            .collect(),
    );
    contents.index = 7;
    contents.number = 8;
    for line in &mut contents.lines {
        line.page_index = 7;
        line.page_number = 8;
    }
    let contents_adapter =
        PdfResolutionInput::from_pages(&[contents], &PdfPrimitiveEvidence::default());
    assert!(contents_adapter.evidence.iter().any(|item| item
        .observations
        .contains(&CandidateObservationV2::ContentsRow)));

    let mut transcript = test_page(
        (1..=25)
            .map(|number| {
                test_line(
                    &number.to_string(),
                    [112.0, 60.0 + f64::from(number) * 20.0, 140.0, 74.0],
                    vec![],
                )
            })
            .chain((1..=25).map(|number| {
                let mut line = test_line(
                    &format!("Counsel continues speaking on transcript line {number}."),
                    [154.0, 60.0 + f64::from(number) * 20.0, 500.0, 74.0],
                    vec![],
                );
                line.region_type = "body".to_owned();
                line
            }))
            .collect(),
    );
    transcript.index = 3;
    transcript.number = 4;
    for line in &mut transcript.lines {
        line.page_index = 3;
        line.page_number = 4;
    }
    let transcript_adapter =
        PdfResolutionInput::from_pages(&[transcript], &PdfPrimitiveEvidence::default());
    assert!(!transcript_adapter.evidence.is_empty());
    assert!(transcript_adapter.evidence.iter().all(|item| item
        .observations
        .contains(&CandidateObservationV2::TranscriptLineNumber)));

    let index = test_page(
        (1..=5)
            .flat_map(|number| {
                [
                    test_line(
                        &format!("term-{number}"),
                        [72.0, 80.0 + f64::from(number) * 20.0, 150.0, 94.0],
                        vec![],
                    ),
                    test_line(
                        &format!("[{number}] {}:{}", number + 20, number + 1),
                        [180.0, 80.0 + f64::from(number) * 20.0, 300.0, 94.0],
                        vec![],
                    ),
                ]
            })
            .collect(),
    );
    assert!(index_pages(&[index]).contains(&0));
}

#[test]
fn typed_note_pairs_keep_every_exact_reference_anchor() {
    let pages: Vec<Page> = serde_json::from_value(json!([{
        "id": "page-1",
        "index": 0,
        "number": 1,
        "width": 600.0,
        "height": 800.0,
        "lines": [
            {
                "id": "reference",
                "page_index": 0,
                "page_number": 1,
                "source_index": 0,
                "reading_order": 0,
                "block_index": 0,
                "text": "x¹ y¹",
                "bbox": [72.0, 100.0, 200.0, 112.0]
            },
            {
                "id": "label",
                "page_index": 0,
                "page_number": 1,
                "source_index": 1,
                "reading_order": 1,
                "block_index": 1,
                "text": "1 Note body",
                "bbox": [72.0, 700.0, 300.0, 712.0]
            }
        ],
        "regions": []
    }]))
    .unwrap();
    let index = PdfTextIndex::from_pages(&pages);
    let (pairs, diagnostics) = map_note_pairs(
        &index,
        &[NotePairClaim {
            pair_id: "pair-1".to_owned(),
            label: "1".to_owned(),
            kind: NotePairKind::Footnote,
            label_anchor: legal_pdf_core::SourceAnchor {
                line_id: "label".to_owned(),
                start: 0,
                end: 1,
            },
            reference_anchors: vec![
                legal_pdf_core::SourceAnchor {
                    line_id: "reference".to_owned(),
                    start: 1,
                    end: 2,
                },
                legal_pdf_core::SourceAnchor {
                    line_id: "reference".to_owned(),
                    start: 4,
                    end: 5,
                },
            ],
            body_line_ids: vec!["label".to_owned()],
        }],
    )
    .unwrap();

    assert!(diagnostics.is_empty());
    assert_eq!(pairs[0].label.range, ScalarRange { start: 6, end: 7 });
    assert_eq!(pairs[0].references.len(), 2);
    assert_eq!(
        pairs[0]
            .references
            .iter()
            .map(|reference| reference.range)
            .collect::<Vec<_>>(),
        [
            ScalarRange { start: 1, end: 2 },
            ScalarRange { start: 4, end: 5 }
        ]
    );
    assert_eq!(pairs[0].body.line_ids, ["label"]);
}

#[test]
fn incomplete_pairer_products_abstain_without_losing_the_footnote_product() {
    let pages: Vec<Page> = serde_json::from_value(json!([{
        "id": "page-1", "index": 0, "number": 1, "width": 600.0, "height": 800.0,
        "lines": [
            {"id":"reference","page_index":0,"page_number":1,"source_index":0,"reading_order":0,"block_index":0,"text":"1","bbox":[0.0,0.0,1.0,1.0]},
            {"id":"empty","page_index":0,"page_number":1,"source_index":1,"reading_order":1,"block_index":1,"text":"","bbox":[0.0,2.0,1.0,3.0]},
            {"id":"body","page_index":0,"page_number":1,"source_index":2,"reading_order":2,"block_index":2,"text":"1 body","bbox":[0.0,4.0,10.0,5.0]}
        ], "regions": []
    }])).unwrap();
    let index = PdfTextIndex::from_pages(&pages);
    let mut pairs = (0..346)
        .map(|number| NotePairClaim {
            pair_id: format!("no-body-{number:03}"),
            label: "1".to_owned(),
            kind: NotePairKind::Footnote,
            label_anchor: legal_pdf_core::SourceAnchor {
                line_id: "body".to_owned(),
                start: 0,
                end: 1,
            },
            reference_anchors: vec![legal_pdf_core::SourceAnchor {
                line_id: "reference".to_owned(),
                start: 0,
                end: 1,
            }],
            body_line_ids: Vec::new(),
        })
        .collect::<Vec<_>>();
    pairs.push(NotePairClaim {
        pair_id: "zero-label".to_owned(),
        label: "1".to_owned(),
        kind: NotePairKind::Footnote,
        label_anchor: legal_pdf_core::SourceAnchor {
            line_id: "empty".to_owned(),
            start: 0,
            end: 0,
        },
        reference_anchors: vec![legal_pdf_core::SourceAnchor {
            line_id: "reference".to_owned(),
            start: 0,
            end: 1,
        }],
        body_line_ids: vec!["body".to_owned()],
    });
    pairs.push(NotePairClaim {
        pair_id: "zero-reference".to_owned(),
        label: "1".to_owned(),
        kind: NotePairKind::Footnote,
        label_anchor: legal_pdf_core::SourceAnchor {
            line_id: "body".to_owned(),
            start: 0,
            end: 1,
        },
        reference_anchors: vec![legal_pdf_core::SourceAnchor {
            line_id: "empty".to_owned(),
            start: 0,
            end: 0,
        }],
        body_line_ids: vec!["body".to_owned()],
    });

    let (claims, diagnostics) = map_note_pairs(&index, &pairs).unwrap();

    assert!(claims.is_empty());
    assert_eq!(diagnostics.len(), 348);
    assert!(diagnostics
        .iter()
        .all(|item| item.code == "note_pair_unmaterialized"));
    assert_eq!(diagnostics[346].candidate_ids, ["zero-label"]);
    assert_eq!(diagnostics[347].candidate_ids, ["zero-reference"]);
}

fn test_line(text: &str, bbox: [f64; 4], spans: Vec<Span>) -> Line {
    Line {
        id: String::new(),
        page_index: 0,
        page_number: 1,
        source_index: 0,
        reading_order: 0,
        block_index: 0,
        text: text.to_owned(),
        bbox,
        spans,
        words: vec![],
        detached_references: vec![],
        exclude_from_body: false,
        suppress_footnote_label: false,
        note_region_mode: String::new(),
        region_id: String::new(),
        region_type: "unknown".to_owned(),
        source: "native".to_owned(),
    }
}

fn sized_line(text: &str, bbox: [f64; 4], size: f64) -> Line {
    test_line(
        text,
        bbox,
        vec![Span {
            id: String::new(),
            text: text.to_owned(),
            bbox,
            font: String::new(),
            size,
            flags: 0,
            superscript: false,
            start: 0,
            end: text.chars().count(),
        }],
    )
}

#[test]
fn paragraph_join_consumes_every_source_hyphen_marker() {
    for marker in ['-', '\u{00ad}', '\u{00ac}'] {
        let first = test_line(&format!("judg{marker}"), [0.0, 0.0, 10.0, 10.0], vec![]);
        let second = test_line("ment", [0.0, 12.0, 10.0, 22.0], vec![]);
        assert_eq!(join_lines(&[&first, &second]).0, "judgment");
    }
}

fn test_page(mut lines: Vec<Line>) -> Page {
    for (index, line) in lines.iter_mut().enumerate() {
        line.id = format!("p0001-l{:04}", index + 1);
        line.source_index = index + 1;
        line.reading_order = index + 1;
    }
    Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }
}

fn mark_source_body(lines: &mut [Line]) {
    for line in lines {
        line.region_type = "body".to_owned();
    }
}

#[test]
fn note_order_puts_a_raised_label_before_its_nearby_text() {
    for marker in ["17", "**"] {
        let mut label = test_line(marker, [39.8, 420.55, 49.8, 430.56], vec![]);
        label.words.push(Word {
            id: String::new(),
            text: marker.to_owned(),
            bbox: [39.8, 420.61, 49.8, 426.72],
            start: 0,
            end: marker.chars().count(),
        });
        let body = test_line(
            "Dominique Moran, Carceral Geography",
            [57.5, 420.50, 363.7, 430.56],
            vec![],
        );
        let mut lines = vec![body, label];

        order_note_lines(&mut lines, 612.0);

        assert_eq!(lines[0].text, marker);
    }
}

#[test]
fn note_order_puts_a_detached_label_before_a_distant_same_row_fragment() {
    let mut label = test_line("68", [95.8, 554.3, 105.9, 566.1], vec![]);
    label.words.push(Word {
        id: String::new(),
        text: "68".to_owned(),
        bbox: [95.8, 555.3, 105.9, 562.0],
        start: 0,
        end: 2,
    });
    let body = test_line("of", [411.7, 554.3, 422.3, 566.1], vec![]);
    let mut lines = vec![body, label];

    order_note_lines(&mut lines, 612.0);

    assert_eq!(lines[0].text, "68");
}

#[test]
fn note_order_reads_two_columns_column_by_column() {
    let mut lines = Vec::new();
    for row in 0..3 {
        lines.push(test_line(
            &format!("right {row}"),
            [
                340.0,
                400.0 + row as f64 * 12.0,
                540.0,
                410.0 + row as f64 * 12.0,
            ],
            vec![],
        ));
        lines.push(test_line(
            &format!("left {row}"),
            [
                70.0,
                400.0 + row as f64 * 12.0,
                270.0,
                410.0 + row as f64 * 12.0,
            ],
            vec![],
        ));
    }

    order_note_lines(&mut lines, 612.0);

    assert_eq!(
        lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>(),
        ["left 0", "left 1", "left 2", "right 0", "right 1", "right 2"]
    );
}

#[test]
fn note_number_margin_is_not_a_text_column() {
    let mut lines = Vec::new();
    for row in 0..6 {
        let y = 400.0 + row as f64 * 12.0;
        lines.push(test_line(
            "citation body",
            [105.0, y, 405.0, y + 10.0],
            vec![],
        ));
        lines.push(test_line(
            &(row + 1).to_string(),
            [50.0, y, 60.0, y + 8.0],
            vec![],
        ));
    }
    assert_eq!(column_model(&lines, 612.0).kind, "margin_column");

    order_note_lines(&mut lines, 612.0);

    for (row, pair) in lines.chunks_exact(2).enumerate() {
        assert_eq!(pair[0].text, (row + 1).to_string());
        assert_eq!(pair[1].text, "citation body");
    }
}

#[test]
fn hanging_citation_fragments_are_not_a_second_note_column() {
    let mut lines = vec![
        test_line("43", [95.0, 100.0, 109.0, 110.0], vec![]),
        test_line("Fashion ID GmbH", [275.0, 100.0, 422.0, 110.0], vec![]),
        test_line(
            "continuation across the row",
            [95.0, 112.0, 422.0, 122.0],
            vec![],
        ),
        test_line("See also", [95.0, 124.0, 132.0, 134.0], vec![]),
        test_line("Wirtschaftsakademie", [249.0, 124.0, 422.0, 134.0], vec![]),
        test_line("C-", [95.0, 136.0, 105.0, 146.0], vec![]),
        test_line("Jehovan todistajat", [340.0, 136.0, 422.0, 146.0], vec![]),
        test_line("C-25/17", [95.0, 148.0, 205.0, 158.0], vec![]),
    ];

    assert_ne!(
        column_model_with_furniture(&lines, 600.0, false).kind,
        "two_column"
    );
    order_note_lines(&mut lines, 600.0);

    assert_eq!(
        lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>(),
        [
            "43",
            "Fashion ID GmbH",
            "continuation across the row",
            "See also",
            "Wirtschaftsakademie",
            "C-",
            "Jehovan todistajat",
            "C-25/17",
        ]
    );
}

#[test]
fn column_model_ignores_far_edge_furniture() {
    let mut lines = Vec::new();
    for row in 0..3 {
        lines.push(test_line(
            "left",
            [
                70.0,
                100.0 + row as f64 * 12.0,
                270.0,
                110.0 + row as f64 * 12.0,
            ],
            vec![],
        ));
        lines.push(test_line(
            "right",
            [
                340.0,
                100.0 + row as f64 * 12.0,
                540.0,
                110.0 + row as f64 * 12.0,
            ],
            vec![],
        ));
    }
    let mut page_number = test_line("27", [580.0, 760.0, 595.0, 770.0], vec![]);
    page_number.region_type = "footer".to_owned();
    lines.push(page_number);

    assert_eq!(column_model(&lines, 612.0).kind, "two_column");
}

#[test]
fn column_model_prefers_a_clear_page_gutter_over_larger_internal_gaps() {
    let mut lines = Vec::new();
    for row in 0..8 {
        let y = 100.0 + row as f64 * 12.0;
        if row < 4 {
            lines.push(test_line("left text", [54.0, y, 380.0, y + 10.0], vec![]));
        }
        lines.push(test_line("page", [408.0, y, 422.0, y + 10.0], vec![]));
        lines.push(test_line("right text", [540.0, y, 900.0, y + 10.0], vec![]));
        if row < 3 {
            lines.push(test_line("short", [540.0, y, 560.0, y + 10.0], vec![]));
        }
    }

    let model = column_model(&lines, 972.0);

    assert_eq!(model.kind, "two_column");
    assert!((450.0..520.0).contains(&model.split_x));
}

#[test]
fn centered_title_furniture_does_not_hide_the_page_gutter() {
    let mut lines = vec![
        test_line("volume and page", [265.0, 30.0, 342.0, 40.0], vec![]),
        test_line("author", [275.0, 70.0, 335.0, 80.0], vec![]),
    ];
    for row in 0..5 {
        let y = 100.0 + row as f64 * 12.0;
        lines.push(test_line("left", [70.0, y, 290.0, y + 10.0], vec![]));
        lines.push(test_line("right", [324.0, y, 542.0, y + 10.0], vec![]));
    }

    let model = column_model_with_furniture(&lines, 612.0, true);

    assert_eq!(model.kind, "two_column");
}

#[test]
fn table_grid_is_not_rewritten_as_columns() {
    let mut lines = vec![test_line(
        "Table 2. Results",
        [60.0, 70.0, 200.0, 80.0],
        vec![],
    )];
    for row in 0..4 {
        let y = 100.0 + row as f64 * 14.0;
        for (column, x) in [60.0, 180.0, 300.0, 420.0].into_iter().enumerate() {
            lines.push(test_line(
                &format!("r{row}c{column}"),
                [x, y, x + 50.0, y + 10.0],
                vec![],
            ));
        }
    }

    assert_eq!(table_evidence(&lines, 600.0).lines.len(), 16);
    assert_eq!(
        arbitrate_body_order(&mut lines, 600.0, 800.0).reason,
        "table_grid"
    );
}

#[test]
fn textual_table_caption_does_not_force_geometry_order() {
    let mut lines = vec![test_line(
        "Table 1. Contents",
        [60.0, 70.0, 200.0, 80.0],
        vec![],
    )];
    for row in 0..6 {
        let y = 100.0 + row as f64 * 14.0;
        lines.extend([
            test_line("I", [60.0, y, 70.0, y + 10.0], vec![]),
            test_line("Section title", [100.0, y, 280.0, y + 10.0], vec![]),
            test_line("Appendix", [360.0, y, 430.0, y + 10.0], vec![]),
        ]);
    }

    assert!(!contents_grid(&lines, 600.0));
}

#[test]
fn margin_notes_do_not_cut_off_the_main_text_column() {
    let mut lines = Vec::new();
    for row in 0..12 {
        let y = 400.0 + row as f64 * 24.0;
        lines.push(sized_line(
            if row == 6 {
                "1988 to a pair of inventors in the main text"
            } else {
                "Main-column prose continues below the adjacent notes"
            },
            [145.0, y, 425.0, y + 10.0],
            9.0,
        ));
    }
    for row in 0..6 {
        let y = 500.0 + row as f64 * 24.0;
        lines.push(sized_line(
            &format!("{} Citation text", row + 1),
            [37.0, y, 110.0, y + 8.0],
            7.0,
        ));
    }
    lines.push(sized_line(
        "25 Margin citation",
        [427.0, 730.0, 505.0, 738.0],
        7.0,
    ));
    assert_eq!(column_model(&lines, 540.0).kind, "margin_column");
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 540.0,
        height: 792.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("Main-column"))
        .all(|line| line.region_type == "body"));
    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text.starts_with("1988"))
        .is_some_and(|line| line.region_type == "body"));
    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.ends_with("Citation text"))
        .all(|line| line.region_type == "footnote"));
}

#[test]
fn an_early_right_margin_note_lane_is_not_body_prose() {
    let mut lines = Vec::new();
    for row in 0..8 {
        let y = 250.0 + row as f64 * 32.0;
        lines.push(sized_line(
            "Main-column prose remains independent of its notes",
            [70.0, y, 410.0, y + 10.0],
            9.0,
        ));
    }
    for row in 0..5 {
        let y = 320.0 + row as f64 * 24.0;
        lines.push(sized_line(
            &format!("{} Margin citation", row + 20),
            [427.0, y, 505.0, y + 8.0],
            7.0,
        ));
    }
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 540.0,
        height: 792.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("Main-column"))
        .all(|line| line.region_type == "body"));
    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.ends_with("Margin citation"))
        .all(|line| line.region_type == "footnote"));
}

#[test]
fn embedded_contents_locators_are_not_document_headings() {
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines: ["I", "II", "III", "A"]
            .into_iter()
            .enumerate()
            .map(|(row, label)| {
                let y = 100.0 + row as f64 * 20.0;
                sized_line(
                    &format!("{label}. Section title\u{2003}{}", row + 10),
                    [60.0, y, 500.0, y + 12.0],
                    10.0,
                )
            })
            .collect(),
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0].lines.iter().all(|line| line.region_type == "body"));
}

#[test]
fn contents_grid_does_not_scramble_prose_or_hide_an_author_note() {
    let mut lines = vec![
        sized_line("ARTICLE TITLE", [84.0, 30.0, 400.0, 44.0], 14.0),
        sized_line("AUTHOR", [84.0, 70.0, 180.0, 81.0], 10.0),
        sized_line("ABSTRACT", [84.0, 420.0, 140.0, 430.0], 8.0),
    ];
    for row in 0..4 {
        let y = 445.0 + row as f64 * 14.0;
        lines.push(sized_line(
            "The abstract is ordinary prose, not a heading.",
            [84.0, y, 420.0, y + 11.0],
            10.0,
        ));
    }
    for row in 0..6 {
        let y = 110.0 + row as f64 * 35.0;
        lines.extend([
            sized_line("I", [84.0, y, 94.0, y + 9.0], 8.0),
            sized_line("SECTION", [120.0, y, 220.0, y + 9.0], 8.0),
            sized_line(&(168 + row).to_string(), [404.0, y, 420.0, y + 9.0], 8.0),
        ]);
    }
    lines.extend([
        sized_line("*", [84.0, 570.0, 90.0, 579.0], 8.0),
        sized_line(
            "Author affiliation and acknowledgments",
            [102.0, 570.0, 400.0, 579.0],
            8.0,
        ),
    ]);
    assert!(contents_grid(&lines, 486.0));
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 486.0,
        height: 702.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[Some(220.0)]);

    let order = |text: &str| {
        pages[0]
            .lines
            .iter()
            .find(|line| line.text == text)
            .map(|line| line.reading_order)
            .unwrap()
    };
    assert!(order("SECTION") < order("ABSTRACT"));
    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("The abstract"))
        .all(|line| line.region_type == "body"));
    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text == "*")
        .is_some_and(|line| line.region_type == "footnote"));
    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text.starts_with("Author affiliation"))
        .is_some_and(|line| line.region_type == "footnote"));
}

#[test]
fn two_column_note_prose_is_not_a_table_grid() {
    let mut lines = Vec::new();
    for row in 0..4 {
        let y = 100.0 + row as f64 * 14.0;
        lines.extend([
            test_line(&(row + 1).to_string(), [50.0, y, 60.0, y + 10.0], vec![]),
            test_line(
                "A full citation body with enough prose to be a note",
                [70.0, y, 280.0, y + 10.0],
                vec![],
            ),
            test_line(&(row + 5).to_string(), [330.0, y, 340.0, y + 10.0], vec![]),
            test_line(
                "Another complete citation body in the right column",
                [350.0, y, 570.0, y + 10.0],
                vec![],
            ),
        ]);
    }

    let evidence = table_evidence(&lines, 600.0);
    assert!(!evidence.strong());
    assert!(!evidence.continuation());
}

#[test]
fn attached_note_sequence_is_not_a_table_grid() {
    let mut lines = Vec::new();
    for row in 0..8 {
        let y = 500.0 + row as f64 * 12.0;
        lines.extend([
            test_line(
                &format!("{} Citation text", row + 91),
                [60.0, y, 260.0, y + 10.0],
                vec![],
            ),
            test_line("20", [330.0, y, 370.0, y + 10.0], vec![]),
            test_line("40", [430.0, y, 470.0, y + 10.0], vec![]),
        ]);
    }

    let evidence = table_evidence(&lines, 600.0);

    assert!(!strong_table_evidence(&evidence, &lines));
}

#[test]
fn table_numbers_do_not_start_a_footnote_region() {
    let mut lines = vec![sized_line(
        "Table 2. Results",
        [60.0, 70.0, 200.0, 80.0],
        10.0,
    )];
    for row in 0..4 {
        let y = 100.0 + row as f64 * 14.0;
        let texts = [
            "Income".to_owned(),
            "Low".to_owned(),
            (30 + row).to_string(),
            (20 + row).to_string(),
        ];
        for (text, x) in texts.into_iter().zip([60.0, 160.0, 260.0, 360.0]) {
            lines.push(sized_line(&text, [x, y, x + 50.0, y + 9.0], 8.0));
        }
    }
    lines.push(sized_line(
        "† Table-specific note",
        [60.0, 160.0, 240.0, 169.0],
        8.0,
    ));
    lines.push(sized_line(
        "2004 study of the following issue",
        [60.0, 190.0, 240.0, 200.0],
        8.0,
    ));
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[Some(130.0)]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text != "† Table-specific note")
        .all(|line| line.region_type != "footnote"));
    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text == "† Table-specific note")
        .is_some_and(|line| line.region_type == "footnote" && !line.suppress_footnote_label));
    let order = |text: &str| {
        pages[0]
            .lines
            .iter()
            .find(|line| line.text == text)
            .map(|line| line.reading_order)
            .unwrap()
    };
    assert!(order("† Table-specific note") < order("2004 study of the following issue"));
    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text == "2004 study of the following issue")
        .is_some_and(|line| line.region_type == "body"));
}

#[test]
fn separator_keeps_an_unlabelled_note_continuation_with_the_notes() {
    let mut lines = (0..12)
        .map(|row| {
            sized_line(
                "Ordinary main text continues above the footnotes",
                [
                    100.0,
                    100.0 + row as f64 * 24.0,
                    480.0,
                    111.0 + row as f64 * 24.0,
                ],
                10.5,
            )
        })
        .collect::<Vec<_>>();
    lines.extend([
        sized_line(
            "Continuation of the preceding note below the separator",
            [130.0, 462.0, 480.0, 472.0],
            9.0,
        ),
        sized_line(
            "2) for fraudulent and wrongful trading",
            [130.0, 578.0, 390.0, 588.0],
            9.0,
        ),
    ]);
    for row in 0..4 {
        let y = 590.0 + row as f64 * 12.0;
        lines.push(sized_line(
            &format!("{} Citation", row + 28),
            [137.0, y, 300.0, y + 8.0],
            6.0,
        ));
    }
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 612.0,
        height: 792.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    let diagnostics = classify_pages(&mut pages, &[Some(451.0)]);

    assert!(pages[0]
        .lines
        .iter()
        .find(|line| line.text.starts_with("Continuation"))
        .is_some_and(|line| line.region_type == "footnote"));
    assert!(!diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FOOTNOTE_REGION_UNCERTAIN"));
}

#[test]
fn captioned_tables_continue_across_pages_without_becoming_notes() {
    let page = |index: usize, mut lines: Vec<Line>| {
        for line in &mut lines {
            line.page_index = index;
            line.page_number = (index + 1) as u32;
        }
        Page {
            id: format!("p{:04}", index + 1),
            index,
            number: (index + 1) as u32,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    };
    let mut continuation = Vec::new();
    for row in 0..8 {
        let y = 100.0 + row as f64 * 14.0;
        continuation.extend([
            sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
            sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
            sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
        ]);
    }
    let mut sparse_cell = sized_line(
        "continued text in a tall cell",
        [360.0, 107.0, 500.0, 116.0],
        8.0,
    );
    sparse_cell.region_type = "footer".to_owned();
    continuation.push(sparse_cell);
    for row in 0..6 {
        let y = 600.0 + row as f64 * 12.0;
        continuation.extend([
            sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
            sized_line("Genuine footnote", [85.0, y, 260.0, y + 8.0], 7.0),
        ]);
    }
    let mut pages = vec![
        page(0, {
            let mut first = vec![sized_line(
                "Table 1: Provincial results",
                [60.0, 580.0, 240.0, 590.0],
                10.0,
            )];
            for row in 0..6 {
                let y = 600.0 + row as f64 * 14.0;
                first.extend([
                    sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
                    sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
                    sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
                ]);
            }
            first
        }),
        page(1, continuation),
    ];

    classify_pages(&mut pages, &[None, Some(580.0)]);

    assert!(pages[1]
        .lines
        .iter()
        .find(|line| line.text == "continued text in a tall cell")
        .is_some_and(|line| line.region_type == "body" && line.note_region_mode.is_empty()));
    assert!(pages[1]
        .lines
        .iter()
        .filter(|line| line.text == "Genuine footnote")
        .all(|line| line.region_type == "footnote"));
}

#[test]
fn bottom_footnotes_do_not_extend_a_table_onto_the_next_page() {
    let page = |index: usize, lines: Vec<Line>| Page {
        id: format!("p{:04}", index + 1),
        index,
        number: (index + 1) as u32,
        width: 600.0,
        height: 800.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    };
    let mut second = (0..8)
        .map(|row| {
            sized_line(
                "Ordinary prose on the page after a completed table",
                [
                    60.0,
                    100.0 + row as f64 * 16.0,
                    500.0,
                    111.0 + row as f64 * 16.0,
                ],
                10.0,
            )
        })
        .collect::<Vec<_>>();
    for row in 0..6 {
        let y = 600.0 + row as f64 * 12.0;
        second.extend([
            sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
            sized_line("Citation text", [85.0, y, 300.0, y + 8.0], 7.0),
        ]);
    }
    let mut pages = vec![
        page(
            0,
            vec![sized_line(
                "Table 1: Results",
                [60.0, 600.0, 240.0, 610.0],
                10.0,
            )],
        ),
        page(1, second),
    ];

    classify_pages(&mut pages, &[None, Some(580.0)]);

    assert!(pages[1]
        .lines
        .iter()
        .filter(|line| line.text == "Citation text")
        .all(|line| line.region_type == "footnote"));
    assert!(pages[1]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("Ordinary prose"))
        .all(|line| line.region_type == "body"));
}

#[test]
fn a_table_ending_midpage_does_not_mark_the_next_page_as_a_continuation() {
    let page = |index: usize, lines: Vec<Line>| Page {
        id: format!("p{:04}", index + 1),
        index,
        number: (index + 1) as u32,
        width: 600.0,
        height: 800.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    };
    let mut first = vec![sized_line(
        "Table 1: Results",
        [60.0, 70.0, 240.0, 80.0],
        10.0,
    )];
    for row in 0..6 {
        let y = 100.0 + row as f64 * 14.0;
        first.extend([
            sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
            sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
            sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
        ]);
    }
    let mut second = (0..8)
        .map(|row| {
            sized_line(
                "Ordinary prose on the following page",
                [
                    60.0,
                    100.0 + row as f64 * 16.0,
                    500.0,
                    111.0 + row as f64 * 16.0,
                ],
                10.0,
            )
        })
        .collect::<Vec<_>>();
    for row in 0..6 {
        let y = 600.0 + row as f64 * 12.0;
        second.extend([
            sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
            sized_line("Footnote text", [85.0, y, 300.0, y + 8.0], 7.0),
        ]);
    }
    let mut pages = vec![page(0, first), page(1, second)];

    classify_pages(&mut pages, &[None, Some(580.0)]);

    assert!(pages[1]
        .lines
        .iter()
        .filter(|line| line.text == "Footnote text")
        .all(|line| line.region_type == "footnote"));
}

#[test]
fn detached_drop_cap_moves_before_its_paragraph_line() {
    let mut lines = vec![
        sized_line("crucial opening line", [101.0, 360.0, 400.0, 372.0], 10.0),
        sized_line("continuation", [68.0, 374.0, 400.0, 386.0], 10.0),
        sized_line("A", [68.0, 350.0, 101.0, 407.0], 48.0),
    ];

    repair_drop_caps(&mut lines);

    assert_eq!(lines[0].text, "A");
    assert_eq!(lines[1].text, "crucial opening line");
}

#[test]
fn two_column_pages_keep_each_columns_notes_after_its_body() {
    let lines = |prefix: &str, y: f64| {
        (0..3)
            .flat_map(|row| {
                [
                    test_line(
                        &format!("{prefix} left {row}"),
                        [
                            70.0,
                            y + row as f64 * 12.0,
                            270.0,
                            y + 10.0 + row as f64 * 12.0,
                        ],
                        vec![],
                    ),
                    test_line(
                        &format!("{prefix} right {row}"),
                        [
                            340.0,
                            y + row as f64 * 12.0,
                            540.0,
                            y + 10.0 + row as f64 * 12.0,
                        ],
                        vec![],
                    ),
                ]
            })
            .collect::<Vec<_>>()
    };
    let mut body = lines("body", 100.0);
    column_order(&mut body, 305.0);
    let mut notes: Vec<_> = (0..3)
        .flat_map(|row| {
            [
                test_line(
                    &format!("note left {row}"),
                    [
                        70.0,
                        700.0 + row as f64 * 12.0,
                        270.0,
                        710.0 + row as f64 * 12.0,
                    ],
                    vec![],
                ),
                test_line(
                    &format!("note right {row}"),
                    [
                        340.0,
                        760.0 + row as f64 * 12.0,
                        540.0,
                        770.0 + row as f64 * 12.0,
                    ],
                    vec![],
                ),
            ]
        })
        .collect();
    column_order(&mut notes, 305.0);

    let result = weave_note_columns(body, notes, 612.0);

    assert_eq!(
        result
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>(),
        [
            "body left 0",
            "body left 1",
            "body left 2",
            "note left 0",
            "note left 1",
            "note left 2",
            "body right 0",
            "body right 1",
            "body right 2",
            "note right 0",
            "note right 1",
            "note right 2",
        ]
    );

    let mut body = lines("body", 100.0);
    column_order(&mut body, 305.0);
    let result = weave_note_columns(
        body,
        vec![
            test_line("*", [54.0, 700.0, 64.0, 710.0], vec![]),
            test_line("full-width note", [72.0, 700.0, 540.0, 710.0], vec![]),
        ],
        612.0,
    );
    assert_eq!(result[result.len() - 2].text, "*");
    assert_eq!(result[result.len() - 1].text, "full-width note");
}

#[test]
fn labels_normalize_unicode_superscripts() {
    assert_eq!(normalize_label("⁰¹²"), "12");
    assert_eq!(label_prefix("  12. Note").unwrap().label, "12");
    assert_eq!(label_prefix("2024 decision").unwrap().label, "2024");
    assert!(line_start_label_prefix("12").is_none());
    assert_eq!(label_prefix("12").unwrap().label, "12");
    assert_eq!(label_prefix("**** Note").unwrap().label, "****");
    let embedded = line_start_label_prefix("2endnote 2This is a note").unwrap();
    assert_eq!(embedded.label, "2");
    assert_eq!(
        char_slice("2endnote 2This is a note", embedded.end, 25),
        "This is a note"
    );
    assert!(label_prefix("*Not a note").is_none());
    assert!(line_start_label_prefix("3.2. Good neighbours").is_none());
}

#[test]
fn compact_note_bodies_are_not_repeated_footers() {
    let page = |index: usize, label: usize| {
        let mut body = test_line("Body prose", [72.0, 100.0, 300.0, 110.0], vec![]);
        body.spans.push(Span {
            id: String::new(),
            text: body.text.clone(),
            bbox: body.bbox,
            font: String::new(),
            size: 10.0,
            flags: 0,
            superscript: false,
            start: 0,
            end: body.text.chars().count(),
        });
        let text = format!("{label}. Ibid at {label}.");
        let mut note = test_line(&text, [72.0, 730.0, 170.0, 737.0], vec![]);
        note.spans.push(Span {
            id: String::new(),
            text: note.text.clone(),
            bbox: note.bbox,
            font: String::new(),
            size: 7.0,
            flags: 0,
            superscript: false,
            start: 0,
            end: note.text.chars().count(),
        });
        Page {
            id: format!("p{:04}", index + 1),
            index,
            number: u32::try_from(index + 1).unwrap(),
            width: 612.0,
            height: 792.0,
            lines: vec![body, note],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    };
    let mut pages = vec![page(0, 33), page(1, 46), page(2, 57)];

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines[1].region_type == "unknown"));
}

#[test]
fn repeated_detached_citation_shortforms_remain_note_lines() {
    let mut pages = (0..4)
        .map(|index| {
            let mut body = sized_line(
                "Ordinary article body establishes the document font.",
                [72.0, 300.0, 500.0, 312.0],
                10.0,
            );
            body.id = format!("body-{index}");
            let mut label = sized_line(&(51 + index).to_string(), [50.0, 730.0, 56.0, 736.0], 5.0);
            label.id = format!("label-{index}");
            let mut note = sized_line("Ibid.", [72.0, 730.0, 92.0, 738.0], 8.0);
            note.id = format!("note-{index}");
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![body, label, note],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        })
        .collect::<Vec<_>>();

    mark_repeated_furniture(&mut pages);

    assert!(pages.iter().all(|page| page.lines[1..]
        .iter()
        .all(|line| line.region_type == "unknown")));
}

#[test]
fn attached_top_note_labels_are_not_repeated_headers() {
    let page = |index: usize, label: usize| {
        let marker = test_line(&label.to_string(), [40.0, 70.0, 52.0, 80.0], vec![]);
        let body = test_line(
            if index == 0 {
                "First note"
            } else {
                "Second note"
            },
            [60.0, 70.0, 180.0, 82.0],
            vec![],
        );
        Page {
            id: format!("p{:04}", index + 1),
            index,
            number: u32::try_from(index + 1).unwrap(),
            width: 612.0,
            height: 792.0,
            lines: vec![marker, body],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    };
    let mut pages = vec![page(0, 41), page(1, 42)];

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines[0].region_type == "unknown"));
}

#[test]
fn repeated_top_paragraph_enumerators_are_not_headers() {
    let mut pages = (0..5)
        .map(|index| Page {
            id: format!("p{:04}", index + 1),
            index,
            number: u32::try_from(index + 1).unwrap(),
            width: 612.0,
            height: 792.0,
            lines: vec![
                sized_line(&format!("{}.", 18 + index), [40.0, 72.0, 52.0, 82.0], 10.0),
                sized_line(
                    "Paragraph text begins on the same baseline.",
                    [60.0, 72.0, 500.0, 84.0],
                    10.0,
                ),
            ],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        })
        .collect::<Vec<_>>();

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines[0].region_type == "unknown"));
}

#[test]
fn attached_page_numbers_remain_repeated_headers() {
    let page = |index: usize, number: usize| {
        let marker_text = number.to_string();
        let marker = sized_line(&marker_text, [40.0, 40.0, 52.0, 50.0], 8.0);
        let heading = sized_line("Journal title", [60.0, 40.0, 180.0, 52.0], 8.0);
        Page {
            id: format!("p{:04}", index + 1),
            index,
            number: u32::try_from(index + 1).unwrap(),
            width: 612.0,
            height: 792.0,
            lines: vec![marker, heading],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    };
    let mut pages = vec![page(0, 240), page(1, 242)];

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines.iter().all(|line| line.region_type == "header")));
}

#[test]
fn repeated_edge_text_does_not_sweep_in_a_geometry_outlier() {
    let mut pages = (0..4)
        .map(|index| {
            let top = if index == 3 { 70.0 } else { 20.0 };
            let mut header = sized_line("ALBERTA LAW REVIEW", [100.0, top, 400.0, top + 10.0], 8.0);
            header.id = format!("header-{index}");
            let mut body = sized_line(
                "Unique body prose remains body evidence.",
                [72.0, 300.0, 500.0, 312.0],
                10.0,
            );
            body.id = format!("body-{index}");
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![header, body],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        })
        .collect::<Vec<_>>();

    mark_repeated_furniture(&mut pages);

    assert!(pages[..3]
        .iter()
        .all(|page| page.lines[0].region_type == "header"));
    assert_eq!(pages[3].lines[0].region_type, "unknown");
}

#[test]
fn repeated_edge_text_uses_the_stable_cluster_not_a_title_outlier() {
    let mut pages = (0..5)
        .map(|index| {
            let mut header = sized_line("CIRCULAR PRIORITIES", [100.0, 30.0, 300.0, 40.0], 8.0);
            header.id = format!("header-{index}");
            let mut lines = vec![header];
            if index == 0 {
                let mut title = sized_line("CIRCULAR PRIORITIES", [100.0, 70.0, 300.0, 84.0], 14.0);
                title.id = "article-title".to_owned();
                lines.push(title);
            }
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines,
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        })
        .collect::<Vec<_>>();

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines[0].region_type == "header"));
    assert_eq!(pages[0].lines[1].region_type, "unknown");
}

#[test]
fn alternating_sequential_bottom_folios_override_same_row_footer_text() {
    let mut pages = (0..5)
        .map(|index| {
            let x = if index % 2 == 0 { 570.0 } else { 20.0 };
            let mut folio = sized_line(&(51 + index).to_string(), [x, 730.0, x + 20.0, 740.0], 8.0);
            folio.id = format!("folio-{index}");
            let mut footer = sized_line(
                "Same-baseline journal footer",
                [80.0, 730.0, 430.0, 740.0],
                8.0,
            );
            footer.id = format!("footer-{index}");
            let mut body = sized_line(
                "Body prose stays ordinary text.",
                [72.0, 300.0, 500.0, 312.0],
                10.0,
            );
            body.id = format!("body-{index}");
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![body, footer, folio],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        })
        .collect::<Vec<_>>();

    mark_repeated_furniture(&mut pages);

    assert!(pages
        .iter()
        .all(|page| page.lines[2].region_type == "footer"));
    assert!(pages
        .iter()
        .all(|page| page.lines[0].region_type == "unknown"));
}

#[test]
fn propositions_remove_durable_markers() {
    let text = "First rule. Second rule⟦FN:pair⟧ continues.";
    assert_eq!(sentence_at(text, 23), "Second rule continues.");
    assert_eq!(
        sentence_at("It was so held.” Next point.", 18),
        "Next point."
    );
    assert_eq!(
        sentence_at("The proceeding ended.⟦FN:12⟧", 21),
        "The proceeding ended."
    );
    assert_eq!(sentence_at("R v X at para.20", 15), "R v X at para.20");
}

#[test]
fn interleaved_columns_are_repaired_to_column_order() {
    let mut title = test_line("full-width title", [50.0, 20.0, 550.0, 30.0], vec![]);
    title.id = "title".to_owned();
    let mut author = test_line("author", [360.0, 50.0, 500.0, 60.0], vec![]);
    author.id = "author".to_owned();
    let mut lines = vec![title, author];
    for row in 0..6 {
        let y = 100.0 + row as f64 * 12.0;
        let mut left = test_line("left column prose", [60.0, y, 240.0, y + 10.0], vec![]);
        left.id = format!("left-{row}");
        let mut right = test_line("right column prose", [360.0, y, 540.0, y + 10.0], vec![]);
        right.id = format!("right-{row}");
        lines.extend([left, right]);
    }
    let decision = arbitrate_body_order(&mut lines, 600.0, 800.0);
    assert_eq!(decision.repair, OrderRepair::Column);
    assert_eq!(lines[0].id, "title");
    assert_eq!(lines[1].id, "author");
    assert!(lines[2..8].iter().all(|line| line.bbox[0] < 300.0));
    assert!(lines[8..].iter().all(|line| line.bbox[0] > 300.0));
}

#[test]
fn misplaced_preamble_alone_does_not_justify_a_column_repair() {
    let mut lines = Vec::new();
    for (x, name) in [(60.0, "left"), (360.0, "right")] {
        for row in 0..3 {
            let y = 100.0 + row as f64 * 12.0;
            lines.push(test_line(name, [x, y, x + 180.0, y + 10.0], vec![]));
        }
    }
    lines.push(test_line("title", [50.0, 20.0, 550.0, 30.0], vec![]));
    lines.push(test_line("author", [360.0, 50.0, 500.0, 60.0], vec![]));

    let decision = arbitrate_body_order(&mut lines, 600.0, 800.0);

    assert_eq!(decision.repair, OrderRepair::None);
    assert_eq!(lines[0].text, "left");
}

#[test]
fn endnotes_read_columns_in_sequence() {
    let mut lines = Vec::new();
    for row in 0..6 {
        let y = 100.0 + row as f64 * 12.0;
        let mut left = test_line(
            &format!("{} left endnote prose", row + 1),
            [60.0, y, 240.0, y + 10.0],
            vec![],
        );
        left.id = format!("left-{row}");
        left.region_type = "footnote".to_owned();
        left.note_region_mode = "endnote".to_owned();
        let mut right = test_line(
            &format!("{} right endnote prose", row + 7),
            [360.0, y, 540.0, y + 10.0],
            vec![],
        );
        right.id = format!("right-{row}");
        right.region_type = "footnote".to_owned();
        right.note_region_mode = "endnote".to_owned();
        lines.extend([left, right]);
    }
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines,
        regions: Vec::new(),
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    let diagnostics = order_pages(&mut pages);

    assert!(diagnostics.is_empty());
    assert_eq!(pages[0].lines[0].id, "left-0");
    assert_eq!(pages[0].lines[1].id, "left-1");
    assert_eq!(pages[0].lines[6].id, "right-0");
}

#[test]
fn detached_reference_fits_the_word_gap_not_the_note_margin() {
    let host = test_line(
        "higher. They",
        [335.0, 93.9, 396.0, 103.9],
        vec![
            Span {
                id: "left".to_owned(),
                text: "higher.".to_owned(),
                bbox: [335.0, 93.9, 366.6, 103.9],
                font: String::new(),
                size: 10.0,
                flags: 0,
                superscript: false,
                start: 0,
                end: 7,
            },
            Span {
                id: "right".to_owned(),
                text: "They".to_owned(),
                bbox: [377.8, 93.9, 396.0, 103.9],
                font: String::new(),
                size: 10.0,
                flags: 0,
                superscript: false,
                start: 8,
                end: 12,
            },
        ],
    );
    let inline_marker = test_line("40", [367.5, 94.2, 373.2, 100.0], vec![]);
    let margin_label = test_line("40", [54.1, 94.2, 59.8, 100.0], vec![]);

    assert_eq!(
        detached_reference_target(0, &[inline_marker, host.clone()], 10.0),
        Some((1, 7))
    );
    assert_eq!(
        detached_reference_target(0, &[margin_label, host], 10.0),
        None
    );
}

#[test]
fn endnote_mode_carries_to_the_next_numbered_note_page() {
    let mut first = test_line("1 First note", [60.0, 100.0, 300.0, 110.0], vec![]);
    first.region_type = "footnote".to_owned();
    first.note_region_mode = "endnote".to_owned();
    let mut second = test_line("2 Second note", [60.0, 100.0, 300.0, 110.0], vec![]);
    second.region_type = "footnote".to_owned();
    second.page_index = 1;
    second.page_number = 2;
    let mut pages = vec![
        Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines: vec![first],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        },
        Page {
            id: "p0002".to_owned(),
            index: 1,
            number: 2,
            width: 600.0,
            height: 800.0,
            lines: vec![second],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        },
    ];

    infer_note_region_modes(&mut pages);

    assert_eq!(pages[1].lines[0].note_region_mode, "endnote");
}

#[test]
fn endnote_heading_uses_a_separate_cut_for_each_column() {
    let mut body = test_line(
        "Article body before the notes",
        [60.0, 100.0, 280.0, 110.0],
        vec![],
    );
    body.source_index = 1;
    let mut notes = test_line("Notes", [60.0, 340.0, 120.0, 360.0], vec![]);
    notes.source_index = 2;
    let mut first = test_line("*", [60.0, 365.0, 70.0, 375.0], vec![]);
    first.source_index = 3;
    let first_body = test_line("First note", [80.0, 365.0, 280.0, 375.0], vec![]);
    let continuation = test_line(
        "Continuation from the prior note",
        [350.0, 105.0, 570.0, 115.0],
        vec![],
    );
    let eighth = test_line("8", [330.0, 130.0, 340.0, 140.0], vec![]);
    let eighth_body = test_line("Eighth note", [350.0, 130.0, 570.0, 140.0], vec![]);
    let right_tail = test_line("More note text", [350.0, 300.0, 570.0, 310.0], vec![]);
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 612.0,
        height: 792.0,
        lines: vec![
            body,
            notes,
            first,
            first_body,
            continuation,
            eighth,
            eighth_body,
            right_tail,
        ],
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    let by_text: HashMap<_, _> = pages[0]
        .lines
        .iter()
        .map(|line| (line.text.as_str(), line))
        .collect();
    assert_eq!(by_text["Article body before the notes"].region_type, "body");
    assert!(by_text["Notes"].note_region_mode.is_empty());
    assert_eq!(by_text["Notes"].region_type, "heading");
    assert_eq!(
        by_text["Continuation from the prior note"].note_region_mode,
        "endnote"
    );
    assert_eq!(by_text["First note"].note_region_mode, "endnote");
}

#[test]
fn an_early_body_number_does_not_turn_bottom_footnotes_into_endnotes() {
    let mut lines = vec![
        sized_line(
            "19. The ordinary paragraph continues",
            [60.0, 180.0, 400.0, 192.0],
            10.0,
        ),
        sized_line("More body prose", [60.0, 200.0, 400.0, 212.0], 10.0),
    ];
    for number in 13..=18 {
        let y = 560.0 + f64::from(number - 13) * 20.0;
        lines.push(sized_line(
            &format!("{number} Citation text"),
            [60.0, y, 400.0, y + 9.0],
            7.0,
        ));
    }
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.region_type == "footnote")
        .all(|line| line.note_region_mode == "footnote"));
}

#[test]
fn a_compact_lower_sequence_is_footnotes_without_a_drawn_rule() {
    let mut lines: Vec<_> = (0..10)
        .map(|row| {
            let y = 120.0 + row as f64 * 20.0;
            sized_line("Ordinary article body", [60.0, y, 500.0, y + 12.0], 10.0)
        })
        .collect();
    for number in 91..=95 {
        let y = 560.0 + f64::from(number - 91) * 20.0;
        if number == 93 {
            lines.push(sized_line(
                "2021) 35 at 41).",
                [60.0, y - 10.0, 180.0, y - 1.0],
                8.5,
            ));
        }
        lines.push(sized_line(
            &format!("{number} Citation text"),
            [60.0, y, 500.0, y + 10.0],
            8.5,
        ));
    }
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with('9'))
        .all(|line| line.region_type == "footnote" && line.note_region_mode == "footnote"));
}

#[test]
fn a_single_lower_note_is_backed_by_its_superscript_reference() {
    let mut body = sized_line(
        "Ordinary article body with a reference",
        [60.0, 120.0, 500.0, 132.0],
        10.0,
    );
    body.spans.push(Span {
        id: String::new(),
        text: "21".to_owned(),
        bbox: [400.0, 116.0, 410.0, 124.0],
        font: String::new(),
        size: 6.0,
        flags: 0,
        superscript: true,
        start: body.text.chars().count(),
        end: body.text.chars().count(),
    });
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 600.0,
        height: 800.0,
        lines: vec![
            body,
            sized_line(
                "More ordinary article body",
                [60.0, 140.0, 500.0, 152.0],
                10.0,
            ),
            sized_line("Still more article body", [60.0, 160.0, 500.0, 172.0], 10.0),
            sized_line("21 Citation text", [60.0, 400.0, 500.0, 410.0], 8.5),
            sized_line("Citation continuation", [75.0, 412.0, 500.0, 422.0], 8.5),
        ],
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0].lines[3..]
        .iter()
        .all(|line| line.region_type == "footnote" && line.note_region_mode == "footnote"));
}

#[test]
fn two_reference_backed_margin_notes_do_not_cut_through_main_prose() {
    let mut lines: Vec<_> = (0..10)
        .map(|row| {
            let y = 380.0 + row as f64 * 24.0;
            sized_line(
                "Main-column prose continues beside the margin notes",
                [145.0, y, 425.0, y + 10.0],
                9.0,
            )
        })
        .collect();
    for (line, label) in lines.iter_mut().take(2).zip(["39", "40"]) {
        line.spans.push(Span {
            id: String::new(),
            text: label.to_owned(),
            bbox: [400.0, line.bbox[1] - 4.0, 410.0, line.bbox[1] + 4.0],
            font: String::new(),
            size: 6.0,
            flags: 0,
            superscript: true,
            start: line.text.chars().count(),
            end: line.text.chars().count(),
        });
    }
    lines.extend([
        sized_line("39 First margin note", [37.0, 500.0, 110.0, 508.0], 7.0),
        sized_line("40 Second margin note", [37.0, 550.0, 110.0, 558.0], 7.0),
    ]);
    let mut pages = vec![Page {
        id: "p0001".to_owned(),
        index: 0,
        number: 1,
        width: 540.0,
        height: 792.0,
        lines,
        regions: vec![],
        source: "native".to_owned(),
        text_quality: 1.0,
        printed_label: None,
        printed_label_source: None,
        printed_label_line_id: None,
    }];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("Main-column"))
        .all(|line| line.region_type == "body"));
    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("39 ") || line.text.starts_with("40 "))
        .all(|line| line.region_type == "footnote"));
}

#[test]
fn smaller_quoted_text_does_not_make_normal_prose_a_heading() {
    let mut lines = Vec::new();
    for row in 0..20 {
        let y = 100.0 + row as f64 * 12.0;
        lines.push(sized_line(
            "Indented quoted passage",
            [90.0, y, 500.0, y + 10.0],
            9.0,
        ));
    }
    for row in 0..10 {
        let y = 350.0 + row as f64 * 14.0;
        lines.push(sized_line(
            "Normal narrative prose continues here.",
            [60.0, y, 520.0, y + 12.0],
            11.0,
        ));
    }
    lines.push(sized_line("IV. Rulings", [60.0, 510.0, 220.0, 526.0], 14.0));
    for (index, line) in lines.iter_mut().enumerate() {
        line.region_type = if index < 20 {
            "block_quote"
        } else if index < 30 {
            "body"
        } else {
            "heading"
        }
        .to_owned();
    }
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("Normal"))
        .all(|line| line.region_type == "body"));
    assert_eq!(pages[0].lines.last().unwrap().region_type, "heading");
}

#[test]
fn region_dependent_lanes_require_a_complete_source_contract() {
    let mut pages = vec![test_page(vec![
        sized_line("Known body", [60.0, 100.0, 300.0, 112.0], 11.0),
        sized_line("Unknown peer", [60.0, 120.0, 300.0, 132.0], 11.0),
    ])];
    pages[0].lines[0].region_type = "body".to_owned();
    assert!(!source_regions_available(&pages));

    pages[0].lines[1].region_type = "text".to_owned();
    assert!(source_regions_available(&pages));
}

#[test]
fn source_roles_admit_display_headings_without_promoting_authors() {
    let mut pages = vec![test_page(vec![
        sized_line(
            "CONSTITUTIONAL PRINCIPLES",
            [60.0, 100.0, 360.0, 112.0],
            11.0,
        ),
        sized_line("JANE EXAMPLE", [60.0, 125.0, 240.0, 137.0], 11.0),
        sized_line(
            "Ordinary narrative text ends here.",
            [60.0, 160.0, 520.0, 172.0],
            11.0,
        ),
    ])];
    pages[0].lines[0].region_type = "text".to_owned();
    pages[0].lines[1].region_type = "author".to_owned();
    pages[0].lines[2].region_type = "text".to_owned();

    classify_pages(&mut pages, &[None]);

    assert_eq!(pages[0].lines[0].region_type, "heading");
    assert_eq!(pages[0].lines[1].region_type, "body");
}

#[test]
fn clean_repeated_heading_grammar_promotes_nested_ladders_without_visual_tuning() {
    let mut lines = (0..12)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    for (row, text) in [
        "I. First Part",
        "A. First Issue",
        "B. Second Issue",
        "II. Second Part",
    ]
    .into_iter()
    .enumerate()
    {
        let y = 330.0 + row as f64 * 25.0;
        lines.push(sized_line(text, [60.0, y, 280.0, y + 12.0], 11.0));
    }
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.contains("Part") || line.text.contains("Issue"))
        .all(|line| line.region_type == "heading"));
}

#[test]
fn source_regions_allow_same_style_wrapped_heading_continuations() {
    let mut lines = (0..10)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    let mut heading = sized_line(
        "I. A Complete Account Of",
        [60.0, 300.0, 340.0, 312.0],
        12.0,
    );
    heading.spans[0].flags = 16;
    let mut continuation = sized_line("The Governing Framework", [80.0, 313.0, 360.0, 325.0], 12.0);
    continuation.spans[0].flags = 16;
    lines.extend([
        heading,
        continuation,
        sized_line(
            "Ordinary prose begins after the display heading.",
            [60.0, 345.0, 520.0, 357.0],
            11.0,
        ),
    ]);
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    let heading_lines = pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("I.") || line.text == "The Governing Framework")
        .collect::<Vec<_>>();
    assert_eq!(heading_lines.len(), 2);
    assert!(heading_lines
        .iter()
        .all(|line| line.region_type == "heading"));
    assert_eq!(heading_lines[0].region_id, heading_lines[1].region_id);
}

#[test]
fn dirty_heading_ladder_abstains_instead_of_promoting_examples() {
    let mut lines = (0..10)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    lines.push(sized_line(
        "I. First Part",
        [60.0, 300.0, 280.0, 312.0],
        11.0,
    ));
    lines.push(sized_line(
        "I. Duplicate Part",
        [60.0, 325.0, 300.0, 337.0],
        11.0,
    ));
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("I."))
        .all(|line| line.region_type == "body"));
}

#[test]
fn long_numeric_ladder_is_not_promoted_as_document_headings() {
    let mut lines = (0..10)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    lines.push(sized_line(
        "15. Historical Note",
        [60.0, 300.0, 280.0, 312.0],
        11.0,
    ));
    lines.push(sized_line(
        "16. Further Note",
        [60.0, 325.0, 280.0, 337.0],
        11.0,
    ));
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("15.") || line.text.starts_with("16."))
        .all(|line| line.region_type != "heading"));
}

#[test]
fn body_flow_vetoes_a_visually_bold_false_heading() {
    let mut lines = (0..10)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    let mut candidate = sized_line("I. This Is Actually", [60.0, 300.0, 280.0, 312.0], 11.0);
    candidate.spans[0].flags = 16;
    lines.push(candidate);
    lines.push(sized_line(
        "continued prose from the same sentence.",
        [60.0, 314.0, 520.0, 326.0],
        11.0,
    ));
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert_eq!(
        pages[0]
            .lines
            .iter()
            .find(|line| line.text.starts_with("I."))
            .unwrap()
            .region_type,
        "body"
    );
}

#[test]
fn citation_and_destination_shapes_never_enter_the_heading_ladder() {
    let mut lines = (0..10)
        .map(|row| {
            let y = 80.0 + row as f64 * 18.0;
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            )
        })
        .collect::<Vec<_>>();
    for (row, text) in ["I. Example v Sample 123", "II. Destination 42"]
        .into_iter()
        .enumerate()
    {
        let mut line = sized_line(
            text,
            [
                60.0,
                300.0 + row as f64 * 25.0,
                320.0,
                312.0 + row as f64 * 25.0,
            ],
            11.0,
        );
        line.spans[0].flags = 16;
        lines.push(line);
    }
    mark_source_body(&mut lines);
    let mut pages = vec![test_page(lines)];

    classify_pages(&mut pages, &[None]);

    assert!(pages[0]
        .lines
        .iter()
        .filter(|line| line.text.starts_with("I.") || line.text.starts_with("II."))
        .all(|line| line.region_type == "body"));
}

#[test]
fn endnote_sequence_includes_column_continuations_above_the_next_label() {
    let mut first = test_line("1", [60.0, 100.0, 70.0, 110.0], vec![]);
    first.source_index = 2;
    let mut first_body = test_line("First note", [80.0, 100.0, 300.0, 110.0], vec![]);
    first_body.source_index = 3;
    let mut second_lines = Vec::new();
    for (label, x, y) in [
        (2, 60.0, 100.0),
        (3, 60.0, 200.0),
        (4, 60.0, 300.0),
        (5, 330.0, 100.0),
        (6, 330.0, 200.0),
        (7, 330.0, 300.0),
    ] {
        let mut marker = test_line(&label.to_string(), [x, y, x + 10.0, y + 10.0], vec![]);
        let mut body = test_line("Note body", [x + 20.0, y, x + 240.0, y + 10.0], vec![]);
        marker.page_index = 1;
        marker.page_number = 2;
        body.page_index = 1;
        body.page_number = 2;
        second_lines.extend([marker, body]);
    }
    let mut continuation = test_line(
        "Continuation from the prior column",
        [350.0, 70.0, 570.0, 80.0],
        vec![],
    );
    continuation.page_index = 1;
    continuation.page_number = 2;
    second_lines.insert(6, continuation);
    let mut pages = vec![
        Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines: vec![
                test_line("Notes", [60.0, 70.0, 120.0, 85.0], vec![]),
                first,
                first_body,
            ],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        },
        Page {
            id: "p0002".to_owned(),
            index: 1,
            number: 2,
            width: 600.0,
            height: 800.0,
            lines: second_lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        },
    ];

    classify_pages(&mut pages, &[None, Some(90.0)]);

    assert!(pages[1]
        .lines
        .iter()
        .all(|line| line.note_region_mode == "endnote"));
}

#[test]
fn crossref_shortform_uses_python_word_boundaries_at_join_controls() {
    let text = "\u{200c}Quebec Water Policy, supra note 3";
    let start = text.find("supra").unwrap();
    assert_eq!(crossref_shortform(text, start), "Quebec Water Policy");

    let text = "\u{200c}Godin, supra note 41";
    let start = text.find("supra").unwrap();
    assert_eq!(crossref_shortform(text, start), "Godin");
}
