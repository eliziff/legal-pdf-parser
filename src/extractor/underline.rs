//! Geometric underline detection.
//!
//! PDFs have no underline font flag — underlines are drawn as separate
//! graphics: stroked horizontal lines (`l`/`S` operators) or thin filled
//! rectangles (`re`/`f`). This pass correlates those graphics with text
//! items after extraction: an item is underlined when a horizontal
//! line/thin rect sits just below its baseline and covers most of its
//! horizontal extent.
//!
//! Repeated same-span rules are treated as table/form rulings rather than
//! underlines, which avoids marking every cell in ruled tables.

use std::collections::HashSet;

use crate::types::{ItemType, PdfRect, TextItem};

/// Max thickness (pt) for a stroked line / filled rect to count as an
/// underline rule rather than a border or decorative band.
const MAX_RULE_THICKNESS: f32 = 2.0;

/// Fraction of the item's width that the rule must cover horizontally.
const MIN_X_OVERLAP: f32 = 0.6;

/// Same-span rules repeated at this many y-levels are usually table/form
/// rulings, not semantic underlines.
const MIN_REPEATED_RULE_LEVELS: usize = 3;

/// Vertical tolerance for considering two rules to be on the same row edge.
const RULE_Y_DEDUP_EPS: f32 = 2.0;

/// Horizontal span similarity required when clustering repeated rulings.
const RULE_SPAN_OVERLAP_RATIO: f32 = 0.8;
const RULE_SPAN_WIDTH_RATIO: f32 = 1.5;

/// Multiple separated rule segments on one row are usually per-column table
/// header/body separators.
const MIN_SEGMENTED_ROW_RULES: usize = 3;
const MIN_SEGMENTED_ROW_GAPS: usize = 2;
const SEGMENTED_ROW_GAP_MIN: f32 = 12.0;

/// A single rule under several widely separated items is usually a table
/// header/body separator, not a sentence underline.
const MIN_TABULAR_RULE_ITEMS: usize = 3;
const MIN_TABULAR_RULE_GAPS: usize = 2;
const TABULAR_RULE_GAP_EM: f32 = 2.0;

#[derive(Clone)]
pub(crate) struct UnderlineLine {
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) x2: f32,
    pub(crate) y2: f32,
    pub(crate) stroke_width: f32,
    pub(crate) page: u32,
}

/// A horizontal rule candidate in page coordinates (PDF y-up).
#[derive(Clone)]
struct Rule {
    x1: f32,
    x2: f32,
    y: f32,
}

impl Rule {
    fn width(&self) -> f32 {
        self.x2 - self.x1
    }
}

fn rules_from_graphics(rects: &[PdfRect], lines: &[UnderlineLine], page: u32) -> Vec<Rule> {
    let mut rules: Vec<Rule> = Vec::new();
    for l in lines {
        if l.page != page {
            continue;
        }
        // Horizontal stroked line (tolerate slight skew).
        if l.stroke_width <= MAX_RULE_THICKNESS && (l.y1 - l.y2).abs() <= MAX_RULE_THICKNESS {
            let (x1, x2) = if l.x1 <= l.x2 {
                (l.x1, l.x2)
            } else {
                (l.x2, l.x1)
            };
            if x2 - x1 > 1.0 {
                rules.push(Rule {
                    x1,
                    x2,
                    y: (l.y1 + l.y2) / 2.0,
                });
            }
        }
    }
    for r in rects {
        if r.page != page {
            continue;
        }
        // Thin filled rect used as an underline rule. Extents are
        // normalized first: `re` operands pass through the CTM, so
        // width/height can be negative (flipped axes / negative scale) —
        // without normalization negative-width rules are missed and
        // negative-height bands sneak past the thickness check.
        let (x1, x2) = if r.width >= 0.0 {
            (r.x, r.x + r.width)
        } else {
            (r.x + r.width, r.x)
        };
        if r.height.abs() <= MAX_RULE_THICKNESS && x2 - x1 > 1.0 {
            rules.push(Rule {
                x1,
                x2,
                y: r.y + r.height / 2.0,
            });
        }
    }
    rules
}

fn discard_repeated_ruling_rules(rules: Vec<Rule>) -> Vec<Rule> {
    if rules.len() < MIN_REPEATED_RULE_LEVELS {
        return rules;
    }

    rules
        .iter()
        .filter(|rule| {
            !is_repeated_ruling_rule(rule, &rules) && !is_segmented_row_ruling_rule(rule, &rules)
        })
        .cloned()
        .collect()
}

fn is_repeated_ruling_rule(rule: &Rule, rules: &[Rule]) -> bool {
    let mut y_levels: Vec<f32> = rules
        .iter()
        .filter(|other| has_similar_span(rule, other))
        .map(|other| other.y)
        .collect();

    y_levels.sort_by(|a, b| a.total_cmp(b));
    y_levels.dedup_by(|a, b| (*a - *b).abs() <= RULE_Y_DEDUP_EPS);
    y_levels.len() >= MIN_REPEATED_RULE_LEVELS
}

fn is_segmented_row_ruling_rule(rule: &Rule, rules: &[Rule]) -> bool {
    let mut row_rules: Vec<&Rule> = rules
        .iter()
        .filter(|other| (other.y - rule.y).abs() <= RULE_Y_DEDUP_EPS)
        .collect();

    if row_rules.len() < MIN_SEGMENTED_ROW_RULES {
        return false;
    }

    row_rules.sort_by(|a, b| a.x1.total_cmp(&b.x1));
    let large_gaps = row_rules
        .windows(2)
        .filter(|pair| pair[1].x1 - pair[0].x2 > SEGMENTED_ROW_GAP_MIN)
        .count();

    large_gaps >= MIN_SEGMENTED_ROW_GAPS
}

fn has_similar_span(a: &Rule, b: &Rule) -> bool {
    let a_width = a.width();
    let b_width = b.width();
    if a_width <= 1.0 || b_width <= 1.0 {
        return false;
    }

    let width_ratio = a_width.max(b_width) / a_width.min(b_width);
    if width_ratio > RULE_SPAN_WIDTH_RATIO {
        return false;
    }

    let overlap = a.x2.min(b.x2) - a.x1.max(b.x1);
    overlap >= a_width.min(b_width) * RULE_SPAN_OVERLAP_RATIO
}

fn tabular_row_separator_rule_indices(rules: &[Rule], items: &[TextItem]) -> HashSet<usize> {
    let mut tabular_rules = HashSet::new();

    for (rule_idx, rule) in rules.iter().enumerate() {
        let mut matched_items: Vec<&TextItem> = items
            .iter()
            .filter(|item| is_underline_candidate(item) && rule_matches_item(rule, item))
            .collect();

        if matched_items.len() < MIN_TABULAR_RULE_ITEMS {
            continue;
        }

        matched_items.sort_by(|a, b| a.x.total_cmp(&b.x));
        let large_gaps = matched_items
            .windows(2)
            .filter(|pair| {
                let left = pair[0];
                let right = pair[1];
                let gap = right.x - (left.x + left.width);
                let font_size = left.font_size.max(right.font_size).max(1.0);
                gap > font_size * TABULAR_RULE_GAP_EM
            })
            .count();

        if large_gaps >= MIN_TABULAR_RULE_GAPS {
            tabular_rules.insert(rule_idx);
        }
    }

    tabular_rules
}

fn is_underline_candidate(item: &TextItem) -> bool {
    matches!(item.item_type, ItemType::Text) && !item.text.trim().is_empty() && item.width > 0.0
}

fn rule_matches_item(rule: &Rule, item: &TextItem) -> bool {
    // Vertical window: underlines sit at or slightly below the baseline.
    // Fonts draw them at roughly 5-15% of the em below; allow up to 35%
    // (min 3pt) below and 1pt above for rounding.
    let below = (item.font_size * 0.35).max(3.0);
    let y_min = item.y - below;
    let y_max = item.y + 1.0;
    if rule.y < y_min || rule.y > y_max {
        return false;
    }

    let ix1 = item.x;
    let ix2 = item.x + item.width;
    let min_overlap = item.width * MIN_X_OVERLAP;
    let overlap = rule.x2.min(ix2) - rule.x1.max(ix1);
    overlap >= min_overlap
}

/// Strikeout window: a rule crossing the glyphs. Strikethroughs sit at
/// roughly 20-35% of the em above the baseline (about half the x-height);
/// accept a band well inside the glyph body so baseline underlines and
/// overlines never qualify.
fn rule_strikes_item(rule: &Rule, item: &TextItem) -> bool {
    let y_min = item.y + item.font_size * 0.12;
    let y_max = item.y + item.font_size * 0.55;
    if rule.y < y_min || rule.y > y_max {
        return false;
    }

    let ix1 = item.x;
    let ix2 = item.x + item.width;
    let min_overlap = item.width * MIN_X_OVERLAP;
    let overlap = rule.x2.min(ix2) - rule.x1.max(ix1);
    overlap >= min_overlap
}

/// Mark `is_underline` on text items that have a horizontal rule just
/// below their baseline, and `is_strikeout` on items whose glyphs a rule
/// crosses at mid x-height. `items`, `rects`, and `lines` are a single
/// page's extraction output (all in PDF coordinates, y-up, where
/// `TextItem::y` is the text baseline).
pub(crate) fn mark_underlined_items(
    items: &mut [TextItem],
    rects: &[PdfRect],
    lines: &[UnderlineLine],
    page: u32,
) {
    let rules = discard_repeated_ruling_rules(rules_from_graphics(rects, lines, page));
    if rules.is_empty() {
        return;
    }
    let tabular_rules = tabular_row_separator_rule_indices(&rules, items);

    for item in items.iter_mut() {
        if !is_underline_candidate(item) {
            continue;
        }

        for (rule_idx, rule) in rules.iter().enumerate() {
            if tabular_rules.contains(&rule_idx) {
                continue;
            }
            if rule_matches_item(rule, item) {
                item.is_underline = true;
            }
            if rule_strikes_item(rule, item) {
                item.is_strikeout = true;
            }
            if item.is_underline && item.is_strikeout {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ItemType;

    fn item(text: &str, x: f32, y: f32, width: f32, font_size: f32) -> TextItem {
        TextItem {
            text: text.to_string(),
            x,
            y,
            width,
            height: font_size,
            font: "F1".to_string(),
            font_size,
            page: 1,
            is_bold: false,
            is_italic: false,
            is_underline: false,
            is_strikeout: false,
            item_type: ItemType::Text,
            mcid: None,
        }
    }

    fn hline(x1: f32, x2: f32, y: f32) -> UnderlineLine {
        UnderlineLine {
            x1,
            y1: y,
            x2,
            y2: y,
            stroke_width: 1.0,
            page: 1,
        }
    }

    fn thin_rect(x: f32, y: f32, width: f32) -> PdfRect {
        PdfRect {
            x,
            y,
            width,
            height: 0.8,
            page: 1,
        }
    }

    #[test]
    fn stroked_line_under_baseline_marks_underline() {
        let mut items = vec![item("underlined", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(99.0, 161.0, 498.5)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(items[0].is_underline);
    }

    #[test]
    fn thin_filled_rect_under_baseline_marks_underline() {
        let mut items = vec![item("underlined", 100.0, 500.0, 60.0, 10.0)];
        let rects = vec![thin_rect(100.0, 497.8, 60.0)];
        mark_underlined_items(&mut items, &rects, &[], 1);
        assert!(items[0].is_underline);
    }

    #[test]
    fn long_rule_under_multiple_items_marks_each() {
        // One underline drawn under a whole sentence: every overlapped
        // item gets the flag.
        let mut items = vec![
            item("first", 100.0, 500.0, 40.0, 10.0),
            item("second", 145.0, 500.0, 50.0, 10.0),
        ];
        let lines = vec![hline(98.0, 200.0, 498.0)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(items[0].is_underline);
        assert!(items[1].is_underline);
    }

    #[test]
    fn line_far_below_baseline_is_not_an_underline() {
        // A horizontal rule 30pt below (section divider) must not mark.
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(90.0, 300.0, 470.0)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn thick_stroked_line_is_not_an_underline() {
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let mut line = hline(99.0, 161.0, 498.5);
        line.stroke_width = 4.0;

        mark_underlined_items(&mut items, &[], &[line], 1);

        assert!(!items[0].is_underline);
    }

    #[test]
    fn mid_glyph_rule_marks_strikeout_not_underline() {
        // Rule at ~30% of the em above the baseline crosses the glyphs.
        let mut items = vec![item("struck out", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(99.0, 161.0, 503.0)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(items[0].is_strikeout);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn baseline_rule_marks_underline_not_strikeout() {
        let mut items = vec![item("underlined", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(99.0, 161.0, 498.5)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(items[0].is_underline);
        assert!(!items[0].is_strikeout);
    }

    #[test]
    fn overline_is_neither_underline_nor_strikeout() {
        // Rule just above the cap height (overline / next line's rule).
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(99.0, 161.0, 507.0)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(!items[0].is_underline);
        assert!(!items[0].is_strikeout);
    }

    #[test]
    fn thin_filled_rect_at_mid_glyph_marks_strikeout() {
        let mut items = vec![item("struck out", 100.0, 500.0, 60.0, 10.0)];
        let rects = vec![thin_rect(100.0, 502.6, 60.0)];
        mark_underlined_items(&mut items, &rects, &[], 1);
        assert!(items[0].is_strikeout);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn line_above_baseline_is_not_an_underline() {
        // Strikethrough / overline geometry must not mark.
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![hline(90.0, 300.0, 505.0)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn insufficient_horizontal_overlap_is_not_an_underline() {
        // Rule under only a quarter of the item (e.g. neighboring cell
        // border) must not mark.
        let mut items = vec![item("wide text item", 100.0, 500.0, 100.0, 10.0)];
        let lines = vec![hline(100.0, 125.0, 498.5)];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn negative_width_rect_is_normalized_and_marks_underline() {
        // A CTM with negative x-scale (or negative `re` operands) produces
        // rects whose width is negative; the rule extents must normalize.
        let mut items = vec![item("underlined", 100.0, 500.0, 60.0, 10.0)];
        let rects = vec![PdfRect {
            x: 160.0,
            y: 497.8,
            width: -60.0,
            height: 0.8,
            page: 1,
        }];
        mark_underlined_items(&mut items, &rects, &[], 1);
        assert!(items[0].is_underline);
    }

    #[test]
    fn negative_height_band_is_not_an_underline() {
        // A 14pt band expressed with negative height must not pass the
        // thickness check via sign trickery.
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let rects = vec![PdfRect {
            x: 95.0,
            y: 509.0,
            width: 80.0,
            height: -14.0,
            page: 1,
        }];
        mark_underlined_items(&mut items, &rects, &[], 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn thick_band_is_not_an_underline() {
        // A highlight bar / filled cell background (tall rect) must not mark.
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let rects = vec![PdfRect {
            x: 95.0,
            y: 495.0,
            width: 80.0,
            height: 14.0,
            page: 1,
        }];
        mark_underlined_items(&mut items, &rects, &[], 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn vertical_line_is_not_an_underline() {
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let lines = vec![UnderlineLine {
            x1: 120.0,
            y1: 498.0,
            x2: 120.0,
            y2: 400.0,
            stroke_width: 1.0,
            page: 1,
        }];
        mark_underlined_items(&mut items, &[], &lines, 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn other_pages_graphics_do_not_mark() {
        let mut items = vec![item("text", 100.0, 500.0, 60.0, 10.0)];
        let mut line = hline(99.0, 161.0, 498.5);
        line.page = 2;
        mark_underlined_items(&mut items, &[], &[line], 1);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn repeated_table_row_rules_do_not_mark_cell_text() {
        let mut items = vec![
            item("A", 110.0, 500.0, 20.0, 10.0),
            item("B", 110.0, 480.0, 20.0, 10.0),
            item("C", 110.0, 460.0, 20.0, 10.0),
        ];
        let lines = vec![
            hline(100.0, 150.0, 498.0),
            hline(100.0, 150.0, 478.0),
            hline(100.0, 150.0, 458.0),
        ];

        mark_underlined_items(&mut items, &[], &lines, 1);

        assert!(items.iter().all(|item| !item.is_underline));
    }

    #[test]
    fn row_separator_under_spaced_column_labels_is_not_an_underline() {
        let mut items = vec![
            item("Date", 100.0, 500.0, 25.0, 10.0),
            item("Rate", 200.0, 500.0, 25.0, 10.0),
            item("Yield", 300.0, 500.0, 30.0, 10.0),
        ];
        let lines = vec![hline(90.0, 340.0, 498.0)];

        mark_underlined_items(&mut items, &[], &lines, 1);

        assert!(items.iter().all(|item| !item.is_underline));
    }

    #[test]
    fn same_row_spaced_rule_segments_do_not_mark_column_labels() {
        let mut items = vec![
            item("Date", 100.0, 500.0, 25.0, 10.0),
            item("Rate", 200.0, 500.0, 25.0, 10.0),
            item("Yield", 300.0, 500.0, 30.0, 10.0),
        ];
        let lines = vec![
            hline(98.0, 128.0, 498.0),
            hline(198.0, 228.0, 498.0),
            hline(298.0, 333.0, 498.0),
        ];

        mark_underlined_items(&mut items, &[], &lines, 1);

        assert!(items.iter().all(|item| !item.is_underline));
    }
}
