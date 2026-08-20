use crate::ppdoc::PPDocDetection;
use legal_pdf_core::model::{Line, Page};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

// Exact production-default port of Text-Fidelity-Project's
// tools/ocr/layout_regioning/ppdoc/region_postprocess.py at clean HEAD
// d8b25257687b3b9aad644dec42cca966b45675ff. Optional experimental rules that
// were disabled by that project's production adapter are intentionally absent.
const BLOCK_QUOTE_THRESHOLD: f64 = 0.68;
const BLOCK_QUOTE_CONTEXT_THRESHOLD: f64 = 0.45;
const BLOCK_QUOTE_MIN_LINES: usize = 2;
const OVERLAP_THRESHOLD: f64 = 0.35;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionDetection {
    pub(crate) label: String,
    pub(crate) score: f32,
    pub(crate) bbox: [f64; 4],
    pub(crate) raw_index: usize,
    pub(crate) order: usize,
}

#[derive(Debug, Clone, Default)]
struct RegionContext {
    line_count: usize,
    text: String,
    line_bboxes: Vec<[f64; 4]>,
}

#[derive(Debug, Clone)]
struct TextRow {
    region_index: usize,
    raw_index: usize,
    bbox: [f64; 4],
    column: usize,
    width: f64,
    line_count: usize,
    position: usize,
    footnote_spillover: bool,
    region_score: f64,
    is_inset: bool,
}

pub(crate) fn scale_detections(
    page_width: f64,
    page_height: f64,
    image_width: u32,
    image_height: u32,
    detections: &[PPDocDetection],
) -> Vec<RegionDetection> {
    if page_width <= 0.0 || page_height <= 0.0 || image_width == 0 || image_height == 0 {
        return Vec::new();
    }
    let x_scale = page_width / f64::from(image_width);
    let y_scale = page_height / f64::from(image_height);
    let mut regions = detections
        .iter()
        .enumerate()
        .map(|(index, detection)| RegionDetection {
            label: detection.label.clone(),
            score: detection.score,
            bbox: [
                f64::from(detection.bbox[0]) * x_scale,
                f64::from(detection.bbox[1]) * y_scale,
                f64::from(detection.bbox[2]) * x_scale,
                f64::from(detection.bbox[3]) * y_scale,
            ],
            raw_index: index + 1,
            order: index,
        })
        .collect::<Vec<_>>();
    order_regions(&mut regions, page_width, page_height);
    regions
}

pub(crate) fn postprocess_document(pages: &[Page], regions_by_page: &mut [Vec<RegionDetection>]) {
    debug_assert_eq!(pages.len(), regions_by_page.len());
    let contexts: Vec<HashMap<usize, RegionContext>> = pages
        .iter()
        .zip(regions_by_page.iter())
        .map(|(page, regions)| region_contexts(&page.lines, regions))
        .collect();

    for ((page, regions), page_contexts) in pages
        .iter()
        .zip(regions_by_page.iter_mut())
        .zip(contexts.iter())
    {
        apply_block_quote_heuristic(page, regions, page_contexts);
        apply_hard_validity(page, regions, page_contexts);
    }
    apply_byline_filter(pages, regions_by_page);
    apply_repeat_headers_footers(pages, regions_by_page, &contexts);
    apply_edge_digits(pages, regions_by_page, &contexts);
    apply_sequenced_edge_digits(pages, regions_by_page, &contexts);
    apply_roman_titles(regions_by_page, &contexts);
    for ((page, regions), page_contexts) in pages
        .iter()
        .zip(regions_by_page.iter_mut())
        .zip(contexts.iter())
    {
        apply_footnote_sandwich(page, regions);
        apply_top_footnotes(page, regions);
        apply_full_width_block_quote_demotion(page, regions, page_contexts);
        apply_overlap_cleanup(regions);
        order_regions(regions, page.width, page.height);
    }
}

pub(crate) fn best_region_index(line: [f64; 4], regions: &[RegionDetection]) -> Option<usize> {
    let line_area = bbox_area(line);
    if line_area == 0.0 {
        return None;
    }
    let center = bbox_center(line);
    let containing = regions
        .iter()
        .enumerate()
        .filter(|(_, region)| contains_point(region.bbox, center))
        .min_by(|(_, left), (_, right)| {
            bbox_area(left.bbox)
                .total_cmp(&bbox_area(right.bbox))
                .then_with(|| left.raw_index.cmp(&right.raw_index))
        })
        .map(|(index, _)| index);
    if containing.is_some() {
        return containing;
    }

    let best = regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            let intersection = intersection_area(line, region.bbox);
            (intersection > 0.0).then_some((
                index,
                intersection / line_area.max(1.0),
                intersection,
                region.raw_index,
            ))
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.2.total_cmp(&right.2))
                .then_with(|| right.3.cmp(&left.3))
        });
    best.filter(|candidate| candidate.1 >= 0.10)
        .map(|candidate| candidate.0)
}

fn region_contexts(lines: &[Line], regions: &[RegionDetection]) -> HashMap<usize, RegionContext> {
    let mut contexts: HashMap<usize, RegionContext> = regions
        .iter()
        .map(|region| (region.raw_index, RegionContext::default()))
        .collect();
    for line in lines {
        if line.exclude_from_body || line.text.trim().is_empty() || bbox_area(line.bbox) == 0.0 {
            continue;
        }
        let Some(region_index) = best_region_index(line.bbox, regions) else {
            continue;
        };
        let context = contexts
            .get_mut(&regions[region_index].raw_index)
            .expect("every region has a context");
        context.line_count += 1;
        context.line_bboxes.push(line.bbox);
        if !context.text.is_empty() {
            context.text.push('\n');
        }
        context.text.push_str(line.text.trim());
    }
    contexts
}

fn apply_block_quote_heuristic(
    page: &Page,
    regions: &mut [RegionDetection],
    contexts: &HashMap<usize, RegionContext>,
) {
    let footnotes: Vec<([f64; 4], usize)> = regions
        .iter()
        .filter(|region| is_footnote(&region.label))
        .map(|region| (region.bbox, region_column(region, page.width)))
        .collect();
    let mut rows: Vec<TextRow> = regions
        .iter()
        .enumerate()
        .filter(|(_, region)| region.label == "text")
        .map(|(region_index, region)| {
            let column = region_column(region, page.width);
            let spillover = region.score <= 0.25
                && footnote_after_distance(region.bbox, column, page.height, &footnotes)
                    .is_some_and(|distance| distance <= (page.height * 0.006).max(10.0));
            TextRow {
                region_index,
                raw_index: region.raw_index,
                bbox: region.bbox,
                column,
                width: bbox_width(region.bbox),
                line_count: contexts
                    .get(&region.raw_index)
                    .map_or(0, |context| context.line_count),
                position: 0,
                footnote_spillover: spillover,
                region_score: 0.0,
                is_inset: false,
            }
        })
        .collect();
    if rows.len() < 3 {
        return;
    }
    rows.sort_by(|left, right| {
        region_order_values(left.bbox, "text", left.raw_index, page.height).cmp(
            &region_order_values(right.bbox, "text", right.raw_index, page.height),
        )
    });
    for (position, row) in rows.iter_mut().enumerate() {
        row.position = position;
    }

    let mut refs = HashMap::<usize, (f64, f64, f64)>::new();
    for column in 0..=1 {
        let column_rows: Vec<&TextRow> = rows.iter().filter(|row| row.column == column).collect();
        if column_rows.len() < 3 {
            continue;
        }
        refs.insert(
            column,
            (
                percentile(column_rows.iter().map(|row| row.bbox[0]), 0.10, 0.0),
                percentile(
                    column_rows.iter().map(|row| row.bbox[2]),
                    0.90,
                    page.width.max(1.0),
                ),
                percentile(
                    column_rows.iter().map(|row| row.width),
                    0.90,
                    page.width.max(1.0),
                ),
            ),
        );
    }

    let page_width = page.width.max(1.0);
    let mut scored_positions = Vec::new();
    for row in &mut rows {
        if row.footnote_spillover {
            continue;
        }
        let Some((left, right, body_width)) = refs.get(&row.column).copied() else {
            continue;
        };
        let left_score = clamp01((row.bbox[0] - left) / (page_width * 0.035).max(22.0));
        let right_score = clamp01((right - row.bbox[2]) / (page_width * 0.030).max(22.0));
        let narrow_score = clamp01((body_width - row.width) / (body_width * 0.18).max(22.0));
        row.region_score = 0.50 * narrow_score + 0.35 * left_score + 0.15 * right_score;
        row.is_inset = row.region_score >= BLOCK_QUOTE_CONTEXT_THRESHOLD;
        if row.is_inset {
            scored_positions.push(row.position);
        }
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    for position in scored_positions {
        let extend = groups
            .last()
            .and_then(|group| group.last())
            .is_some_and(|previous| {
                rows[*previous].column == rows[position].column && position == *previous + 1
            });
        if extend {
            groups.last_mut().expect("group exists").push(position);
        } else {
            groups.push(vec![position]);
        }
    }

    let mut relabel = HashSet::new();
    for group in groups {
        let mut scores: Vec<f64> = group
            .iter()
            .map(|position| rows[*position].region_score)
            .collect();
        let group_score =
            median(&mut scores, 0.0) + ((group.len().saturating_sub(1) as f64) * 0.025).min(0.10);
        let start = group[0];
        let end = *group.last().expect("group is nonempty");
        let before = start.checked_sub(1).and_then(|position| rows.get(position));
        let after = rows.get(end + 1);
        let column = rows[start].column;
        let body_before = before.is_some_and(|row| row.column == column && !row.is_inset);
        let body_after = after.is_some_and(|row| row.column == column && !row.is_inset);
        let surrounded = body_before && body_after;
        let adjacent = body_before || body_after;
        let group_bbox = group.iter().fold(
            [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ],
            |mut bbox, position| {
                let row = &rows[*position];
                bbox[0] = bbox[0].min(row.bbox[0]);
                bbox[1] = bbox[1].min(row.bbox[1]);
                bbox[2] = bbox[2].max(row.bbox[2]);
                bbox[3] = bbox[3].max(row.bbox[3]);
                bbox
            },
        );
        let footnote_boundary = body_before
            && footnote_after_distance(group_bbox, column, page.height, &footnotes).is_some();
        let total_lines: usize = group
            .iter()
            .map(|position| rows[*position].line_count)
            .sum();
        if total_lines < BLOCK_QUOTE_MIN_LINES {
            continue;
        }
        if total_lines <= 2 && (group_score < BLOCK_QUOTE_THRESHOLD + 0.12 || !surrounded) {
            continue;
        }
        let required = if surrounded || footnote_boundary {
            BLOCK_QUOTE_CONTEXT_THRESHOLD
        } else {
            BLOCK_QUOTE_THRESHOLD
        };
        if group_score < required {
            continue;
        }
        let strong_edge = adjacent
            && group_score >= BLOCK_QUOTE_THRESHOLD + 0.08
            && total_lines >= BLOCK_QUOTE_MIN_LINES.max(3);
        if !surrounded && !footnote_boundary && !strong_edge {
            continue;
        }
        relabel.extend(group.iter().map(|position| rows[*position].raw_index));
    }

    for row in rows {
        if row.footnote_spillover {
            regions[row.region_index].label = "footnote".to_owned();
        } else if relabel.contains(&row.raw_index) {
            regions[row.region_index].label = "block_quote".to_owned();
        }
    }
    order_regions(regions, page.width, page.height);
}

fn apply_hard_validity(
    page: &Page,
    regions: &mut [RegionDetection],
    contexts: &HashMap<usize, RegionContext>,
) {
    for region in regions {
        let context = contexts.get(&region.raw_index);
        let text = context.map_or("", |value| value.text.as_str());
        let compact = digit_text(text);
        if region.label == "number"
            && (in_middle_half(region, page.height)
                || (!compact.is_empty() && !compact.chars().all(|ch| ch.is_ascii_digit())))
        {
            region.label = "text".to_owned();
        } else if matches!(region.label.as_str(), "header" | "footer")
            && in_middle_half(region, page.height)
        {
            region.label = "text".to_owned();
        } else if matches!(region.label.as_str(), "doc_title" | "abstract") && page.index + 1 > 3 {
            region.label = "text".to_owned();
        } else if region.label == "block_quote"
            && context.is_some_and(|value| (1..=2).contains(&value.line_count))
        {
            region.label = if heading_style_corroborated(text) {
                "paragraph_title"
            } else {
                "text"
            }
            .to_owned();
        }
    }
}

fn apply_byline_filter(pages: &[Page], regions_by_page: &mut [Vec<RegionDetection>]) {
    let page_count = pages.len();
    for (page, regions) in pages.iter().zip(regions_by_page.iter_mut()) {
        let page_number = page.index + 1;
        if page_number <= 3 || page_number == page_count {
            continue;
        }
        for region in regions.iter_mut().filter(|region| region.label == "byline") {
            region.label = "text".to_owned();
        }
    }
}

fn apply_repeat_headers_footers(
    pages: &[Page],
    regions_by_page: &mut [Vec<RegionDetection>],
    contexts: &[HashMap<usize, RegionContext>],
) {
    let mut groups = BTreeMap::<(bool, String), Vec<(usize, usize)>>::new();
    for (page_position, ((page, regions), page_contexts)) in pages
        .iter()
        .zip(regions_by_page.iter())
        .zip(contexts.iter())
        .enumerate()
    {
        for region in regions {
            if !repeat_candidate(&region.label) || in_middle_half(region, page.height) {
                continue;
            }
            let ratio = region_y_ratio(region, page.height);
            let top = if ratio <= 0.20 {
                true
            } else if ratio >= 0.80 {
                false
            } else {
                continue;
            };
            let key = repeat_text_key(
                page_contexts
                    .get(&region.raw_index)
                    .map_or("", |context| context.text.as_str()),
            );
            if !key.is_empty() {
                groups
                    .entry((top, key))
                    .or_default()
                    .push((page_position, region.raw_index));
            }
        }
    }
    for ((top, _), rows) in groups {
        let distinct: HashSet<usize> = rows.iter().map(|row| row.0).collect();
        if distinct.len() < 3 || distinct.len() as f64 / (pages.len().max(1) as f64) < 0.40 {
            continue;
        }
        for (page_position, raw_index) in rows {
            if let Some(region) = region_by_raw_mut(&mut regions_by_page[page_position], raw_index)
            {
                region.label = if top { "header" } else { "footer" }.to_owned();
            }
        }
    }
}

fn apply_edge_digits(
    pages: &[Page],
    regions_by_page: &mut [Vec<RegionDetection>],
    contexts: &[HashMap<usize, RegionContext>],
) {
    for ((page, regions), page_contexts) in pages
        .iter()
        .zip(regions_by_page.iter_mut())
        .zip(contexts.iter())
    {
        for region in regions {
            let ratio = region_y_ratio(region, page.height);
            if candidate_number(&region.label)
                && (ratio <= 0.20 || ratio >= 0.80)
                && exact_page_digits(
                    page_contexts
                        .get(&region.raw_index)
                        .map_or("", |context| context.text.as_str()),
                )
                .is_some()
            {
                region.label = "number".to_owned();
            }
        }
    }
}

fn apply_sequenced_edge_digits(
    pages: &[Page],
    regions_by_page: &mut [Vec<RegionDetection>],
    contexts: &[HashMap<usize, RegionContext>],
) {
    let mut groups = BTreeMap::<(bool, i64), Vec<(usize, usize)>>::new();
    for (page_position, ((page, regions), page_contexts)) in pages
        .iter()
        .zip(regions_by_page.iter())
        .zip(contexts.iter())
        .enumerate()
    {
        for region in regions {
            if !candidate_number(&region.label) {
                continue;
            }
            let ratio = region_y_ratio(region, page.height);
            let top = if ratio <= 0.40 {
                true
            } else if ratio >= 0.60 {
                false
            } else {
                continue;
            };
            let Some(digits) = exact_page_digits(
                page_contexts
                    .get(&region.raw_index)
                    .map_or("", |context| context.text.as_str()),
            ) else {
                continue;
            };
            groups
                .entry((top, digits - (page.index + 1) as i64))
                .or_default()
                .push((page_position, region.raw_index));
        }
    }
    for (_, rows) in groups {
        let distinct: HashSet<usize> = rows.iter().map(|row| row.0).collect();
        if distinct.len() < 3 {
            continue;
        }
        for (page_position, raw_index) in rows {
            if let Some(region) = region_by_raw_mut(&mut regions_by_page[page_position], raw_index)
            {
                region.label = "number".to_owned();
            }
        }
    }
}

fn apply_roman_titles(
    regions_by_page: &mut [Vec<RegionDetection>],
    contexts: &[HashMap<usize, RegionContext>],
) {
    for (regions, page_contexts) in regions_by_page.iter_mut().zip(contexts.iter()) {
        for region in regions {
            if is_footnote(&region.label)
                || is_page_number(&region.label)
                || is_visual(&region.label)
            {
                continue;
            }
            let Some(context) = page_contexts.get(&region.raw_index) else {
                continue;
            };
            if (1..=2).contains(&context.line_count)
                && roman_title_text(&context.text)
                && heading_style_corroborated(&context.text)
            {
                region.label = "paragraph_title".to_owned();
            }
        }
    }
}

fn apply_footnote_sandwich(page: &Page, regions: &mut [RegionDetection]) {
    let mut rows_by_column = [Vec::<usize>::new(), Vec::<usize>::new()];
    let mut ordered: Vec<usize> = (0..regions.len()).collect();
    ordered.sort_by(|left, right| {
        region_column(&regions[*left], page.width)
            .cmp(&region_column(&regions[*right], page.width))
            .then_with(|| regions[*left].bbox[1].total_cmp(&regions[*right].bbox[1]))
            .then_with(|| regions[*left].bbox[0].total_cmp(&regions[*right].bbox[0]))
            .then_with(|| regions[*left].raw_index.cmp(&regions[*right].raw_index))
    });
    for index in ordered {
        if region_y_ratio(&regions[index], page.height) >= 0.60 {
            rows_by_column[region_column(&regions[index], page.width)].push(index);
        }
    }
    let mut relabel = HashSet::new();
    for rows in rows_by_column {
        for start in 0..rows.len().saturating_sub(2) {
            if !is_footnote(&regions[rows[start]].label)
                || !is_footnote(&regions[rows[start + 1]].label)
            {
                continue;
            }
            let mut middle = Vec::<usize>::new();
            for candidate in rows.iter().skip(start + 2).copied() {
                if is_footnote(&regions[candidate].label) {
                    relabel.extend(
                        middle.iter().copied().filter(|index| {
                            !is_footnote_sandwich_protected(&regions[*index].label)
                        }),
                    );
                    break;
                }
                middle.push(candidate);
            }
        }
    }
    for index in relabel {
        regions[index].label = "footnote".to_owned();
    }
}

fn apply_top_footnotes(page: &Page, regions: &mut [RegionDetection]) {
    for region in regions {
        if is_footnote(&region.label) && region_y_ratio(region, page.height) <= 0.30 {
            region.label = "text".to_owned();
        }
    }
}

fn apply_full_width_block_quote_demotion(
    page: &Page,
    regions: &mut [RegionDetection],
    contexts: &HashMap<usize, RegionContext>,
) {
    let mut ordered: Vec<usize> = (0..regions.len()).collect();
    ordered.sort_by(|left, right| region_order_cmp(&regions[*left], &regions[*right], page.height));
    let footnote_columns: HashSet<usize> = regions
        .iter()
        .filter(|region| is_footnote(&region.label))
        .map(|region| region_column(region, page.width))
        .collect();
    let mut refs = HashMap::<usize, (f64, f64, f64, f64)>::new();
    for column in 0..=1 {
        let body: Vec<usize> = ordered
            .iter()
            .copied()
            .filter(|index| {
                let region = &regions[*index];
                let y = region_y_ratio(region, page.height);
                region_column(region, page.width) == column
                    && matches!(region.label.as_str(), "text" | "content")
                    && contexts
                        .get(&region.raw_index)
                        .is_some_and(|context| context.line_count >= 1)
                    && (0.08..=0.92).contains(&y)
            })
            .collect();
        if body.len() < 3 {
            continue;
        }
        let line_heights = body.iter().flat_map(|index| {
            contexts
                .get(&regions[*index].raw_index)
                .into_iter()
                .flat_map(|context| context.line_bboxes.iter().copied())
                .map(bbox_height)
        });
        refs.insert(
            column,
            (
                percentile(body.iter().map(|index| regions[*index].bbox[0]), 0.10, 0.0),
                percentile(
                    body.iter().map(|index| regions[*index].bbox[2]),
                    0.90,
                    page.width.max(1.0),
                ),
                percentile(
                    body.iter().map(|index| bbox_width(regions[*index].bbox)),
                    0.90,
                    page.width.max(1.0),
                ),
                percentile(line_heights, 0.50, 0.0),
            ),
        );
    }

    let mut rows_by_column = [Vec::<usize>::new(), Vec::<usize>::new()];
    for index in ordered {
        rows_by_column[region_column(&regions[index], page.width)].push(index);
    }
    let page_width = page.width.max(1.0);
    let mut relabel = Vec::new();
    for (column, rows) in rows_by_column.iter().enumerate() {
        let Some((body_left, body_right, body_width, body_line_height)) =
            refs.get(&column).copied()
        else {
            continue;
        };
        for (position, index) in rows.iter().copied().enumerate() {
            let region = &regions[index];
            if region.label != "block_quote" {
                continue;
            }
            let Some(context) = contexts.get(&region.raw_index) else {
                continue;
            };
            let y0_ratio = region.bbox[1] / page.height.max(1.0);
            if context.line_count < 3
                || (page.index + 1 <= 2 && y0_ratio < 0.35)
                || (y0_ratio > 0.60 && footnote_columns.contains(&column))
            {
                continue;
            }
            let before = position.checked_sub(1).map(|value| rows[value]);
            let after = rows.get(position + 1).copied();
            if ![before, after]
                .into_iter()
                .flatten()
                .any(|neighbor| matches!(regions[neighbor].label.as_str(), "text" | "content"))
            {
                continue;
            }
            let before_text = before
                .and_then(|neighbor| contexts.get(&regions[neighbor].raw_index))
                .map_or("", |context| context.text.as_str());
            if before.is_some_and(|neighbor| {
                matches!(regions[neighbor].label.as_str(), "text" | "content")
            }) && before_text.trim_end().ends_with(':')
                && !list_item_text(&context.text)
            {
                continue;
            }
            let candidate_line_height = percentile(
                context.line_bboxes.iter().copied().map(bbox_height),
                0.50,
                0.0,
            );
            if body_line_height > 0.0
                && candidate_line_height > 0.0
                && candidate_line_height < body_line_height * 0.95
            {
                continue;
            }
            let width = bbox_width(region.bbox);
            let left_score = clamp01((region.bbox[0] - body_left) / (page_width * 0.035).max(22.0));
            let right_score =
                clamp01((body_right - region.bbox[2]) / (page_width * 0.030).max(22.0));
            let narrow_score = clamp01((body_width - width) / (body_width * 0.18).max(22.0));
            let inset_score = 0.50 * narrow_score + 0.35 * left_score + 0.15 * right_score;
            let width_ratio = width / body_width.max(1.0);
            let left_delta = (region.bbox[0] - body_left).abs() / page_width;
            let right_delta = (region.bbox[2] - body_right).abs() / page_width;
            if inset_score <= 0.18
                && (0.88..=1.22).contains(&width_ratio)
                && left_delta <= 0.055
                && right_delta <= 0.070
            {
                relabel.push(index);
            }
        }
    }
    for index in relabel {
        regions[index].label = "text".to_owned();
    }
}

fn apply_overlap_cleanup(regions: &mut Vec<RegionDetection>) {
    let mut removed = HashSet::new();
    for left_index in 0..regions.len() {
        let left_raw = regions[left_index].raw_index;
        if removed.contains(&left_raw) {
            continue;
        }
        for right_index in left_index + 1..regions.len() {
            let right_raw = regions[right_index].raw_index;
            if removed.contains(&right_raw) {
                continue;
            }
            let intersection =
                intersection_area(regions[left_index].bbox, regions[right_index].bbox);
            let smaller =
                bbox_area(regions[left_index].bbox).min(bbox_area(regions[right_index].bbox));
            if intersection <= 0.0 || smaller <= 0.0 || intersection / smaller < OVERLAP_THRESHOLD {
                continue;
            }
            let left_wins =
                overlap_rank_cmp(&regions[left_index], &regions[right_index]) != Ordering::Greater;
            let loser_index = if left_wins { right_index } else { left_index };
            let loser_coverage = intersection / bbox_area(regions[loser_index].bbox).max(1.0);
            if loser_coverage < OVERLAP_THRESHOLD {
                continue;
            }
            removed.insert(regions[loser_index].raw_index);
            if loser_index == left_index {
                break;
            }
        }
    }
    regions.retain(|region| !removed.contains(&region.raw_index));
}

fn order_regions(regions: &mut [RegionDetection], width: f64, height: f64) {
    let _ = width;
    regions.sort_by(|left, right| region_order_cmp(left, right, height));
    for (order, region) in regions.iter_mut().enumerate() {
        region.order = order;
    }
}

fn region_order_cmp(left: &RegionDetection, right: &RegionDetection, height: f64) -> Ordering {
    region_order_values(left.bbox, &left.label, left.raw_index, height).cmp(&region_order_values(
        right.bbox,
        &right.label,
        right.raw_index,
        height,
    ))
}

#[derive(Debug)]
struct RegionOrderValues(usize, f64, f64, f64, usize);

impl PartialEq for RegionOrderValues {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RegionOrderValues {}

impl PartialOrd for RegionOrderValues {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionOrderValues {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
            .then_with(|| self.2.total_cmp(&other.2))
            .then_with(|| self.3.total_cmp(&other.3))
            .then_with(|| self.4.cmp(&other.4))
    }
}

fn region_order_values(
    bbox: [f64; 4],
    label: &str,
    raw_index: usize,
    height: f64,
) -> RegionOrderValues {
    let height = height.max(1.0);
    let y_ratio = (bbox[1] + bbox[3]) * 0.5 / height;
    let y0_ratio = bbox[1] / height;
    let y1_ratio = bbox[3] / height;
    let group = if matches!(label, "header" | "footer" | "number" | "formula_number")
        && y_ratio < 0.20
    {
        0
    } else if is_footnote(label)
        && (y0_ratio > 0.68 || (y0_ratio < 0.20 && y_ratio > 0.42 && y1_ratio > 0.55))
    {
        2
    } else if matches!(label, "header" | "footer" | "number" | "formula_number") && y_ratio > 0.45 {
        3
    } else {
        1
    };
    RegionOrderValues(
        group,
        bbox[1],
        (bbox[0] + bbox[2]) * 0.5,
        bbox[0],
        raw_index,
    )
}

fn overlap_rank_cmp(left: &RegionDetection, right: &RegionDetection) -> Ordering {
    overlap_priority(&left.label)
        .cmp(&overlap_priority(&right.label))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| bbox_area(right.bbox).total_cmp(&bbox_area(left.bbox)))
}

fn overlap_priority(label: &str) -> usize {
    match label {
        "footnote" => 0,
        "reference_content" => 1,
        "vision_footnote" => 2,
        "reference" => 3,
        "block_quote" => 4,
        "paragraph_title" => 5,
        "doc_title" => 6,
        "byline" => 7,
        "text" => 8,
        "content" => 9,
        "vertical_text" => 10,
        "header" => 11,
        "footer" => 12,
        "number" => 13,
        "formula_number" => 14,
        _ => 999,
    }
}

fn region_by_raw_mut(
    regions: &mut [RegionDetection],
    raw_index: usize,
) -> Option<&mut RegionDetection> {
    regions
        .iter_mut()
        .find(|region| region.raw_index == raw_index)
}

fn bbox_area(bbox: [f64; 4]) -> f64 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn bbox_width(bbox: [f64; 4]) -> f64 {
    (bbox[2] - bbox[0]).max(0.0)
}

fn bbox_height(bbox: [f64; 4]) -> f64 {
    (bbox[3] - bbox[1]).max(0.0)
}

fn bbox_center(bbox: [f64; 4]) -> (f64, f64) {
    ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
}

fn contains_point(bbox: [f64; 4], point: (f64, f64)) -> bool {
    bbox[0] <= point.0 && point.0 <= bbox[2] && bbox[1] <= point.1 && point.1 <= bbox[3]
}

fn intersection_area(left: [f64; 4], right: [f64; 4]) -> f64 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0)
}

fn region_y_ratio(region: &RegionDetection, height: f64) -> f64 {
    bbox_center(region.bbox).1 / height.max(1.0)
}

fn region_column(region: &RegionDetection, width: f64) -> usize {
    usize::from(width > 0.0 && bbox_center(region.bbox).0 > width * 0.54)
}

fn in_middle_half(region: &RegionDetection, height: f64) -> bool {
    (0.25..=0.75).contains(&region_y_ratio(region, height))
}

fn horizontal_overlap_ratio(left: [f64; 4], right: [f64; 4]) -> f64 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        / bbox_width(left).min(bbox_width(right)).max(1.0)
}

fn footnote_after_distance(
    bbox: [f64; 4],
    column: usize,
    height: f64,
    footnotes: &[([f64; 4], usize)],
) -> Option<f64> {
    footnotes
        .iter()
        .filter(|(candidate, candidate_column)| {
            let gap = candidate[1] - bbox[3];
            *candidate_column == column
                && (-12.0..=(height * 0.035).max(90.0)).contains(&gap)
                && horizontal_overlap_ratio(bbox, *candidate) >= 0.08
        })
        .map(|(candidate, _)| candidate[1] - bbox[3])
        .min_by(f64::total_cmp)
}

fn percentile(values: impl IntoIterator<Item = f64>, fraction: f64, default: f64) -> f64 {
    let mut values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return default;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * clamp01(fraction)).round_ties_even() as usize;
    values[index]
}

fn median(values: &mut [f64], default: f64) -> f64 {
    if values.is_empty() {
        return default;
    }
    values.sort_by(f64::total_cmp);
    if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn digit_text(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn exact_page_digits(text: &str) -> Option<i64> {
    let compact = digit_text(text);
    ((1..=3).contains(&compact.len()) && compact.chars().all(|ch| ch.is_ascii_digit()))
        .then(|| compact.parse().expect("one to three ASCII digits"))
}

fn repeat_text_key(text: &str) -> String {
    let lowercase = text.to_ascii_lowercase();
    let tokens: Vec<&str> = ascii_words_regex()
        .find_iter(&lowercase)
        .map(|matched| matched.as_str())
        .collect();
    if tokens.iter().map(|token| token.len()).sum::<usize>() < 6 {
        String::new()
    } else {
        tokens.join(" ")
    }
}

fn roman_title_text(text: &str) -> bool {
    roman_title_regex().is_match(text)
}

fn list_item_text(text: &str) -> bool {
    list_item_regex().is_match(text)
}

fn heading_style_corroborated(text: &str) -> bool {
    let compact = whitespace_regex().replace_all(text, " ");
    let compact = compact.trim();
    if compact.is_empty() || compact.len() > 90 || footnote_marker_regex().is_match(compact) {
        return false;
    }
    let letters: Vec<char> = compact.chars().filter(|ch| ch.is_alphabetic()).collect();
    if !letters.is_empty() && letters.iter().all(|ch| ch.is_uppercase()) {
        return true;
    }
    if !heading_enumerator_regex().is_match(compact) || compact.ends_with('.') {
        return false;
    }
    let tokens: Vec<&str> = heading_token_regex()
        .find_iter(compact)
        .map(|matched| matched.as_str())
        .collect();
    !tokens.is_empty()
        && tokens
            .iter()
            .filter(|token| token.starts_with(|ch: char| ch.is_ascii_uppercase()))
            .count() as f64
            / tokens.len() as f64
            >= 0.60
}

fn candidate_number(label: &str) -> bool {
    !is_footnote(label) && !is_visual(label) && label != "number"
}

fn repeat_candidate(label: &str) -> bool {
    !is_footnote(label) && !is_page_number(label) && !is_visual(label)
}

fn is_footnote(label: &str) -> bool {
    matches!(
        label,
        "footnote" | "vision_footnote" | "reference" | "reference_content"
    )
}

fn is_page_number(label: &str) -> bool {
    matches!(label, "number" | "formula_number")
}

fn is_visual(label: &str) -> bool {
    matches!(
        label,
        "image" | "chart" | "header_image" | "footer_image" | "seal" | "table"
    )
}

fn is_footnote_sandwich_protected(label: &str) -> bool {
    is_footnote(label) || matches!(label, "header" | "footer" | "number" | "formula_number")
}

fn regex_once(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static PPdoc regex is valid"))
}

fn whitespace_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"\s+")
}

fn ascii_words_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"[a-z]+")
}

fn roman_title_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"(?i)^\s*(?:I|II)[).]\s+")
}

fn list_item_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"^\s*(?:[•*.-]|\(?[A-Za-z0-9]{1,3}[).])\s+")
}

fn footnote_marker_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"^\s*(?:\(?\d{1,3}\)?[.)]?|[*†‡§¶]+)\s+\S")
}

fn heading_enumerator_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"^\s*(?:[IVXLC]{1,6}|[A-Za-z]|\d{1,2})\s*[.)]\s+\S")
}

fn heading_token_regex() -> &'static Regex {
    static VALUE: OnceLock<Regex> = OnceLock::new();
    regex_once(&VALUE, r"[A-Za-z][A-Za-z'`-]+")
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{Line, Page};

    fn region(label: &str, raw_index: usize, bbox: [f64; 4]) -> RegionDetection {
        RegionDetection {
            label: label.to_owned(),
            score: 0.9,
            bbox,
            raw_index,
            order: raw_index,
        }
    }

    fn line(id: usize, text: &str, bbox: [f64; 4]) -> Line {
        Line {
            id: format!("l{id}"),
            page_index: 0,
            page_number: 1,
            source_index: id,
            reading_order: id,
            block_index: id,
            text: text.to_owned(),
            bbox,
            spans: Vec::new(),
            words: Vec::new(),
            detached_references: Vec::new(),
            exclude_from_body: false,
            suppress_footnote_label: false,
            note_region_mode: String::new(),
            region_id: String::new(),
            region_type: "unknown".to_owned(),
            source: "native".to_owned(),
        }
    }

    fn page(index: usize, regions: &[RegionDetection], texts: &[(usize, &[&str])]) -> Page {
        let mut lines = Vec::new();
        for (raw_index, chunks) in texts {
            let item = regions
                .iter()
                .find(|region| region.raw_index == *raw_index)
                .unwrap();
            for (offset, text) in chunks.iter().enumerate() {
                let y0 = item.bbox[1] + 8.0 + offset as f64 * 16.0;
                lines.push(line(
                    lines.len() + 1,
                    text,
                    [
                        item.bbox[0] + 5.0,
                        y0,
                        item.bbox[2] - 5.0,
                        (y0 + 10.0).min(item.bbox[3] - 2.0),
                    ],
                ));
            }
        }
        Page {
            id: format!("p{}", index + 1),
            index,
            number: (index + 1) as u32,
            width: 1000.0,
            height: 1000.0,
            lines,
            regions: Vec::new(),
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    }

    fn label(regions: &[RegionDetection], raw_index: usize) -> Option<&str> {
        regions
            .iter()
            .find(|region| region.raw_index == raw_index)
            .map(|region| region.label.as_str())
    }

    #[test]
    fn assignment_matches_python_smallest_container_then_ten_percent_overlap() {
        let regions = vec![
            region("image", 1, [0.0, 0.0, 100.0, 100.0]),
            region("paragraph_title", 2, [10.0, 18.0, 90.0, 32.0]),
        ];
        assert_eq!(
            best_region_index([12.0, 20.0, 88.0, 30.0], &regions),
            Some(1)
        );
        assert_eq!(
            best_region_index([101.0, 0.0, 111.0, 100.0], &regions),
            None
        );
    }

    #[test]
    fn production_rules_match_text_fidelity_validity_and_heading_fixtures() {
        let regions = vec![
            region("number", 1, [100.0, 450.0, 900.0, 490.0]),
            region("number", 2, [100.0, 50.0, 900.0, 80.0]),
            region("number", 3, [20.0, 50.0, 80.0, 80.0]),
            region("doc_title", 4, [100.0, 120.0, 900.0, 170.0]),
            region("abstract", 5, [100.0, 180.0, 900.0, 240.0]),
            region("block_quote", 6, [100.0, 300.0, 900.0, 360.0]),
            region("block_quote", 7, [100.0, 380.0, 900.0, 420.0]),
            region("text", 8, [100.0, 500.0, 900.0, 550.0]),
        ];
        let page = page(
            3,
            &regions,
            &[
                (1, &["123"]),
                (2, &["12A"]),
                (3, &["42"]),
                (4, &["Later running title"]),
                (5, &["Late abstract"]),
                (6, &["Short quote line 1", "Short quote line 2"]),
                (7, &["PART TWO"]),
                (8, &["II. Background"]),
            ],
        );
        let mut all = vec![regions];
        postprocess_document(&[page], &mut all);
        assert_eq!(label(&all[0], 1), Some("text"));
        assert_eq!(label(&all[0], 2), Some("text"));
        assert_eq!(label(&all[0], 3), Some("number"));
        assert_eq!(label(&all[0], 4), Some("text"));
        assert_eq!(label(&all[0], 5), Some("text"));
        assert_eq!(label(&all[0], 6), Some("text"));
        assert_eq!(label(&all[0], 7), Some("paragraph_title"));
        assert_eq!(label(&all[0], 8), Some("paragraph_title"));
    }

    #[test]
    fn production_rules_match_cross_page_repeat_and_sequence_fixtures() {
        let mut pages = Vec::new();
        let mut all = Vec::new();
        for index in 0..3 {
            let regions = vec![
                region("text", 1, [100.0, 70.0, 900.0, 110.0]),
                region("text", 2, [50.0, 340.0, 120.0, 370.0]),
                region("text", 3, [700.0, 340.0, 760.0, 370.0]),
            ];
            pages.push(page(
                index,
                &regions,
                &[
                    (1, &[&format!("{} Alberta Law Review", 723 + index)]),
                    (2, &[&format!("{}", 723 + index)]),
                    (3, &["99"]),
                ],
            ));
            all.push(regions);
        }
        postprocess_document(&pages, &mut all);
        assert!(all
            .iter()
            .all(|regions| label(regions, 1) == Some("header")));
        assert!(all
            .iter()
            .all(|regions| label(regions, 2) == Some("number")));
        assert!(all.iter().all(|regions| label(regions, 3) == Some("text")));
    }

    #[test]
    fn production_rules_match_three_line_inset_block_quote_fixture() {
        let regions = vec![
            region("text", 1, [100.0, 100.0, 900.0, 160.0]),
            region("text", 2, [100.0, 200.0, 900.0, 260.0]),
            region("text", 3, [135.0, 300.0, 860.0, 368.0]),
            region("text", 4, [100.0, 600.0, 900.0, 660.0]),
        ];
        let page = page(
            0,
            &regions,
            &[
                (1, &["Body paragraph before the quote runs full measure."]),
                (2, &["Another full-measure body paragraph sits here."]),
                (
                    3,
                    &[
                        "inset quote line 1",
                        "inset quote line 2",
                        "inset quote line 3",
                    ],
                ),
                (4, &["Body paragraph after the quote runs full measure."]),
            ],
        );
        let mut all = vec![regions];
        postprocess_document(&[page], &mut all);
        assert_eq!(label(&all[0], 3), Some("block_quote"));
    }

    #[test]
    fn production_rules_demote_full_measure_block_quote_fixture() {
        let regions = vec![
            region("text", 1, [100.0, 100.0, 900.0, 160.0]),
            region("text", 2, [100.0, 200.0, 900.0, 260.0]),
            region("block_quote", 3, [100.0, 300.0, 900.0, 368.0]),
            region("text", 4, [100.0, 600.0, 900.0, 660.0]),
        ];
        let page = page(
            2,
            &regions,
            &[
                (
                    1,
                    &["Body paragraph before the candidate runs full measure."],
                ),
                (2, &["Another full-measure body paragraph sits here."]),
                (
                    3,
                    &["ordinary line 1", "ordinary line 2", "ordinary line 3"],
                ),
                (
                    4,
                    &["Body paragraph after the candidate runs full measure."],
                ),
            ],
        );
        let mut all = vec![regions];
        postprocess_document(&[page], &mut all);
        assert_eq!(label(&all[0], 3), Some("text"));
    }

    #[test]
    fn production_rules_match_footnote_sandwich_fixture() {
        let regions = vec![
            region("footnote", 1, [100.0, 80.0, 900.0, 120.0]),
            region("footnote", 2, [100.0, 620.0, 900.0, 650.0]),
            region("footnote", 3, [100.0, 660.0, 900.0, 690.0]),
            region("text", 4, [100.0, 700.0, 900.0, 730.0]),
            region("footer", 5, [100.0, 830.0, 900.0, 850.0]),
            region("footnote", 6, [100.0, 880.0, 900.0, 910.0]),
        ];
        let page = page(
            0,
            &regions,
            &[
                (1, &["not really a note"]),
                (2, &["1 First note"]),
                (3, &["2 Second note"]),
                (4, &["continued note text"]),
                (5, &["Journal footer"]),
                (6, &["3 Third note"]),
            ],
        );
        let mut all = vec![regions];
        postprocess_document(&[page], &mut all);
        assert_eq!(label(&all[0], 1), Some("text"));
        assert_eq!(label(&all[0], 4), Some("footnote"));
        assert_eq!(label(&all[0], 5), Some("footer"));
    }

    #[test]
    fn production_rules_match_overlap_priority_fixture() {
        let regions = vec![
            region("text", 1, [100.0, 600.0, 900.0, 760.0]),
            region("footnote", 2, [100.0, 620.0, 900.0, 740.0]),
        ];
        let page = page(
            0,
            &regions,
            &[(1, &["body text"]), (2, &["1 Footnote text"])],
        );
        let mut all = vec![regions];
        postprocess_document(&[page], &mut all);
        assert_eq!(label(&all[0], 1), None);
        assert_eq!(label(&all[0], 2), Some("footnote"));
    }
}
