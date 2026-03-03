//! Line preprocessing: heading merging, drop cap handling, and repeated line removal.

use std::collections::{HashMap, HashSet};

use crate::types::TextLine;

use super::analysis::detect_header_level;

/// Merge consecutive heading lines at the same level into a single line.
///
/// When a heading wraps across multiple text lines (e.g., "About Glenair, the Mission-Critical"
/// and "Interconnect Company"), each fragment becomes a separate `# Header` in the output.
/// This function detects consecutive lines at the same heading tier on the same page
/// with a small Y gap and merges them into one line.
pub(crate) fn merge_heading_lines(
    lines: Vec<TextLine>,
    base_size: f32,
    heading_tiers: &[f32],
) -> Vec<TextLine> {
    if lines.is_empty() {
        return lines;
    }

    let mut result: Vec<TextLine> = Vec::with_capacity(lines.len());

    for line in lines {
        let line_font = line.items.first().map(|i| i.font_size).unwrap_or(base_size);
        let line_level = detect_header_level(line_font, base_size, heading_tiers);

        // Check if the previous line is a heading at the same level on the same page
        let should_merge = if let (Some(prev), Some(curr_level)) = (result.last(), line_level) {
            let prev_font = prev.items.first().map(|i| i.font_size).unwrap_or(base_size);
            let prev_level = detect_header_level(prev_font, base_size, heading_tiers);
            let same_page = prev.page == line.page;
            let same_level = prev_level == Some(curr_level);
            let y_gap = prev.y - line.y;
            // Merge if gap is within ~2x the font size (normal line wrap spacing)
            let close_enough = y_gap > 0.0 && y_gap < line_font * 2.0;
            same_page && same_level && close_enough
        } else {
            false
        };

        if should_merge {
            // Append this line's items to the previous line
            let prev = result.last_mut().unwrap();
            // Add a space-bearing TextItem to separate the merged text
            if let Some(first_item) = line.items.first() {
                let mut space_item = first_item.clone();
                space_item.text = format!(" {}", space_item.text.trim_start());
                prev.items.push(space_item);
            }
            for item in line.items.into_iter().skip(1) {
                prev.items.push(item);
            }
        } else {
            result.push(line);
        }
    }

    result
}

/// Merge drop caps with the appropriate line.
/// A drop cap is a single large letter at the start of a paragraph.
/// Due to PDF coordinate sorting, the drop cap may appear AFTER the line it belongs to.
pub(crate) fn merge_drop_caps(lines: Vec<TextLine>, base_size: f32) -> Vec<TextLine> {
    let mut result: Vec<TextLine> = Vec::with_capacity(lines.len());

    for line in &lines {
        let text = line.text();
        let trimmed = text.trim();

        // Check if this looks like a drop cap:
        // 1. Single character (or single char + space)
        // 2. Much larger than base font (3x or more)
        // 3. The character is uppercase
        let is_drop_cap = trimmed.len() <= 2
            && line.items.first().map(|i| i.font_size).unwrap_or(0.0) >= base_size * 2.5
            && trimmed
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);

        if is_drop_cap {
            let drop_char = trimmed.chars().next().unwrap();

            // Find the first line that starts with lowercase and is at the START of a paragraph
            // (i.e., preceded by a header or non-lowercase-starting line)
            let mut target_idx: Option<usize> = None;

            for (idx, prev_line) in result.iter().enumerate() {
                if prev_line.page != line.page {
                    continue;
                }

                let prev_text = prev_line.text();
                let prev_trimmed = prev_text.trim();

                // Check if this line starts with lowercase
                if prev_trimmed
                    .chars()
                    .next()
                    .map(|c| c.is_lowercase())
                    .unwrap_or(false)
                {
                    // Check if previous line exists and doesn't start with lowercase
                    // (meaning this is the start of a paragraph)
                    let is_para_start = if idx == 0 {
                        true
                    } else {
                        let before = result[idx - 1].text();
                        let before_trimmed = before.trim();
                        !before_trimmed
                            .chars()
                            .next()
                            .map(|c| c.is_lowercase())
                            .unwrap_or(true)
                    };

                    if is_para_start {
                        target_idx = Some(idx);
                        break;
                    }
                }
            }

            // Merge with the target line
            if let Some(idx) = target_idx {
                if let Some(first_item) = result[idx].items.first_mut() {
                    let prev_text = first_item.text.trim().to_string();
                    first_item.text = format!("{}{}", drop_char, prev_text);
                }
            }
            // Don't add the drop cap line itself
            continue;
        }

        result.push(line.clone());
    }

    result
}

/// Normalize whitespace in a string for comparison: trim and collapse internal runs of whitespace.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalize text for frequency comparison: collapse whitespace and strip leading/trailing
/// digit sequences (page numbers). E.g., "Chapter 3 — Page 5" and "Chapter 3 — Page 6"
/// both normalize to "Chapter 3 — Page".
fn normalize_for_comparison(s: &str) -> String {
    let ws = normalize_whitespace(s);
    let trimmed = ws
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .trim_start();
    let trimmed = trimmed
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end();
    trimmed.to_string()
}

/// Returns true if the line looks like a list item or heading (should not be stripped).
fn is_structural_line(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('#')
        || t.starts_with("- ")
        || t.starts_with("* ")
        || t.starts_with("• ")
        || t.chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
            && (t.contains(". ") || t.contains(") "))
}

/// Returns true if a line consists entirely of a single repeated character
/// (e.g., "----------", "**************", "============").
fn is_decorative_separator(text: &str) -> bool {
    let mut chars = text.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    chars.all(|c| c == first)
}

/// Strip lines that repeat on many distinct pages (running headers/footers).
///
/// A line is considered a repeated header/footer if:
/// 1. Its normalized text appears on `>= max(3, page_count * 30%)` distinct pages
/// 2. It is at least 10 characters long
/// 3. It doesn't look like a structural element (heading, list item)
/// 4. It consistently appears in the top or bottom N distinct Y positions
/// 5. Its Y positions across pages have low variance (consistent placement),
///    distinguishing true headers/footers from table content that happens to
///    land near page margins
/// 6. It is not a decorative separator (repeated single character)
///
/// Additionally, TextLines at the same Y position on a page are grouped into
/// "Y-bands." When any member of a Y-band is stripped, all siblings in that
/// band are also stripped. This handles split column headers where individual
/// fragments may not independently meet the frequency threshold.
///
/// Page numbers are stripped from line text before comparison, so headers like
/// "Chapter 3 — Page 5" and "Chapter 3 — Page 6" are treated as the same text.
pub(crate) fn strip_repeated_lines(lines: Vec<TextLine>, page_count: u32) -> Vec<TextLine> {
    if lines.is_empty() || page_count < 3 {
        return lines;
    }

    // Compute Y range per page (min_y, max_y)
    let mut page_y_range: HashMap<u32, (f32, f32)> = HashMap::new();
    for line in &lines {
        let entry = page_y_range.entry(line.page).or_insert((line.y, line.y));
        if line.y < entry.0 {
            entry.0 = line.y;
        }
        if line.y > entry.1 {
            entry.1 = line.y;
        }
    }

    // Build sorted Y values per page, so we can check line rank (position from edge)
    let mut page_sorted_ys: HashMap<u32, Vec<f32>> = HashMap::new();
    for line in &lines {
        page_sorted_ys.entry(line.page).or_default().push(line.y);
    }
    for ys in page_sorted_ys.values_mut() {
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ys.dedup();
    }

    // A line is in the page margin if it's among the first or last N distinct
    // Y positions on that page. This is more robust than a percentage-based zone
    // because it catches actual edge lines regardless of how much content fills
    // the page. N=5 accommodates multi-line headers/footers and repeated form
    // column headers (e.g., 5-row IRS form headers) that sit just inside the
    // page margin.
    const EDGE_LINE_COUNT: usize = 5;

    /// Returns true if the given Y position is among the first or last N distinct
    /// Y positions on the specified page.
    fn is_y_at_edge(y: f32, page: u32, page_sorted_ys: &HashMap<u32, Vec<f32>>, n: usize) -> bool {
        let ys = match page_sorted_ys.get(&page) {
            Some(ys) => ys,
            None => return false,
        };
        if ys.len() <= n * 2 {
            // Page has very few lines — everything is near the edge
            return true;
        }
        // Check if this Y is among the first or last N
        let pos = match ys.iter().position(|&py| (py - y).abs() < 0.1) {
            Some(p) => p,
            None => return false,
        };
        pos < n || pos >= ys.len() - n
    }

    // Average page span for normalizing Y variance
    let avg_span = {
        let total: f32 = page_y_range.values().map(|(lo, hi)| hi - lo).sum();
        if page_y_range.is_empty() {
            1.0
        } else {
            (total / page_y_range.len() as f32).max(1.0)
        }
    };

    // Build Y-bands: group line indices by (page, quantized_y).
    // Lines at the same Y position (within ~0.1pt) on the same page form a band.
    let mut y_bands: HashMap<(u32, i32), Vec<usize>> = HashMap::new();
    for (idx, line) in lines.iter().enumerate() {
        let y_bucket = (line.y * 10.0).round() as i32;
        y_bands.entry((line.page, y_bucket)).or_default().push(idx);
    }

    // Build frequency maps using normalize_for_comparison.
    // Individual line text -> distinct pages
    let mut freq: HashMap<String, HashSet<u32>> = HashMap::new();
    let mut y_positions: HashMap<String, Vec<f32>> = HashMap::new();
    for line in &lines {
        if !is_y_at_edge(line.y, line.page, &page_sorted_ys, EDGE_LINE_COUNT) {
            continue;
        }
        let text = line.text();
        let normalized = normalize_for_comparison(&text);
        if normalized.len() < 10 || is_decorative_separator(&normalized) {
            continue;
        }
        freq.entry(normalized.clone())
            .or_default()
            .insert(line.page);
        y_positions.entry(normalized).or_default().push(line.y);
    }

    // Coalesced row text -> distinct pages (for multi-member Y-bands).
    // This catches split column headers where individual fragments don't meet
    // the frequency threshold but the combined row does.
    let mut band_freq: HashMap<String, HashSet<u32>> = HashMap::new();
    let mut band_y_positions: HashMap<String, Vec<f32>> = HashMap::new();
    for (&(page, _), indices) in &y_bands {
        if indices.len() < 2 {
            continue; // single-line bands are already in the individual map
        }
        let band_y = lines[indices[0]].y;
        if !is_y_at_edge(band_y, page, &page_sorted_ys, EDGE_LINE_COUNT) {
            continue;
        }
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        let coalesced: String = sorted_indices
            .iter()
            .map(|&i| lines[i].text())
            .collect::<Vec<_>>()
            .join(" ");
        let normalized = normalize_for_comparison(&coalesced);
        if normalized.len() < 10 || is_decorative_separator(&normalized) {
            continue;
        }
        band_freq
            .entry(normalized.clone())
            .or_default()
            .insert(page);
        band_y_positions.entry(normalized).or_default().push(band_y);
    }

    // Compute threshold
    let threshold = 3u32.max(page_count * 30 / 100);

    // Check Y-position consistency: headers/footers appear at the same position
    // on every page, table content varies. Require normalized stddev < 5% of
    // average page span.
    let has_consistent_y = |text: &str, positions: &HashMap<String, Vec<f32>>| -> bool {
        let pos = match positions.get(text) {
            Some(p) if p.len() >= 2 => p,
            _ => return true, // single occurrence — allow
        };
        let n = pos.len() as f32;
        let mean = pos.iter().sum::<f32>() / n;
        let variance = pos.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / n;
        let stddev = variance.sqrt();
        stddev / avg_span < 0.05
    };

    // Identify candidates from individual frequency map
    let candidates: HashSet<String> = freq
        .into_iter()
        .filter(|(text, pages)| {
            pages.len() as u32 >= threshold
                && !is_structural_line(text)
                && has_consistent_y(text, &y_positions)
        })
        .map(|(text, _)| text)
        .collect();

    // Identify candidates from coalesced band frequency map
    let band_candidates: HashSet<String> = band_freq
        .into_iter()
        .filter(|(text, pages)| {
            pages.len() as u32 >= threshold
                && !is_structural_line(text)
                && has_consistent_y(text, &band_y_positions)
        })
        .map(|(text, _)| text)
        .collect();

    if candidates.is_empty() && band_candidates.is_empty() {
        return lines;
    }

    // Build removal set.
    // A line is removed if it's at an edge position and:
    //   (a) its individual text matches a candidate, OR
    //   (b) its Y-band's coalesced text matches a band candidate, OR
    //   (c) any sibling in its Y-band was removed (propagation).
    let mut removal_set: HashSet<usize> = HashSet::new();

    // (a) Lines matching individual candidates at edge positions
    for (idx, line) in lines.iter().enumerate() {
        if !is_y_at_edge(line.y, line.page, &page_sorted_ys, EDGE_LINE_COUNT) {
            continue;
        }
        let text = line.text();
        let normalized = normalize_for_comparison(&text);
        if candidates.contains(&normalized) {
            removal_set.insert(idx);
        }
    }

    // (b) Lines in Y-bands whose coalesced text matches a band candidate
    for (&(page, _), indices) in &y_bands {
        if indices.len() < 2 {
            continue;
        }
        let band_y = lines[indices[0]].y;
        if !is_y_at_edge(band_y, page, &page_sorted_ys, EDGE_LINE_COUNT) {
            continue;
        }
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        let coalesced: String = sorted_indices
            .iter()
            .map(|&i| lines[i].text())
            .collect::<Vec<_>>()
            .join(" ");
        let normalized = normalize_for_comparison(&coalesced);
        if band_candidates.contains(&normalized) {
            for &idx in &sorted_indices {
                removal_set.insert(idx);
            }
        }
    }

    // (c) Y-band sibling propagation: if any member is removed, remove all
    //     members (provided the band is at an edge position).
    for (&(page, _), indices) in &y_bands {
        let band_y = lines[indices[0]].y;
        if !is_y_at_edge(band_y, page, &page_sorted_ys, EDGE_LINE_COUNT) {
            continue;
        }
        if indices.iter().any(|idx| removal_set.contains(idx)) {
            for &idx in indices {
                removal_set.insert(idx);
            }
        }
    }

    if removal_set.is_empty() {
        return lines;
    }

    lines
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| !removal_set.contains(idx))
        .map(|(_, line)| line)
        .collect()
}
