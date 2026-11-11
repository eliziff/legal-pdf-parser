//! The document's stylesheet, as DOC reads it: a legal document is set from a handful of templates (a look, where the
//! line stands, the numbering it opens on), and what one line of a template is, the template's other lines mostly are.
//! Each template is measured over all its lines: how many and over how many pages, whether its markers count on, how
//! many of its lines the document's bookmarks name (each entry found as a line of the page it names, `anchors`) and at
//! what level, how far it repeats at the pages' edges, how far it stands beside another line on its row (a marginal
//! note's lane). The templates the bookmarks name are ordered among themselves (one above another where its occurrences
//! restart the other's count and enclose its occurrences, by its look's weight where neither tells), the bookmarks'
//! levels fixing the templates they name.

use super::*;

pub struct Stylesheet {
    /// Each line's template, in reading order.
    pub template: Vec<u32>,
    pub templates: Vec<TemplateView>,
    /// Each line the outline names: its level there (0 for none).
    pub anchor: Vec<u16>,
}

/// What the stylesheet measures of a template.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemplateView {
    pub lines: f32,
    pub pages: f32,
    pub monotone: f32,
    pub anchored: f32,
    pub anchor_level: f32,
    pub level: f32,
    pub edge: f32,
    /// The share of its lines set beside another line on their row (a marginal note's lane).
    pub beside: f32,
}

/// A template row's width (`Stylesheet::rows`), as DOC's stylesheet stage reads it.
pub const ROW: usize = 20;

/// A line feature's place among `LINE_FEATURES`.
pub(crate) fn feature(name: &str) -> usize {
    LINE_FEATURES.iter().position(|n| *n == name).expect("a line feature")
}

/// The commonest of the counted keys (the least of those as common), 0 for none.
fn commonest<K: Copy + Ord + Into<f32>>(counts: &HashMap<K, u32>) -> f32 {
    counts.iter().max_by_key(|(key, n)| (**n, std::cmp::Reverse(**key))).map_or(0.0, |(key, _)| (*key).into())
}

/// Each line's template, numbered from 0 in order of first appearance: its look, where it stands (at
/// the margin, indented a little or more, centred, set right) and the numbering family its marker
/// counts in.
fn template_ids(table: &LineTable, measures: &Measures) -> Vec<u32> {
    let (indent, centre, width, right) = (feature("indent"), feature("centre"), feature("width"), feature("right_gap"));
    let counts: Vec<usize> = ["count_1", "count_1.1", "count_1.1.1", "count_1.1.1.1", "count_a", "count_A", "count_i", "count_I", "count_bullet"]
        .iter().map(|name| feature(name)).collect();
    let mut ids: HashMap<(u32, u8, u8), u32> = HashMap::new();
    (0..table.readings.len()).map(|k| {
        let row = measures.row(k);
        let place = if row[centre].abs() < 0.04 && row[width] < 0.85 { 3 } else if row[indent] > 0.4 && row[right] < 0.05 { 4 }
            else if row[indent] < 0.02 { 0 } else if row[indent] < 0.08 { 1 } else { 2 };
        let family = counts.iter().position(|&c| row[c] > 0.5).map_or(9, |c| c as u8);
        let next = ids.len() as u32;
        *ids.entry((table.looks[k], place, family)).or_insert(next)
    }).collect()
}

impl Stylesheet {
    /// The stylesheet of the lines `table` reads.
    #[allow(clippy::too_many_arguments)]
    pub fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &crate::decide::Geometry, outline: &[PdfOutlineEntry], table: &LineTable, measures: &Measures) -> Self {
        let template = template_ids(table, measures);
        let n = table.readings.len();
        let (size, bold, upper, edge, beside) = (feature("size"), feature("bold"), feature("upper"), feature("edge_repeated"), feature("beside"));
        let count = template.iter().map(|&t| t as usize + 1).max().unwrap_or(0);
        // The document's bookmarks, as lines: its printed contents the decision links itself.
        let anchoring = crate::decide::anchors(pages, slots, evidence, fences, geometry, &fences.role, outline, &[]);
        let mut anchor = vec![0u16; n];
        for found in &anchoring.anchors {
            for slot in &found.lines {
                anchor[fences.place[slot.index()] as usize] = found.level.max(1);
            }
        }
        let mut views = vec![TemplateView::default(); count];
        let mut page_sets: Vec<HashSet<usize>> = vec![HashSet::new(); count];
        let mut anchor_levels: Vec<HashMap<u16, u32>> = vec![HashMap::new(); count];
        let mut last_value: Vec<Option<(String, u32)>> = vec![None; count];
        let mut counted = vec![(0u32, 0u32); count];
        let mut weight = vec![0f32; count];
        for k in 0..n {
            let t = template[k] as usize;
            let row = measures.row(k);
            let view = &mut views[t];
            view.lines += 1.0;
            view.edge += row[edge];
            view.beside += row[beside];
            page_sets[t].insert(slots.page(fences.order[k]));
            if anchor[k] > 0 {
                view.anchored += 1.0;
                *anchor_levels[t].entry(anchor[k]).or_default() += 1;
            }
            if let Some(reading) = table.readings[k].first() {
                if let Some((family, value)) = &last_value[t] {
                    counted[t].1 += 1;
                    counted[t].0 += u32::from(*family == reading.family && reading.value == value + 1);
                }
                last_value[t] = Some((reading.family.clone(), reading.value));
            }
            weight[t] += row[size] * 4.0 + row[bold] * 2.0 + row[upper];
        }
        for (t, view) in views.iter_mut().enumerate() {
            let lines = view.lines.max(1.0);
            view.edge /= lines;
            view.beside /= lines;
            view.anchored /= lines;
            weight[t] /= lines;
            view.monotone = if counted[t].1 > 0 { counted[t].0 as f32 / counted[t].1 as f32 } else { -1.0 };
            view.anchor_level = commonest(&anchor_levels[t]);
            view.pages = page_sets[t].len() as f32;
        }
        let levels = order_templates(&template, table, &views, &weight);
        for (view, level) in views.iter_mut().zip(levels) {
            view.level = level;
            view.lines = (1.0 + view.lines).ln();
            view.pages = (1.0 + view.pages).ln();
        }
        Self { template, templates: views, anchor }
    }

    /// Each template's row as DOC's stylesheet stage reads it: its measures, the mean of its lines' look and place, the
    /// share of them opening on a marker, and where in the document it first stands. Three cells hold what the export
    /// was fitted on and no reader now gives: the share of its lines the walker's local scores held headings, which
    /// were uniform over the classes (summed as they were), and how the walker's first reading read it, none.
    pub fn rows(&self, table: &LineTable, measures: &Measures) -> Vec<[f32; ROW]> {
        let count = self.templates.len();
        let at = ["size", "bold", "upper", "centre", "indent", "length", "width"].map(feature);
        let n = table.readings.len();
        let uniform = (1.0 / CLASSES as f64) as f32;
        let (mut sums, mut lines, mut heading, mut first) = (vec![[0f32; 8]; count], vec![0f32; count], vec![0f32; count], vec![None; count]);
        for k in 0..n {
            let t = self.template[k] as usize;
            let row = measures.row(k);
            sums[t].iter_mut().zip(&at).for_each(|(sum, &f)| *sum += row[f]);
            sums[t][7] += f32::from(u8::from(!table.readings[k].is_empty()));
            lines[t] += 1.0;
            heading[t] += uniform;
            first[t].get_or_insert(k);
        }
        (0..count).map(|t| {
            let v = &self.templates[t];
            let mut r = [0f32; ROW];
            r[..11].copy_from_slice(&[v.lines, v.pages, heading[t] / lines[t].max(1.0), v.monotone, v.anchored, v.anchor_level, v.level, v.edge, v.beside, 0.0, 0.0]);
            r[11..19].iter_mut().zip(sums[t]).for_each(|(out, sum)| *out = sum / lines[t].max(1.0));
            r[19] = first[t].map_or(1.0, |k| k as f32 / n.max(1) as f32);
            r
        }).collect()
    }
}

/// The heading levels the templates the bookmarks name take among themselves (0 for one they do not): template a
/// stands above b where, after a's occurrences, b's count starts over at one more than the other way round, and where
/// b's occurrences fall between a's more than a's between b's; where neither tells, the weightier look stands above. A
/// template the bookmarks name mostly at one level keeps it.
fn order_templates(template: &[u32], table: &LineTable, views: &[TemplateView], weight: &[f32]) -> Vec<f32> {
    let count = views.len();
    let candidates: Vec<usize> = (0..count).filter(|&t| views[t].anchored >= 0.3).collect();
    if candidates.is_empty() { return vec![0.0; count]; }
    let index: HashMap<usize, usize> = candidates.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let m = candidates.len();
    // Restarts: after an occurrence of a, whether b's next marker reads 1; and nesting: whether b
    // occurs between two occurrences of a.
    let mut restarts = vec![vec![(0u32, 0u32); m]; m];
    let mut between = vec![vec![0u32; m]; m];
    let mut seen_since: Vec<Vec<bool>> = vec![vec![false; m]; m];
    let mut after: Vec<Vec<bool>> = vec![vec![false; m]; m];
    let mut occurred = vec![false; m];
    for k in 0..template.len() {
        let Some(&b) = index.get(&(template[k] as usize)) else { continue };
        let first = table.readings[k].first().map(|r| r.value);
        for a in (0..m).filter(|&a| a != b && occurred[a]) {
            if after[a][b] {
                if let Some(value) = first {
                    restarts[a][b].1 += 1;
                    restarts[a][b].0 += u32::from(value == 1);
                }
                after[a][b] = false;
            }
            seen_since[a][b] = true;
        }
        for other in 0..m {
            between[b][other] += u32::from(other != b && seen_since[b][other]);
            (seen_since[b][other], after[b][other]) = (false, true);
        }
        occurred[b] = true;
    }
    let share = |(hit, all): (u32, u32)| if all == 0 { 0.0 } else { hit as f32 / all as f32 };
    let above = |a: usize, b: usize| -> f32 {
        let reset = share(restarts[a][b]) - share(restarts[b][a]);
        let nest = (between[a][b] as f32 - between[b][a] as f32) / (between[a][b] + between[b][a]).max(1) as f32;
        let look = (weight[candidates[a]] - weight[candidates[b]]).clamp(-1.0, 1.0) * 0.25;
        reset + 0.5 * nest + look
    };
    // Each candidate's standing: how many it stands above, less how many above it.
    let mut standing: Vec<(f32, usize)> = (0..m).map(|a| ((0..m).filter(|&b| b != a).map(|b| above(a, b).signum()).sum(), a)).collect();
    standing.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    let mut levels = vec![0.0; count];
    let (mut level, mut previous) = (0.0, f32::INFINITY);
    for &(score, a) in &standing {
        if score < previous { (level, previous) = (level + 1.0, score); }
        levels[candidates[a]] = level;
    }
    // The bookmarks' levels stand where they name a template.
    candidates.iter().filter(|&&t| views[t].anchor_level > 0.0).for_each(|&t| levels[t] = views[t].anchor_level);
    levels
}
