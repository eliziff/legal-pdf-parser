//! The document's stylesheet, as DOC reads it: a legal document is set from a handful of templates (a look, where the
//! line stands, the numbering it opens on), and what one line of a template is, the template's other lines mostly are.
//! Each template is measured over all its lines: how many and over how many pages, whether its markers count on, how
//! many of its lines the document's outline names (the lines t0's decision found its entries on) and at what level (a
//! template's level where a third of its lines or more are named), how far it is t0's furniture, how far it stands
//! beside another line on its row (a marginal note's lane).

use crate::model::machine::lines::Measures;
use crate::model::machine::lines::LineTable;
use crate::model::machine::lines::LINE_FEATURES;
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

/// An outline anchor's levels DOC reads (0 for none): a deeper level reads as the deepest.
pub const ANCHORS: u16 = 10;

/// A line feature's place among `LINE_FEATURES`.
pub fn feature(name: &str) -> usize {
    LINE_FEATURES
        .iter()
        .position(|n| *n == name)
        .expect("a line feature")
}

/// The commonest of the counted keys (the least of those as common), 0 for none.
fn commonest<K: Copy + Ord + Into<f32>>(counts: &HashMap<K, u32>) -> f32 {
    counts
        .iter()
        .max_by_key(|(key, n)| (**n, std::cmp::Reverse(**key)))
        .map_or(0.0, |(key, _)| (*key).into())
}

/// Each line's template, numbered from 0 in order of first appearance: its look, where it stands (at
/// the margin, indented a little or more, centred, set right) and the numbering family its marker
/// counts in.
fn template_ids(table: &LineTable, measures: &Measures) -> Vec<u32> {
    let (indent, centre, width, right) = (
        feature("indent"),
        feature("centre"),
        feature("width"),
        feature("right_gap"),
    );
    let counts: Vec<usize> = [
        "count_1",
        "count_1.1",
        "count_1.1.1",
        "count_1.1.1.1",
        "count_a",
        "count_A",
        "count_i",
        "count_I",
        "count_bullet",
    ]
    .iter()
    .map(|name| feature(name))
    .collect();
    let mut ids: HashMap<(u32, u8, u8), u32> = HashMap::new();
    (0..table.readings.len())
        .map(|k| {
            let row = measures.row(k);
            let place = if row[centre].abs() < 0.04 && row[width] < 0.85 {
                3
            } else if row[indent] > 0.4 && row[right] < 0.05 {
                4
            } else if row[indent] < 0.02 {
                0
            } else if row[indent] < 0.08 {
                1
            } else {
                2
            };
            let family = counts
                .iter()
                .position(|&c| row[c] > 0.5)
                .map_or(9, |c| c as u8);
            let next = ids.len() as u32;
            *ids.entry((table.looks[k], place, family)).or_insert(next)
        })
        .collect()
}

impl Stylesheet {
    /// The stylesheet of the lines `table` reads, its anchors the outline's lines t0 found (`Decision::anchors`).
    pub fn read(
        slots: &Slots,
        fences: &Fences,
        table: &LineTable,
        measures: &Measures,
        decision: &Decision,
    ) -> Self {
        let template = template_ids(table, measures);
        let n = table.readings.len();
        let (edge, beside) = (feature("edge_repeated"), feature("beside"));
        let count = template.iter().map(|&t| t as usize + 1).max().unwrap_or(0);
        let mut anchor = vec![0u16; n];
        for found in &decision.anchors {
            for slot in &found.lines {
                anchor[fences.place[slot.index()] as usize] = found.level.clamp(1, ANCHORS - 1);
            }
        }
        let mut views = vec![TemplateView::default(); count];
        let mut page_sets: Vec<HashSet<usize>> = vec![HashSet::new(); count];
        let mut anchor_levels: Vec<HashMap<u16, u32>> = vec![HashMap::new(); count];
        let mut last_value: Vec<Option<(String, u32)>> = vec![None; count];
        let mut counted = vec![(0u32, 0u32); count];
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
                    counted[t].0 +=
                        u32::from(*family == reading.family && reading.value == value + 1);
                }
                last_value[t] = Some((reading.family.clone(), reading.value));
            }
        }
        for (t, view) in views.iter_mut().enumerate() {
            let lines = view.lines.max(1.0);
            view.edge /= lines;
            view.beside /= lines;
            view.anchored /= lines;
            view.monotone = if counted[t].1 > 0 {
                counted[t].0 as f32 / counted[t].1 as f32
            } else {
                -1.0
            };
            view.anchor_level = commonest(&anchor_levels[t]);
            view.level = if view.anchored >= 0.3 {
                view.anchor_level
            } else {
                0.0
            };
            view.lines = (1.0 + view.lines).ln();
            view.pages = (1.0 + page_sets[t].len() as f32).ln();
        }
        Self {
            template,
            templates: views,
            anchor,
        }
    }

    /// Each template's row as DOC's stylesheet stage reads it: its measures, the mean of its lines' look and place, the
    /// share of them opening on a marker, and where in the document it first stands. Three cells hold what the export
    /// was fitted on and no reader now gives: a heading share uniform over the classes, and two zeros. Each cell within
    /// ±8, a cell of no number 0: the reader's range.
    pub fn rows(&self, table: &LineTable, measures: &Measures) -> Vec<[f32; ROW]> {
        let count = self.templates.len();
        let at = [
            "size", "bold", "upper", "centre", "indent", "length", "width",
        ]
        .map(feature);
        let n = table.readings.len();
        let uniform = (1.0 / CLASSES as f64) as f32;
        let (mut sums, mut lines, mut first) =
            (vec![[0f32; 8]; count], vec![0f32; count], vec![None; count]);
        for k in 0..n {
            let t = self.template[k] as usize;
            let row = measures.row(k);
            sums[t]
                .iter_mut()
                .zip(&at)
                .for_each(|(sum, &f)| *sum += row[f]);
            sums[t][7] += f32::from(u8::from(!table.readings[k].is_empty()));
            lines[t] += 1.0;
            first[t].get_or_insert(k);
        }
        (0..count)
            .map(|t| {
                let v = &self.templates[t];
                let mut r = [0f32; ROW];
                r[..11].copy_from_slice(&[
                    v.lines,
                    v.pages,
                    uniform,
                    v.monotone,
                    v.anchored,
                    v.anchor_level,
                    v.level,
                    v.edge,
                    v.beside,
                    0.0,
                    0.0,
                ]);
                r[11..19]
                    .iter_mut()
                    .zip(sums[t])
                    .for_each(|(out, sum)| *out = sum / lines[t].max(1.0));
                r[19] = first[t].map_or(1.0, |k| k as f32 / n.max(1) as f32);
                r.map(|v| {
                    if v.is_finite() {
                        v.clamp(-8.0, 8.0)
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    }
}
