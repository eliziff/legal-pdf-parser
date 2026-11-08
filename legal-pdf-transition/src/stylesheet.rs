//! The document's stylesheet, read whole before the parse and again after a first reading: a legal
//! document is set from a handful of templates (a look, where the line stands, the numbering it
//! opens on), and what one line of a template is, the template's other lines mostly are. Each template
//! is measured over all its lines: how many and over how many pages, how far the local scores hold its
//! lines headings, whether its markers count on, how many of its lines the document's own outline names
//! (its bookmarks and its printed contents list, each entry found as a line of the page it names,
//! `levels::anchors`) and at what level, how far it repeats at the pages' edges, how far it stands beside
//! another line on its row (a marginal note's lane). Templates that may be
//! headings are ordered among themselves (one above another where its occurrences restart the other's
//! count and enclose its occurrences, by its look's weight where neither tells), the outline's levels
//! fixing the templates it names; each line reads its template's level. After a first reading, each
//! template also carries how its lines were read (`Stylesheet::read_back`): the share made headings and
//! the level they mostly took, so the second reading decides each line with the document's view of its
//! template. The regime (paragraph-numbered, sectioned, transcribed) is measured with it, over each
//! constituent document the first reading opened.

use super::*;

pub struct Stylesheet {
    /// Each line's template, in reading order.
    pub template: Vec<u32>,
    pub templates: Vec<TemplateView>,
    /// Each line the outline names: its level there (0 for none).
    pub anchor: Vec<u16>,
    /// Each line opening a row that reads as a contents list's (a title and the locator it prints, the
    /// row the lines on one baseline): the title, normalized as titles are matched. A contents list
    /// holds the lines that have one.
    pub contents_titles: Vec<Option<String>>,
    /// Each line's regime: the shares of its document's lines opening on a bracketed number, on a
    /// section's number, on a transcript's line number, on a speaker's label; its document the whole
    /// source before a first reading, the constituent that reading opened after it.
    pub regime: Vec<[f32; 4]>,
    /// Each line's own four openings.
    opens: Vec<[bool; 4]>,
}

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
pub fn template_ids(table: &LineTable) -> Vec<u32> {
    let (indent, centre, width, right) = (feature("indent"), feature("centre"), feature("width"), feature("right_gap"));
    let counts: Vec<usize> = ["count_1", "count_1.1", "count_1.1.1", "count_1.1.1.1", "count_a", "count_A", "count_i", "count_I", "count_bullet"]
        .iter().map(|name| feature(name)).collect();
    let mut ids: HashMap<(u32, u8, u8), u32> = HashMap::new();
    (0..table.order.len()).map(|k| {
        let row = table.row(k);
        let place = if row[centre].abs() < 0.04 && row[width] < 0.85 { 3 } else if row[indent] > 0.4 && row[right] < 0.05 { 4 }
            else if row[indent] < 0.02 { 0 } else if row[indent] < 0.08 { 1 } else { 2 };
        let family = counts.iter().position(|&c| row[c] > 0.5).map_or(9, |c| c as u8);
        let next = ids.len() as u32;
        *ids.entry((table.looks[k], place, family)).or_insert(next)
    }).collect()
}

/// Each line opening a row read as a contents list's (`contents_lists::row_of` over the lines one baseline
/// band holds on its page, in reading order): its title normalized as titles are matched
/// (`bookmarks::normalize_title`), three characters or more.
pub fn contents_rows(pages: &[Page], slots: &Slots, table: &LineTable) -> Vec<Option<String>> {
    use legal_pdf_headings::contents_lists::{row_bands, row_of};
    let n = table.order.len();
    let line = |at: usize| slots.line(pages, table.order[at]);
    let mut titles = vec![None; n];
    let mut k = 0;
    while k < n {
        let page = slots.page(table.order[k]);
        let end = (k..n).find(|&e| slots.page(table.order[e]) != page).unwrap_or(n);
        for band in row_bands(&(k..end).collect::<Vec<usize>>(), |at| line(at).bbox) {
            let row = row_of(band.iter().map(|&at| line(at)).collect());
            let title = row.locator.is_some().then(|| legal_pdf_headings::headings::bookmarks::normalize_title(&row.title));
            if let Some(title) = title.filter(|title| title.chars().count() >= 3) { titles[band[0]] = Some(title); }
        }
        k = end;
    }
    titles
}

/// Whether a line may open a unit as the encoder's attention reads the document: it opens on a marker,
/// or it is short (sixty characters or fewer).
pub fn opener(table: &LineTable, k: usize) -> bool {
    !table.readings[k].is_empty() || table.row(k)[feature("length")] <= 0.8222
}

impl Stylesheet {
    /// The stylesheet of the lines `table` reads, each line's template `template` (`template_ids`).
    pub fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &legal_pdf_decide::Geometry, outline: &[PdfOutlineEntry], table: &LineTable, template: Vec<u32>, locals: &[Local]) -> Self {
        let n = table.order.len();
        let (size, bold, upper, edge, beside) = (feature("size"), feature("bold"), feature("upper"), feature("edge_repeated"), feature("beside"));
        let count = template.iter().map(|&t| t as usize + 1).max().unwrap_or(0);
        // The outline's anchors: the bookmarks and the confirmed contents entries, as lines.
        // The document's bookmarks alone: its printed contents the parser links itself (`State::answering`).
        let anchoring = legal_pdf_decide::anchors(pages, slots, evidence, fences, geometry, &fences.role, outline, &[]);
        let mut anchor = vec![0u16; n];
        for found in &anchoring.anchors {
            for slot in &found.lines {
                anchor[fences.place[slot.index()] as usize] = found.level.max(1);
            }
        }
        // A line on a row that reads as a contents list's: a title and the locator it prints.
        let contents_titles = contents_rows(pages, slots, table);
        // Each template's measures.
        let heading_class = usize::from(HEADING) * 2 + 1;
        let mut views = vec![TemplateView::default(); count];
        let mut page_sets: Vec<HashSet<usize>> = vec![HashSet::new(); count];
        let mut anchor_levels: Vec<HashMap<u16, u32>> = vec![HashMap::new(); count];
        let mut last_value: Vec<Option<(String, u32)>> = vec![None; count];
        let mut counted = vec![(0u32, 0u32); count];
        let mut weight = vec![0f32; count];
        for k in 0..n {
            let t = template[k] as usize;
            let row = table.row(k);
            let view = &mut views[t];
            view.lines += 1.0;
            view.heading += locals[k].probabilities[heading_class] as f32;
            view.edge += row[edge];
            view.beside += row[beside];
            page_sets[t].insert(slots.page(table.order[k]));
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
            view.heading /= lines;
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
        // The regime: what the lines open on.
        let opens: Vec<[bool; 4]> = (0..n).map(|k| {
            let at = table.order[k].index();
            let first = table.readings[k].first();
            [
                first.is_some_and(|r| r.family.contains("[]")),
                first.is_some_and(|r| r.family.ends_with(":1") && !r.family.contains("[]")),
                evidence.line_number[at],
                evidence.words[at] & WORD_SPEAKER != 0,
            ]
        }).collect();
        let mut sheet = Self { template, templates: views, anchor, contents_titles, regime: vec![[0.0; 4]; n], opens };
        sheet.regimes(&[]);
        sheet
    }

    /// Each line's regime over its document, the documents opening at `starts` (places in reading order).
    fn regimes(&mut self, starts: &[usize]) {
        let n = self.opens.len();
        let mut bounds: Vec<usize> = std::iter::once(0).chain(starts.iter().copied().filter(|&k| k > 0 && k < n)).collect();
        bounds.push(n);
        for pair in bounds.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            let lines = (to - from).max(1) as f32;
            let mut shares = [0f32; 4];
            (from..to).for_each(|k| shares.iter_mut().zip(self.opens[k]).for_each(|(share, open)| *share += f32::from(u8::from(open)) / lines));
            self.regime[from..to].fill(shares);
        }
    }

    /// Each template as a first reading read its lines: the share made headings and their commonest level;
    /// and each line's regime over the constituent document that reading put it in.
    pub fn read_back(&mut self, moves: &[Move]) {
        let starts: Vec<usize> = moves.iter().enumerate().filter(|(_, m)| m.starts).map(|(k, _)| k).collect();
        self.regimes(&starts);
        let count = self.templates.len();
        let (mut headings, mut lines) = (vec![0f32; count], vec![0f32; count]);
        let mut levels: Vec<HashMap<i8, u32>> = vec![HashMap::new(); count];
        for (k, m) in moves.iter().enumerate() {
            let t = self.template[k] as usize;
            lines[t] += 1.0;
            if m.region == HEADING && m.opens {
                headings[t] += 1.0;
                *levels[t].entry(m.level).or_default() += 1;
            }
        }
        for (t, view) in self.templates.iter_mut().enumerate() {
            view.read_heading = headings[t] / lines[t].max(1.0);
            view.read_level = commonest(&levels[t]);
        }
    }

    pub fn of(&self, k: usize) -> &TemplateView {
        &self.templates[self.template[k] as usize]
    }
}

/// The heading levels the templates that may be headings take among themselves (0 for one that may
/// not): template a stands above b where, after a's occurrences, b's count starts over at one more than
/// the other way round, and where b's occurrences fall between a's more than a's between b's; where
/// neither tells, the weightier look stands above. A template the outline names mostly at one level
/// keeps it, the others ranked around the named ones.
fn order_templates(template: &[u32], table: &LineTable, views: &[TemplateView], weight: &[f32]) -> Vec<f32> {
    let count = views.len();
    let candidates: Vec<usize> = (0..count).filter(|&t| views[t].heading >= 0.25 || views[t].anchored >= 0.3).collect();
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
    // The outline's levels stand where it names a template.
    candidates.iter().filter(|&&t| views[t].anchored >= 0.3 && views[t].anchor_level > 0.0).for_each(|&t| levels[t] = views[t].anchor_level);
    levels
}
