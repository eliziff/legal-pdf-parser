//! What Detect-Order-Construct is fitted on, read once from the gold's rows (the grader's `labels`) and written into the
//! dump (`targets`), so the fit reads its targets and decodes no gold: each line's role (`region * 2 + opens`), its
//! flags and its links on its page; the document's regions (a unit's lines on one page, by their first line), each
//! region's continuation, a note's region pointing at the region holding its paired reference, the reading order the
//! gold ranks; and the heading and body trees (each node's parent, left sibling and gap under its parent).
//!
//! Row columns: region, start (0 continues, 1 opens, 2 resumes), document, reference, level step, marker, resumed row
//! (a line place), depth step.

use super::*;
use super::tensors::Tensor;

/// A gold region: body text read unmarked (`UNMARKED_BODY`) is text.
fn region_of(g: i64) -> i64 {
    if g == UNMARKED_BODY { 1 } else { g }
}

/// A heading's or body unit's gap under its parent is one of 1..=GAPS levels.
const GAPS: i64 = 4;

/// Each unit opening's absolute level (column 4) or depth (column 7) of `wanted` regions, its steps chained (10 the
/// same), a document start beginning again; 0 elsewhere or unknown.
fn absolute(gold: &[[i64; 8]], column: usize, wanted: &[u8]) -> Vec<i64> {
    let (mut out, mut current) = (vec![0i64; gold.len()], 0i64);
    for (k, g) in gold.iter().enumerate() {
        if g[2] == 1 {
            current = 0;
        }
        let region = region_of(g[0]);
        if region > 0 && wanted.contains(&(region as u8)) && g[1] == 1 {
            if g[column] < 0 {
                current = 0;
                continue;
            }
            current = (current.max(1) + g[column] - 10).clamp(1, 9);
            out[k] = current;
        }
    }
    out
}

/// For the regions `nodes` (their first lines, in reading order) with `levels`, each one's parent and left sibling
/// among the earlier nodes (indexes into `nodes`; its own index for none), the nearest open node of a smaller level
/// and that parent's last child before it, and its gap under the parent (less one).
fn tree(levels: &[i64], nodes: &[usize], starts: &[bool]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut parent, mut sibling, mut gaps) = (Vec::new(), Vec::new(), Vec::new());
    let (mut stack, mut last_child): (Vec<usize>, HashMap<Option<usize>, usize>) = (Vec::new(), HashMap::new());
    for (i, &k) in nodes.iter().enumerate() {
        let from = if i > 0 { nodes[i - 1] + 1 } else { 0 };
        if (from..=k).any(|j| starts[j]) {
            stack.clear();
            last_child.clear();
        }
        let level = levels[k];
        while stack.last().is_some_and(|&top| levels[nodes[top]] >= level) {
            stack.pop();
        }
        let p = stack.last().copied();
        parent.push(p.map_or(i, |p| p) as i64);
        sibling.push(last_child.get(&p).map_or(i, |&s| s) as i64);
        gaps.push((level - p.map_or(0, |p| levels[nodes[p]])).clamp(1, GAPS) - 1);
        last_child.insert(p, i);
        stack.push(i);
    }
    (parent, sibling, gaps)
}

/// The fit's targets from the gold's `rows`, each line's page (`pages`), a note's paired reference line (`note_refs`),
/// each line's gold reading rank (`ranks`), its reference marks (`references`, their values first) and note number;
/// with what the fit reads of the parse's `inputs` (the dump's arrays) at the gold's regions and tree nodes, so the fit
/// derives nothing: each region's lines and box, each node's measures as the reader gives them to the trees. And the
/// document's summary, what the fit knows of it before reading its arrays: its lines, regions, body nodes, each page's
/// lines, its arrays' widths and the heads its gold has.
pub fn targets(gold: &[[i64; 8]], pages: &[usize], note_refs: &[i64], ranks: &[i64], references: &[Vec<(u32, u32, u32)>],
    note_numbers: &[i64], inputs: &[Tensor]) -> (Vec<Tensor>, serde_json::Value) {
    let n = gold.len();
    let regions: Vec<i64> = gold.iter().map(|g| if g[0] >= 0 { region_of(g[0]) } else { -1 }).collect();
    let role: Vec<i64> = (0..n).map(|k| if regions[k] >= 0 { regions[k] * 2 + i64::from(gold[k][1] == 1) } else { -100 }).collect();
    let free: Vec<bool> = gold.iter().map(|g| g[0] == UNMARKED_BODY).collect();
    let marked: Vec<i64> = (0..n).map(|k| if regions[k] >= 0 { i64::from(gold[k][1] == 1 && gold[k][5] == 1) } else { -100 }).collect();
    let start: Vec<i64> = (0..n).map(|k| if regions[k] >= 0 { i64::from(gold[k][2] == 1) } else { -100 }).collect();
    // Each labelled line's unit (the index of the unit's first line): a line opens a unit, goes on with the unit open in
    // its stream, or resumes the unit its pointer names.
    let mut units = vec![-1i64; n];
    let mut open: HashMap<usize, usize> = HashMap::new();
    for k in 0..n {
        let region = region_of(gold[k][0]);
        if region <= 0 {
            continue;
        }
        let stream = stream_of(region as u8);
        let (begins, pointer) = (gold[k][1], gold[k][6]);
        units[k] = if begins == 2 && (0..k as i64).contains(&pointer) && units[pointer as usize] >= 0 {
            units[pointer as usize]
        } else if begins == 0 && open.get(&stream).is_some_and(|&u| region_of(gold[u][0]) == region) {
            open[&stream] as i64
        } else {
            k as i64
        };
        open.insert(stream, units[k] as usize);
    }
    // Detect's regions: a unit's lines on one page, by the first of them; each line's successor and predecessor there.
    let mut region = vec![-1i64; n];
    let mut firsts: HashMap<(i64, usize), usize> = HashMap::new();
    for k in (0..n).filter(|&k| units[k] >= 0) {
        region[k] = *firsts.entry((units[k], pages[k])).or_insert(k) as i64;
    }
    let (mut successor, mut predecessor) = (vec![-100i64; n], vec![-100i64; n]);
    let mut last: HashMap<i64, usize> = HashMap::new();
    for k in (0..n).filter(|&k| region[k] >= 0) {
        match last.get(&region[k]) {
            Some(&before) => {
                successor[before] = k as i64;
                predecessor[k] = before as i64;
            }
            None => predecessor[k] = k as i64,
        }
        successor[k] = k as i64;
        last.insert(region[k], k);
    }
    // Order: the regions in reading order (their first lines); a unit's later regions going on in its earlier ones.
    let mut ordered: Vec<usize> = region.iter().filter(|&&r| r >= 0).map(|&r| r as usize).collect::<HashSet<_>>().into_iter().collect();
    ordered.sort_unstable();
    let index: HashMap<usize, usize> = ordered.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    let region_of_line: Vec<i64> = region.iter().map(|&r| if r >= 0 { index[&(r as usize)] as i64 } else { -1 }).collect();
    let m = ordered.len();
    let mut unit_regions: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut slot: HashMap<i64, usize> = HashMap::new();
    for (i, &r) in ordered.iter().enumerate() {
        let at = *slot.entry(units[r]).or_insert_with(|| { unit_regions.push((units[r], Vec::new())); unit_regions.len() - 1 });
        unit_regions[at].1.push(i);
    }
    let mut goes_on = vec![-100i64; m];
    for (_, parts) in &unit_regions {
        for pair in parts.windows(2) {
            goes_on[pair[0]] = pair[1] as i64;
        }
        let tail = *parts.last().expect("a unit has a region");
        goes_on[tail] = tail as i64;
    }
    let is_note = |g: i64| { let r = region_of(g); r == NOTE as i64 || r == NOTE_QUOTE as i64 };
    // A note's region points at the region holding its paired reference (its own where it has none).
    let note_of: Vec<i64> = ordered.iter().enumerate().map(|(i, &r)| {
        if !is_note(gold[r][0]) || gold[r][1] != 1 {
            return -100;
        }
        let at = note_refs[r];
        if at >= 0 && region_of_line[at as usize] >= 0 { region_of_line[at as usize] } else { i as i64 }
    }).collect();
    let region_role: Vec<i64> = ordered.iter().map(|&r| regions[r].max(0)).collect();
    // Construct: the headings' tree, and the body's.
    let starts: Vec<bool> = gold.iter().map(|g| g[2] == 1).collect();
    let levels = absolute(gold, 4, &[HEADING]);
    let headings: Vec<usize> = (0..m).filter(|&i| { let r = ordered[i]; region_of(gold[r][0]) == HEADING as i64 && gold[r][1] == 1 && levels[r] > 0 }).collect();
    let (heading_parent, heading_sibling, heading_gap) = tree(&levels, &headings.iter().map(|&i| ordered[i]).collect::<Vec<_>>(), &starts);
    let depths = absolute(gold, 7, &RANKED_BODY);
    let in_body = |r: usize| { let region = region_of(gold[r][0]); region > 0 && RANKED_BODY.contains(&(region as u8)) };
    let opening: Vec<usize> = (0..m).filter(|&i| in_body(ordered[i]) && gold[ordered[i]][1] == 1).collect();
    // A gold that gives its body no depths (A2AJ's alignments) says only that its numbered paragraphs are the court's
    // numbering at the body's top, each one's left sibling among the nodes from the paragraph before it.
    let partial = !opening.is_empty() && !opening.iter().any(|&i| depths[ordered[i]] > 0);
    let (bodies, body_parent, body_sibling, body_sibling_from, body_gap) = if partial {
        let h = opening.len();
        let (mut parent, mut sibling, mut from, mut gap) = (vec![-100i64; h], vec![-100i64; h], vec![-100i64; h], vec![-100i64; h]);
        let (mut prev, mut last): (Option<usize>, i64) = (None, -1);
        for (at, &i) in opening.iter().enumerate() {
            let r = ordered[i];
            if ((last + 1) as usize..=r).any(|j| starts[j]) {
                prev = None;
            }
            last = r as i64;
            if region_of(gold[r][0]) == 1 && gold[r][5] == 1 {
                (parent[at], gap[at]) = (at as i64, 0);
                if let Some(p) = prev {
                    (sibling[at], from[at]) = (at as i64 - 1, p as i64);
                }
                prev = Some(at);
            }
        }
        (opening, parent, sibling, from, gap)
    } else {
        let bodies: Vec<usize> = opening.into_iter().filter(|&i| depths[ordered[i]] > 0).collect();
        let (parent, sibling, gap) = tree(&depths, &bodies.iter().map(|&i| ordered[i]).collect::<Vec<_>>(), &starts);
        let from = sibling.clone();
        (bodies, parent, sibling, from, gap)
    };
    // The regions' places: each one's page, its successor in the gold's reading (none where the gold ranks none), a
    // note's number and the marks of it each region sets.
    let region_pages: Vec<usize> = ordered.iter().map(|&r| pages[r]).collect();
    let mut rank = vec![i64::MAX; m];
    for k in (0..n).filter(|&k| region_of_line[k] >= 0 && ranks[k] >= 0) {
        let i = region_of_line[k] as usize;
        rank[i] = rank[i].min(ranks[k]);
    }
    let mut known: Vec<usize> = (0..m).filter(|&i| rank[i] < i64::MAX).collect();
    known.sort_by_key(|&i| (rank[i], i));
    let mut region_order = vec![-100i64; m];
    for pair in known.windows(2) {
        region_order[pair[0]] = pair[1] as i64;
    }
    if let Some(&tail) = known.last() {
        region_order[tail] = tail as i64;
    }
    let region_number: Vec<i64> = ordered.iter().map(|&r| if is_note(gold[r][0]) { note_numbers[r] } else { -1 }).collect();
    let mut marks: Vec<Vec<u32>> = vec![Vec::new(); m];
    for k in (0..n).filter(|&k| region_of_line[k] >= 0) {
        let i = region_of_line[k] as usize;
        if REFERRING.contains(&(region_role[i] as u8)) {
            marks[i].extend(references[k].iter().map(|&(value, _, _)| value));
        }
    }
    marks.iter_mut().for_each(|found| { found.sort_unstable(); found.dedup(); });
    // The region pairs Order scores: every two regions within a page of each other, and a note's region against every
    // region setting its number however far (endnotes); sorted by region, then partner. Each pair's whether the note's
    // number is a mark the other sets, and each region head's gold as a pair (-100 none, -1 outside the region's pairs).
    let mut by_page: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, &page) in region_pages.iter().enumerate() {
        match by_page.iter_mut().find(|(p, _)| *p == page) {
            Some((_, members)) => members.push(i),
            None => by_page.push((page, vec![i])),
        }
    }
    let on_page = |page: i64| by_page.iter().find(|(p, _)| *p as i64 == page).map_or(&[][..], |(_, m)| m.as_slice());
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (page, members) in &by_page {
        let near: Vec<usize> = (-1i64..=1).flat_map(|d| on_page(*page as i64 + d).iter().copied()).collect();
        pairs.extend(members.iter().flat_map(|&a| near.iter().map(move |&b| (a, b))));
    }
    let mut setting: HashMap<u32, Vec<usize>> = HashMap::new();
    for (b, found) in marks.iter().enumerate() {
        found.iter().for_each(|&value| setting.entry(value).or_default().push(b));
    }
    for a in (0..m).filter(|&a| region_number[a] >= 0) {
        for &b in setting.get(&(region_number[a] as u32)).map_or(&[][..], Vec::as_slice) {
            if region_pages[a].abs_diff(region_pages[b]) > 1 {
                pairs.push((a, b));
            }
        }
    }
    pairs.sort();
    let pair_match: Vec<bool> = pairs.iter().map(|&(a, b)| a != b && region_number[a] >= 0 && marks[b].binary_search(&(region_number[a] as u32)).is_ok()).collect();
    let at: HashMap<(usize, usize), usize> = pairs.iter().enumerate().map(|(p, &pair)| (pair, p)).collect();
    let as_pair = |gold: &[i64]| -> Vec<i64> {
        gold.iter().enumerate().map(|(r, &t)| if t < 0 { -100 } else { at.get(&(r, t as usize)).map_or(-1, |&p| p as i64) }).collect()
    };
    let (pair_i, pair_j): (Vec<usize>, Vec<usize>) = pairs.iter().copied().unzip();
    let us = |name: &str, v: &[usize]| Tensor::i32(name, vec![v.len()], v.iter().map(|&x| x as i64));
    let is = |name: &str, v: &[i64]| Tensor::ints(name, v);
    // The parse's arrays the regions and nodes read.
    let input = |name: &str| inputs.iter().find(|t| t.name == name);
    let ints = |name: &str| input(name).map(Tensor::values).unwrap_or_default().into_iter().map(|v| v as i64).collect::<Vec<i64>>();
    let boxes: Vec<f32> = input("boxes").map(Tensor::values).unwrap_or_default().into_iter().map(|v| v as f32).collect();
    let (width, text) = (input("own").and_then(|t| t.shape.get(1)).copied().unwrap_or(0), input("tokens").is_some());
    let (templates, anchors, looks) = (ints("templates"), ints("anchors"), ints("looks"));
    let (family, value, prefix) = (ints("marker_family"), ints("marker_value"), ints("marker_prefix"));
    // Each node's left edge and type size as the reader's trees read them: the line's own, not on the features' scales.
    let (x0, size): (Vec<f32>, Vec<f32>) = (boxes.chunks_exact(4).map(|b| b[0]).collect(), input("size").map(Tensor::f32s).unwrap_or_default());
    let region_lines: Vec<usize> = (0..n).filter(|&k| region_of_line[k] >= 0).collect();
    let line_region: Vec<i64> = region_lines.iter().map(|&k| region_of_line[k]).collect();
    let region_boxes = if boxes.len() == 4 * n { reader::region_boxes(&boxes, region_lines.iter().map(|&k| (k, region_of_line[k] as usize)), m) } else { vec![0.0; 4 * m] };
    // Where each region stands as the reader's trees read it (`reader::sections`): a region opens where no earlier one goes on in it.
    let mut continued = vec![false; m];
    goes_on.iter().enumerate().filter(|&(a, &b)| b >= 0 && b as usize != a).for_each(|(_, &b)| continued[b as usize] = true);
    let (sections, enclosing) = reader::sections(&reader::resets(&ordered, &starts), |i| !continued[i] && region_role[i] == i64::from(HEADING));
    let mut nodes_of = Vec::new();
    for (name, nodes) in [("heading", &headings), ("body", &bodies)] {
        let at = |of: &dyn Fn(usize, usize) -> i64| -> Vec<i64> { nodes.iter().map(|&i| of(i, ordered[i])).collect() };
        let at_f32 = |of: &[f32]| -> Vec<f32> { nodes.iter().map(|&i| of.get(ordered[i]).copied().unwrap_or(0.0)).collect() };
        let line = |of: &[i64]| at(&|_, k| of.get(k).copied().unwrap_or(0));
        nodes_of.extend([
            is(&format!("{name}_templates"), &line(&templates)), is(&format!("{name}_anchors"), &line(&anchors)),
            is(&format!("{name}_family"), &line(&family)), Tensor::i64(&format!("{name}_value"), vec![nodes.len()], line(&value)), is(&format!("{name}_prefix"), &line(&prefix)),
            is(&format!("{name}_section"), &at(&|i, _| sections[i])),
            Tensor::f32(&format!("{name}_x0"), vec![nodes.len()], &at_f32(&x0)), Tensor::f32(&format!("{name}_size"), vec![nodes.len()], &at_f32(&size)),
            is(&format!("{name}_looks"), &line(&looks)), is(&format!("{name}_pages"), &at(&|_, k| pages[k] as i64)),
            is(&format!("{name}_roles"), &at(&|i, _| region_role[i])), is(&format!("{name}_enclosing"), &at(&|i, _| enclosing[i].map_or(-1, |e| e as i64))),
        ]);
    }
    let any = |v: &[i64]| v.iter().any(|&x| x >= 0);
    let has: Vec<&str> = [("role", any(&role)), ("marked", any(&marked)), ("start", any(&start)), ("goes_on", any(&goes_on)), ("note_of", any(&note_of)),
        ("successor", any(&successor)), ("predecessor", any(&predecessor)), ("regions", m > 0), ("order", any(&region_order)),
        ("heading", headings.len() >= 2), ("body", bodies.len() >= 2)].into_iter().filter(|&(_, h)| h).map(|(name, _)| name).collect();
    let page_lines: Vec<usize> = pages.chunk_by(|a, b| a == b).map(<[usize]>::len).collect();
    let summary = serde_json::json!({"lines": n, "regions": m, "bodies": bodies.len(), "page_lines": page_lines, "has": has,
        "width": width, "text": text, "row_width": stylesheet::ROW});
    let mut tensors = vec![
        us("pair_i", &pair_i), us("pair_j", &pair_j), Tensor::flags("pair_match", &pair_match),
        is("goes_on_pair", &as_pair(&goes_on)), is("note_of_pair", &as_pair(&note_of)), is("order_pair", &as_pair(&region_order)),
        is("role", &role), Tensor::flags("free", &free), is("marked", &marked), is("start", &start), is("successor", &successor),
        is("predecessor", &predecessor), us("regions", &ordered), is("region_of_line", &region_of_line), is("goes_on", &goes_on),
        is("note_of", &note_of), is("region_role", &region_role), us("headings", &headings), is("heading_parent", &heading_parent),
        is("heading_sibling", &heading_sibling), is("heading_sibling_from", &heading_sibling), is("heading_gap", &heading_gap),
        us("bodies", &bodies), is("body_parent", &body_parent), is("body_sibling", &body_sibling),
        is("body_sibling_from", &body_sibling_from), is("body_gap", &body_gap), Tensor::flags("partial_bodies", &[partial]),
        us("region_pages", &region_pages), is("region_order", &region_order),
        us("region_lines", &region_lines), is("line_region", &line_region), Tensor::f32("region_boxes", vec![m, 4], &region_boxes),
    ];
    tensors.extend(nodes_of);
    (tensors, summary)
}
