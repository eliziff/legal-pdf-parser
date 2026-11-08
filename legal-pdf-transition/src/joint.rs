//! The joint stylesheet solve: the document's templates given their roles together, the reading the parse
//! makes under them. A template that may be headings is held to one role, its lines no heading or headings
//! at one level; each line's move is still the ranker's, a move against its template's role costing
//! `penalty`. An assignment's worth is its reading's log-probability (each move's among its line's
//! candidates, softmaxed), less its penalties, plus `weight` times each template's lines times the log of
//! the template prior's probability of its role (a forest over the template's measures, `template_rows`,
//! fitted on the gold's majority role and level of each template). The assignment starts from the free
//! reading's (each template's commonest reading) and climbs one template at a time, every role of each
//! tried by walking the document again from the template's first line, until a sweep changes none.

use super::*;

/// The template prior's measures of a template.
pub const PRIOR: usize = 20;
/// The templates held: the likeliest headings, by their lines.
const HELD: usize = 16;
/// Each template's measures as the prior reads them: its stylesheet view, the mean of its lines' look and
/// place, the share of them opening on a marker, and where in the document it first stands.
pub fn template_rows(sheet: &Stylesheet, table: &LineTable, measures: &Measures) -> Vec<[f32; PRIOR]> {
    let count = sheet.templates.len();
    let at = ["size", "bold", "upper", "centre", "indent", "length", "width"].map(feature);
    let n = table.readings.len();
    let (mut sums, mut lines, mut first) = (vec![[0f32; 8]; count], vec![0f32; count], vec![None; count]);
    for k in 0..n {
        let t = sheet.template[k] as usize;
        let row = measures.row(k);
        sums[t].iter_mut().zip(&at).for_each(|(sum, &f)| *sum += row[f]);
        sums[t][7] += f32::from(u8::from(!table.readings[k].is_empty()));
        lines[t] += 1.0;
        first[t].get_or_insert(k);
    }
    (0..count).map(|t| {
        let v = &sheet.templates[t];
        let mut r = [0f32; PRIOR];
        r[..11].copy_from_slice(&[v.lines, v.pages, v.heading, v.monotone, v.anchored, v.anchor_level, v.level, v.edge, v.beside, v.read_heading, v.read_level]);
        r[11..19].iter_mut().zip(sums[t]).for_each(|(out, sum)| *out = sum / lines[t].max(1.0));
        r[19] = first[t].map_or(1.0, |k| k as f32 / n.max(1) as f32);
        r
    }).collect()
}

/// The templates the solve gives roles: those the local scores or the first reading hold headings, or the
/// outline names, the `HELD` with the most lines.
pub fn held_templates(sheet: &Stylesheet) -> Vec<u32> {
    let mut held: Vec<u32> = (0..sheet.templates.len() as u32).filter(|&t| {
        let v = &sheet.templates[t as usize];
        v.heading >= 0.25 || v.read_heading > 0.0 || v.anchored > 0.0
    }).collect();
    held.sort_by(|&a, &b| sheet.templates[b as usize].lines.total_cmp(&sheet.templates[a as usize].lines).then(a.cmp(&b)));
    held.truncate(HELD);
    held
}
