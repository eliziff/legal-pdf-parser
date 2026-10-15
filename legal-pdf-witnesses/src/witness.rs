//! Every counted line's witnesses, read against its page and its document.

use legal_pdf_core::model::{Line, Page};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

/// The witnesses, in the order of each row's values. Flags are 1 or 0; measures are ratios to the
/// document's body (type size, line height), fractions of the page's text measure, or counts.
pub const NAMES: &[&str] = &[
    // The page and document
    "recognized",
    "page_frac",
    "page_first",
    "odd_width",
    "source_switch",
    "bold_share",
    "repeats",
    // Type
    "bold",
    "italic",
    "own_font",
    "size",
    "underlined",
    "prev_bold",
    "next_bold",
    // Geometry
    "height",
    "width",
    "indent",
    "rindent",
    "centered",
    "gap_above",
    "gap_below",
    "gap_above_none",
    "gap_below_none",
    // Words
    "chars",
    "words",
    "caps",
    "first_upper",
    "first_lower",
    "letters",
    "end_colon",
    "end_period",
    "end_comma",
    "end_semicolon",
    "end_paren",
    "end_dash",
    "prev_end_colon",
    "prev_end_period",
    "prev_end_comma",
    "prev_end_semicolon",
    "prev_end_dash",
    "next_lower",
    "lead_in",
    "court",
    "address",
    "party",
    "citation",
    "defined_term",
    "dot_leader",
    // The court-forms store: a form's title or number, a court's name, a form's printed boilerplate
    "form_title",
    "form_number",
    "known_court",
    "form_boilerplate",
    // Markers
    "marker_any",
    "marker_only",
    "m_bullet",
    "m_paren_num",
    "m_paren_letter",
    "m_paren_roman",
    "m_decimal",
    "m_num",
    "m_num_bare",
    "m_letter",
    "m_roman",
    "m_word",
    "backbone",
    // The document's style ladder and outline
    "style_rank",
    "style_lines",
    "body_style",
    "in_contents",
    // The readers' verdicts so far
    "reader_heading",
    "quoted",
    "table_cell",
    "translation",
    "layout_title",
    "layout_list",
    "layout_known",
    // The line's place in its region (the layout model's, or the readers' where none ran)
    "region_first",
    "region_last",
    "region_lines",
    "tag_known",
    "tag_heading",
    "tag_title",
    "tag_paragraph",
    "tag_list",
    "tag_table",
    "tag_note",
    "tag_reference",
    "tag_caption",
    "tag_artifact",
    "tag_quotation",
];

pub fn index(name: &str) -> Option<usize> {
    NAMES.iter().position(|known| *known == name)
}

/// What the structure stage has read that witnesses use.
#[derive(Default)]
pub struct Context<'a> {
    pub quoted: Option<&'a HashSet<String>>,
    pub table_cells: Option<&'a HashSet<String>>,
    pub contents: Option<&'a HashSet<String>>,
    pub translation: Option<&'a HashSet<String>>,
    /// A layout model's label for each line, where one ran.
    pub layout: Option<&'a HashMap<String, String>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Row {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_range: Option<legal_structure_model::ScalarRange>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<SpanWitness>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<(usize, usize, bool)>,
    pub page: usize,
    pub slot: usize,
    pub id: String,
    pub text: String,
    pub recognized: bool,
    pub values: Vec<f32>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub rows: Vec<Row>,
}

impl Table {
    pub fn by_id(&self) -> HashMap<&str, &Row> {
        self.rows.iter().map(|row| (row.id.as_str(), row)).collect()
    }
}

/// Feature text uses only the unit's own source membership, never intervening envelope text.
pub(crate) fn membership_text(
    node: &legal_structure_model::StructureNode,
    text: &legal_structure_model::ScalarText<'_>,
) -> Option<String> {
    let ranges = if node.ranges.is_empty() {
        std::slice::from_ref(&node.range)
    } else {
        &node.ranges
    };
    let ranges = ranges.iter().map(|range| Some(legal_structure_model::ScalarRange {
        start: text.utf16_at_scalar(range.start)?,
        end: text.utf16_at_scalar(range.end)?,
    })).collect::<Option<Vec<_>>>()?;
    Some(text.membership_text_utf16(&ranges))
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SpanWitness {
    pub(crate) start: usize,
    pub(crate) end: usize,
    size: f32,
    bold: bool,
    italic: bool,
    own_font: bool,
    /// Rank, source-line frequency and body-style membership for this exact type.
    style: [f32; 3],
    role: Option<String>,
}

/// Bind known source IDs to their established coordinates; never find repeated text heuristically.
pub fn bind_source_fragments(
    table: &mut Table,
    text: &str,
    fragments: &[legal_structure_model::SourceFragment],
) -> Result<(), String> {
    let coordinates = legal_structure_model::ScalarText::new(text);
    let by_id = fragments
        .iter()
        .map(|fragment| (fragment.id.as_str(), fragment.range))
        .collect::<HashMap<_, _>>();
    for row in &mut table.rows {
        let range = by_id
            .get(row.id.as_str())
            .ok_or("Witness line has no source fragment")?;
        let start = coordinates
            .scalar_at_utf16(range.start)
            .ok_or("Invalid witness fragment start")?;
        let end = coordinates
            .scalar_at_utf16(range.end)
            .ok_or("Invalid witness fragment end")?;
        if coordinates.slice(start..end) != Some(row.text.as_str()) {
            return Err("Witness text differs from identified source fragment".into());
        }
        row.source_range = Some(legal_structure_model::ScalarRange { start, end });
    }
    Ok(())
}

fn exact_type(values: &mut [f32], row: Option<&Row>, ranges: &[(usize, usize)]) {
    let mut weight = 0usize;
    let mut size = 0.0;
    let mut bold = 0usize;
    let mut italic = 0usize;
    let mut own_font = 0usize;
    let mut style_weight = 0usize;
    let mut style = [0.0; 3];
    let mut tags = HashSet::new();
    let overlap = |start: usize, end: usize| {
        ranges
            .iter()
            .map(|(lo, hi)| end.min(*hi).saturating_sub(start.max(*lo)))
            .sum::<usize>()
    };
    if let Some(row) = row {
        for span in &row.spans {
            let count = overlap(span.start, span.end);
            if count == 0 {
                continue;
            }
            if let Some(role) = span.role.as_deref() {
                tags.insert(role);
            }
            if span.size > 0.0 {
                style_weight += count;
                for (sum, value) in style.iter_mut().zip(span.style) { *sum += value * count as f32; }
                weight += count;
                size += span.size * count as f32;
                bold += usize::from(span.bold) * count;
                italic += usize::from(span.italic) * count;
                own_font += usize::from(span.own_font) * count;
            }
        }
    }
    for (name, value) in [
        ("size", size / weight.max(1) as f32),
        ("bold", bold as f32 / weight.max(1) as f32),
        ("italic", italic as f32 / weight.max(1) as f32),
        ("own_font", own_font as f32 / weight.max(1) as f32),
        ("style_rank", style[0] / style_weight.max(1) as f32),
        ("style_lines", style[1] / style_weight.max(1) as f32),
        ("body_style", style[2] / style_weight.max(1) as f32),
    ] {
        values[index(name).unwrap()] = value;
    }
    let (mut words, mut underlined) = (0, 0);
    if let Some(row) = row {
        for (start, end, marked) in &row.words {
            let count = overlap(*start, *end);
            words += count;
            if *marked {
                underlined += count;
            }
        }
    }
    values[index("underlined").unwrap()] = underlined as f32 / words.max(1) as f32;
    for (name, present) in [
        ("tag_known", !tags.is_empty()),
        (
            "tag_heading",
            tags.iter()
                .any(|role| matches!(*role, "H" | "H1" | "H2" | "H3" | "H4" | "H5" | "H6")),
        ),
        ("tag_title", tags.contains("Title")),
        ("tag_paragraph", tags.contains("P")),
        (
            "tag_list",
            tags.iter()
                .any(|role| matches!(*role, "L" | "LI" | "Lbl" | "LBody")),
        ),
        (
            "tag_table",
            tags.iter()
                .any(|role| matches!(*role, "Table" | "TR" | "TH" | "TD")),
        ),
        ("tag_note", tags.contains("Note")),
        (
            "tag_reference",
            tags.iter()
                .any(|role| matches!(*role, "Reference" | "Link")),
        ),
        ("tag_caption", tags.contains("Caption")),
        ("tag_artifact", tags.contains("Artifact")),
        (
            "tag_quotation",
            tags.iter()
                .any(|role| matches!(*role, "BlockQuote" | "Quote")),
        ),
    ] {
        values[index(name).unwrap()] = if present { 1.0 } else { 0.0 };
    }
}

pub(crate) fn local_span_values(
    row: Option<&Row>,
    text: &str,
    start: usize,
    end: usize,
) -> Vec<f32> {
    let mut values = span_values(row, text);
    let opening_end = row.and_then(|row| {
        row.spans
            .iter()
            .filter(|span| span.start <= start && start < span.end)
            .map(|span| span.end.min(end))
            .min()
    });
    let ranges = opening_end
        .map(|end| vec![(start, end)])
        .unwrap_or_default();
    exact_type(&mut values, row, &ranges);
    values
}

pub(crate) fn node_values(
    row: Option<&Row>,
    node: &legal_structure_model::StructureNode,
    coordinates: &legal_structure_model::ScalarText<'_>,
) -> Vec<f32> {
    let mut values = span_values(row, &membership_text(node, coordinates).unwrap_or_default());
    let ranges = row
        .and_then(|row| row.source_range)
        .map(|source| {
            let parts = if node.ranges.is_empty() {
                std::slice::from_ref(&node.range)
            } else {
                &node.ranges
            };
            parts
                .iter()
                .filter_map(|part| {
                    let start = part.start.max(source.start);
                    let end = part.end.min(source.end);
                    (start < end).then_some((start - source.start, end - source.start))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    exact_type(&mut values, row, &ranges);
    values
}

/// Physical typography stays with the source line; lexical evidence belongs to the exact unit.
pub(crate) fn span_values(row: Option<&Row>, text: &str) -> Vec<f32> {
    let mut values = row.map_or_else(|| vec![0.0; NAMES.len()], |row| row.values.clone());
    let text = text.trim();
    let flag = |value: bool| if value { 1.0 } else { 0.0 };
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let capitals = text.chars().filter(|c| c.is_uppercase()).count();
    let first = text.chars().next();
    let end = text.chars().last();
    for (name, value) in [
        ("chars", text.chars().count() as f32),
        ("words", text.split_whitespace().count() as f32),
        (
            "caps",
            if letters > 0 {
                capitals as f32 / letters as f32
            } else {
                0.0
            },
        ),
        ("letters", letters as f32),
        (
            "first_upper",
            flag(first.is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())),
        ),
        ("first_lower", flag(first.is_some_and(char::is_lowercase))),
        ("end_colon", flag(end == Some(':'))),
        ("end_period", flag(end == Some('.'))),
        ("end_comma", flag(end == Some(','))),
        ("end_semicolon", flag(end == Some(';'))),
        ("end_paren", flag(end == Some(')'))),
        ("end_dash", flag(end.is_some_and(|c| "-–—".contains(c)))),
        (
            "marker_any",
            flag(families(text).iter().any(|marked| *marked)),
        ),
        ("marker_only", flag(bare().is_match(text))),
    ] {
        if let Some(at) = index(name) {
            values[at] = value;
        }
    }
    for (name, marked) in [
        "m_bullet",
        "m_paren_num",
        "m_paren_letter",
        "m_paren_roman",
        "m_decimal",
        "m_num",
        "m_num_bare",
        "m_letter",
        "m_roman",
        "m_word",
    ]
    .into_iter()
    .zip(families(text))
    {
        if let Some(at) = index(name) {
            values[at] = flag(marked);
        }
    }
    values
}

fn regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("witness pattern"))
}

macro_rules! pattern {
    ($name:ident, $pattern:expr) => {
        fn $name() -> &'static Regex {
            static RE: OnceLock<Regex> = OnceLock::new();
            regex(&RE, $pattern)
        }
    };
}

pattern!(
    bullet,
    r"^\s*[•·▪◦‣●○■□➢✓\u{f0b7}\u{f0a7}\u{f0d8}\u{f076}*–-]\s"
);
pattern!(paren_num, r"^\s*\(\d{1,3}\)");
pattern!(paren_letter, r"^\s*\([a-z]{1,2}\)");
pattern!(paren_roman, r"^\s*\((?:[ivx]{1,5})\)");
pattern!(decimal, r"^\s*\d{1,3}(?:\.\d{1,3})+\.?\s");
pattern!(num, r"^\s*\d{1,3}[.)]\s");
pattern!(num_bare, r"^\s*\d{1,3}\s+[A-Z]");
pattern!(letter, r"^\s*[a-zA-Z][.)]\s");
pattern!(roman, r"^\s*[IVXivx]{1,5}[.)]\s");
pattern!(
    word_marker,
    r"(?i)^\s*(part|section|article|chapter|schedule|division|rule|regulation)\s+[\dIVX]"
);
pattern!(
    bare,
    r"^\s*(\(?[0-9]{1,3}[.)]?|\(?[ivxlcIVXLC]{1,6}[.)]|\(?[A-Za-z][.)]|\([a-zA-Z0-9]{1,4}\)|[•·▪◦‣\-–*\u{f0b7}\u{f0a7}\u{f0d8}])\s*$"
);
pattern!(
    court,
    r"(?i)^\W*(?:in\s+the\s+matter\s+of\b|between\W*$|before\W*$|in\s+the\s+.*\b(?:court|tribunal|registry)\b|(?:the\s+)?(?:supreme|high|superior|federal|district|provincial|county|circuit|family|magistrates?'?|labou?r|constitutional|appeal|appellate)\s+court\b|court\s+of\s+(?:appeal|queen's|king's|justice|session)\b|.*\btribunal\W*$)"
);
pattern!(
    address,
    r"(?i)(?:\b(?:tel(?:ephone)?|phone|fax|facsimile|cell)\b\.?\s*(?:no\.?)?\s*:?\s*[+(]?\d|\bp\.?\s?o\.?\s+box\b|\bdocex\b|[\w.+-]+@[\w-]+\.\w|\bsuite\s+\d|^\d{1,5}[a-z]?\s+[\w' -]+\s+(?:street|st|road|rd|avenue|ave|drive|dr|boulevard|blvd|lane|way|place|crescent|court)\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b)"
);
pattern!(
    party,
    r"(?i)\b(?:applicants?|respondents?|appellants?|plaintiffs?|defendants?|petitioners?|claimants?|accused|interven[eo]rs?)\W*$"
);
pattern!(
    citation,
    r"\s(?:v\.?|vs\.?)\s+[A-Z]|\[\d{4}\]\s*\w|\(\d{4}\),?\s+\d+\s+[A-Z]|\b\d+\s+[A-Z][A-Za-z.]{1,12}\s+\d+\b|\bS\.?C\.?R\.?\b|\bR\.S\.[A-Z]"
);
pattern!(
    defined,
    "[\"\u{201c}][^\"\u{201d}]{1,60}[\"\u{201d}]\\s+(?:means|includes|has the meaning)"
);
pattern!(leader, r"(?:\.{3,}|…|\s{3,})\s*\d+\s*$");
pattern!(
    form_number,
    r"(?i)^\W*(?:form|formule|formulaire)\s+[0-9]+[a-z]?(?:[.-][0-9a-z]+)*"
);

const LEAD_WORDS: &[&str] = &[
    "following",
    "include",
    "includes",
    "including",
    "means",
    "namely",
    "are",
    "is",
    "be",
    "to",
    "that",
    "whether",
];

fn families(text: &str) -> [bool; 10] {
    [
        bullet().is_match(text),
        paren_num().is_match(text),
        paren_letter().is_match(text),
        paren_roman().is_match(text),
        decimal().is_match(text),
        num().is_match(text),
        num_bare().is_match(text),
        letter().is_match(text),
        roman().is_match(text),
        word_marker().is_match(text),
    ]
}

fn family(text: &str) -> Option<usize> {
    families(text).iter().position(|f| *f)
}

fn counted(line: &Line) -> bool {
    !line.exclude_from_body
        && line.note_region_mode.is_empty()
        && matches!(line.region_type.as_str(), "body" | "heading")
        && !line.text.trim().is_empty()
}

fn bold(line: &Line) -> Option<bool> {
    let spans = line
        .spans
        .iter()
        .filter(|span| !span.text.trim().is_empty())
        .collect::<Vec<_>>();
    (!spans.is_empty()).then(|| {
        spans
            .iter()
            .all(|span| span.flags & 16 != 0 || span.font.to_lowercase().contains("bold"))
    })
}

fn italic(line: &Line) -> bool {
    let spans = line
        .spans
        .iter()
        .filter(|span| !span.text.trim().is_empty())
        .collect::<Vec<_>>();
    !spans.is_empty()
        && spans
            .iter()
            .all(|span| span.flags & 2 != 0 || span.font.to_lowercase().contains("ital"))
}

fn size(line: &Line) -> f64 {
    line.spans.iter().map(|span| span.size).fold(0.0, f64::max)
}

fn underlined(line: &Line) -> f32 {
    let total = line
        .words
        .iter()
        .map(|word| word.text.chars().count())
        .sum::<usize>();
    let marked = line
        .words
        .iter()
        .filter(|word| word.underlined)
        .map(|word| word.text.chars().count())
        .sum::<usize>();
    if total == 0 {
        0.0
    } else {
        marked as f32 / total as f32
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn normal(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect()
}

fn letters_key(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic())
        .take(40)
        .collect()
}

/// Every counted line's witnesses, in reading order.
pub fn witnesses(pages: &[Page], context: &Context) -> Table {
    collect_witnesses(pages, context, false)
}

/// Every physical source line, including lines without a reader classification.
pub fn source_witnesses(pages: &[Page], context: &Context) -> Table {
    collect_witnesses(pages, context, true)
}

fn collect_witnesses(pages: &[Page], context: &Context, all_source: bool) -> Table {
    let eligible = |line: &Line| all_source || counted(line);
    let lines = pages
        .iter()
        .enumerate()
        .flat_map(|(p, page)| {
            page.lines
                .iter()
                .enumerate()
                .filter(|(_, l)| eligible(l))
                .map(move |(s, l)| (p, s, l))
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Table { rows: Vec::new() };
    }
    let body_size = median(
        lines
            .iter()
            .map(|(_, _, l)| size(l))
            .filter(|s| *s > 0.0)
            .collect(),
    );
    let body_height = median(
        lines
            .iter()
            .map(|(_, _, l)| l.bbox[3] - l.bbox[1])
            .filter(|h| *h > 0.0)
            .collect(),
    )
    .max(1.0);
    let styled = lines
        .iter()
        .filter_map(|(_, _, l)| bold(l))
        .collect::<Vec<_>>();
    let bold_share = styled.iter().filter(|b| **b).count() as f32 / styled.len().max(1) as f32;
    let fonts = lines
        .iter()
        .filter_map(|(_, _, l)| l.spans.first().map(|s| s.font.clone()))
        .fold(HashMap::new(), |mut m, f| {
            *m.entry(f).or_insert(0usize) += 1;
            m
        });
    let body_font = fonts
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(f, _)| f.clone())
        .unwrap_or_default();
    let page_count = pages.len().max(1) as f32;
    // Text repeated page after page: running heads and titles.
    let mut seen_on = HashMap::<String, HashSet<usize>>::new();
    for (p, _, l) in &lines {
        seen_on.entry(normal(&l.text)).or_default().insert(*p);
    }
    // The style ladder: each style (bold, size step, capitals), ranked by prominence.
    let style = |l: &Line| {
        let letters = l.text.chars().filter(|c| c.is_alphabetic()).count();
        let caps =
            letters > 0 && l.text.chars().filter(|c| c.is_uppercase()).count() * 10 >= letters * 9;
        let step = if body_size > 0.0 && size(l) > 0.0 {
            (size(l) / body_size * 10.0).round() as i64
        } else {
            ((l.bbox[3] - l.bbox[1]) / body_height * 10.0).round() as i64
        };
        (bold(l) == Some(true), step, caps)
    };
    let mut style_count = HashMap::<(bool, i64, bool), usize>::new();
    for (_, _, l) in &lines {
        *style_count.entry(style(l)).or_default() += 1;
    }
    let mut ranked = style_count.keys().copied().collect::<Vec<_>>();
    ranked.sort_by_key(|(b, step, caps)| -(step * 4 + i64::from(*b) * 2 + i64::from(*caps)));
    let rank = ranked
        .iter()
        .enumerate()
        .map(|(n, st)| (*st, n))
        .collect::<HashMap<_, _>>();
    let body_style = style_count
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(st, _)| *st);
    // Backbone counts: a marker family spread over a third of the document's pages.
    let mut family_pages = HashMap::<usize, HashSet<usize>>::new();
    for (p, _, l) in &lines {
        if let Some(f) = family(&l.text) {
            family_pages.entry(f).or_default().insert(*p);
        }
    }
    let backbone = family_pages
        .iter()
        .filter(|(_, ps)| ps.len() as f32 >= (page_count / 3.0).max(2.0))
        .map(|(f, _)| *f)
        .collect::<HashSet<_>>();
    // The contents list's entries, as the structure stage read them, by their words.
    let contents_keys = context
        .contents
        .map(|ids| {
            lines
                .iter()
                .filter(|(_, _, l)| ids.contains(&l.id))
                .map(|(_, _, l)| letters_key(&leader().replace(&l.text, "")))
                .filter(|k| k.len() >= 4)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    // Pages set at another width than most, and pages switching between scan and type.
    let mut widths = HashMap::<i64, usize>::new();
    for page in pages {
        *widths.entry(page.width.round() as i64).or_default() += 1;
    }
    let usual_width = widths
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(w, _)| *w as f64)
        .unwrap_or(0.0);
    // Each page's text measure.
    let measures = pages
        .iter()
        .map(|page| {
            let mut lefts = page
                .lines
                .iter()
                .filter(|l| eligible(l))
                .map(|l| l.bbox[0])
                .collect::<Vec<_>>();
            let mut rights = page
                .lines
                .iter()
                .filter(|l| eligible(l))
                .map(|l| l.bbox[2])
                .collect::<Vec<_>>();
            if lefts.is_empty() {
                return (0.0, page.width.max(1.0));
            }
            lefts.sort_by(f64::total_cmp);
            rights.sort_by(f64::total_cmp);
            (
                lefts[lefts.len() / 10],
                rights[rights.len() - 1 - rights.len() / 10],
            )
        })
        .collect::<Vec<_>>();
    let lexicon = crate::lexicon::lexicon();
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let mut region_lines = HashMap::<(usize, &str), usize>::new();
    for (p, _, l) in &lines {
        if !l.region_id.is_empty() {
            *region_lines.entry((*p, l.region_id.as_str())).or_default() += 1;
        }
    }
    let same_region = |o: Option<(usize, usize, &Line)>, p: usize, l: &Line| {
        o.is_some_and(|(q, _, o)| q == p && o.region_id == l.region_id)
    };
    let mut rows = Vec::with_capacity(lines.len());
    for (n, &(p, s, l)) in lines.iter().enumerate() {
        let page = &pages[p];
        let text = l.text.trim();
        let (left, right) = measures[p];
        let measure = (right - left).max(1.0);
        let [x0, y0, x1, y1] = l.bbox;
        let prev = n.checked_sub(1).map(|m| lines[m]);
        let next = lines.get(n + 1).copied();
        let above = prev.filter(|(q, _, _)| *q == p).map(|(_, _, o)| o);
        let below = next.filter(|(q, _, _)| *q == p).map(|(_, _, o)| o);
        let gap_above = above.map(|o| ((y0 - o.bbox[3]) / body_height) as f32);
        let gap_below = below.map(|o| ((o.bbox[1] - y1) / body_height) as f32);
        let letters = text.chars().filter(|c| c.is_alphabetic()).count();
        let capitals = text.chars().filter(|c| c.is_uppercase()).count();
        let end = text.chars().last().unwrap_or(' ');
        let prev_text = prev.map_or("", |(_, _, o)| o.text.trim());
        let prev_end = prev_text.chars().last().unwrap_or(' ');
        let prev_words = prev_text
            .rsplit(|c: char| !c.is_alphabetic())
            .filter(|w| !w.is_empty())
            .take(1)
            .map(str::to_lowercase)
            .collect::<Vec<_>>();
        let marks = families(text);
        let own = style(l);
        let recognized = page.source != "native" || l.spans.is_empty();
        let layout = context
            .layout
            .and_then(|labels| labels.get(&l.id))
            .map(String::as_str);
        let contains = |set: Option<&HashSet<String>>| set.is_some_and(|ids| ids.contains(&l.id));
        let values = vec![
            flag(recognized),
            (page.index + 1) as f32 / page_count,
            flag(prev.is_none_or(|(q, _, _)| q != p)),
            flag(usual_width > 0.0 && (page.width - usual_width).abs() > 20.0),
            flag(p > 0 && pages[p - 1].source != page.source),
            bold_share,
            flag(
                seen_on
                    .get(&normal(&l.text))
                    .is_some_and(|ps| ps.len() >= 3),
            ),
            flag(bold(l) == Some(true)),
            flag(italic(l)),
            flag(
                l.spans
                    .first()
                    .is_some_and(|sp| !body_font.is_empty() && sp.font != body_font),
            ),
            if body_size > 0.0 && size(l) > 0.0 {
                (size(l) / body_size) as f32
            } else {
                0.0
            },
            underlined(l),
            flag(prev.is_some_and(|(_, _, o)| bold(o) == Some(true))),
            flag(next.is_some_and(|(_, _, o)| bold(o) == Some(true))),
            ((y1 - y0) / body_height) as f32,
            ((x1 - x0) / measure) as f32,
            ((x0 - left) / measure) as f32,
            ((right - x1) / measure) as f32,
            flag(
                (((x0 + x1) / 2.0 - (left + right) / 2.0) / measure).abs() < 0.08
                    && (x0 - left) / measure > 0.08,
            ),
            gap_above.unwrap_or(-1.0),
            gap_below.unwrap_or(-1.0),
            flag(gap_above.is_none()),
            flag(gap_below.is_none()),
            text.chars().count() as f32,
            text.split_whitespace().count() as f32,
            if letters > 0 {
                capitals as f32 / letters as f32
            } else {
                0.0
            },
            flag(
                text.chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit()),
            ),
            flag(text.chars().next().is_some_and(char::is_lowercase)),
            letters as f32,
            flag(end == ':'),
            flag(end == '.'),
            flag(end == ','),
            flag(end == ';'),
            flag(end == ')'),
            flag(matches!(end, '-' | '\u{2013}' | '\u{2014}')),
            flag(prev_end == ':'),
            flag(prev_end == '.'),
            flag(prev_end == ','),
            flag(prev_end == ';'),
            flag(matches!(prev_end, '-' | '\u{2013}' | '\u{2014}')),
            flag(next.is_some_and(|(_, _, o)| o.text.trim_start().starts_with(char::is_lowercase))),
            flag(
                matches!(prev_end, ':' | '-' | '\u{2013}' | '\u{2014}')
                    || prev_words
                        .first()
                        .is_some_and(|w| LEAD_WORDS.contains(&w.as_str())),
            ),
            flag(court().is_match(text)),
            flag(address().is_match(text)),
            flag(party().is_match(text)),
            flag(citation().is_match(text)),
            flag(defined().is_match(text)),
            flag(leader().is_match(text)),
            flag(lexicon.form_title(text)),
            flag(form_number().is_match(text)),
            flag(lexicon.court(text)),
            flag(lexicon.boilerplate(text)),
            flag(marks.iter().any(|m| *m)),
            flag(bare().is_match(text)),
            flag(marks[0]),
            flag(marks[1]),
            flag(marks[2]),
            flag(marks[3]),
            flag(marks[4]),
            flag(marks[5]),
            flag(marks[6]),
            flag(marks[7]),
            flag(marks[8]),
            flag(marks[9]),
            flag(family(text).is_some_and(|f| backbone.contains(&f))),
            rank.get(&own).copied().unwrap_or(0) as f32,
            style_count.get(&own).copied().unwrap_or(0) as f32,
            flag(Some(own) == body_style),
            flag(
                !contains(context.contents)
                    && letters_key(text).len() >= 4
                    && contents_keys.contains(&letters_key(text)),
            ),
            flag(l.region_type == "heading"),
            flag(contains(context.quoted)),
            flag(contains(context.table_cells)),
            flag(contains(context.translation)),
            flag(matches!(
                layout,
                Some("paragraph_title" | "doc_title" | "heading")
            )),
            flag(matches!(layout, Some("list" | "list_item"))),
            flag(layout.is_some()),
            flag(!l.region_id.is_empty() && !same_region(prev, p, l)),
            flag(!l.region_id.is_empty() && !same_region(next, p, l)),
            region_lines
                .get(&(p, l.region_id.as_str()))
                .copied()
                .unwrap_or(0) as f32,
        ];
        let mut values = values;
        let coordinates = legal_structure_model::ScalarText::new(&l.text);
        let tags = l
            .spans
            .iter()
            .filter(|span| {
                span.start < span.end
                    && coordinates.slice(span.start..span.end) == Some(span.text.as_str())
            })
            .filter_map(|span| span.structure_role.as_deref())
            .collect::<std::collections::HashSet<_>>();
        values.extend([
            flag(!tags.is_empty()),
            flag(
                tags.iter()
                    .any(|role| matches!(*role, "H" | "H1" | "H2" | "H3" | "H4" | "H5" | "H6")),
            ),
            flag(tags.contains("Title")),
            flag(tags.contains("P")),
            flag(
                tags.iter()
                    .any(|role| matches!(*role, "L" | "LI" | "Lbl" | "LBody")),
            ),
            flag(
                tags.iter()
                    .any(|role| matches!(*role, "Table" | "TR" | "TH" | "TD")),
            ),
            flag(tags.contains("Note")),
            flag(
                tags.iter()
                    .any(|role| matches!(*role, "Reference" | "Link")),
            ),
            flag(tags.contains("Caption")),
            flag(tags.contains("Artifact")),
            flag(
                tags.iter()
                    .any(|role| matches!(*role, "BlockQuote" | "Quote")),
            ),
        ]);
        debug_assert_eq!(values.len(), NAMES.len());
        let spans = l
            .spans
            .iter()
            .filter(|span| {
                span.start < span.end
                    && coordinates.slice(span.start..span.end) == Some(span.text.as_str())
            })
            .map(|span| SpanWitness {
                start: span.start,
                end: span.end,
                size: if body_size > 0.0 {
                    (span.size / body_size) as f32
                } else {
                    0.0
                },
                bold: span.flags & 16 != 0 || span.font.to_lowercase().contains("bold"),
                italic: span.flags & 2 != 0 || span.font.to_lowercase().contains("ital"),
                own_font: span.font != body_font,
                style: [0.0; 3],
                role: span.structure_role.clone(),
            })
            .collect();
        let words = l
            .words
            .iter()
            .filter(|word| {
                word.start < word.end
                    && coordinates.slice(word.start..word.end) == Some(word.text.as_str())
            })
            .map(|word| (word.start, word.end, word.underlined))
            .collect();
        rows.push(Row {
            source_range: None,
            spans,
            words,
            page: page.index,
            slot: s,
            id: l.id.clone(),
            text: l.text.clone(),
            recognized,
            values,
        });
    }
    exact_styles(&mut rows);
    Table { rows }
}

/// Whole-source style frequencies use aligned spans, so inline headings and body text
/// on the same physical line do not inherit one another's prominence or frequency.
fn exact_styles(rows: &mut [Row]) {
    let key = |span: &SpanWitness| ((span.size * 100.0).round() as i32, span.bold, span.italic);
    let mut lines = BTreeMap::<_, HashSet<usize>>::new();
    let mut characters = BTreeMap::<_, usize>::new();
    for (index, row) in rows.iter().enumerate() {
        let text = legal_structure_model::ScalarText::new(&row.text);
        for span in &row.spans {
            let count = text.slice(span.start..span.end).unwrap_or_default().chars()
                .filter(|character| !character.is_whitespace()).count();
            if count == 0 || span.size <= 0.0 { continue; }
            lines.entry(key(span)).or_default().insert(index);
            *characters.entry(key(span)).or_default() += count;
        }
    }
    let ranks = characters.keys().rev().enumerate().map(|(rank, style)| (*style, rank))
        .collect::<BTreeMap<_, _>>();
    let body = characters.iter().max_by_key(|(style, count)| (**count, **style)).map(|(style, _)| *style);
    for row in rows {
        for span in &mut row.spans {
            if span.size <= 0.0 { span.style = [0.0; 3]; continue; }
            let style = key(span);
            span.style = [ranks.get(&style).copied().unwrap_or(0) as f32,
                lines.get(&style).map_or(0, HashSet::len) as f32,
                if body == Some(style) { 1.0 } else { 0.0 }];
        }
    }
}
