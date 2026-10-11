//! Every counted line's witnesses, read against its page and its document.

use legal_pdf_core::model::{Line, Page};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// The witnesses, in the order of each row's values. Flags are 1 or 0; measures are ratios to the
/// document's body (type size, line height), fractions of the page's text measure, or counts.
pub const NAMES: &[&str] = &[
    // The page and document
    "recognized", "page_frac", "page_first", "odd_width", "source_switch", "bold_share", "repeats",
    // Type
    "bold", "italic", "own_font", "size", "underlined", "prev_bold", "next_bold",
    // Geometry
    "height", "width", "indent", "rindent", "centered", "gap_above", "gap_below", "gap_above_none", "gap_below_none",
    // Words
    "chars", "words", "caps", "first_upper", "first_lower", "letters",
    "end_colon", "end_period", "end_comma", "end_semicolon", "end_paren", "end_dash",
    "prev_end_colon", "prev_end_period", "prev_end_comma", "prev_end_semicolon", "prev_end_dash", "next_lower",
    "lead_in", "court", "address", "party", "citation", "defined_term", "dot_leader",
    // The court-forms store: a form's title or number, a court's name, a form's printed boilerplate
    "form_title", "form_number", "known_court", "form_boilerplate",
    // Markers
    "marker_any", "marker_only", "m_bullet", "m_paren_num", "m_paren_letter", "m_paren_roman", "m_decimal",
    "m_num", "m_num_bare", "m_letter", "m_roman", "m_word", "backbone",
    // The document's style ladder and outline
    "style_rank", "style_lines", "body_style", "in_contents",
    // The readers' verdicts so far
    "reader_heading", "quoted", "table_cell", "translation", "layout_title", "layout_list", "layout_known",
    // The layout model's other labels, and the line's place in its region (the layout model's, or the
    // readers' where none ran)
    "layout_text", "layout_quote", "layout_note", "layout_table", "layout_caption", "layout_furniture",
    "layout_reference", "layout_byline", "layout_formula", "region_first", "region_last", "region_lines",
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

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Row {
    pub page: usize,
    pub slot: usize,
    pub id: String,
    pub text: String,
    pub recognized: bool,
    pub values: Vec<f32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub rows: Vec<Row>,
}

impl Table {
    pub fn by_id(&self) -> HashMap<&str, &Row> {
        self.rows.iter().map(|row| (row.id.as_str(), row)).collect()
    }
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

pattern!(bullet, r"^\s*[•·▪◦‣●○■□➢✓\u{f0b7}\u{f0a7}\u{f0d8}\u{f076}*–-]\s");
pattern!(paren_num, r"^\s*\(\d{1,3}\)");
pattern!(paren_letter, r"^\s*\([a-z]{1,2}\)");
pattern!(paren_roman, r"^\s*\((?:[ivx]{1,5})\)");
pattern!(decimal, r"^\s*\d{1,3}(?:\.\d{1,3})+\.?\s");
pattern!(num, r"^\s*\d{1,3}[.)]\s");
pattern!(num_bare, r"^\s*\d{1,3}\s+[A-Z]");
pattern!(letter, r"^\s*[a-zA-Z][.)]\s");
pattern!(roman, r"^\s*[IVXivx]{1,5}[.)]\s");
pattern!(word_marker, r"(?i)^\s*(part|section|article|chapter|schedule|division|rule|regulation)\s+[\dIVX]");
pattern!(bare, r"^\s*(\(?[0-9]{1,3}[.)]?|\(?[ivxlcIVXLC]{1,6}[.)]|\(?[A-Za-z][.)]|\([a-zA-Z0-9]{1,4}\)|[•·▪◦‣\-–*\u{f0b7}\u{f0a7}\u{f0d8}])\s*$");
pattern!(court, r"(?i)^\W*(?:in\s+the\s+matter\s+of\b|between\W*$|before\W*$|in\s+the\s+.*\b(?:court|tribunal|registry)\b|(?:the\s+)?(?:supreme|high|superior|federal|district|provincial|county|circuit|family|magistrates?'?|labou?r|constitutional|appeal|appellate)\s+court\b|court\s+of\s+(?:appeal|queen's|king's|justice|session)\b|.*\btribunal\W*$)");
pattern!(address, r"(?i)(?:\b(?:tel(?:ephone)?|phone|fax|facsimile|cell)\b\.?\s*(?:no\.?)?\s*:?\s*[+(]?\d|\bp\.?\s?o\.?\s+box\b|\bdocex\b|[\w.+-]+@[\w-]+\.\w|\bsuite\s+\d|^\d{1,5}[a-z]?\s+[\w' -]+\s+(?:street|st|road|rd|avenue|ave|drive|dr|boulevard|blvd|lane|way|place|crescent|court)\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b)");
pattern!(party, r"(?i)\b(?:applicants?|respondents?|appellants?|plaintiffs?|defendants?|petitioners?|claimants?|accused|interven[eo]rs?)\W*$");
pattern!(citation, r"\s(?:v\.?|vs\.?)\s+[A-Z]|\[\d{4}\]\s*\w|\(\d{4}\),?\s+\d+\s+[A-Z]|\b\d+\s+[A-Z][A-Za-z.]{1,12}\s+\d+\b|\bS\.?C\.?R\.?\b|\bR\.S\.[A-Z]");
pattern!(defined, "[\"\u{201c}][^\"\u{201d}]{1,60}[\"\u{201d}]\\s+(?:means|includes|has the meaning)");
pattern!(leader, r"(?:\.{3,}|…|\s{3,})\s*\d+\s*$");
pattern!(form_number, r"(?i)^\W*(?:form|formule|formulaire)\s+[0-9]+[a-z]?(?:[.-][0-9a-z]+)*");

const LEAD_WORDS: &[&str] = &["following", "include", "includes", "including", "means", "namely", "are", "is", "be", "to", "that", "whether"];

fn families(text: &str) -> [bool; 10] {
    [
        bullet().is_match(text), paren_num().is_match(text), paren_letter().is_match(text), paren_roman().is_match(text),
        decimal().is_match(text), num().is_match(text), num_bare().is_match(text), letter().is_match(text),
        roman().is_match(text), word_marker().is_match(text),
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
    let spans = line.spans.iter().filter(|span| !span.text.trim().is_empty()).collect::<Vec<_>>();
    (!spans.is_empty()).then(|| spans.iter().all(|span| span.flags & 16 != 0 || span.font.to_lowercase().contains("bold")))
}

fn italic(line: &Line) -> bool {
    let spans = line.spans.iter().filter(|span| !span.text.trim().is_empty()).collect::<Vec<_>>();
    !spans.is_empty() && spans.iter().all(|span| span.flags & 2 != 0 || span.font.to_lowercase().contains("ital"))
}

fn size(line: &Line) -> f64 {
    line.spans.iter().map(|span| span.size).fold(0.0, f64::max)
}

fn underlined(line: &Line) -> f32 {
    let total = line.words.iter().map(|word| word.text.chars().count()).sum::<usize>();
    let marked = line.words.iter().filter(|word| word.underlined).map(|word| word.text.chars().count()).sum::<usize>();
    if total == 0 { 0.0 } else { marked as f32 / total as f32 }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn normal(text: &str) -> String {
    text.trim().to_lowercase().chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect()
}

fn letters_key(text: &str) -> String {
    text.to_lowercase().chars().filter(|c| c.is_alphabetic()).take(40).collect()
}

/// Every counted line's witnesses, in reading order.
pub fn witnesses(pages: &[Page], context: &Context) -> Table {
    let lines = pages
        .iter()
        .enumerate()
        .flat_map(|(p, page)| page.lines.iter().enumerate().filter(|(_, l)| counted(l)).map(move |(s, l)| (p, s, l)))
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Table { rows: Vec::new() };
    }
    let body_size = median(lines.iter().map(|(_, _, l)| size(l)).filter(|s| *s > 0.0).collect());
    let body_height = median(lines.iter().map(|(_, _, l)| l.bbox[3] - l.bbox[1]).filter(|h| *h > 0.0).collect()).max(1.0);
    let styled = lines.iter().filter_map(|(_, _, l)| bold(l)).collect::<Vec<_>>();
    let bold_share = styled.iter().filter(|b| **b).count() as f32 / styled.len().max(1) as f32;
    let fonts = lines.iter().filter_map(|(_, _, l)| l.spans.first().map(|s| s.font.clone())).fold(HashMap::new(), |mut m, f| {
        *m.entry(f).or_insert(0usize) += 1;
        m
    });
    let body_font = fonts.iter().max_by_key(|(_, n)| **n).map(|(f, _)| f.clone()).unwrap_or_default();
    let page_count = pages.len().max(1) as f32;
    // Text repeated page after page: running heads and titles.
    let mut seen_on = HashMap::<String, HashSet<usize>>::new();
    for (p, _, l) in &lines {
        seen_on.entry(normal(&l.text)).or_default().insert(*p);
    }
    // The style ladder: each style (bold, size step, capitals), ranked by prominence.
    let style = |l: &Line| {
        let letters = l.text.chars().filter(|c| c.is_alphabetic()).count();
        let caps = letters > 0 && l.text.chars().filter(|c| c.is_uppercase()).count() * 10 >= letters * 9;
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
    let rank = ranked.iter().enumerate().map(|(n, st)| (*st, n)).collect::<HashMap<_, _>>();
    let body_style = style_count.iter().max_by_key(|(_, n)| **n).map(|(st, _)| *st);
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
            lines.iter().filter(|(_, _, l)| ids.contains(&l.id)).map(|(_, _, l)| letters_key(&leader().replace(&l.text, ""))).filter(|k| k.len() >= 4).collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    // Pages set at another width than most, and pages switching between scan and type.
    let mut widths = HashMap::<i64, usize>::new();
    for page in pages {
        *widths.entry(page.width.round() as i64).or_default() += 1;
    }
    let usual_width = widths.iter().max_by_key(|(_, n)| **n).map(|(w, _)| *w as f64).unwrap_or(0.0);
    // Each page's text measure.
    let measures = pages
        .iter()
        .map(|page| {
            let mut lefts = page.lines.iter().filter(|l| counted(l)).map(|l| l.bbox[0]).collect::<Vec<_>>();
            let mut rights = page.lines.iter().filter(|l| counted(l)).map(|l| l.bbox[2]).collect::<Vec<_>>();
            if lefts.is_empty() {
                return (0.0, page.width.max(1.0));
            }
            lefts.sort_by(f64::total_cmp);
            rights.sort_by(f64::total_cmp);
            (lefts[lefts.len() / 10], rights[rights.len() - 1 - rights.len() / 10])
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
    let same_region = |o: Option<(usize, usize, &Line)>, p: usize, l: &Line| o.is_some_and(|(q, _, o)| q == p && o.region_id == l.region_id);
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
        let layout = context.layout.and_then(|labels| labels.get(&l.id)).map(String::as_str);
        let contains = |set: Option<&HashSet<String>>| set.is_some_and(|ids| ids.contains(&l.id));
        let values = vec![
            flag(recognized),
            (page.index + 1) as f32 / page_count,
            flag(prev.is_none_or(|(q, _, _)| q != p)),
            flag(usual_width > 0.0 && (page.width - usual_width).abs() > 20.0),
            flag(p > 0 && pages[p - 1].source != page.source),
            bold_share,
            flag(seen_on.get(&normal(&l.text)).is_some_and(|ps| ps.len() >= 3)),
            flag(bold(l) == Some(true)),
            flag(italic(l)),
            flag(l.spans.first().is_some_and(|sp| !body_font.is_empty() && sp.font != body_font)),
            if body_size > 0.0 && size(l) > 0.0 { (size(l) / body_size) as f32 } else { 0.0 },
            underlined(l),
            flag(prev.is_some_and(|(_, _, o)| bold(o) == Some(true))),
            flag(next.is_some_and(|(_, _, o)| bold(o) == Some(true))),
            ((y1 - y0) / body_height) as f32,
            ((x1 - x0) / measure) as f32,
            ((x0 - left) / measure) as f32,
            ((right - x1) / measure) as f32,
            flag((((x0 + x1) / 2.0 - (left + right) / 2.0) / measure).abs() < 0.08 && (x0 - left) / measure > 0.08),
            gap_above.unwrap_or(-1.0),
            gap_below.unwrap_or(-1.0),
            flag(gap_above.is_none()),
            flag(gap_below.is_none()),
            text.chars().count() as f32,
            text.split_whitespace().count() as f32,
            if letters > 0 { capitals as f32 / letters as f32 } else { 0.0 },
            flag(text.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())),
            flag(text.chars().next().is_some_and(char::is_lowercase)),
            letters as f32,
            flag(end == ':'), flag(end == '.'), flag(end == ','), flag(end == ';'), flag(end == ')'),
            flag(matches!(end, '-' | '\u{2013}' | '\u{2014}')),
            flag(prev_end == ':'), flag(prev_end == '.'), flag(prev_end == ','), flag(prev_end == ';'),
            flag(matches!(prev_end, '-' | '\u{2013}' | '\u{2014}')),
            flag(next.is_some_and(|(_, _, o)| o.text.trim_start().starts_with(char::is_lowercase))),
            flag(matches!(prev_end, ':' | '-' | '\u{2013}' | '\u{2014}') || prev_words.first().is_some_and(|w| LEAD_WORDS.contains(&w.as_str()))),
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
            flag(marks[0]), flag(marks[1]), flag(marks[2]), flag(marks[3]), flag(marks[4]),
            flag(marks[5]), flag(marks[6]), flag(marks[7]), flag(marks[8]), flag(marks[9]),
            flag(family(text).is_some_and(|f| backbone.contains(&f))),
            rank.get(&own).copied().unwrap_or(0) as f32,
            style_count.get(&own).copied().unwrap_or(0) as f32,
            flag(Some(own) == body_style),
            flag(!contains(context.contents) && letters_key(text).len() >= 4 && contents_keys.contains(&letters_key(text))),
            flag(l.region_type == "heading"),
            flag(contains(context.quoted)),
            flag(contains(context.table_cells)),
            flag(contains(context.translation)),
            flag(matches!(layout, Some("paragraph_title" | "doc_title" | "heading"))),
            flag(matches!(layout, Some("list" | "list_item"))),
            flag(layout.is_some()),
            flag(matches!(layout, Some("text" | "content" | "body" | "abstract" | "vertical_text"))),
            flag(matches!(layout, Some("block_quote" | "quote"))),
            flag(matches!(layout, Some("footnote" | "vision_footnote" | "note"))),
            flag(layout.is_some_and(|label| label.starts_with("table") || label == "form" || label == "key_value_region")),
            flag(matches!(layout, Some("caption" | "figure_title" | "chart_title" | "table_title"))),
            flag(matches!(layout, Some("header" | "footer" | "page_header" | "page_footer" | "number" | "header_image" | "footer_image"))),
            flag(matches!(layout, Some("reference" | "reference_content" | "document_index"))),
            flag(matches!(layout, Some("byline" | "aside_text"))),
            flag(matches!(layout, Some("formula" | "formula_number" | "algorithm" | "code"))),
            flag(!l.region_id.is_empty() && !same_region(prev, p, l)),
            flag(!l.region_id.is_empty() && !same_region(next, p, l)),
            region_lines.get(&(p, l.region_id.as_str())).copied().unwrap_or(0) as f32,
        ];
        debug_assert_eq!(values.len(), NAMES.len());
        rows.push(Row { page: page.index, slot: s, id: l.id.clone(), text: text.chars().take(80).collect(), recognized, values });
    }
    Table { rows }
}
