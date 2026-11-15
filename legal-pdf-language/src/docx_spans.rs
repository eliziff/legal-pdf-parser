//! Writes into a Word document's body paragraphs and footnotes at spans of their text as `docx_to_toa_text_units`
//! read it (a paragraph by its path, a note by its id; UTF-16 offsets): a span replaced as a tracked change,
//! highlighted, or linked; and tracked revisions rejected by id. Only the runs an edit touches are rewritten.

use legal_pdf_core::{Error, Result};
use quick_xml::events::Event;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
pub struct DocxUnit {
    pub id: String,
    pub text: String,
    pub path: Option<Vec<usize>>,
}

#[derive(Deserialize, Clone)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DocxEdit {
    Replace { replacement: String },
    Highlight { color: String },
    Link { url: String },
}

/// A write into a unit's span, `expected` its text there as read.
#[derive(Deserialize, Clone)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct DocxSpanWrite {
    pub unit_id: String,
    pub start: usize,
    pub end: usize,
    pub expected: String,
    #[serde(flatten)]
    pub edit: DocxEdit,
}

pub struct DocxSpansWritten {
    pub bytes: Vec<u8>,
    pub written: Vec<bool>,
    pub revisions: Vec<u32>,
}

/// An element: its name, and where its open tag, content and whole lie.
pub(crate) struct Node {
    pub name: String,
    pub start: usize,
    pub open_end: usize,
    pub inner_end: usize,
    pub end: usize,
    pub children: Vec<Node>,
}

impl Node {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|child| child.name == name)
    }
}

pub(crate) fn parse(xml: &str) -> Result<Vec<Node>> {
    let mut reader = quick_xml::Reader::from_str(xml);
    let (mut roots, mut stack) = (Vec::new(), Vec::<Node>::new());
    loop {
        let start = reader.buffer_position() as usize;
        let event = reader.read_event().map_err(|error| Error::Message(format!("DOCX XML error: {error}")))?;
        let end = reader.buffer_position() as usize;
        let node = |name: &str| Node { name: name.to_owned(), start, open_end: end, inner_end: end, end, children: Vec::new() };
        let closed = match event {
            Event::Start(tag) => { stack.push(node(tag.name().as_ref())); None }
            Event::Empty(tag) => Some(node(tag.name().as_ref())),
            Event::End(_) => stack.pop().map(|mut open| { (open.inner_end, open.end) = (start, end); open }),
            Event::Eof if stack.is_empty() => return Ok(roots),
            Event::Eof => return Err(Error::Message("DOCX XML has an unclosed element".to_owned())),
            _ => None,
        };
        if let Some(closed) = closed {
            stack.last_mut().map_or(&mut roots, |parent| &mut parent.children).push(closed);
        }
    }
}

pub(crate) fn escaped(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

pub(crate) fn text_of(xml: &str, node: &Node) -> Vec<u16> {
    quick_xml::escape::unescape(&xml[node.open_end..node.inner_end]).unwrap_or_default().encode_utf16().collect()
}

pub(crate) fn text_element(tag: &str, text: &str) -> String {
    let space = if text.starts_with(char::is_whitespace) || text.ends_with(char::is_whitespace) { " xml:space=\"preserve\"" } else { "" };
    format!("<{tag}{space}>{}</{tag}>", escaped(text))
}

/// A run: its element, its parent, its text's place in the unit, and where each tab, break or field mark in it ends.
pub(crate) struct Run<'a> {
    pub node: &'a Node,
    pub parent: &'a Node,
    pub start: usize,
    pub end: usize,
    marks: Vec<usize>,
}

/// A unit's text (UTF-16) and runs; deleted text and text boxes (and older readers' fallback) are not its own.
pub(crate) fn reading<'a>(xml: &str, roots: &'a [Node]) -> (Vec<u16>, Vec<Run<'a>>) {
    fn read<'a>(node: &'a Node, parent: Option<&'a Node>, xml: &str, text: &mut Vec<u16>, runs: &mut Vec<Run<'a>>) {
        match node.name.as_str() {
            "w:del" | "w:txbxContent" | "mc:Fallback" => {}
            "w:t" => text.extend(text_of(xml, node)),
            "w:tab" => text.push(9),
            "w:br" | "w:cr" => text.push(10),
            "w:r" if parent.is_some() => {
                let (start, mut marks) = (text.len(), Vec::new());
                for child in &node.children {
                    read(child, Some(node), xml, text, &mut Vec::new());
                    if matches!(child.name.as_str(), "w:tab" | "w:br" | "w:cr" | "w:fldChar" | "w:instrText") {
                        marks.push(text.len());
                    }
                }
                runs.push(Run { node, parent: parent.unwrap(), start, end: text.len(), marks });
            }
            _ => node.children.iter().for_each(|child| read(child, Some(node), xml, text, runs)),
        }
    }
    let (mut text, mut runs) = (Vec::new(), Vec::new());
    roots.iter().for_each(|root| read(root, None, xml, &mut text, &mut runs));
    (text, runs)
}

/// The unit with each run that prints text across `start` or `end` cut there, one run a child; None where the span
/// crosses a tab, a break or a field.
fn cut(xml: &str, start: usize, end: usize) -> Result<Option<String>> {
    let roots = parse(xml)?;
    let (_, runs) = reading(xml, &roots);
    if runs.iter().any(|run| run.start < end && run.marks.iter().any(|mark| start < *mark)) {
        return Ok(None);
    }
    let mut output = xml.to_owned();
    let reached = |run: &&Run| run.start < end.max(start + 1) && start < run.end && (start < end || run.start < start);
    for run in runs.iter().rev().filter(|run| run.end > run.start).filter(reached) {
        let open = &xml[run.node.start..run.node.open_end];
        let properties = run.node.child("w:rPr").map_or("", |rpr| &xml[rpr.start..rpr.end]);
        let (mut at, mut pieces) = (run.start, String::new());
        for child in run.node.children.iter().filter(|child| child.name != "w:rPr") {
            if child.name != "w:t" {
                pieces += &format!("{open}{properties}{}</w:r>", &xml[child.start..child.end]);
                continue;
            }
            let text = text_of(xml, child);
            let mut cuts = [start, end].into_iter().filter(|cut| at < *cut && *cut < at + text.len()).map(|cut| cut - at).collect::<Vec<_>>();
            cuts.dedup();
            for (from, to) in std::iter::once(0).chain(cuts.clone()).zip(cuts.into_iter().chain([text.len()])) {
                pieces += &format!("{open}{properties}{}</w:r>", text_element("w:t", &String::from_utf16_lossy(&text[from..to])));
            }
            at += text.len();
        }
        output.replace_range(run.node.start..run.node.end, &pieces);
    }
    Ok(Some(output))
}

/// A run property's place among a run's properties (CT_RPr's order), which Word holds a document to.
const ORDER: [&str; 39] = ["w:rStyle", "w:rFonts", "w:b", "w:bCs", "w:i", "w:iCs", "w:caps", "w:smallCaps", "w:strike",
    "w:dstrike", "w:outline", "w:shadow", "w:emboss", "w:imprint", "w:noProof", "w:snapToGrid", "w:vanish", "w:webHidden",
    "w:color", "w:spacing", "w:w", "w:kern", "w:position", "w:sz", "w:szCs", "w:highlight", "w:u", "w:effect", "w:bdr",
    "w:shd", "w:fitText", "w:vertAlign", "w:rtl", "w:cs", "w:em", "w:lang", "w:eastAsianLayout", "w:specVanish", "w:oMath"];

/// The run with each `(property, value)` set, in its place among the others.
fn with_properties(run: &Node, xml: &str, set: &[(&str, &str)]) -> String {
    let rank = |name: &str| ORDER.iter().position(|known| *known == name).unwrap_or(ORDER.len());
    let existing = run.child("w:rPr");
    let mut properties = existing.into_iter().flat_map(|rpr| &rpr.children).filter(|child| set.iter().all(|(name, _)| *name != child.name))
        .map(|child| (rank(&child.name), xml[child.start..child.end].to_owned())).collect::<Vec<_>>();
    properties.extend(set.iter().map(|(name, value)| (rank(name), format!("<{name} w:val=\"{value}\"/>"))));
    properties.sort_by_key(|(rank, _)| *rank);
    let (before, after) = existing.map_or((run.open_end, run.open_end), |rpr| (rpr.start, rpr.end));
    format!("{}<w:rPr>{}</w:rPr>{}", &xml[run.start..before], properties.into_iter().map(|(_, xml)| xml).collect::<String>(), &xml[after..run.end])
}

const RELATIONSHIPS: [&str; 2] = ["word/_rels/document.xml.rels", "word/_rels/footnotes.xml.rels"];

/// What a document's writes share: the next revision id and those written, by whom and when, whether links take the
/// Hyperlink style, and each part's relationships (linked where a link was added).
struct Writer<'a> {
    next: u32,
    revisions: Vec<u32>,
    author: &'a str,
    date: String,
    styled: bool,
    relationships: [(String, bool); 2],
}

impl Writer<'_> {
    fn revision(&mut self, tag: &str, content: &str) -> String {
        self.revisions.push(self.next);
        self.next += 1;
        format!("<{tag} w:id=\"{}\" w:author=\"{}\" w:date=\"{}\">{content}</{tag}>", self.next - 1, escaped(self.author), self.date)
    }

    /// A relationship of `part`'s to the address, by the id it is given.
    fn relate(&mut self, part: usize, url: &str) -> String {
        static ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"Id="rId(\d+)""#).expect("relationship"));
        let (xml, linked) = &mut self.relationships[part];
        let id = format!("rId{}", ID.captures_iter(xml).filter_map(|id| id[1].parse::<u32>().ok()).max().unwrap_or(0) + 1);
        let relationship = format!("<Relationship Id=\"{id}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink\" Target=\"{}\" TargetMode=\"External\"/>", escaped(url));
        let at = xml.rfind("</Relationships>").unwrap_or(xml.len());
        xml.insert_str(at, &relationship);
        *linked = true;
        id
    }
}

static DELETED_TEXT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<(/?)w:t([\s>])").expect("w:t"));

/// The unit with one write made on runs cut at its ends; None where no run prints the span's ends, or a link would not
/// wrap runs side by side in one paragraph.
fn apply(xml: &str, part: usize, start: usize, end: usize, edit: &DocxEdit, writer: &mut Writer) -> Result<Option<String>> {
    let roots = parse(xml)?;
    let (_, runs) = reading(xml, &roots);
    let inside = (0..runs.len()).filter(|at| runs[*at].end > runs[*at].start && start <= runs[*at].start && runs[*at].end <= end).collect::<Vec<_>>();
    let (first, last) = (inside.first().copied(), inside.last().copied());
    if start < end && (first.map(|at| runs[at].start) != Some(start) || last.map(|at| runs[at].end) != Some(end))
        || !matches!(edit, DocxEdit::Replace { .. }) && inside.is_empty() {
        return Ok(None);
    }
    let whole = |at: usize| &xml[runs[at].node.start..runs[at].node.end];
    // Each edit's runs and what replaces them, in the unit's order.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    match edit {
        DocxEdit::Replace { replacement } => {
            for &at in &inside {
                let deleted = writer.revision("w:del", &DELETED_TEXT.replace_all(whole(at), "<${1}w:delText$2"));
                edits.push((at, at, deleted));
            }
            // The insertion: after the last run deleted; else after the run ending where it begins, or before the one
            // starting there; in that run's properties.
            let anchor = last.map(|at| (at, true)).or_else(|| runs.iter().position(|run| run.end == start && run.start < start).map(|at| (at, true)))
                .or_else(|| runs.iter().position(|run| run.start == start && run.end > start).map(|at| (at, false)));
            let Some((at, after)) = anchor else { return Ok(None) };
            if !replacement.is_empty() {
                let properties = runs[at].node.child("w:rPr").map_or("", |rpr| &xml[rpr.start..rpr.end]);
                let inserted = writer.revision("w:ins", &format!("<w:r>{properties}{}</w:r>", text_element("w:t", replacement)));
                match edits.iter_mut().find(|(from, ..)| *from == at) {
                    Some((.., with)) => with.push_str(&inserted),
                    None if after => edits.push((at, at, format!("{}{inserted}", whole(at)))),
                    None => edits.push((at, at, format!("{inserted}{}", whole(at)))),
                }
            }
        }
        DocxEdit::Highlight { color } => edits.extend(inside.iter().map(|&at| (at, at, with_properties(runs[at].node, xml, &[("w:highlight", color)])))),
        DocxEdit::Link { url } => {
            let (from, to, parent) = (first.unwrap(), last.unwrap(), runs[first.unwrap()].parent);
            if parent.name != "w:p" || (from..=to).any(|at| !std::ptr::eq(runs[at].parent, parent))
                || (from..to).any(|at| !xml[runs[at].node.end..runs[at + 1].node.start].trim().is_empty()) {
                return Ok(None);
            }
            let style: &[(&str, &str)] = if writer.styled { &[("w:rStyle", "Hyperlink")] } else { &[("w:color", "0563C1"), ("w:u", "single")] };
            let linked = (from..=to).map(|at| with_properties(runs[at].node, xml, style)).collect::<String>();
            edits.push((from, to, format!("<w:hyperlink r:id=\"{}\" w:history=\"1\">{linked}</w:hyperlink>", writer.relate(part, url))));
        }
    }
    let mut output = xml.to_owned();
    edits.sort_by_key(|(from, ..)| std::cmp::Reverse(runs[*from].node.start));
    for (from, to, with) in edits {
        output.replace_range(runs[from].node.start..runs[to].node.end, &with);
    }
    Ok(Some(output))
}

pub(crate) static ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\sw:id=["'](-?\d+)["']"#).expect("id"));

pub(crate) fn part(files: &[(String, Vec<u8>)], name: &str) -> Option<String> {
    files.iter().find(|(file, _)| file == name).map(|(_, bytes)| String::from_utf8_lossy(bytes).into_owned())
}

/// Writes into the units as the reading read them: replacements as tracked changes by `author`, highlights, links (in
/// the document's Hyperlink style where it has one). Highlights go first, then each unit's writes last first, so each
/// span is where it was read. A write is left out, and said to be, where its span crosses a tab, a break or a field, a
/// link would not wrap runs side by side, or its text is no longer as read; a unit no longer as read refuses the document.
/// The document's body and each unit's element in its part's reading (0 the document, 1 the notes): a paragraph by its
/// path among the body's elements, a note by its id; a unit whose text is no longer as read refuses the document.
pub(crate) fn units_in<'a>(parts: &[String; 2], trees: &'a [Vec<Node>; 2], units: &[DocxUnit]) -> Result<(&'a Node, Vec<(usize, &'a Node)>)> {
    let body = trees[0].iter().flat_map(|root| &root.children).find(|child| child.name == "w:body")
        .ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;
    let found = units.iter().map(|unit| {
        let found = match &unit.path {
            Some(path) => path.iter().try_fold(body, |node, index| node.children.get(*index)).map(|node| (0, node)),
            None => trees[1].iter().flat_map(|root| &root.children).find(|note| note.name == "w:footnote"
                && ID.captures(&parts[1][note.start..note.open_end]).is_some_and(|id| unit.id == format!("footnote:{}", &id[1]))).map(|note| (1, note)),
        };
        match found {
            Some((part, node)) if String::from_utf16_lossy(&reading(&parts[part], std::slice::from_ref(node)).0) == unit.text => Ok((part, node)),
            _ => Err(Error::Message(format!("Reviewed text no longer matches {}.", unit.id))),
        }
    }).collect::<Result<Vec<_>>>()?;
    Ok((body, found))
}

/// The package's files and the two parts its units live in, as text.
pub(crate) fn open(bytes: &[u8]) -> Result<(Vec<(String, Vec<u8>)>, [String; 2])> {
    let files = crate::docx::read_docx_files(bytes, None)?;
    let parts = [part(&files, "word/document.xml").ok_or_else(|| Error::Message("DOCX has no word/document.xml".to_owned()))?,
        part(&files, "word/footnotes.xml").unwrap_or_default()];
    Ok((files, parts))
}

pub fn write_docx_spans(bytes: &[u8], units: &[DocxUnit], writes: &[DocxSpanWrite], author: &str) -> Result<DocxSpansWritten> {
    let (mut files, mut parts) = open(bytes)?;
    let trees = [parse(&parts[0])?, parse(&parts[1]).unwrap_or_default()];
    let (body, found) = units_in(&parts, &trees, units)?;
    // Each unit's XML, edited in place, and its range in its part.
    let mut held = units.iter().zip(found).map(|(unit, (part, node))| (unit.id.as_str(), (part, node.start, node.end, parts[part][node.start..node.end].to_owned())))
        .collect::<HashMap<_, _>>();
    let highest = |xml: &str| ID.captures_iter(xml).filter_map(|id| id[1].parse::<u32>().ok()).max().unwrap_or(0);
    // Each part's relationships, an empty set where it has none, opened so a relationship can go before its close.
    static EMPTY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<Relationships([^>]*)/>").expect("relationships"));
    let relationships = RELATIONSHIPS.map(|path| (EMPTY.replace(&part(&files, path).unwrap_or_else(|| concat!("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n",
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>").to_owned()), "<Relationships$1></Relationships>").into_owned(), false));
    let mut writer = Writer { next: highest(&parts[0][body.start..body.end]).max(highest(&parts[1])) + 1, revisions: Vec::new(), author,
        date: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        styled: part(&files, "word/styles.xml").is_some_and(|styles| styles.contains("w:styleId=\"Hyperlink\"")), relationships };
    let mut order = (0..writes.len()).collect::<Vec<_>>();
    let highlight = |at: usize| matches!(writes[at].edit, DocxEdit::Highlight { .. });
    order.sort_by(|&a, &b| highlight(b).cmp(&highlight(a)).then_with(|| writes[a].unit_id.cmp(&writes[b].unit_id))
        .then(writes[b].start.cmp(&writes[a].start)).then(writes[b].end.cmp(&writes[a].end)));
    let mut written = vec![false; writes.len()];
    for index in order {
        let DocxSpanWrite { unit_id, start, end, expected, edit } = &writes[index];
        let Some((part, _, _, xml)) = held.get_mut(unit_id.as_str()) else { continue };
        let text = reading(xml, &parse(xml)?).0;
        if start > end || *end > text.len() || String::from_utf16_lossy(&text[*start..*end]) != *expected {
            continue;
        }
        if let Some(edited) = cut(xml, *start, *end)?.map(|cut| apply(&cut, *part, *start, *end, edit, &mut writer)).transpose()?.flatten() {
            (*xml, written[index]) = (edited, true);
        }
    }
    // Each part's units spliced back, last first, so every earlier range still holds; a part with links declares their
    // relationship ids' namespace, as Word's own parts do.
    for (index, (name, root)) in [("word/document.xml", "<w:document"), ("word/footnotes.xml", "<w:footnotes")].into_iter().enumerate() {
        let mut units = held.values().filter(|(part, ..)| *part == index).collect::<Vec<_>>();
        units.sort_by_key(|(_, start, ..)| std::cmp::Reverse(*start));
        for (_, start, end, xml) in &units {
            parts[index].replace_range(*start..*end, xml);
        }
        if writer.relationships[index].1 && !parts[index].contains("xmlns:r=") {
            parts[index] = parts[index].replacen(root, &format!("{root} xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""), 1);
        }
        if !units.is_empty() {
            crate::docx::replace_file(&mut files, name, std::mem::take(&mut parts[index]).into_bytes());
        }
    }
    for (path, (xml, linked)) in RELATIONSHIPS.into_iter().zip(writer.relationships) {
        if linked {
            crate::docx::replace_file(&mut files, path, xml.into_bytes());
        }
    }
    Ok(DocxSpansWritten { bytes: crate::docx::write_docx_files(&files)?, written, revisions: writer.revisions })
}
