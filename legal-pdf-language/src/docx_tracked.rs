//! An assistant's edits to a Word document, as tracked changes: each edit found by its text and context on the
//! document's accepted-text stream (paragraphs joined by "\n"; UTF-16 offsets), cut to its minimal changes, and written
//! as `w:del`/`w:ins` by its author, or directly into its runs ("auto"); new paragraphs inserted as tracked; and tracked
//! changes accepted or rejected by id. Only the runs an edit touches are rewritten; every other byte stays.

use crate::docx::{read_docx_files, replace_file, write_docx_files};
use crate::docx_spans::{escaped, parse, part, text_of, Node, ID};
use diff_match_patch_rs::{Compat, DiffMatchPatch, Ops};
use legal_pdf_core::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

/// An edit: the text it replaces and its context, or exact accepted-stream offsets a trusted caller pins.
#[derive(Deserialize, Clone)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
pub struct EditInput {
    #[serde(default)]
    pub find: String,
    #[serde(default)]
    pub replace: String,
    #[serde(default)]
    pub context_before: String,
    #[serde(default)]
    pub context_after: String,
    pub reason: Option<String>,
    pub exact_start: Option<usize>,
    pub exact_end: Option<usize>,
}

#[derive(Deserialize, Serialize, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    Manual,
    Auto,
}

#[derive(Serialize, Clone, Debug)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct EditDiffSegment {
    pub kind: DiffKind,
    pub text: String,
}

#[derive(Serialize, Clone, Copy, Debug)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "lowercase")]
pub enum DiffKind {
    Equal,
    Delete,
    Insert,
}

/// A change written: its id (and its deletion's and insertion's revision ids), what it deleted and inserted, its
/// context and reason, and the edit's diff.
#[derive(Serialize, Clone, Debug)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
#[serde(rename_all = "camelCase")]
pub struct AppliedChange {
    pub id: String,
    pub del_id: Option<String>,
    pub ins_id: Option<String>,
    pub deleted_text: String,
    pub inserted_text: String,
    pub context_before: String,
    pub context_after: String,
    pub reason: Option<String>,
    pub diff: Vec<EditDiffSegment>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct EditError {
    pub index: usize,
    pub reason: String,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct TrackedEdits {
    #[serde(skip)]
    #[cfg_attr(feature = "binding-types", ts(skip))]
    pub bytes: Vec<u8>,
    pub changes: Vec<AppliedChange>,
    pub errors: Vec<EditError>,
    #[serde(rename = "emissionMode")]
    pub mode: EditMode,
}

/// New paragraphs and where they go: before or after the body paragraph showing `anchor_text` (its `occurrence`th,
/// from 1, where several do), else at the document's start or end.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
#[serde(rename_all = "camelCase")]
pub struct BlockInsert {
    pub blocks: Vec<String>,
    pub position: BlockPosition,
    pub anchor_text: Option<String>,
    pub occurrence: Option<usize>,
}

#[derive(Deserialize, PartialEq)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "lowercase")]
pub enum BlockPosition {
    Before,
    After,
}

pub(crate) type Units = Vec<u16>;

pub(crate) fn units(text: &str) -> Units {
    text.encode_utf16().collect()
}

pub(crate) fn string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// A run an edit can rewrite: its element, its place among the paragraph's children, its accepted text's place, its
/// properties, whether a content control holds it, and its text elements' places.
struct EditRun<'a> {
    node: &'a Node,
    child: usize,
    start: usize,
    end: usize,
    properties: &'a str,
    protected: bool,
    texts: Vec<(usize, usize)>,
}

/// A paragraph as an edit and a comparison read it: its element children, its place on the accepted stream, its
/// accepted text (pending insertions in, deletions out), the text it shows, the text a comparison reads (its runs'
/// rendered text outside fields' and controls' chrome), whether it holds an object outside its runs, and its runs.
pub(crate) struct Paragraph<'a> {
    pub node: &'a Node,
    children: Vec<&'a Node>,
    start: usize,
    accepted: Units,
    visible: Units,
    pub compare: Units,
    pub objects: bool,
    runs: Vec<EditRun<'a>>,
}

#[derive(Clone, Copy)]
struct State {
    del: bool,
    protected: bool,
    edit: bool,
    compare: bool,
}

static SKIP: LazyLock<Regex> = LazyLock::new(|| Regex::new(
    r"^(?:w:(?:pPr|rPr|sectPr|tblPr|trPr|tcPr|tblGrid|sdtPr|sdtEndPr|bookmarkStart|bookmarkEnd|proofErr|commentReference|commentRangeStart|commentRangeEnd)|mc:Fallback)$").expect("skip"));
static OBJECT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^w:(?:footnoteReference|endnoteReference|drawing|object|pict|fldChar|fldSimple)$").expect("object"));

pub(crate) fn paragraph<'a>(xml: &'a str, node: &'a Node) -> Paragraph<'a> {
    fn visit<'a>(xml: &'a str, node: &'a Node, child: usize, state: State, paragraph: &mut Paragraph<'a>) {
        if SKIP.is_match(&node.name) {
            return;
        }
        paragraph.objects |= OBJECT.is_match(&node.name);
        match node.name.as_str() {
            "w:r" => {
                if state.del {
                    return;
                }
                let start = paragraph.accepted.len();
                let mut texts = Vec::new();
                for item in &node.children {
                    let rendered = match item.name.as_str() {
                        "w:t" => {
                            let text = text_of(xml, item);
                            if state.edit {
                                texts.push((paragraph.accepted.len(), paragraph.accepted.len() + text.len()));
                                paragraph.accepted.extend(&text);
                            }
                            paragraph.visible.extend(&text);
                            text
                        }
                        "w:tab" => { paragraph.visible.push(9); vec![9] }
                        "w:cr" => vec![10],
                        // A break with attributes (a page or column break) shows, but renders no line.
                        "w:br" => { paragraph.visible.push(10); if xml[item.start..item.open_end].contains('=') { Vec::new() } else { vec![10] } }
                        _ => Vec::new(),
                    };
                    if state.compare { paragraph.compare.extend(rendered); }
                }
                if state.edit {
                    let properties = node.child("w:rPr").map_or("", |rpr| &xml[rpr.start..rpr.end]);
                    paragraph.runs.push(EditRun { node, child, start, end: paragraph.accepted.len(), properties, protected: state.protected, texts });
                }
            }
            "w:ins" => node.children.iter().for_each(|item| visit(xml, item, child, state, paragraph)),
            "w:del" => node.children.iter().for_each(|item| visit(xml, item, child, State { del: true, edit: false, compare: false, ..state }, paragraph)),
            "w:sdt" => for item in &node.children {
                let content = item.name == "w:sdtContent";
                visit(xml, item, child, State { protected: content || state.protected, edit: content && state.edit, compare: content && state.compare, ..state }, paragraph);
            },
            name => {
                let edit = state.edit && matches!(name, "w:sdtContent" | "w:hyperlink");
                let compare = state.compare && matches!(name, "w:sdtContent" | "w:hyperlink" | "w:smartTag" | "w:customXml");
                node.children.iter().for_each(|item| visit(xml, item, child, State { edit, compare, ..state }, paragraph));
            }
        }
    }
    let mut found = Paragraph { node, children: node.children.iter().collect(), start: 0, accepted: Vec::new(), visible: Vec::new(),
        compare: Vec::new(), objects: false, runs: Vec::new() };
    let state = State { del: false, protected: false, edit: true, compare: true };
    for (child, item) in node.children.iter().enumerate() {
        visit(xml, item, child, state, &mut found);
    }
    found
}
/// The body's paragraphs, in order through its tables and content controls, each placed on the accepted stream.
pub(crate) fn paragraphs<'a>(xml: &'a str, body: &'a Node) -> Vec<Paragraph<'a>> {
    fn collect<'a>(xml: &'a str, node: &'a Node, found: &mut Vec<Paragraph<'a>>) {
        match node.name.as_str() {
            "w:p" => {
                let mut indexed = paragraph(xml, node);
                indexed.start = found.last().map_or(0, |last| last.start + last.accepted.len() + 1);
                found.push(indexed);
            }
            "w:tbl" | "w:tr" | "w:tc" | "w:sdt" | "w:sdtContent" => node.children.iter().for_each(|child| collect(xml, child, found)),
            _ => {}
        }
    }
    let mut found = Vec::new();
    body.children.iter().for_each(|child| collect(xml, child, &mut found));
    found
}

pub(crate) fn body(roots: &[Node]) -> Option<&Node> {
    roots.iter().flat_map(|root| &root.children).find(|child| child.name == "w:body")
}

fn stream(paragraphs: &[Paragraph]) -> Units {
    paragraphs.iter().map(|paragraph| paragraph.accepted.clone()).collect::<Vec<_>>().join(&10)
}

static STORIES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^word/(?:document|footnotes|endnotes|comments|header\d*|footer\d*)\.xml$").expect("stories"));

/// Each story part's tracked changes, in reading order: its path, text, and each `w:ins`/`w:del` with its id.
fn revisions(files: &[(String, Vec<u8>)]) -> Result<Vec<(String, String, Vec<(&'static str, String)>)>> {
    let mut paths = files.iter().map(|(path, _)| path.as_str()).filter(|path| STORIES.is_match(path)).collect::<Vec<_>>();
    paths.sort_unstable();
    paths.into_iter().map(|path| {
        let xml = part(files, path).unwrap_or_default();
        let mut changes = Vec::new();
        fn visit(xml: &str, node: &Node, changes: &mut Vec<(&'static str, String)>) {
            if let (kind @ ("w:ins" | "w:del"), Some(id)) = (node.name.as_str(), ID.captures(&xml[node.start..node.open_end])) {
                changes.push((if kind == "w:ins" { "ins" } else { "del" }, id[1].to_owned()));
            }
            node.children.iter().for_each(|child| visit(xml, child, changes));
        }
        parse(&xml)?.iter().for_each(|root| visit(&xml, root, &mut changes));
        Ok((path.to_owned(), xml, changes))
    }).collect()
}

pub(crate) fn highest(files: &[(String, Vec<u8>)]) -> Result<u64> {
    Ok(revisions(files)?.iter().flat_map(|(.., changes)| changes).filter_map(|(_, id)| id.parse::<u64>().ok()).max().unwrap_or(0))
}

/// The document's accepted text, paragraphs joined by "\n", as edits are found on it; empty without a body.
pub fn docx_body_text(bytes: &[u8]) -> Result<String> {
    Ok(string(&body_units(bytes)?))
}

fn body_units(bytes: &[u8]) -> Result<Units> {
    let files = read_docx_files(bytes, None)?;
    let Some(xml) = part(&files, "word/document.xml") else { return Ok(Vec::new()) };
    let roots = parse(&xml)?;
    Ok(body(&roots).map_or_else(Vec::new, |body| stream(&paragraphs(&xml, body))))
}

/// Every tracked change in the document's stories, in reading order.
pub fn docx_tracked_change_ids(bytes: &[u8]) -> Result<Vec<(&'static str, String)>> {
    Ok(revisions(&read_docx_files(bytes, None)?)?.into_iter().flat_map(|(.., changes)| changes).collect())
}

/// Text as edits are matched against it: typographic quotes and dashes plain, each run of whitespace one space; with
/// each kept unit's place in the original.
pub(crate) fn normalized(text: &[u16]) -> (Units, Vec<usize>) {
    let space = |unit: u16| char::from_u32(u32::from(unit)).is_some_and(legal_structure::javascript_whitespace);
    let (mut norm, mut places, mut previous) = (Vec::new(), Vec::new(), false);
    for (at, &unit) in text.iter().enumerate() {
        let unit = match unit { 0x2018 | 0x2019 | 0x2032 => 0x27, 0x201C | 0x201D | 0x2033 => 0x22, 0x2013 | 0x2014 => 0x2D, 0xA0 | 0x200B => 0x20, other => other };
        let blank = space(unit);
        if blank && previous { continue; }
        norm.push(if blank { 0x20 } else { unit });
        places.push(at);
        previous = blank;
    }
    (norm, places)
}

fn find_from(haystack: &[u16], needle: &[u16], from: usize) -> Option<usize> {
    if needle.is_empty() { return (from <= haystack.len()).then_some(from); }
    (from..=haystack.len().saturating_sub(needle.len())).find(|&at| haystack[at..].starts_with(needle))
}

/// The one place an edit's text stands between its contexts, as original offsets; "none" or "ambiguous" otherwise.
fn locate(body: &(Units, Vec<usize>), length: usize, edit: &EditInput) -> std::result::Result<(usize, usize), &'static str> {
    let (find, before, after) = (normalized(&units(&edit.find)).0, normalized(&units(&edit.context_before)).0, normalized(&units(&edit.context_after)).0);
    let needle = if !find.is_empty() { &find } else if !before.is_empty() { &before } else { &after };
    let offset = if find.is_empty() && !before.is_empty() { before.len() } else { 0 };
    let (mut found, mut from) = (None, 0);
    while let Some(index) = find_from(&body.0, needle, from).filter(|_| from + needle.len() <= body.0.len()) {
        from = index + 1;
        let position = index + offset;
        let context = position.checked_sub(before.len());
        let end = position + find.len();
        if context.is_none_or(|start| body.0[start..position] != before[..]) || body.0.get(end..end + after.len()).is_none_or(|text| text != &after[..]) {
            continue;
        }
        if found.is_some() { return Err("ambiguous"); }
        found = Some(position);
    }
    let found = found.ok_or("none")?;
    let start = body.1.get(found).copied().unwrap_or(length);
    Ok((start, if find.is_empty() { start } else { body.1[found + find.len() - 1] + 1 }))
}

/// One edit's minimal changes (diff-match-patch, with semantic cleanup unless the edit crosses a text boundary): each a
/// deleted range of `find` and the text inserted at its start; and the edit's diff.
fn minimal(find: &[u16], replace: &[u16], anchored: bool) -> (Vec<(usize, Units, Units)>, Vec<EditDiffSegment>) {
    let parts = difference(&string(find), &string(replace), !anchored);
    let segments = parts.iter().map(|(op, text)| EditDiffSegment { text: string(text),
        kind: match op { Ops::Equal => DiffKind::Equal, Ops::Delete => DiffKind::Delete, Ops::Insert => DiffKind::Insert } }).collect();
    (clusters(&parts, false), segments)
}

/// Two texts' difference as diff-match-patch reads it, character by character (semantic cleanup where `cleanup`),
/// each piece in UTF-16 units.
pub(crate) fn difference(old: &str, new: &str, cleanup: bool) -> Vec<(Ops, Units)> {
    let mut dmp = DiffMatchPatch::new();
    dmp.set_checklines(false);
    dmp.set_timeout(None);
    let mut diffs = dmp.diff_main::<Compat>(old, new).unwrap_or_default();
    if cleanup { DiffMatchPatch::diff_cleanup_semantic(&mut diffs); }
    diffs.iter().map(|diff| (diff.op(), units(&diff.data().iter().collect::<String>()))).collect()
}

/// Adjacent changes grouped, each at its offset in the old text (or, with `new`, the new): its offset, the text it
/// deletes and the text it inserts.
pub(crate) fn clusters(parts: &[(Ops, Units)], new: bool) -> Vec<(usize, Units, Units)> {
    let (mut found, mut offset) = (Vec::<(usize, Units, Units)>::new(), 0);
    for (op, piece) in parts {
        if *op == Ops::Equal { offset += piece.len(); continue; }
        let placed = |(at, deleted, inserted): &(usize, Units, Units)| at + if new { inserted.len() } else { deleted.len() } == offset;
        if !found.last().is_some_and(placed) { found.push((offset, Vec::new(), Vec::new())); }
        let cluster = found.last_mut().expect("cluster");
        if *op == Ops::Delete { cluster.1.extend(piece) } else { cluster.2.extend(piece) }
        if (*op == Ops::Insert) == new { offset += piece.len(); }
    }
    found
}

/// A change planned in a paragraph: its deleted range, the text inserted at its start, and their revision ids.
struct Planned {
    start: usize,
    end: usize,
    inserted: Units,
    del: Option<String>,
    ins: Option<String>,
}

/// The first of `spans` ending after `at`.
fn span_at(spans: impl Iterator<Item = usize>, at: usize) -> usize {
    spans.take_while(|end| *end <= at).count()
}

/// A run holding `text` in `properties`, a newline written as a line break.
fn text_run(open: &str, properties: &str, text: &[u16], tag: &str) -> String {
    let content = string(text).split_inclusive('\n').flat_map(|part| {
        let (text, line) = part.strip_suffix('\n').map_or((part, false), |text| (text, true));
        [(!text.is_empty()).then(|| format!("<{tag} xml:space=\"preserve\">{}</{tag}>", escaped(text))), line.then(|| "<w:br/>".to_owned())]
    }).flatten().collect::<String>();
    format!("{open}{properties}{content}</w:r>")
}

fn revision(tag: &str, id: &str, author: &str, date: &str, content: &str) -> String {
    format!("<{tag} w:id=\"{id}\" w:author=\"{}\" w:date=\"{date}\">{content}</{tag}>", escaped(author))
}

/// The runs a paragraph's sorted, separate changes touch, each rewritten: as tracked changes ("manual"), or with its text
/// changed in place ("auto").
fn rewrite(xml: &str, paragraph: &Paragraph, plan: &[Planned], date: &str, author: &str, mode: EditMode) -> Vec<(usize, usize, String)> {
    let length = paragraph.accepted.len();
    let run_at = |at: usize| span_at(paragraph.runs.iter().map(|run| run.end), at.min(length.saturating_sub(1)));
    let mut edits = Vec::new();
    for (index, run) in paragraph.runs.iter().enumerate() {
        let changes = plan.iter().filter(|change| change.start < run.end && change.end > run.start
            || !change.inserted.is_empty() && run_at(change.start) == index).collect::<Vec<_>>();
        if changes.is_empty() { continue; }
        let open = &xml[run.node.start..run.node.open_end];
        let text_node = |change: &Planned| span_at(run.texts.iter().map(|(_, end)| *end), change.start.min(run.end.saturating_sub(1)));
        let (mut output, mut text_index) = (String::new(), 0);
        if mode == EditMode::Auto { output.push_str(open); }
        for child in &run.node.children {
            if child.name != "w:t" {
                if mode == EditMode::Auto { output.push_str(&xml[child.start..child.end]); }
                else if child.name != "w:rPr" { output.push_str(&format!("{open}{}{}</w:r>", run.properties, &xml[child.start..child.end])); }
                continue;
            }
            let node = text_index;
            text_index += 1;
            let (node_start, node_end) = run.texts[node];
            let (mut cursor, mut direct) = (node_start, Vec::new());
            let unchanged = |from: usize, to: usize, output: &mut String, direct: &mut Units| if from < to {
                if mode == EditMode::Auto { direct.extend(&paragraph.accepted[from..to]); }
                else { output.push_str(&text_run(open, run.properties, &paragraph.accepted[from..to], "w:t")); }
            };
            for change in &changes {
                let (start, end) = (node_start.max(change.start), node_end.min(change.end));
                let insert = !change.inserted.is_empty() && run_at(change.start) == index && text_node(change) == node;
                if end <= start && !insert { continue; }
                unchanged(cursor, start, &mut output, &mut direct);
                if insert {
                    if mode == EditMode::Auto { direct.extend(&change.inserted); }
                    else { output.push_str(&revision("w:ins", change.ins.as_deref().unwrap_or_default(), author, date, &text_run(open, run.properties, &change.inserted, "w:t"))); }
                }
                if mode == EditMode::Manual && start < end {
                    output.push_str(&revision("w:del", change.del.as_deref().unwrap_or_default(), author, date,
                        &text_run(open, run.properties, &paragraph.accepted[start..end], "w:delText")));
                }
                cursor = cursor.max(start).max(end);
            }
            unchanged(cursor, node_end, &mut output, &mut direct);
            if mode == EditMode::Auto {
                if direct[..] == paragraph.accepted[node_start..node_end] { output.push_str(&xml[child.start..child.end]); continue; }
                // The text element keeps its own attributes; a newline is a line break beside it.
                let tag = xml[child.start..child.open_end].trim_end_matches("/>").trim_end_matches('>').replace(" xml:space=\"preserve\"", "");
                let content = text_run("", "", &direct, "w:t").trim_end_matches("</w:r>").to_owned();
                output.push_str(&if direct.is_empty() { format!("{tag} xml:space=\"preserve\"></w:t>") }
                    else { content.replace("<w:t xml:space=\"preserve\">", &format!("{tag} xml:space=\"preserve\">")) });
            }
        }
        if mode == EditMode::Auto { output.push_str("</w:r>"); }
        edits.push((run.node.start, run.node.end, output));
    }
    edits
}

/// Rewriting re-emits accepted text as plain runs, which would silently accept a pending revision among the rewritten
/// children.
fn rewrites_revision(paragraph: &Paragraph, plan: &[(usize, usize)]) -> bool {
    if paragraph.accepted.is_empty() || paragraph.runs.is_empty() { return false; }
    let length = paragraph.accepted.len();
    let run_at = |at: usize| span_at(paragraph.runs.iter().map(|run| run.end), at.min(length - 1)).min(paragraph.runs.len() - 1);
    let first = run_at(plan[0].0);
    let last = plan.iter().map(|(start, end)| run_at((*start).max(end.saturating_sub(1)))).max().unwrap_or(first).max(first);
    fn revised(node: &Node) -> bool {
        matches!(node.name.as_str(), "w:ins" | "w:del" | "w:moveFrom" | "w:moveTo") || node.children.iter().any(revised)
    }
    paragraph.children[paragraph.runs[first].child..=paragraph.runs[last].child].iter().any(|child| revised(child))
}

fn touches_content_control(paragraph: &Paragraph, start: usize, end: usize) -> bool {
    // At an insertion boundary either adjoining character can belong to a control.
    let (start, end) = if start == end { (start.saturating_sub(1), end + 1) } else { (start, end) };
    paragraph.runs.iter().filter(|run| run.end > start && run.start < end).any(|run| run.start < run.end && run.protected)
}

/// Writes the edits, each found on the accepted stream (or at its pinned offsets), as tracked changes by `author`, or
/// directly ("auto"). An edit copied across several paragraphs is pinned to each paragraph it changes. An edit is left
/// out, and says why, where it is not found or found more than once, crosses paragraphs, touches a content control or
/// a pending revision, or overlaps another.
pub fn apply_tracked_edits(bytes: &[u8], edits: &[EditInput], author: &str, mode: EditMode) -> Result<TrackedEdits> {
    let mut files = read_docx_files(bytes, None)?;
    let date = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    let mut xml = part(&files, "word/document.xml").ok_or_else(|| Error::Message("document.xml missing from docx".to_owned()))?;
    let roots = parse(&xml)?;
    let body = body(&roots).ok_or_else(|| Error::Message("w:body missing from docx".to_owned()))?;
    let paragraphs = paragraphs(&xml, body);
    let text = stream(&paragraphs);
    let norm = normalized(&text);
    let (mut errors, mut concrete, mut diffs) = (Vec::new(), Vec::new(), BTreeMap::new());
    let mut fail = |index: usize, reason: &str| errors.push(EditError { index, reason: reason.to_owned() });
    for (index, source) in edits.iter().enumerate() {
        let (find, replace) = (source.find.replace("\r\n", "\n").replace('\r', "\n"), source.replace.replace("\r\n", "\n").replace('\r', "\n"));
        let multiline = find.contains('\n') || replace.contains('\n');
        if find.is_empty() && (replace.is_empty() || source.context_before.is_empty() && source.context_after.is_empty() && source.exact_start.is_none()) {
            fail(index, if replace.is_empty() { "Empty edit." } else { "Pure insertion requires context_before or context_after." });
            continue;
        }
        let matched = match (source.exact_start, source.exact_end) {
            (Some(start), Some(end)) => Ok((start, end)),
            _ => locate(&norm, text.len(), &EditInput { find: find.clone(), ..source.clone() }),
        };
        let (start, end) = match matched {
            Ok(found) => found,
            Err(error) => {
                fail(index, match (multiline, error) {
                    (true, "ambiguous") => "Ambiguous match for the multi-paragraph edit; the document is unchanged.",
                    (true, _) => "Could not locate the multi-paragraph edit; the document is unchanged.",
                    (false, "ambiguous") => "Ambiguous match for this edit; the document is unchanged.",
                    _ => "Could not locate this edit on the current document text plane; the document is unchanged.",
                });
                continue;
            }
        };
        if !multiline {
            let find = if source.exact_start.is_some() { find } else { string(&text[start.min(text.len())..end.min(text.len())]) };
            concrete.push((index, EditInput { find, replace, exact_start: Some(start), exact_end: Some(end), ..source.clone() }));
            continue;
        }
        let actual = string(&text[start.min(text.len())..end.min(text.len())]);
        diffs.insert(index, minimal(&units(&actual), &units(&replace), false).1);
        let (found, replaced) = (actual.split('\n').collect::<Vec<_>>(), replace.split('\n').collect::<Vec<_>>());
        if found.len() != replaced.len() {
            fail(index, &format!("Multi-paragraph replacement must preserve the paragraph count ({} found, {} supplied).", found.len(), replaced.len()));
            continue;
        }
        let (mut cursor, mut changed) = (start, false);
        for (line, (original, replacement)) in found.iter().zip(&replaced).enumerate() {
            let length = original.encode_utf16().count();
            if original != replacement {
                changed = true;
                concrete.push((index, EditInput { find: (*original).to_owned(), replace: (*replacement).to_owned(),
                    context_before: string(&text[cursor.saturating_sub(40)..cursor]), context_after: string(&text[cursor + length..(cursor + length + 40).min(text.len())]),
                    reason: source.reason.clone(), exact_start: Some(cursor), exact_end: Some(cursor + length) }));
            }
            cursor += length + usize::from(line + 1 < found.len());
        }
        if !changed { fail(index, "Replacement does not change the matched text."); }
    }
    let mut next = highest(&files)? + 1;
    let millis = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_millis());
    let (mut plans, mut applied, mut ids) = (BTreeMap::<usize, Vec<Planned>>::new(), BTreeMap::<usize, AppliedChange>::new(), BTreeMap::<usize, (Option<String>, Option<String>)>::new());
    for (index, edit) in concrete {
        let (start, end) = (edit.exact_start.unwrap(), edit.exact_end.unwrap());
        if end < start { fail(index, "Invalid exact edit span."); continue; }
        let Some(at) = paragraphs.iter().position(|paragraph| paragraph.start <= start && start <= paragraph.start + paragraph.accepted.len())
            .filter(|&at| end <= paragraphs[at].start + paragraphs[at].accepted.len()) else { fail(index, "Exact edit span must resolve inside one paragraph."); continue };
        let paragraph = &paragraphs[at];
        let (from, to) = (start - paragraph.start, end - paragraph.start);
        let find = units(&edit.find);
        // The pinned span is the matcher's own coordinate: it must still read exactly as `find`.
        if paragraph.accepted.get(from..to) != Some(&find[..]) { fail(index, "Exact edit span no longer matches the pinned text."); continue; }
        // Equal text on either side of an XML boundary keeps a replacement in its own style, bookmark or note region.
        let anchored = paragraph.runs.iter().any(|run| run.texts.iter().any(|(node, _)| *node > from && *node < to));
        let (clusters, diff) = minimal(&find, &units(&edit.replace), anchored);
        diffs.entry(index).or_insert(diff);
        if clusters.is_empty() { fail(index, "Replacement does not change the matched text."); continue; }
        if clusters.iter().any(|(offset, deleted, _)| touches_content_control(paragraph, from + offset, from + offset + deleted.len())) {
            fail(index, "This edit touches a Word content control. Edit the control in Word or regenerate the draft.");
            continue;
        }
        let existing = plans.get(&at).map_or(&[][..], Vec::as_slice);
        let mut spans = existing.iter().map(|change| (change.start, change.end))
            .chain(clusters.iter().map(|(offset, deleted, _)| (from + offset, from + offset + deleted.len()))).collect::<Vec<_>>();
        spans.sort_unstable();
        if rewrites_revision(paragraph, &spans) {
            fail(index, "This text is inside or beside pending tracked changes, which this edit would silently accept. Use word_python, which keeps existing revisions.");
            continue;
        }
        let revision = ids.entry(index).or_default();
        if mode == EditMode::Manual && clusters.iter().any(|(_, deleted, _)| !deleted.is_empty()) && revision.0.is_none() { revision.0 = Some(next.to_string()); next += 1; }
        if mode == EditMode::Manual && clusters.iter().any(|(.., inserted)| !inserted.is_empty()) && revision.1.is_none() { revision.1 = Some(next.to_string()); next += 1; }
        let planned = clusters.iter().map(|(offset, deleted, inserted)| Planned { start: from + offset, end: from + offset + deleted.len(), inserted: inserted.clone(),
            del: (!deleted.is_empty()).then(|| revision.0.clone()).flatten(), ins: (!inserted.is_empty()).then(|| revision.1.clone()).flatten() }).collect::<Vec<_>>();
        if planned.iter().any(|plan| existing.iter().any(|other| !(plan.end <= other.start || plan.start >= other.end))) {
            fail(index, "Overlaps a previous edit in the same paragraph.");
            continue;
        }
        let joined = |field: usize| clusters.iter().map(|cluster| string(if field == 0 { &cluster.1 } else { &cluster.2 })).filter(|text| !text.is_empty()).collect::<Vec<_>>().join(" … ");
        let (deleted, inserted) = (joined(0), joined(1));
        let (del_id, ins_id) = revision.clone();
        match applied.get_mut(&index) {
            Some(change) => {
                (change.del_id, change.ins_id) = (del_id, ins_id);
                for (field, text) in [(&mut change.deleted_text, deleted), (&mut change.inserted_text, inserted)] {
                    if !text.is_empty() { if !field.is_empty() { field.push('\n'); } field.push_str(&text); }
                }
                change.context_after = edit.context_after.clone();
            }
            None => { applied.insert(index, AppliedChange { id: format!("mike-{index}-{millis}"), del_id, ins_id, deleted_text: deleted, inserted_text: inserted,
                context_before: edit.context_before.clone(), context_after: edit.context_after.clone(), reason: edit.reason.clone(), diff: diffs[&index].clone() }); }
        }
        let plan = plans.entry(at).or_default();
        plan.extend(planned);
        plan.sort_by_key(|change| change.start);
    }
    let mut replaced = plans.iter().flat_map(|(at, plan)| rewrite(&xml, &paragraphs[*at], plan, &date, author, mode)).collect::<Vec<_>>();
    replaced.sort_by_key(|(start, ..)| std::cmp::Reverse(*start));
    let output = replaced.into_iter().fold(xml.clone(), |mut output, (start, end, text)| { output.replace_range(start..end, &text); output });
    xml = output;
    replace_file(&mut files, "word/document.xml", xml.into_bytes());
    Ok(TrackedEdits { bytes: write_docx_files(&files)?, changes: applied.into_values().collect(), errors, mode })
}

/// The new paragraphs where `insert` puts them, each as tracked text with an inserted paragraph mark.
pub fn insert_tracked_blocks(bytes: &[u8], insert: &BlockInsert, author: &str) -> Result<TrackedEdits> {
    let (blocks, after) = (&insert.blocks, insert.position == BlockPosition::After);
    if blocks.is_empty() || blocks.iter().any(|block| block.trim().is_empty()) {
        return Err(Error::Message("insert_blocks requires one or more non-empty blocks".to_owned()));
    }
    if blocks.iter().any(|block| block.contains(['\r', '\n'])) {
        return Err(Error::Message("Each insert_blocks item must be one paragraph without a newline".to_owned()));
    }
    let mut files = read_docx_files(bytes, None)?;
    let mut xml = part(&files, "word/document.xml").ok_or_else(|| Error::Message("document.xml missing from docx".to_owned()))?;
    let roots = parse(&xml)?;
    let body = body(&roots).ok_or_else(|| Error::Message("w:body missing from docx".to_owned()))?;
    let indexed = paragraphs(&xml, body);
    let top = indexed.iter().filter(|paragraph| body.children.iter().any(|child| std::ptr::eq(child, paragraph.node))).collect::<Vec<_>>();
    let (mut before_text, mut after_text) = (String::new(), String::new());
    let at = match insert.anchor_text.as_deref().filter(|anchor| !anchor.trim().is_empty()) {
        Some(anchor) => {
            let needle = normalized(&units(anchor)).0;
            let hits = top.iter().filter(|paragraph| find_from(&normalized(&paragraph.visible).0, &needle, 0).is_some()).collect::<Vec<_>>();
            let chosen = match insert.occurrence { Some(occurrence) => occurrence.checked_sub(1).and_then(|at| hits.get(at)), None if hits.len() == 1 => hits.first(), None => None };
            let Some(chosen) = chosen else {
                return Err(Error::Message(if hits.is_empty() { "insert_blocks anchor paragraph was not found".to_owned() }
                    else { format!("insert_blocks anchor is ambiguous ({} paragraphs); set occurrence", hits.len()) }));
            };
            let visible = &chosen.visible;
            if after { before_text = string(&visible[visible.len().saturating_sub(120)..]); } else { after_text = string(&visible[..visible.len().min(120)]); }
            if after { chosen.node.end } else { chosen.node.start }
        }
        None if !after => top.first().map_or(body.open_end, |paragraph| paragraph.node.start),
        None => body.children.iter().find(|child| child.name == "w:sectPr").map_or(body.inner_end, |section| section.start),
    };
    let date = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    let mut next = highest(&files)? + 1;
    let mut changes = Vec::new();
    let inserted = blocks.iter().map(|block| {
        let id = next.to_string();
        next += 1;
        changes.push(AppliedChange { id: id.clone(), del_id: None, ins_id: Some(id.clone()), deleted_text: String::new(), inserted_text: block.clone(),
            context_before: before_text.clone(), context_after: after_text.clone(), reason: None, diff: vec![EditDiffSegment { kind: DiffKind::Insert, text: block.clone() }] });
        let mark = format!("<w:ins w:id=\"{id}\" w:author=\"{}\" w:date=\"{date}\"/>", escaped(author));
        format!("<w:p><w:pPr><w:rPr>{mark}</w:rPr></w:pPr>{}</w:p>", revision("w:ins", &id, author, &date, &text_run("<w:r>", "", &units(block), "w:t")))
    }).collect::<String>();
    xml.insert_str(at, &inserted);
    replace_file(&mut files, "word/document.xml", xml.into_bytes());
    Ok(TrackedEdits { bytes: write_docx_files(&files)?, changes, errors: Vec::new(), mode: EditMode::Manual })
}

/// Accepts or rejects the tracked changes `ids` names, in every story: accepting an insertion or rejecting a deletion
/// keeps what it wraps (a deletion's text read again), the others drop it, and rejecting an inserted paragraph mark
/// drops its paragraph. None where any id is not a tracked change of the document.
pub fn resolve_tracked_changes(bytes: &[u8], ids: &[String], accept: bool) -> Result<Option<Vec<u8>>> {
    let mut files = read_docx_files(bytes, None)?;
    let stories = revisions(&files)?;
    let wanted = ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let present = stories.iter().flat_map(|(.., changes)| changes.iter().map(|(_, id)| id.as_str())).collect::<HashSet<_>>();
    if wanted.is_empty() || !wanted.is_subset(&present) { return Ok(None); }
    let id_of = |xml: &str, node: &Node| ID.captures(&xml[node.start..node.open_end]).map(|id| id[1].to_owned());
    // `xml[from..to]` with each resolved change among `nodes` kept (a deletion's text read again) or dropped.
    fn render(xml: &str, nodes: &[Node], from: usize, to: usize, out: &mut String, resolve: &dyn Fn(&Node) -> Option<bool>) {
        let mut at = from;
        for node in nodes {
            out.push_str(&xml[at..node.start]);
            match resolve(node) {
                None => render(xml, &node.children, node.start, node.end, out, resolve),
                Some(false) => {}
                Some(true) => {
                    let mut inner = String::new();
                    render(xml, &node.children, node.open_end, node.inner_end, &mut inner, resolve);
                    out.push_str(&if node.name == "w:del" {
                        inner.replace("<w:delText", "<w:t").replace("</w:delText>", "</w:t>").replace("<w:delInstrText", "<w:instrText").replace("</w:delInstrText>", "</w:instrText>")
                    } else { inner });
                }
            }
            at = node.end;
        }
        out.push_str(&xml[at..to]);
    }
    for (path, xml, changes) in &stories {
        if !changes.iter().any(|(_, id)| wanted.contains(id.as_str())) { continue; }
        let roots = parse(xml)?;
        // Some(keep): a change resolved, its content kept or dropped; a rejected paragraph mark drops its paragraph.
        let resolve = |node: &Node| -> Option<bool> {
            match node.name.as_str() {
                "w:p" if !accept => node.child("w:pPr").and_then(|pr| pr.child("w:rPr"))
                    .is_some_and(|rpr| rpr.children.iter().any(|mark| mark.name == "w:ins" && id_of(xml, mark).is_some_and(|id| wanted.contains(id.as_str()))))
                    .then_some(false),
                "w:ins" | "w:del" if id_of(xml, node).is_some_and(|id| wanted.contains(id.as_str())) =>
                    Some(node.name == "w:ins" && accept || node.name == "w:del" && !accept),
                _ => None,
            }
        };
        let mut output = String::new();
        render(xml, &roots, 0, xml.len(), &mut output, &resolve);
        replace_file(&mut files, path, output.into_bytes());
    }
    Ok(Some(write_docx_files(&files)?))
}


/// Where a text op runs on the accepted text: all of it, each (or the `occurrence`th, from 1) place `text` stands, from
/// the first place `from_text` stands to the next of `to_text`, or spans the address layer resolved on the same text.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TextScope {
    WholeDocument,
    FindText { text: String, occurrence: Option<usize> },
    Range { from_text: String, to_text: String },
    Spans { spans: Vec<TextSpan> },
}

#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

/// A text op the assistant asks for: its name, its scope and its own parameters.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct TextOpRequest {
    pub op: String,
    pub scope: TextScope,
    #[serde(flatten)]
    pub params: crate::text_ops::TextOpParams,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct TextOpReport {
    pub op: String,
    pub replacements: usize,
    pub notes: Vec<crate::text_ops::TextOpNote>,
}

/// Text ops written: the edits, each op's report, and how many changes they planned.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct TextOpsApplied {
    #[serde(flatten)]
    pub edits: TrackedEdits,
    pub reports: Vec<TextOpReport>,
    pub replacement_count: usize,
}

/// Each place `needle` stands on the normalized text, as original offsets.
fn hits(body: &(Units, Vec<usize>), length: usize, needle: &str) -> Vec<(usize, usize)> {
    let needle = normalized(&units(needle)).0;
    let (mut found, mut from) = (Vec::new(), 0);
    while let Some(at) = (!needle.is_empty()).then(|| find_from(&body.0, &needle, from)).flatten() {
        from = at + 1;
        found.push((body.1[at], body.1.get(at + needle.len() - 1).map_or(length, |place| place + 1)));
    }
    found
}

fn scope_spans(scope: &TextScope, text: &[u16], norm: &(Units, Vec<usize>)) -> Result<Vec<(usize, usize)>> {
    let refused = |message: String| Error::Message(message);
    let quoted = |text: &str| text.chars().take(80).collect::<String>();
    Ok(match scope {
        TextScope::WholeDocument => vec![(0, text.len())],
        TextScope::Spans { spans } => {
            let mut spans = spans.iter().map(|span| (span.start.min(text.len()), span.end.min(text.len()))).filter(|(start, end)| end > start).collect::<Vec<_>>();
            spans.sort_unstable_by_key(|(start, _)| *start);
            if spans.is_empty() { return Err(refused("Resolved scope is empty".to_owned())); }
            spans
        }
        TextScope::FindText { text: needle, occurrence } => {
            let found = hits(norm, text.len(), needle);
            if found.is_empty() { return Err(refused(format!("Scope text not found: \"{}\". Copy it verbatim from the document.", quoted(needle)))); }
            match occurrence {
                Some(occurrence) => vec![*occurrence.checked_sub(1).and_then(|at| found.get(at))
                    .ok_or_else(|| refused(format!("Scope occurrence {occurrence} not found ({} matches)", found.len())))?],
                None => found,
            }
        }
        TextScope::Range { from_text, to_text } => {
            let start = *hits(norm, text.len(), from_text).first().ok_or_else(|| refused(format!("Range start not found: \"{}\"", quoted(from_text))))?;
            let end = hits(norm, text.len(), to_text).into_iter().find(|hit| hit.0 >= start.1)
                .ok_or_else(|| refused(format!("Range end not found after its start: \"{}\"", quoted(to_text))))?;
            vec![(start.0, end.1)]
        }
    })
}

/// A span's text become `new` as edits: its minimal changes, none across a paragraph, an insertion widened by a
/// neighbour from its own paragraph, and changes at most two characters apart joined.
fn span_changes(text: &[u16], start: usize, end: usize, new: &str) -> Result<Vec<(usize, usize, Units)>> {
    let mut out = Vec::<(usize, usize, Units)>::new();
    for (offset, deleted, inserted) in minimal(&text[start..end], &units(new), true).0 {
        let (deleted, inserted) = (deleted.split(|unit| *unit == 10).collect::<Vec<_>>(), inserted.split(|unit| *unit == 10).collect::<Vec<_>>());
        if deleted.len() != inserted.len() { return Err(Error::Message("A text op moved a paragraph boundary".to_owned())); }
        let mut at = start + offset;
        for (deleted, inserted) in deleted.into_iter().zip(inserted) {
            let (mut from, mut to, mut new) = (at, at + deleted.len(), inserted.to_vec());
            at = to + 1;
            if from == to && new.is_empty() { continue; }
            if from == to {
                if from > 0 && text[from - 1] != 10 { from -= 1; new.insert(0, text[from]); }
                else if let Some(&next) = text.get(to) { new.push(next); to += 1; }
            }
            match out.last_mut() {
                Some(last) if from <= last.1 + 2 && !text[last.1..from.max(last.1)].contains(&10) => {
                    last.2.extend(&text[last.1..from.max(last.1)]);
                    last.2.extend(new);
                    last.1 = to;
                }
                _ => out.push((from, to, new)),
            }
        }
    }
    Ok(out)
}

/// Deterministic text ops written as edits, each change its own: each op run once on each of its scope's spans of the
/// accepted text, and the differences written as `apply_tracked_edits` writes an edit. Ops whose changes overlap, or
/// that move a paragraph, refuse the document.
pub fn apply_text_ops(bytes: &[u8], ops: &[TextOpRequest], author: &str, mode: EditMode) -> Result<TextOpsApplied> {
    let text = body_units(bytes)?;
    let norm = normalized(&text);
    let (mut changes, mut reports) = (Vec::new(), Vec::new());
    for request in ops {
        let mut report = TextOpReport { op: request.op.clone(), replacements: 0, notes: Vec::new() };
        for (start, end) in scope_spans(&request.scope, &text, &norm)? {
            let (new, notes) = crate::text_ops::run_text_op(&request.op, &string(&text[start..end]), &request.params)?;
            let planned = span_changes(&text, start, end, &new)?;
            report.notes.extend(notes);
            report.replacements += planned.len();
            changes.extend(planned);
        }
        reports.push(report);
    }
    changes.sort_by_key(|(start, end, _)| (*start, *end));
    if changes.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(Error::Message("Two ops produced overlapping changes; apply them in separate calls".to_owned()));
    }
    let edits = changes.iter().map(|(start, end, new)| EditInput { find: string(&text[*start..*end]), replace: string(new),
        context_before: String::new(), context_after: String::new(), reason: None, exact_start: Some(*start), exact_end: Some(*end) }).collect::<Vec<_>>();
    let edits = if edits.is_empty() { TrackedEdits { bytes: bytes.to_vec(), changes: Vec::new(), errors: Vec::new(), mode } }
        else { apply_tracked_edits(bytes, &edits, author, mode)? };
    Ok(TextOpsApplied { edits, reports, replacement_count: changes.len() })
}