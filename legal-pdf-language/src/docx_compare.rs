//! An old version's changes shown on the new one: the new document with `w:ins`/`w:del` for what changed from the
//! old, its blocks aligned by their text, a changed paragraph's words diffed (punctuation apart; quote, dash and space
//! variants equal), and what cannot be shown as a revision said as an abstention. Only document.xml is rewritten, and
//! only the paragraphs a change touches.

use crate::docx::{read_docx_files, replace_file, write_docx_files};
use crate::docx_spans::{escaped, parse, part, text_of, Node};
use crate::docx_tracked::{body, clusters, difference, highest, normalized, paragraph, paragraphs, string, units, Units};
use diff_match_patch_rs::Ops;
use legal_pdf_core::{Error, Result};
use legal_structure::{javascript_whitespace, JS_WHITESPACE_CLASS};
use regex::Regex;
use serde::Serialize;
use std::sync::LazyLock;

#[derive(Serialize, Clone, Copy)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "lowercase")]
pub enum CompareKind {
    Insert,
    Delete,
    Replace,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct CompareChange {
    pub kind: CompareKind,
    pub deleted_text: String,
    pub inserted_text: String,
}

/// What the redline leaves unshown, and an excerpt of the text it concerns.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct CompareAbstention {
    pub reason: String,
    pub excerpt: String,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct ComparedVersions {
    #[serde(skip)]
    #[cfg_attr(feature = "binding-types", ts(skip))]
    pub bytes: Vec<u8>,
    pub changes: Vec<CompareChange>,
    pub abstentions: Vec<CompareAbstention>,
}

const MAX_LCS_CELLS: usize = 4_000_000;
const MAX_GAP_CELLS: usize = 10_000;
const PAIR_SIMILARITY: f64 = 0.5;

fn space(unit: u16) -> bool {
    char::from_u32(u32::from(unit)).is_some_and(javascript_whitespace)
}

fn trim(text: &[u16]) -> &[u16] {
    let start = text.iter().position(|unit| !space(*unit)).unwrap_or(text.len());
    let end = text.iter().rposition(|unit| !space(*unit)).map_or(start, |at| at + 1);
    &text[start..end]
}

fn norm_trim(text: &[u16]) -> Units {
    trim(&normalized(text).0).to_vec()
}

fn abstention(reason: String, text: &[u16]) -> CompareAbstention {
    let mut flat = Vec::new();
    for &unit in text {
        if !space(unit) { flat.push(unit) } else if flat.last() != Some(&32) { flat.push(32) }
    }
    let flat = trim(&flat);
    CompareAbstention { reason, excerpt: if flat.len() > 160 { format!("{}…", string(&flat[..160])) } else { string(flat) } }
}

/// The pairs of equal items of their longest common subsequence, after their shared ends; none where the middle is too
/// large to align.
fn align(a: &[Units], b: &[Units]) -> Option<Vec<(usize, usize)>> {
    let prefix = a.iter().zip(b).take_while(|(a, b)| a == b).count();
    let suffix = a[prefix..].iter().rev().zip(b[prefix..].iter().rev()).take_while(|(a, b)| a == b).count();
    let (n, m) = (a.len() - prefix - suffix, b.len() - prefix - suffix);
    let mut pairs = (0..prefix).map(|at| (at, at)).collect::<Vec<_>>();
    if n > 0 && m > 0 {
        if n * m > MAX_LCS_CELLS { return None; }
        let width = m + 1;
        let mut table = vec![0u32; (n + 1) * width];
        for i in 1..=n {
            for j in 1..=m {
                table[i * width + j] = if a[prefix + i - 1] == b[prefix + j - 1] { table[(i - 1) * width + j - 1] + 1 }
                    else { table[(i - 1) * width + j].max(table[i * width + j - 1]) };
            }
        }
        let (mut middle, mut i, mut j) = (Vec::new(), n, m);
        while i > 0 && j > 0 {
            if a[prefix + i - 1] == b[prefix + j - 1] { middle.push((prefix + i - 1, prefix + j - 1)); i -= 1; j -= 1; }
            else if table[(i - 1) * width + j] >= table[i * width + j - 1] { i -= 1 } else { j -= 1 }
        }
        pairs.extend(middle.into_iter().rev());
    }
    pairs.extend((1..=suffix).rev().map(|k| (a.len() - k, b.len() - k)));
    Some(pairs)
}

fn similarity(a: &[u16], b: &[u16]) -> f64 {
    let total = (a.len() + b.len()) as f64;
    if total == 0.0 { return 1.0; }
    if 2.0 * a.len().min(b.len()) as f64 / total < PAIR_SIMILARITY { return 0.0; }
    let equal = difference(&string(a), &string(b), false).iter().filter(|(op, _)| *op == Ops::Equal).map(|(_, text)| text.len()).sum::<usize>();
    2.0 * equal as f64 / total
}

enum Gap {
    Changed(usize, usize),
    Deleted(usize),
    Inserted(usize),
}

/// Unaligned paragraphs paired where similar enough (least edit cost), the rest deleted or inserted.
fn pair_gap(old: &[Units], new: &[Units]) -> Vec<Gap> {
    let (n, m) = (old.len(), new.len());
    if n == 0 || m == 0 || n * m > MAX_GAP_CELLS {
        return (0..n).map(Gap::Deleted).chain((0..m).map(Gap::Inserted)).collect();
    }
    let width = m + 1;
    let (mut cost, mut from) = (vec![0f64; (n + 1) * width], vec![0u8; (n + 1) * width]);
    for i in 1..=n { (cost[i * width], from[i * width]) = (i as f64, 1); }
    for j in 1..=m { (cost[j], from[j]) = (j as f64, 2); }
    for i in 1..=n {
        for j in 1..=m {
            let (delete, insert) = (cost[(i - 1) * width + j] + 1.0, cost[i * width + j - 1] + 1.0);
            let sim = similarity(&old[i - 1], &new[j - 1]);
            let change = if sim >= PAIR_SIMILARITY { cost[(i - 1) * width + j - 1] + 2.0 * (1.0 - sim) } else { f64::INFINITY };
            (cost[i * width + j], from[i * width + j]) = if change <= delete && change <= insert { (change, 3) } else if delete <= insert { (delete, 1) } else { (insert, 2) };
        }
    }
    let (mut ops, mut i, mut j) = (Vec::new(), n, m);
    while i > 0 || j > 0 {
        match from[i * width + j] {
            3 => { ops.push(Gap::Changed(i - 1, j - 1)); i -= 1; j -= 1; }
            1 => { ops.push(Gap::Deleted(i - 1)); i -= 1; }
            _ => { ops.push(Gap::Inserted(j - 1)); j -= 1; }
        }
    }
    ops.reverse();
    ops
}

static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!(
    r#"[{JS_WHITESPACE_CLASS}]+|\.\.\.|\[[^\]]+\]|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|["“”‘’]|[^A-Za-z0-9_{JS_WHITESPACE_CLASS}]|."#)).expect("token"));

/// A text's tokens: a word (with its internal apostrophes), a bracket, an ellipsis, a punctuation mark, or a space;
/// a word or bracket directly beside another fused with it.
fn tokens(text: &str) -> Vec<String> {
    let mergeable = |token: &str| !token.starts_with(javascript_whitespace)
        && (token.contains(|c: char| c.is_ascii_alphanumeric()) || token.starts_with('[') && token.ends_with(']'));
    let mut merged = Vec::<String>::new();
    for found in TOKEN.find_iter(text) {
        match merged.last_mut() {
            Some(last) if mergeable(last) && mergeable(found.as_str()) => last.push_str(found.as_str()),
            _ => merged.push(found.as_str().to_owned()),
        }
    }
    merged
}

/// A token as words compare: every space one, quotes and dashes plain; case kept.
fn token_key(token: &str) -> String {
    const DASHES: &str = "\u{ad}\u{2010}\u{2011}\u{2012}\u{2013}\u{2014}\u{2015}\u{2212}";
    if token.starts_with(javascript_whitespace) { return " ".to_owned(); }
    if token.encode_utf16().count() == 1 {
        let character = token.chars().next().unwrap_or_default();
        if "\"“”«»„".contains(character) { return "\"".to_owned(); }
        if "'‘’‚".contains(character) { return "'".to_owned(); }
        if character == '-' || DASHES.contains(character) { return "-".to_owned(); }
    }
    token.chars().map(|character| match character { '‘' | '’' => '\'', '“' | '”' => '"', c if DASHES.contains(c) => '-', c => c }).collect()
}

/// A changed paragraph's word changes, each placed in the new text: its start and end there, and the old text it deletes.
fn word_changes(old: &[u16], new: &[u16]) -> Vec<(usize, usize, Units)> {
    if old == new { return Vec::new(); }
    let (old, new) = (string(old), string(new));
    let (old_tokens, new_tokens) = (tokens(&old), tokens(&new));
    let mut codes = std::collections::HashMap::<String, char>::new();
    let mut encode = |tokens: &[String]| tokens.iter().map(|token| {
        let next = codes.len() as u32 + 1;
        let code = *codes.entry(token_key(token)).or_insert_with(|| char::from_u32(next).unwrap_or('\u{d7ff}'));
        (next < 0xd7ff).then_some(code)
    }).collect::<Option<String>>();
    let parts = match (encode(&old_tokens), encode(&new_tokens)) {
        (Some(old_code), Some(new_code)) => {
            let (mut at_old, mut at_new) = (0, 0);
            difference(&old_code, &new_code, false).into_iter().map(|(op, encoded)| {
                let count = encoded.len();
                // Equal keys can differ in length: the offsets belong to the new text.
                let text = if op == Ops::Delete { &old_tokens[at_old..at_old + count] } else { &new_tokens[at_new..at_new + count] };
                if op != Ops::Insert { at_old += count; }
                if op != Ops::Delete { at_new += count; }
                (op, units(&text.concat()))
            }).collect()
        }
        _ => difference(&old, &new, true),
    };
    clusters(&parts, true).into_iter().map(|(at, deleted, inserted)| (at, at + inserted.len(), deleted)).collect()
}

/// A paragraph's content as a comparison rewrites it: its text, tabs and breaks, each in its run, and what it keeps
/// whole (a bookmark, a note reference's run); or why it cannot be rewritten.
enum Atom<'a> {
    Text(Units, usize),
    Tab(usize),
    Break(usize),
    Keep(&'a Node),
}

fn atoms<'a>(xml: &'a str, node: &'a Node) -> std::result::Result<(Vec<Atom<'a>>, Vec<&'a str>, Units), String> {
    let (mut atoms, mut properties, mut text) = (Vec::new(), Vec::new(), Vec::new());
    for child in node.children.iter().filter(|child| child.name != "w:pPr") {
        match child.name.as_str() {
            "w:bookmarkStart" | "w:bookmarkEnd" | "w:proofErr" | "w:commentRangeStart" | "w:commentRangeEnd" => atoms.push(Atom::Keep(child)),
            "w:r" => {
                let run = properties.len();
                properties.push(child.child("w:rPr").map_or("", |rpr| &xml[rpr.start..rpr.end]));
                let (mut own, mut own_text, mut whole, mut unsupported) = (Vec::new(), Vec::new(), false, None);
                for item in &child.children {
                    match item.name.as_str() {
                        "w:rPr" | "w:lastRenderedPageBreak" => {}
                        "w:t" => { let value = text_of(xml, item); own_text.extend(&value); own.push(Atom::Text(value, run)); }
                        "w:tab" => { own_text.push(9); own.push(Atom::Tab(run)); }
                        "w:br" if xml[item.start..item.open_end].contains('=') => whole = true,
                        "w:cr" | "w:br" => { own_text.push(10); own.push(Atom::Break(run)); }
                        "w:footnoteReference" | "w:endnoteReference" | "w:commentReference" | "w:drawing" => whole = true,
                        name => { unsupported.get_or_insert_with(|| format!("contains run content <{name}>")); }
                    }
                }
                if let Some(reason) = unsupported { return Err(reason); }
                if whole && !own_text.is_empty() { return Err("a run mixes text with non-text content".to_owned()); }
                if whole { atoms.push(Atom::Keep(child)) } else { atoms.extend(own); text.extend(own_text); }
            }
            name => return Err(format!("contains <{name}>")),
        }
    }
    Ok((atoms, properties, text))
}

fn opened(xml: &str, node: &Node) -> String {
    let open = &xml[node.start..node.open_end];
    open.strip_suffix("/>").map_or_else(|| open.to_owned(), |open| format!("{}>", open.trim_end()))
}

fn closed(node: &Node) -> String {
    format!("</{}>", node.name)
}

/// The revision ids given out, from one past the document's highest, with their author and time.
struct Revisions {
    next: u64,
    author: String,
    date: String,
}

impl Revisions {
    fn wrap(&mut self, tag: &str, content: &str) -> String {
        self.next += 1;
        format!(r#"<{tag} w:id="{}" w:author="{}" w:date="{}">{content}</{tag}>"#, self.next - 1, escaped(&self.author), self.date)
    }
    fn mark(&mut self, tag: &str) -> String {
        self.next += 1;
        format!(r#"<{tag} w:id="{}" w:author="{}" w:date="{}"/>"#, self.next - 1, escaped(&self.author), self.date)
    }
}

/// A paragraph's properties with its mark revision first in their run properties (made where there are none, before a
/// section's properties); for a deleted paragraph, without its section, numbering or other mark revisions.
fn marked(xml: &str, properties: Option<&Node>, mark: &str, deleted: bool) -> String {
    let Some(properties) = properties else { return format!("<w:pPr><w:rPr>{mark}</w:rPr></w:pPr>") };
    let (mut out, mut placed) = (opened(xml, properties), false);
    for child in &properties.children {
        match child.name.as_str() {
            "w:sectPr" | "w:numPr" if deleted => {}
            "w:rPr" => {
                out.push_str(&opened(xml, child));
                out.push_str(mark);
                child.children.iter().filter(|item| !deleted || !matches!(item.name.as_str(), "w:ins" | "w:del")).for_each(|item| out.push_str(&xml[item.start..item.end]));
                out.push_str("</w:rPr>");
                placed = true;
            }
            "w:sectPr" if !placed => { out.push_str(&format!("<w:rPr>{mark}</w:rPr>{}", &xml[child.start..child.end])); placed = true; }
            _ => out.push_str(&xml[child.start..child.end]),
        }
    }
    if !placed { out.push_str(&format!("<w:rPr>{mark}</w:rPr>")); }
    out + "</w:pPr>"
}

/// The old paragraph as a deleted one in the new package: its runs' text, tabs and breaks as deleted text, and what
/// could not be carried across (note references, objects, fields, hyperlinks, content controls); none where it holds
/// nothing carried.
fn deleted_paragraph(xml: &str, node: &Node, revisions: &mut Revisions) -> (Option<String>, Vec<String>) {
    fn visit(xml: &str, nodes: &[Node], runs: &mut String, dropped: &mut Vec<String>) {
        let drop = |what: &str, dropped: &mut Vec<String>| if !dropped.iter().any(|known| known == what) { dropped.push(what.to_owned()) };
        for node in nodes {
            match node.name.as_str() {
                "w:pPr" | "w:del" | "w:bookmarkStart" | "w:bookmarkEnd" | "w:proofErr" | "w:commentRangeStart" | "w:commentRangeEnd" => {}
                "w:r" => {
                    let (mut kids, mut content) = (String::new(), false);
                    for item in &node.children {
                        match item.name.as_str() {
                            "w:rPr" => kids.push_str(&xml[item.start..item.end]),
                            "w:t" => { kids.push_str(&format!(r#"<w:delText xml:space="preserve">{}</w:delText>"#, escaped(&string(&text_of(xml, item))))); content = true; }
                            "w:tab" | "w:br" | "w:cr" | "w:noBreakHyphen" | "w:softHyphen" | "w:sym" => { kids.push_str(&xml[item.start..item.end]); content = true; }
                            "w:lastRenderedPageBreak" | "w:commentReference" => {}
                            "w:footnoteReference" | "w:endnoteReference" => drop("note reference", dropped),
                            "w:drawing" | "w:object" | "w:pict" => drop("image or object", dropped),
                            "w:fldChar" | "w:instrText" => drop("field", dropped),
                            name => drop(name, dropped),
                        }
                    }
                    if content { runs.push_str(&format!("<w:r>{kids}</w:r>")); }
                }
                "w:hyperlink" => { drop("hyperlink", dropped); visit(xml, &node.children, runs, dropped); }
                "w:smartTag" | "w:customXml" | "w:ins" => visit(xml, &node.children, runs, dropped),
                "w:sdt" => {
                    drop("content control", dropped);
                    node.children.iter().filter(|child| child.name == "w:sdtContent").for_each(|content| visit(xml, &content.children, runs, dropped));
                }
                "w:fldSimple" => { drop("field", dropped); visit(xml, &node.children, runs, dropped); }
                name => drop(name, dropped),
            }
        }
    }
    let (mut runs, mut dropped) = (String::new(), Vec::new());
    visit(xml, &node.children, &mut runs, &mut dropped);
    if runs.is_empty() && !dropped.is_empty() { return (None, dropped); }
    let deletion = if runs.is_empty() { String::new() } else { revisions.wrap("w:del", &runs) };
    let mark = revisions.mark("w:del");
    (Some(format!("<w:p>{}{deletion}</w:p>", marked(xml, node.child("w:pPr"), &mark, true))), dropped)
}

/// Why an added paragraph cannot be wrapped as inserted, if it cannot.
fn not_insertable(node: &Node) -> Option<String> {
    const RUN: [&str; 13] = ["w:rPr", "w:t", "w:tab", "w:br", "w:cr", "w:noBreakHyphen", "w:softHyphen", "w:lastRenderedPageBreak",
        "w:footnoteReference", "w:endnoteReference", "w:commentReference", "w:drawing", "w:sym"];
    fn visit(nodes: &[Node], properties: bool) -> Option<String> {
        nodes.iter().find_map(|node| match node.name.as_str() {
            "w:pPr" if properties => None,
            "w:pPr" => Some("contains nested <w:pPr>".to_owned()),
            "w:r" => node.children.iter().find(|item| !RUN.contains(&item.name.as_str())).map(|item| format!("contains run content <{}>", item.name)),
            "w:hyperlink" | "w:smartTag" => visit(&node.children, false),
            "w:sdt" => node.children.iter().filter(|child| child.name == "w:sdtContent").find_map(|content| visit(&content.children, false)),
            "w:ins" | "w:del" | "w:bookmarkStart" | "w:bookmarkEnd" | "w:proofErr" | "w:commentRangeStart" | "w:commentRangeEnd" => None,
            name => Some(format!("contains <{name}>")),
        })
    }
    visit(&node.children, true)
}

/// An added paragraph marked inserted: each group of its runs (in hyperlinks and content controls too) and its mark.
fn inserted_paragraph(xml: &str, node: &Node, revisions: &mut Revisions) -> String {
    fn wrapped(xml: &str, nodes: &[Node], revisions: &mut Revisions) -> String {
        let (mut out, mut group) = (String::new(), String::new());
        for node in nodes.iter().filter(|node| node.name != "w:pPr") {
            if node.name == "w:r" { group.push_str(&xml[node.start..node.end]); continue; }
            if !group.is_empty() { out.push_str(&revisions.wrap("w:ins", &std::mem::take(&mut group))); }
            out.push_str(&match node.name.as_str() {
                "w:hyperlink" | "w:smartTag" => format!("{}{}{}", opened(xml, node), wrapped(xml, &node.children, revisions), closed(node)),
                "w:sdt" => format!("{}{}{}", opened(xml, node), node.children.iter().map(|child| if child.name == "w:sdtContent" {
                    format!("{}{}{}", opened(xml, child), wrapped(xml, &child.children, revisions), closed(child))
                } else { xml[child.start..child.end].to_owned() }).collect::<String>(), closed(node)),
                _ => xml[node.start..node.end].to_owned(),
            });
        }
        if !group.is_empty() { out.push_str(&revisions.wrap("w:ins", &group)); }
        out
    }
    let content = wrapped(xml, &node.children, revisions);
    let mark = revisions.mark("w:ins");
    format!("{}{}{content}</w:p>", opened(xml, node), marked(xml, node.child("w:pPr"), &mark, false))
}

/// A run being written: its run's number, its finished children, and its open text.
struct Open {
    run: usize,
    children: String,
    text: Units,
}

/// A changed paragraph rewritten as the new version with the old one's word changes tracked: each deletion a deleted
/// run styled as the text after it, each insertion its runs wrapped as inserted.
struct Rewrite<'a> {
    properties: &'a [&'a str],
    revisions: &'a mut Revisions,
    out: String,
    insertion: Option<(usize, String)>,
    open: Option<Open>,
}

impl Rewrite<'_> {
    fn flush(&mut self) {
        if let Some(open) = self.open.take() {
            let text = if open.text.is_empty() { String::new() } else { format!(r#"<w:t xml:space="preserve">{}</w:t>"#, escaped(&string(&open.text))) };
            let run = format!("<w:r>{}{}{text}</w:r>", self.properties[open.run], open.children);
            match &mut self.insertion { Some((_, runs)) => runs.push_str(&run), None => self.out.push_str(&run) }
        }
    }
    fn close(&mut self) {
        self.flush();
        if let Some((_, runs)) = self.insertion.take() { let inserted = self.revisions.wrap("w:ins", &runs); self.out.push_str(&inserted); }
    }
    fn select(&mut self, cluster: Option<usize>) {
        if self.insertion.as_ref().map(|(at, _)| *at) == cluster { return; }
        self.close();
        self.insertion = cluster.map(|at| (at, String::new()));
    }
    fn append(&mut self, run: usize, text: Option<&[u16]>, element: &str) {
        if self.open.as_ref().is_none_or(|open| open.run != run) {
            self.flush();
            self.open = Some(Open { run, children: String::new(), text: Vec::new() });
        }
        let open = self.open.as_mut().expect("open run");
        match text {
            Some(text) => open.text.extend(text),
            None => {
                if !open.text.is_empty() { open.children.push_str(&format!(r#"<w:t xml:space="preserve">{}</w:t>"#, escaped(&string(&std::mem::take(&mut open.text))))); }
                open.children.push_str(element);
            }
        }
    }
    fn delete(&mut self, text: &[u16], properties: &str) {
        self.close();
        let pieces = string(text).split_inclusive(['\t', '\n']).flat_map(|piece| {
            let (body, end) = piece.strip_suffix('\t').map(|body| (body, "<w:tab/>")).or_else(|| piece.strip_suffix('\n').map(|body| (body, "<w:br/>"))).unwrap_or((piece, ""));
            [(!body.is_empty()).then(|| format!(r#"<w:delText xml:space="preserve">{}</w:delText>"#, escaped(body))).unwrap_or_default(), end.to_owned()]
        }).collect::<String>();
        let deleted = self.revisions.wrap("w:del", &format!("<w:r>{properties}{pieces}</w:r>"));
        self.out.push_str(&deleted);
    }
}

fn rewritten(xml: &str, node: &Node, atoms: &[Atom], properties: &[&str], changes: &[(usize, usize, Units)], revisions: &mut Revisions) -> (String, Vec<String>) {
    let mut notes = Vec::new();
    let mut writer = Rewrite { properties, revisions, out: String::new(), insertion: None, open: None };
    // A deletion takes the style of the run whose text is at its place (the last text's at the end): that run's number.
    let (mut style_at, mut style_end, mut style) = (0, 0, None::<usize>);
    let mut style_for = |position: usize| {
        while style_at < atoms.len() && style_end <= position {
            match &atoms[style_at] {
                Atom::Text(text, run) if !text.is_empty() => { style_end += text.len(); style = Some(*run); }
                Atom::Tab(run) | Atom::Break(run) => { style_end += 1; style = Some(*run); }
                _ => {}
            }
            style_at += 1;
        }
        style
    };
    let (mut next, mut active) = (0, None::<usize>);
    let mut advance = |position: usize, writer: &mut Rewrite, style_for: &mut dyn FnMut(usize) -> Option<usize>| {
        while next < changes.len() && changes[next].0 <= position {
            if !changes[next].2.is_empty() { let style = style_for(changes[next].0).map_or("", |run| properties[run]); writer.delete(&changes[next].2, style); }
            active = Some(next);
            next += 1;
        }
        if active.is_some_and(|at| changes[at].1 <= position) { active = None; }
        (active, next)
    };
    let mut position = 0;
    for atom in atoms {
        let (run, text, element) = match atom {
            Atom::Keep(kept) => {
                advance(position, &mut writer, &mut style_for);
                if writer.insertion.as_ref().is_some_and(|(at, _)| changes[*at].0 < position && position < changes[*at].1) {
                    notes.push("inline_object_in_inserted_range: an inline object inside inserted text was preserved but not itself marked as inserted".to_owned());
                }
                writer.close();
                writer.out.push_str(&xml[kept.start..kept.end]);
                continue;
            }
            Atom::Text(text, run) => (*run, text.clone(), ""),
            Atom::Tab(run) => (*run, vec![9], "<w:tab/>"),
            Atom::Break(run) => (*run, vec![10], "<w:br/>"),
        };
        let mut offset = 0;
        while offset < text.len() {
            let (cluster, next) = advance(position, &mut writer, &mut style_for);
            let boundary = cluster.map(|at| changes[at].1).or_else(|| changes.get(next).map(|change| change.0)).unwrap_or(usize::MAX);
            let length = (text.len() - offset).min(boundary - position);
            writer.select(cluster);
            writer.append(run, element.is_empty().then(|| &text[offset..offset + length]), element);
            position += length;
            offset += length;
        }
        if text.is_empty() { advance(position, &mut writer, &mut style_for); }
    }
    advance(position, &mut writer, &mut style_for);
    writer.close();
    let properties = node.child("w:pPr").map_or("", |pr| &xml[pr.start..pr.end]);
    (format!("{}{properties}{}</w:p>", opened(xml, node), writer.out), notes)
}

/// A body block: a paragraph, a table or a content control, its place among the body's elements, its text (a
/// paragraph's compare text; a table's or control's paragraphs', each ended by a newline) and its alignment key.
struct Block<'a> {
    kind: &'static str,
    node: &'a Node,
    index: usize,
    text: Units,
    key: Units,
}

fn blocks<'a>(xml: &'a str, body: &'a Node) -> Vec<Block<'a>> {
    body.children.iter().enumerate().filter_map(|(index, node)| {
        let kind = match node.name.as_str() { "w:p" => "p", "w:tbl" => "tbl", "w:sdt" => "sdt", _ => return None };
        let text = if kind == "p" { paragraph(xml, node).compare }
            else { paragraphs(xml, node).into_iter().flat_map(|paragraph| paragraph.compare.into_iter().chain([10])).collect() };
        let key = units(kind).into_iter().chain([58]).chain(norm_trim(&text)).collect();
        Some(Block { kind, node, index, text, key })
    }).collect()
}

/// Only list presence and level compare: a numbering id is the package's own.
fn numbering(node: &Node, xml: &str) -> Option<String> {
    static VALUE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"w:val="([^"]*)""#).expect("value"));
    let numbered = node.child("w:pPr")?.child("w:numPr")?;
    let level = numbered.child("w:ilvl").and_then(|level| VALUE.captures(&xml[level.start..level.open_end]).map(|value| value[1].to_owned()));
    Some(format!("ilvl:{}", level.as_deref().unwrap_or("0")))
}

/// Each story but the main one whose text differs between the versions: only the main story is marked up.
fn other_stories(old: &[(String, Vec<u8>)], new: &[(String, Vec<u8>)]) -> Vec<CompareAbstention> {
    static DELETED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<w:del\b.*?</w:del>").expect("deleted"));
    static TEXT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<w:t\b[^>]*>(.*?)</w:t>").expect("text"));
    let stories = [("headers_changed", "header", r"(?i)^word[/\\]header\d*\.xml$"), ("footers_changed", "footer", r"(?i)^word[/\\]footer\d*\.xml$"),
        ("footnotes_changed", "footnote", r"(?i)^word[/\\]footnotes\.xml$"), ("endnotes_changed", "endnote", r"(?i)^word[/\\]endnotes\.xml$")];
    let text = |files: &[(String, Vec<u8>)], pattern: &Regex| files.iter().filter(|(path, _)| pattern.is_match(path)).map(|(_, bytes)| {
        let xml = DELETED.replace_all(&String::from_utf8_lossy(bytes), "").into_owned();
        TEXT.captures_iter(&xml).map(|found| quick_xml::escape::unescape(&found[1]).map_or_else(|_| found[1].to_owned(), |text| text.into_owned())).collect::<String>()
    }).collect::<Vec<_>>().join("\n");
    stories.into_iter().filter_map(|(code, label, pattern)| {
        let pattern = Regex::new(pattern).expect("story");
        let (before, after) = (units(&text(old, &pattern)), units(&text(new, &pattern)));
        (norm_trim(&before) != norm_trim(&after)).then(|| abstention(format!("{code}: the {label} story differs between the two versions; \
            compare_versions marks up the main document story only, so this difference is not shown in the redline"), if after.is_empty() { &before } else { &after }))
    }).collect()
}

/// The new version with the old one's changes as tracked changes by `author`, each change, and what is not shown.
pub fn compare_docx_versions(old: &[u8], new: &[u8], author: &str) -> Result<ComparedVersions> {
    let (old_files, mut new_files) = (read_docx_files(old, None)?, read_docx_files(new, None)?);
    let document = |files: &[(String, Vec<u8>)], which: &str| part(files, "word/document.xml").ok_or_else(|| Error::Message(format!("{which} docx has no word/document.xml")));
    let (old_xml, new_xml) = (document(&old_files, "old")?, document(&new_files, "new")?);
    let (old_roots, new_roots) = (parse(&old_xml)?, parse(&new_xml)?);
    let missing = |which: &str| Error::Message(format!("{which} docx has no document body"));
    let (old_body, new_body) = (body(&old_roots).ok_or_else(|| missing("old"))?, body(&new_roots).ok_or_else(|| missing("new"))?);
    let mut abstentions = other_stories(&old_files, &new_files);
    let (old_blocks, new_blocks) = (blocks(&old_xml, old_body), blocks(&new_xml, new_body));
    let unchanged = |changes, abstentions| Ok(ComparedVersions { bytes: new.to_vec(), changes, abstentions });
    let Some(pairs) = align(&old_blocks.iter().map(|block| block.key.clone()).collect::<Vec<_>>(), &new_blocks.iter().map(|block| block.key.clone()).collect::<Vec<_>>()) else {
        abstentions.push(abstention(format!("documents_too_divergent: the two versions share too little aligned structure for a safe deterministic \
            comparison ({}×{} blocks); the new version is returned without revision marks", old_blocks.len(), new_blocks.len()),
            new_blocks.first().map_or(&[][..], |block| &block.text)));
        return unchanged(Vec::new(), abstentions);
    };
    let date = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    let mut revisions = Revisions { next: highest(&new_files)? + 1, author: author.to_owned(), date };
    let (mut changes, mut edits) = (Vec::new(), Vec::<(usize, usize, String)>::new());
    let abstain = |reason: String, text: &[u16], abstentions: &mut Vec<CompareAbstention>| abstentions.push(abstention(reason, text));
    let record = |deleted: Units, inserted: Units, changes: &mut Vec<CompareChange>| changes.push(CompareChange {
        kind: if !deleted.is_empty() && !inserted.is_empty() { CompareKind::Replace } else if !deleted.is_empty() { CompareKind::Delete } else { CompareKind::Insert },
        deleted_text: string(&deleted), inserted_text: string(&inserted) });
    // Added at the end, a paragraph goes before the body's section properties.
    let end = new_body.children.len() - usize::from(new_body.children.last().is_some_and(|last| last.name == "w:sectPr"));
    let at = |index: usize| new_body.children.get(index).map_or(new_body.inner_end, |child| child.start);
    let changed = |old: &Block, new: &Block, changes: &mut Vec<CompareChange>, edits: &mut Vec<(usize, usize, String)>, abstentions: &mut Vec<CompareAbstention>, revisions: &mut Revisions| {
        let (before, after) = (numbering(old.node, &old_xml), numbering(new.node, &new_xml));
        if before != after {
            abstain(format!("numbering_change_not_tracked: this paragraph's list numbering changed ({} → {}); numbering property changes are not \
                representable as text revisions and are left unmarked", before.as_deref().unwrap_or("none"), after.as_deref().unwrap_or("none")), &new.text, abstentions);
        }
        if old.text == new.text { return; }
        let (atoms, properties, _) = match atoms(&new_xml, new.node) {
            Ok(found) if found.2 == new.text => found,
            found => {
                let why = found.err().unwrap_or_else(|| "its text has structure the differ cannot round-trip".to_owned());
                abstain(format!("paragraph_not_diffable: a changed paragraph {why}; it is left as the new version's text without revision marks"), &new.text, abstentions);
                return;
            }
        };
        if paragraph(&old_xml, old.node).objects {
            abstain("old_paragraph_objects_not_compared: the old version of a changed paragraph contains non-text content (fields, images, or note \
                references); only its text was compared".to_owned(), &old.text, abstentions);
        }
        let found = word_changes(&old.text, &new.text);
        if found.is_empty() { return; }
        let (rebuilt, notes) = rewritten(&new_xml, new.node, &atoms, &properties, &found, revisions);
        edits.push((new.node.start, new.node.end, rebuilt));
        notes.into_iter().for_each(|note| abstain(note, &new.text, abstentions));
        for (start, end, deleted) in found { record(deleted, new.text[start..end].to_vec(), changes); }
    };
    let (mut old_at, mut new_at) = (0, 0);
    for k in 0..=pairs.len() {
        let (pi, pj) = pairs.get(k).copied().unwrap_or((old_blocks.len(), new_blocks.len()));
        let trailing = new_blocks.get(pj).map_or(end, |block| block.index);
        let (gap_old, gap_new) = (&old_blocks[old_at..pi], &new_blocks[new_at..pj]);
        for (kind, label, code) in [("tbl", "table", "table"), ("sdt", "content control", "content_control")] {
            let (old_kind, new_kind) = (gap_old.iter().filter(|block| block.kind == kind).collect::<Vec<_>>(), gap_new.iter().filter(|block| block.kind == kind).collect::<Vec<_>>());
            let paired = old_kind.len().min(new_kind.len());
            for block in &new_kind[..paired] { abstain(format!("{code}_changed: a {label} differs between the two versions; {label}s are not compared, and the new version's content is included without revision marks"), &block.text, &mut abstentions); }
            for block in &old_kind[paired..] { abstain(format!("{code}_removed: a {label} present in the old version does not appear in the new version; its removal is not shown in the redline"), &block.text, &mut abstentions); }
            for block in &new_kind[paired..] { abstain(format!("{code}_added: a {label} added in the new version is included without revision marks; {label} changes are not tracked"), &block.text, &mut abstentions); }
        }
        let (old_paragraphs, new_paragraphs) = (gap_old.iter().filter(|block| block.kind == "p").collect::<Vec<_>>(), gap_new.iter().filter(|block| block.kind == "p").collect::<Vec<_>>());
        let ops = pair_gap(&old_paragraphs.iter().map(|block| norm_trim(&block.text)).collect::<Vec<_>>(), &new_paragraphs.iter().map(|block| norm_trim(&block.text)).collect::<Vec<_>>());
        // A deleted paragraph goes before the next new paragraph kept or added after it.
        let mut anchors = vec![trailing; ops.len()];
        let mut carry = trailing;
        for (index, op) in ops.iter().enumerate().rev() {
            anchors[index] = carry;
            if let Gap::Changed(_, ni) | Gap::Inserted(ni) = op { carry = new_paragraphs[*ni].index; }
        }
        for (op, anchor) in ops.iter().zip(anchors) {
            match *op {
                Gap::Changed(oi, ni) => changed(old_paragraphs[oi], new_paragraphs[ni], &mut changes, &mut edits, &mut abstentions, &mut revisions),
                Gap::Deleted(oi) => {
                    let old = old_paragraphs[oi];
                    let (node, dropped) = deleted_paragraph(&old_xml, old.node, &mut revisions);
                    let Some(node) = node else {
                        abstain(format!("deleted_paragraph_unrepresentable: a paragraph removed from the old version has no text content (only: {}); \
                            its deletion is not shown in the redline", dropped.join(", ")), &old.text, &mut abstentions);
                        continue;
                    };
                    if !dropped.is_empty() {
                        abstain(format!("deleted_paragraph_content_dropped: a deleted paragraph is shown in the redline but its non-text content ({}) \
                            could not be carried into the new document's package", dropped.join(", ")), &old.text, &mut abstentions);
                    }
                    edits.push((at(anchor), at(anchor), node));
                    record(old.text.clone(), Vec::new(), &mut changes);
                }
                Gap::Inserted(ni) => {
                    let new = new_paragraphs[ni];
                    if let Some(problem) = not_insertable(new.node) {
                        abstain(format!("inserted_paragraph_not_markable: a paragraph added in the new version {problem} and cannot be safely wrapped \
                            in w:ins; it is included without revision marks"), &new.text, &mut abstentions);
                        continue;
                    }
                    edits.push((new.node.start, new.node.end, inserted_paragraph(&new_xml, new.node, &mut revisions)));
                    record(Vec::new(), new.text.clone(), &mut changes);
                }
            }
        }
        let Some(&(pi, pj)) = pairs.get(k) else { break };
        if old_blocks[pi].kind == "p" && new_blocks[pj].kind == "p" {
            changed(&old_blocks[pi], &new_blocks[pj], &mut changes, &mut edits, &mut abstentions, &mut revisions);
        }
        (old_at, new_at) = (pi + 1, pj + 1);
    }
    if edits.is_empty() { return unchanged(changes, abstentions); }
    // Last first; at one place a paragraph's rewrite before the deleted paragraphs put ahead of it.
    edits.sort_by_key(|(start, end, _)| (std::cmp::Reverse(*start), start == end));
    let mut xml = new_xml.clone();
    for (start, end, text) in edits { xml.replace_range(start..end, &text); }
    replace_file(&mut new_files, "word/document.xml", xml.into_bytes());
    Ok(ComparedVersions { bytes: write_docx_files(&new_files)?, changes, abstentions })
}
