//! A table of authorities written into a Word brief, at the citations `docx_to_toa_text_units` read: TA marks Word's own
//! table reads (each entry marked in full at its first citation, by its short name after), or a table appended (Word's
//! TOA fields with their results, a linked list, or a ruled table of where each authority is cited). Every edit is an
//! insertion at a place in the package as read, made last first, so the brief's own bytes are kept.

use crate::docx_spans::{escaped, open, parse, part, reading, text_element, text_of, units_in, DocxUnit, Node};
use crate::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(rename = "DocxPinpointLink"))]
pub struct PinpointLink {
    pub start: usize,
    pub end: usize,
    pub url: String,
}

/// A citation of an authority, at its place in a unit: the entry it gives the table (its long name, italic for its
/// first `italic` characters, its short name, its category), whether it is marked, what follows it (its tab, linked),
/// its pinpoint's link, and its authority.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
#[serde(rename_all = "camelCase")]
pub struct DocxAuthorityMark {
    pub unit_id: String,
    pub offset: usize,
    pub long_name: String,
    pub italic: Option<usize>,
    pub short_name: String,
    pub category: u8,
    pub suffix: Option<String>,
    pub mark: Option<bool>,
    pub tab_url: Option<String>,
    pub pinpoint_link: Option<PinpointLink>,
    pub authority_id: Option<String>,
}

/// An authority as an appended table lists it: its label (italic for its first `italic` characters), its link, and for
/// the ruled table its authority, group and tab.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
#[serde(rename_all = "camelCase")]
pub struct DocxLinkedAuthority {
    pub label: String,
    pub italic: usize,
    pub url: Option<String>,
    pub authority_id: Option<String>,
    pub group: Option<String>,
    pub tab: Option<String>,
}

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "kebab-case")]
pub enum DocxTableDelivery {
    NativeMarks,
    NativeAppend,
    LinkedAppend,
    RuledAppend,
}

fn attribute(xml: &str, node: &Node, name: &str) -> Option<String> {
    static ATTRIBUTE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\s([\w:]+)\s*=\s*(?:"([^"]*)"|'([^']*)')"#).expect("attribute"));
    ATTRIBUTE.captures_iter(&xml[node.start..node.open_end]).find(|found| &found[1] == name)
        .map(|found| quick_xml::escape::unescape(found.get(2).or(found.get(3)).map_or("", |value| value.as_str())).unwrap_or_default().into_owned())
}

fn utf16_slice(text: &str, from: usize, to: usize) -> String {
    let units = text.encode_utf16().collect::<Vec<_>>();
    String::from_utf16_lossy(&units[from.min(units.len())..to.min(units.len())])
}

/// Visits a node, then its children unless `visit` says not; deleted text and text boxes are no unit's own.
fn walk<'a>(node: &'a Node, visit: &mut impl FnMut(&'a Node) -> bool) {
    if node.name != "w:txbxContent" && node.name != "mc:Fallback" && visit(node) {
        node.children.iter().for_each(|child| walk(child, visit));
    }
}

/// Edits of the package's parts: each an insertion (or a replacement of the bytes it names) at its place in a part.
struct Edits(Vec<(usize, usize, usize, String)>);

impl Edits {
    fn insert(&mut self, part: usize, at: usize, text: String) {
        self.0.push((part, at, at, text));
    }

    /// `nodes` put at a text offset of a unit. Text inside a field's result is Word's to replace when it updates the
    /// field, so an offset there inserts after the field; the author's own link keeps only its own text, so an offset in
    /// it inserts after the link; else the run is cut there and the nodes go between its halves.
    fn at_offset(&mut self, part: usize, xml: &str, unit: &Node, offset: usize, nodes: &str) -> Result<()> {
        enum After<'a> { Target, Element(&'a Node) }
        #[derive(Default)]
        struct Walk<'a> { cursor: usize, open: i32, target: Option<(&'a Node, &'a Node)>, local: usize, after: Option<After<'a>>, link: Option<&'a Node> }
        fn descend<'a>(node: &'a Node, parent: Option<&'a Node>, run: Option<&'a Node>, run_parent: Option<&'a Node>,
            mut hyperlink: Option<&'a Node>, xml: &str, offset: usize, state: &mut Walk<'a>) {
            if state.after.is_some() || matches!(node.name.as_str(), "w:del" | "w:txbxContent" | "mc:Fallback") {
                return;
            }
            if node.name == "w:hyperlink" && parent.is_some() {
                hyperlink = Some(node);
            }
            let (next_run, next_parent) = if node.name == "w:r" { (Some(node), parent) } else { (run, run_parent) };
            if node.name == "w:fldChar" {
                let kind = attribute(xml, node, "w:fldCharType");
                state.open = (state.open + match kind.as_deref() { Some("begin") => 1, Some("end") => -1, _ => 0 }).max(0);
                if let (Some(_), 0, Some(run)) = (state.target, state.open, run) {
                    state.after = Some(After::Element(run));
                }
            }
            if node.name == "w:t" && state.target.is_none() {
                let end = state.cursor + text_of(xml, node).len();
                if let (true, Some(run), Some(run_parent)) = (state.cursor <= offset && offset <= end, next_run, next_parent) {
                    (state.target, state.link, state.local) = (Some((node, run)), hyperlink, offset - state.cursor);
                    if state.open == 0 && run_parent.name != "w:fldSimple" {
                        state.after = Some(After::Target);
                    }
                }
                state.cursor = end;
                return;
            }
            if matches!(node.name.as_str(), "w:tab" | "w:br" | "w:cr") {
                state.cursor += 1;
            }
            node.children.iter().for_each(|child| descend(child, Some(node), next_run, next_parent, hyperlink, xml, offset, state));
            if node.name == "w:fldSimple" && state.target.is_some() && state.after.is_none() && parent.is_some() {
                state.after = Some(After::Element(node));
            }
        }
        let mut state = Walk::default();
        descend(unit, None, None, None, None, xml, offset, &mut state);
        let Some((text, run)) = state.target else {
            return Err(crate::Error::Message("A reviewed citation no longer matches the Word document.".to_owned()));
        };
        match (state.after, state.link) {
            (Some(After::Target), Some(link)) => self.insert(part, link.end, nodes.to_owned()),
            (Some(After::Element(element)), _) => self.insert(part, element.end, nodes.to_owned()),
            _ => {
                // The run cut at the offset: its text closed there and opened again after the nodes, in its properties;
                // a cut at the end of its text with nothing after it goes after the run.
                let raw = &xml[text.open_end..text.inner_end];
                if state.local == text_of(xml, text).len() && !run.children.iter().any(|child| child.start >= text.end) {
                    return Ok(self.insert(part, run.end, nodes.to_owned()));
                }
                // The byte where the offset falls in the text's XML, an entity read as the character it stands for.
                let (mut at, mut units) = (0, 0);
                while units < state.local && at < raw.len() {
                    let width = if raw[at..].starts_with('&') { raw[at..].find(';').map_or(1, |end| end + 1) } else { raw[at..].chars().next().map_or(1, char::len_utf8) };
                    units += quick_xml::escape::unescape(&raw[at..at + width]).map_or(1, |character| character.encode_utf16().count());
                    at += width;
                }
                let properties = run.child("w:rPr").map_or("", |rpr| &xml[rpr.start..rpr.end]);
                let open = &xml[run.start..run.open_end];
                self.0.push((part, text.start, text.open_end, "<w:t xml:space=\"preserve\">".to_owned()));
                self.insert(part, text.open_end + at, format!("</w:t></w:r>{nodes}{open}{properties}<w:t xml:space=\"preserve\">"));
            }
        }
        Ok(())
    }

    fn apply(mut self, parts: &mut [String]) {
        self.0.sort_by_key(|(part, start, ..)| (*part, std::cmp::Reverse(*start)));
        self.0.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.1 != a.2 && a.3 == b.3);
        for (part, start, end, text) in self.0 {
            parts[part].replace_range(start..end, &text);
        }
    }
}

/// What a TA field gives Word's table: the text on one line.
fn entry_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A field argument: Word ends a quoted argument at a straight or curly double quote, so each is escaped.
fn field_text(value: &str) -> String {
    value.chars().fold(String::new(), |mut text, character| {
        if matches!(character, '"' | '“' | '”') { text.push('\\'); }
        text.push(character);
        text
    })
}

/// The text of a field's argument after `\name`, quoted or bare, with its escapes read.
fn field_argument(code: &str, name: char) -> Option<String> {
    let at = code.find(&format!("\\{name}")).filter(|at| code[at + 2..].starts_with(char::is_whitespace))?;
    let rest = code[at + 2..].trim_start();
    if !rest.starts_with(['"', '“']) {
        return rest.split_whitespace().next().map(str::to_owned);
    }
    let (mut value, mut characters) = (String::new(), rest.chars().skip(1).peekable());
    while let Some(character) = characters.next() {
        match character {
            '\\' if characters.peek().is_some_and(|next| matches!(next, '"' | '“' | '”')) => value.push(characters.next().unwrap()),
            '"' | '“' | '”' => return Some(value),
            other => value.push(other),
        }
    }
    Some(value)
}

/// A TA mark as Word writes it, no result: its code in parts, an italic part as the citation's style of cause.
fn field_runs(parts: &[(&str, bool)]) -> String {
    let run = |child: String, italic: bool| format!("<w:r>{}{child}</w:r>", if italic { "<w:rPr><w:i/></w:rPr>" } else { "" });
    let mut runs = run("<w:fldChar w:fldCharType=\"begin\"/>".to_owned(), false);
    for (text, italic) in parts {
        runs += &run(format!("<w:instrText xml:space=\"preserve\">{}</w:instrText>", escaped(text)), *italic);
    }
    runs + &run("<w:fldChar w:fldCharType=\"end\"/>".to_owned(), false)
}

/// A field with its result as Word last showed it.
fn field(code: &str, result: &str) -> String {
    format!("<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r><w:r><w:instrText xml:space=\"preserve\"> {} </w:instrText></w:r>\
        <w:r><w:fldChar w:fldCharType=\"separate\"/></w:r><w:r><w:rPr><w:noProof/></w:rPr>{}</w:r><w:r><w:fldChar w:fldCharType=\"end\"/></w:r>",
        escaped(code), text_element("w:t", result))
}

/// Word's sort of a table's entries (ICU's root collation, as the book's).
fn entry_order(left: &str, right: &str) -> std::cmp::Ordering {
    static COLLATOR: LazyLock<icu_collator::CollatorBorrowed<'static>> =
        LazyLock::new(|| icu_collator::Collator::try_new(Default::default(), Default::default()).expect("collator"));
    COLLATOR.compare(left, right)
}

const CATEGORY_NAMES: [(u8, &str); 7] = [(1, "Cases"), (2, "Legislation"), (3, "Other sources"), (4, "Rules"), (5, "Secondary sources"),
    (6, "Regulations"), (7, "Constitutional Provisions")];

fn category_name(category: u8) -> String {
    CATEGORY_NAMES.iter().find(|(known, _)| *known == category).map_or_else(|| category.to_string(), |(_, name)| (*name).to_owned())
}

/// The width of the page's text in twentieths of a point: where Word's table sets its page numbers.
fn text_width(xml: &str, section: Option<&Node>) -> i64 {
    let size = |node: Option<&Node>, name: &str| node.and_then(|node| attribute(xml, node, &format!("w:{name}"))).and_then(|value| value.parse::<i64>().ok()).unwrap_or(0);
    let (page, margins) = (section.and_then(|section| section.child("w:pgSz")), section.and_then(|section| section.child("w:pgMar")));
    let width = size(page, "w").max(if size(page, "w") == 0 { 12_240 } else { 0 })
        - margins.map_or(2_880, |margins| size(Some(margins), "left") + size(Some(margins), "right") + size(Some(margins), "gutter"));
    if width > 0 { width } else { 9_360 }
}

/// The page each place in the brief was on when Word last laid it out (lastRenderedPageBreak), or by its hard page breaks
/// where Word never laid it out; a note is on its reference's page.
fn rendered_pages<'a>(xml: &str, body: &'a Node, units: &HashMap<*const Node, &'a str>) -> impl Fn(&str, usize) -> usize + 'a {
    let mut laid_out = false;
    walk(body, &mut |node| { laid_out |= node.name == "w:lastRenderedPageBreak"; true });
    let (mut page, mut cursor, mut current) = (1, 0, None::<&str>);
    let (mut pages, mut notes) = (HashMap::<String, Vec<(usize, usize)>>::new(), HashMap::<String, usize>::new());
    let mut paragraph_starts = vec![(0, 1)];
    walk(body, &mut |node| {
        if node.name == "w:del" {
            return false;
        }
        match node.name.as_str() {
            "w:p" => {
                if let Some(id) = current.take() { pages.insert(id.to_owned(), std::mem::take(&mut paragraph_starts)); }
                (paragraph_starts, cursor, current) = (vec![(0, page)], 0, units.get(&(node as *const Node)).copied());
            }
            "w:t" => cursor += text_of(xml, node).len(),
            "w:tab" | "w:cr" | "w:br" => cursor += 1,
            "w:footnoteReference" => { notes.insert(format!("footnote:{}", attribute(xml, node, "w:id").unwrap_or_default()), page); }
            _ => {}
        }
        let breaks = if laid_out { node.name == "w:lastRenderedPageBreak" } else {
            node.name == "w:br" && attribute(xml, node, "w:type").as_deref() == Some("page")
                || node.name == "w:pageBreakBefore" && !matches!(attribute(xml, node, "w:val").as_deref(), Some("0" | "false"))
        };
        if breaks {
            page += 1;
            paragraph_starts.push((cursor, page));
        }
        true
    });
    if let Some(id) = current { pages.insert(id.to_owned(), paragraph_starts); }
    move |unit: &str, offset: usize| notes.get(unit).copied().unwrap_or_else(|| pages.get(unit)
        .and_then(|starts| starts.iter().filter(|(at, _)| *at <= offset).last()).map_or(1, |(_, page)| *page))
}

/// The TA fields a unit already holds: where each begins in its text, its category and the entry it gives the table.
fn existing_marks(xml: &str, unit: &Node) -> Vec<(usize, u8, Option<String>)> {
    static CATEGORY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\\c\s+"?(\d+)"#).expect("category"));
    static TA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*TA\b").expect("TA"));
    let (mut found, mut open, mut cursor) = (Vec::new(), Vec::<(usize, String)>::new(), 0);
    walk(unit, &mut |node| {
        match node.name.as_str() {
            "w:del" => return false,
            "w:t" => cursor += text_of(xml, node).len(),
            "w:tab" | "w:br" | "w:cr" => cursor += 1,
            "w:instrText" => if let Some((_, code)) = open.last_mut() { code.push_str(&String::from_utf16_lossy(&text_of(xml, node))); },
            "w:fldChar" => match attribute(xml, node, "w:fldCharType").as_deref() {
                Some("begin") => open.push((cursor, String::new())),
                Some("end") => if let Some((at, code)) = open.pop() {
                    if TA.is_match(&code) {
                        if let Some(category) = CATEGORY.captures(&code).and_then(|found| found[1].parse().ok()) {
                            found.push((at, category, field_argument(&code, 'l')));
                        }
                    }
                },
                _ => {}
            },
            _ => {}
        }
        true
    });
    found
}

/// Where the author's links lie in a unit's text.
fn linked_ranges(xml: &str, unit: &Node) -> Vec<(usize, usize)> {
    fn visit(node: &Node, xml: &str, cursor: &mut usize, ranges: &mut Vec<(usize, usize)>) {
        match node.name.as_str() {
            "w:del" | "w:txbxContent" | "mc:Fallback" => return,
            "w:t" => *cursor += text_of(xml, node).len(),
            "w:tab" | "w:br" | "w:cr" => *cursor += 1,
            _ => {}
        }
        let start = *cursor;
        node.children.iter().for_each(|child| visit(child, xml, cursor, ranges));
        if node.name == "w:hyperlink" { ranges.push((start, *cursor)); }
    }
    let (mut cursor, mut ranges) = (0, Vec::new());
    visit(unit, xml, &mut cursor, &mut ranges);
    ranges
}

/// The body's paragraphs, in order, through its tables and content controls.
fn paragraphs(body: &Node) -> Vec<&Node> {
    fn collect<'a>(node: &'a Node, found: &mut Vec<&'a Node>) {
        match node.name.as_str() {
            "w:p" => found.push(node),
            "w:tbl" | "w:tr" | "w:tc" | "w:sdt" | "w:sdtContent" => node.children.iter().for_each(|child| collect(child, found)),
            _ => {}
        }
    }
    let mut found = Vec::new();
    body.children.iter().for_each(|child| collect(child, &mut found));
    found
}

/// The number Word prints before each numbered paragraph of the body (its level's text, without trailing periods,
/// as a REF field's \n switch gives it; `auto` where Word numbers it), or a number typed at a paragraph's start ("12."
/// then a tab). Headings and subparagraphs ("(a)") are no paragraph numbers: the paragraphs are those at the list
/// level most numbered paragraphs are at.
fn paragraph_numbers(xml: &str, paragraphs: &[&Node], styles: Option<&(String, Vec<Node>)>, numbering: Option<&(String, Vec<Node>)>)
    -> HashMap<usize, (String, bool)> {
    fn roots<'a>(part: Option<&'a (String, Vec<Node>)>, name: &str) -> Vec<(&'a str, &'a Node)> {
        part.map_or(Vec::new(), |(text, nodes)| nodes.iter().filter(|root| root.name == name)
            .flat_map(|root| root.children.iter().map(move |child| (text.as_str(), child))).collect())
    }
    let value = |part: &str, node: Option<&Node>| node.and_then(|node| attribute(part, node, "w:val"));
    let styles = roots(styles, "w:styles").into_iter().filter_map(|(text, style)| Some((attribute(text, style, "w:styleId")?, (text, style)))).collect::<HashMap<_, _>>();
    let definitions = roots(numbering, "w:numbering");
    let nums = definitions.iter().filter(|(_, node)| node.name == "w:num").filter_map(|(text, node)| Some((attribute(text, node, "w:numId")?, (*text, *node)))).collect::<HashMap<_, _>>();
    let abstracts = definitions.iter().filter(|(_, node)| node.name == "w:abstractNum").filter_map(|(text, node)| Some((attribute(text, node, "w:abstractNumId")?, (*text, *node)))).collect::<HashMap<_, _>>();
    fn numbered_by(part: &str, properties: Option<&Node>, styles: &HashMap<String, (&str, &Node)>, depth: usize) -> (Option<String>, Option<String>, bool) {
        let value = |part: &str, node: Option<&Node>| node.and_then(|node| attribute(part, node, "w:val"));
        let style = styles.get(&value(part, properties.and_then(|properties| properties.child("w:pStyle"))).unwrap_or_else(|| "Normal".to_owned()));
        let own = properties.and_then(|properties| properties.child("w:numPr"));
        let inherited = style.filter(|_| depth < 10).map(|(text, style)| numbered_by(text, style.child("w:pPr"), styles, depth + 1));
        let name = style.and_then(|(text, style)| value(text, style.child("w:name"))).unwrap_or_default().to_lowercase();
        let outline = value(part, properties.and_then(|properties| properties.child("w:outlineLvl"))).unwrap_or_else(|| "9".to_owned());
        (value(part, own.and_then(|own| own.child("w:numId"))).or_else(|| inherited.as_ref().and_then(|(id, ..)| id.clone())),
         value(part, own.and_then(|own| own.child("w:ilvl"))).or_else(|| inherited.as_ref().and_then(|(_, level, _)| level.clone())),
         name.starts_with("heading") || name.starts_with("title") || outline != "9" || inherited.is_some_and(|(.., heading)| heading))
    }
    let format = |count: i64, kind: &str| {
        let kind = kind.to_lowercase();
        if kind.contains("letter") {
            let letter = char::from(b'A' + ((count - 1).rem_euclid(26)) as u8).to_string().repeat(((count - 1) / 26 + 1) as usize);
            return if kind.starts_with("lower") { letter.to_lowercase() } else { letter };
        }
        if kind.contains("roman") {
            let (mut roman, mut rest) = (String::new(), count);
            for (size, letters) in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")] {
                while rest >= size { roman += letters; rest -= size; }
            }
            return if kind.starts_with("lower") { roman.to_lowercase() } else { roman };
        }
        count.to_string()
    };
    static TYPED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(\d{1,4})(?:\.[\t ]+|\t+)\S").expect("typed"));
    static LEVEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"%(\d)").expect("level"));
    let (mut counts, mut begun, mut found) = (HashMap::<String, Vec<Option<i64>>>::new(), HashSet::new(), Vec::new());
    for (index, paragraph) in paragraphs.iter().enumerate() {
        let (num_id, ilvl, heading) = numbered_by(xml, paragraph.child("w:pPr"), &styles, 0);
        let num = num_id.as_ref().and_then(|id| nums.get(id));
        let abstract_id = num.and_then(|(text, num)| value(text, num.child("w:abstractNumId")));
        if let (Some((num_text, num)), Some((text, abstract_node))) = (num, abstract_id.as_ref().and_then(|id| abstracts.get(id))) {
            let level = ilvl.and_then(|level| level.parse::<usize>().ok()).unwrap_or(0);
            let lvl = |at: usize| abstract_node.children.iter().find(|node| node.name == "w:lvl" && attribute(text, node, "w:ilvl").and_then(|value| value.parse().ok()) == Some(at));
            let start = |at: usize| num.children.iter().find(|node| node.name == "w:lvlOverride" && attribute(num_text, node, "w:ilvl").and_then(|value| value.parse().ok()) == Some(at))
                .and_then(|node| value(num_text, node.child("w:startOverride"))).or_else(|| value(text, lvl(at).and_then(|lvl| lvl.child("w:start"))))
                .and_then(|value| value.parse::<i64>().ok()).unwrap_or(1);
            let count = counts.entry(abstract_id.clone().unwrap()).or_default();
            // A list whose numbering restarts starts again where it is first used.
            if begun.insert(num_id.clone().unwrap()) {
                for node in num.children.iter().filter(|node| node.name == "w:lvlOverride") {
                    if let Some(at) = attribute(num_text, node, "w:ilvl").and_then(|value| value.parse::<usize>().ok()).filter(|at| *at < count.len()) { count[at] = None; }
                }
            }
            count.resize(count.len().max(level + 1), None);
            count[level] = Some(count[level].unwrap_or(start(level) - 1) + 1);
            count.truncate(level + 1);
            let kind = value(text, lvl(level).and_then(|lvl| lvl.child("w:numFmt"))).unwrap_or_else(|| "decimal".to_owned());
            if !heading && kind != "bullet" && kind != "none" {
                let template = value(text, lvl(level).and_then(|lvl| lvl.child("w:lvlText"))).unwrap_or_else(|| format!("%{}.", level + 1));
                let printed = LEVEL.replace_all(&template, |captures: &regex::Captures| {
                    let at = captures[1].parse::<usize>().unwrap_or(1).saturating_sub(1);
                    format(count.get(at).copied().flatten().unwrap_or_else(|| start(at)), &value(text, lvl(at).and_then(|lvl| lvl.child("w:numFmt"))).unwrap_or_default())
                }).trim_end_matches(|c: char| c == '.' || c.is_whitespace()).to_owned();
                found.push((index, printed, true, level));
            }
        } else if !heading {
            if let Some(typed) = TYPED.captures(&String::from_utf16_lossy(&reading(xml, std::slice::from_ref(*paragraph)).0)) {
                found.push((index, typed[1].to_owned(), false, 0));
            }
        }
    }
    // A paragraph number is a number; a list's "(a)" is part of the paragraph before it.
    found.retain(|(_, text, ..)| !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()));
    // The level most numbered paragraphs are at; of levels as many, the first met.
    let mut levels = Vec::<(usize, usize)>::new();
    for (.., level) in &found {
        match levels.iter_mut().find(|(known, _)| known == level) { Some((_, count)) => *count += 1, None => levels.push((*level, 1)) }
    }
    let level = levels.iter().fold(None::<(usize, usize)>, |best, &(level, count)| match best { Some((_, most)) if most >= count => best, _ => Some((level, count)) })
        .map(|(level, _)| level);
    found.into_iter().filter(|(.., at)| Some(*at) == level).map(|(index, text, auto, _)| (index, (text, auto))).collect()
}

/// The styles Word gives a table of authorities, for a brief that has none: each entry's pages at the margin after a
/// dotted leader.
fn table_styles(width: i64) -> [(&'static str, String); 2] {
    let common = "<w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/><w:uiPriority w:val=\"99\"/><w:semiHidden/><w:unhideWhenUsed/>";
    [("TOAHeading", format!("<w:style w:type=\"paragraph\" w:styleId=\"TOAHeading\"><w:name w:val=\"toa heading\"/>{common}\
        <w:pPr><w:spacing w:before=\"120\"/></w:pPr><w:rPr><w:rFonts w:asciiTheme=\"majorHAnsi\" w:eastAsiaTheme=\"majorEastAsia\" \
        w:hAnsiTheme=\"majorHAnsi\" w:cstheme=\"majorBidi\"/><w:b/><w:bCs/><w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/></w:rPr></w:style>")),
     ("TableofAuthorities", format!("<w:style w:type=\"paragraph\" w:styleId=\"TableofAuthorities\"><w:name w:val=\"table of authorities\"/>{common}\
        <w:pPr><w:tabs><w:tab w:val=\"right\" w:leader=\"dot\" w:pos=\"{width}\"/></w:tabs><w:ind w:left=\"220\" w:right=\"720\" w:hanging=\"220\"/></w:pPr></w:style>"))]
}

/// Settings Word writes after updateFields (CT_Settings order); extension settings come last.
const SETTINGS_AFTER_UPDATE_FIELDS: [&str; 19] = ["w:hdrShapeDefaults", "w:footnotePr", "w:endnotePr", "w:compat", "w:docVars", "w:rsids",
    "w:attachedSchema", "w:themeFontLang", "w:clrSchemeMapping", "w:doNotIncludeSubdocsInStats", "w:doNotAutoCompressPictures",
    "w:forceUpgrade", "w:captions", "w:readModeInkLockDown", "w:smartTagType", "w:shapeDefaults", "w:doNotEmbedSmartTags",
    "w:decimalSymbol", "w:listSeparator"];

/// A run of text in `format` (its run properties' children).
fn run(text: &str, format: &str) -> String {
    format!("<w:r>{}{}</w:r>", if format.is_empty() { String::new() } else { format!("<w:rPr>{format}</w:rPr>") }, text_element("w:t", text))
}

/// A label's italic head and plain rest, as runs.
fn label_runs(label: &str, italic: usize, plain: &str) -> String {
    let length = label.encode_utf16().count();
    [(utf16_slice(label, 0, italic), "<w:i/>"), (utf16_slice(label, italic, length), plain)].into_iter()
        .filter(|(text, _)| !text.is_empty()).map(|(text, format)| run(&text, format)).collect()
}

const HEADING: &str = "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p><w:p><w:pPr><w:pStyle w:val=\"Heading1\"/><w:numPr><w:ilvl w:val=\"0\"/>\
    <w:numId w:val=\"0\"/></w:numPr><w:ind w:left=\"0\" w:right=\"0\" w:firstLine=\"0\"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>Table of Authorities</w:t></w:r></w:p>";

/// Under each category's heading, a TOA field holding the table Word would show for it already as its result: each
/// entry on its own line, sorted as Word sorts them, its style of cause in italics, its pages after a dotted leader.
fn table_fields(entries: &BTreeMap<u8, HashMap<String, (usize, Vec<usize>)>>, width: i64) -> String {
    let paragraph = |style: &str, extra: &str, runs: String| format!("<w:p><w:pPr><w:pStyle w:val=\"{style}\"/>{extra}</w:pPr>{runs}</w:p>");
    let text = |value: &str, format: &str| if value == "\t" { format!("<w:r><w:rPr>{format}<w:noProof/></w:rPr><w:tab/></w:r>") }
        else { format!("<w:r><w:rPr>{format}<w:noProof/></w:rPr>{}</w:r>", text_element("w:t", value)) };
    let char = |kind: &str| format!("<w:r><w:fldChar w:fldCharType=\"{kind}\"/></w:r>");
    let indent = format!("<w:tabs><w:tab w:val=\"right\" w:leader=\"dot\" w:pos=\"{width}\"/></w:tabs><w:ind w:left=\"220\" w:right=\"720\" w:hanging=\"220\"/>");
    entries.iter().map(|(category, names)| {
        let mut names = names.iter().collect::<Vec<_>>();
        names.sort_by(|left, right| entry_order(left.0, right.0));
        let count = names.len();
        paragraph("TOAHeading", "<w:keepNext/><w:spacing w:before=\"240\" w:after=\"120\"/>", text(&category_name(*category), "<w:b/>"))
            + &names.into_iter().enumerate().map(|(index, (name, (italic, pages)))| {
                let mut pages = pages.clone();
                pages.sort_unstable();
                pages.dedup();
                let length = name.encode_utf16().count();
                let runs = [if index == 0 { format!("{}<w:r><w:instrText xml:space=\"preserve\"> TOA \\c &quot;{category}&quot; </w:instrText></w:r>{}", char("begin"), char("separate")) } else { String::new() },
                    if *italic > 0 { text(&utf16_slice(name, 0, *italic), "<w:i/>") } else { String::new() }, text(&utf16_slice(name, *italic, length), ""),
                    if pages.is_empty() { String::new() } else { text("\t", "") + &text(&pages.iter().map(usize::to_string).collect::<Vec<_>>().join(", "), "") },
                    if index + 1 == count { char("end") } else { String::new() }].concat();
                paragraph("TableofAuthorities", &indent, runs)
            }).collect::<String>()
    }).collect()
}

/// Each authority with its hyperlink, printed too for a reader on paper: the citation in black, the link in link blue.
fn linked_table(entries: &[DocxLinkedAuthority]) -> String {
    let blue = "<w:color w:val=\"0563C1\"/><w:u w:val=\"single\"/>";
    let mut table = "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p><w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr>".to_owned() + &run("TABLE OF AUTHORITIES", "") + "</w:p>";
    for (index, entry) in entries.iter().enumerate() {
        let href = entry.url.as_deref().map(str::trim).filter(|url| url.starts_with("https://") || url.starts_with("http://"));
        let citation = label_runs(&entry.label, entry.italic, "");
        let body = match href {
            Some(href) => {
                let hyperlink = |runs: String| format!("<w:fldSimple w:instr=\" HYPERLINK &quot;{}&quot; \">{runs}</w:fldSimple>", escaped(href));
                hyperlink(citation) + "<w:r><w:br/></w:r>" + &hyperlink(run(href, blue))
            }
            None => citation,
        };
        table += &format!("<w:p>{}{body}</w:p>", run(&format!("{}. ", index + 1), ""));
    }
    table
}

/// The authorities as a ruled table, as court filings set them out: the tab each is behind, the authority and where the
/// brief cites it, under a header row Word repeats on each page, each group under a row of its own.
fn ruled_table(entries: &[(&DocxLinkedAuthority, Vec<String>)], width: i64, by_paragraph: bool) -> String {
    let tabs = entries.iter().any(|(entry, _)| entry.tab.is_some());
    let columns = [if tabs { vec![864] } else { vec![] }, vec![width - if tabs { 864 } else { 0 } - 1872, 1872]].concat();
    let paragraph = |runs: &str, align: &str, keep: bool| format!("<w:p><w:pPr>{}<w:spacing w:before=\"40\" w:after=\"40\" w:line=\"240\" w:lineRule=\"auto\"/>\
        <w:ind w:left=\"0\" w:right=\"0\" w:firstLine=\"0\"/><w:jc w:val=\"{align}\"/></w:pPr>{runs}</w:p>", if keep { "<w:keepNext/>" } else { "" });
    let cell = |content: String, width: i64, extra: &str| format!("<w:tc><w:tcPr><w:tcW w:w=\"{width}\" w:type=\"dxa\"/>{extra}</w:tcPr>{content}</w:tc>");
    let row = |cells: String, properties: &str| format!("<w:tr><w:trPr><w:cantSplit/>{properties}</w:trPr>{cells}</w:tr>");
    let shade = "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"D9D9D9\"/>";
    let titles = [if tabs { vec!["Tab"] } else { vec![] }, vec!["Authority", if by_paragraph { "Cited at paragraph(s)" } else { "Cited at page(s)" }]].concat();
    let header = row(titles.iter().enumerate().map(|(index, title)| cell(paragraph(&run(title, "<w:b/>"), if tabs && index == 0 { "center" } else { "left" }, false), columns[index], shade)).collect(), "<w:tblHeader/>");
    let groups = entries.iter().map(|(entry, _)| &entry.group).collect::<HashSet<_>>().len() > 1;
    let mut rows = String::new();
    for (index, (entry, cited)) in entries.iter().enumerate() {
        // A group's heading row stays with its first authority.
        if groups && (index == 0 || entries[index - 1].0.group != entry.group) {
            rows += &row(cell(paragraph(&run(entry.group.as_deref().unwrap_or_default(), "<w:b/>"), "left", true), width,
                &format!("<w:gridSpan w:val=\"{}\"/>", columns.len())), "");
        }
        let tab = if tabs { cell(paragraph(&entry.tab.as_deref().map_or(String::new(), |tab| run(tab, "")), "center", false), columns[0], "") } else { String::new() };
        rows += &row(tab + &cell(paragraph(&label_runs(&entry.label, entry.italic, ""), "left", false), columns[columns.len() - 2], "")
            + &cell(paragraph(&cited.join(&run(", ", "")), "left", false), columns[columns.len() - 1], ""), "");
    }
    let borders = ["top", "left", "bottom", "right", "insideH", "insideV"].map(|side| format!("<w:{side} w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>")).concat();
    format!("<w:tbl><w:tblPr><w:tblW w:w=\"{width}\" w:type=\"dxa\"/><w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type=\"fixed\"/>\
        <w:tblCellMar><w:top w:w=\"29\" w:type=\"dxa\"/><w:left w:w=\"108\" w:type=\"dxa\"/><w:bottom w:w=\"29\" w:type=\"dxa\"/><w:right w:w=\"108\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr>\
        <w:tblGrid>{}</w:tblGrid>{header}{rows}</w:tbl>", columns.iter().map(|column| format!("<w:gridCol w:w=\"{column}\"/>")).collect::<String>())
}

/// Writes the table of authorities `delivery` asks for into the brief, at the citations `marks` names in `units` (as the
/// reading read them): TA marks Word's table reads (a citation the author marked already keeps that mark), its tab after
/// each citation and its pinpoint linked where asked; and the table appended where asked. Word is asked to update the
/// table's fields on opening, as their page numbers are its own layout.
pub fn apply_table_of_authorities(bytes: &[u8], units: &[DocxUnit], marks: &[DocxAuthorityMark], delivery: DocxTableDelivery,
    linked: &[DocxLinkedAuthority]) -> Result<Vec<u8>> {
    let (mut files, mut parts) = open(bytes)?;
    let trees = [parse(&parts[0])?, parse(&parts[1]).unwrap_or_default()];
    let (body, found) = units_in(&parts, &trees, units)?;
    let targets = units.iter().zip(found).map(|(unit, (part, node))| (unit.id.as_str(), (part, node))).collect::<HashMap<_, _>>();
    let document = parts[0].as_str();
    let page_at = rendered_pages(document, body, &targets.iter().filter(|(_, (part, _))| *part == 0).map(|(id, (_, node))| (*node as *const Node, *id)).collect());
    let existing = targets.iter().map(|(id, (part, node))| (*id, existing_marks(&parts[*part], node))).collect::<HashMap<_, _>>();
    // Each entry is marked in full once, at its first citation, and by its short name after, as Word's Mark Citation does;
    // a short name two entries share would join them, so those keep their full name as their short name too.
    let native = matches!(delivery, DocxTableDelivery::NativeMarks | DocxTableDelivery::NativeAppend);
    let placed = (0..marks.len()).filter(|&at| native && marks[at].mark != Some(false)
        && !existing.get(marks[at].unit_id.as_str()).is_some_and(|found| found.iter().any(|(offset, ..)| *offset == marks[at].offset))).collect::<Vec<_>>();
    let mut owners = HashMap::<String, HashSet<String>>::new();
    for &at in &placed {
        owners.entry(entry_text(&marks[at].short_name).to_lowercase()).or_default().insert(entry_text(&marks[at].long_name));
    }
    let (mut firsts, mut seen) = (HashSet::new(), HashSet::new());
    for &at in &placed {
        if seen.insert(entry_text(&marks[at].long_name)) { firsts.insert(at); }
    }
    let mut edits = Edits(Vec::new());
    // A ruled table says where each authority is cited: at the brief's numbered paragraphs where it numbers the
    // paragraphs that cite them, else at its pages; each place a field Word updates (REF, PAGEREF) to a bookmark.
    let ruled = if delivery == DocxTableDelivery::RuledAppend { (0..marks.len()).filter(|&at| marks[at].authority_id.is_some()).collect::<Vec<_>>() } else { Vec::new() };
    let paragraphs = paragraphs(body);
    let parsed = |files: &[(String, Vec<u8>)], name: &str| part(files, name).and_then(|text| parse(&text).ok().map(|nodes| (text, nodes)));
    let numbers = if ruled.is_empty() { HashMap::new() } else { paragraph_numbers(document, &paragraphs, parsed(&files, "word/styles.xml").as_ref(), parsed(&files, "word/numbering.xml").as_ref()) };
    let mut note_at = HashMap::new();
    let mut numbered_at = vec![None; paragraphs.len()];
    for (index, paragraph) in paragraphs.iter().enumerate() {
        walk(paragraph, &mut |node| { if node.name == "w:footnoteReference" { note_at.insert(format!("footnote:{}", attribute(document, node, "w:id").unwrap_or_default()), index); } true });
        numbered_at[index] = if numbers.contains_key(&index) { Some(index) } else if index > 0 { numbered_at[index - 1] } else { None };
    }
    // A citation after the last numbered paragraph (a schedule, the brief's own table) is at none.
    for index in (0..paragraphs.len()).rev() {
        if numbers.contains_key(&index) { break; }
        numbered_at[index] = None;
    }
    let paragraph_of = |mark: &DocxAuthorityMark| if mark.unit_id.starts_with("body:") {
        targets.get(mark.unit_id.as_str()).and_then(|(_, node)| paragraphs.iter().position(|paragraph| std::ptr::eq(*paragraph, *node)))
    } else { note_at.get(&mark.unit_id).copied() };
    let numbered = |mark: &DocxAuthorityMark| paragraph_of(mark).and_then(|at| numbered_at[at]);
    let by_paragraph = !ruled.is_empty() && ruled.iter().filter(|&&at| numbered(&marks[at]).is_some()).count() * 2 >= ruled.len();
    let mut bookmark_ids = 0;
    let mut bookmark_names = HashSet::new();
    for (xml, root) in std::iter::once((document, body)).chain(targets.values().map(|(part, node)| (parts[*part].as_str(), *node))) {
        walk(root, &mut |node| {
            if node.name == "w:bookmarkStart" {
                bookmark_ids = bookmark_ids.max(attribute(xml, node, "w:id").and_then(|id| id.parse().ok()).unwrap_or(0));
                bookmark_names.insert(attribute(xml, node, "w:name").unwrap_or_default());
            }
            true
        });
    }
    let mut bookmark = || {
        loop {
            bookmark_ids += 1;
            let name = format!("_Toa{bookmark_ids}");
            if !bookmark_names.contains(&name) {
                return (name.clone(), format!("<w:bookmarkStart w:id=\"{bookmark_ids}\" w:name=\"{name}\"/>"), format!("<w:bookmarkEnd w:id=\"{bookmark_ids}\"/>"));
            }
        }
    };
    let (mut cited_at, mut page_marks, mut paragraph_marks) = (HashMap::<&str, Vec<String>>::new(), HashMap::new(), BTreeMap::new());
    let authorities = ruled.iter().map(|&at| marks[at].authority_id.as_deref().unwrap())
        .fold(Vec::new(), |mut kept, id| { if !kept.contains(&id) { kept.push(id); } kept });
    for authority in authorities {
        let cited = ruled.iter().copied().filter(|&at| marks[at].authority_id.as_deref() == Some(authority)).collect::<Vec<_>>();
        let places = if by_paragraph {
            let mut at = cited.iter().filter_map(|&mark| numbered(&marks[mark])).collect::<Vec<_>>();
            at.sort_unstable();
            at.dedup();
            at.into_iter().map(|paragraph| {
                let (text, auto) = &numbers[&paragraph];
                if !auto { return format!("<w:r>{}</w:r>", text_element("w:t", text)); }
                let (name, ..) = paragraph_marks.entry(paragraph).or_insert_with(&mut bookmark).clone();
                field(&format!("REF {name} \\n \\h"), text)
            }).collect()
        } else {
            let mut pages = BTreeMap::new();
            for &mark in &cited {
                pages.entry(page_at(&marks[mark].unit_id, marks[mark].offset)).or_insert(mark);
            }
            pages.into_iter().map(|(page, mark)| {
                let target = bookmark();
                let name = target.0.clone();
                page_marks.insert(mark, target);
                field(&format!("PAGEREF {name} \\h"), &page.to_string())
            }).collect()
        };
        cited_at.insert(authority, places);
    }
    for (paragraph, (_, start, end)) in &paragraph_marks {
        let node = paragraphs[*paragraph];
        edits.insert(0, node.child("w:pPr").map_or(node.open_end, |pr| pr.end), start.clone());
        edits.insert(0, node.inner_end, end.clone());
    }
    let mut order = (0..marks.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| marks[b].unit_id.cmp(&marks[a].unit_id).then(marks[b].offset.cmp(&marks[a].offset)));
    for at in order {
        let mark = &marks[at];
        let &(part, unit) = targets.get(mark.unit_id.as_str())
            .ok_or_else(|| crate::Error::Message(format!("Reviewed Word location is missing: {}.", mark.unit_id)))?;
        let xml = parts[part].as_str();
        if let Some((_, start, end)) = page_marks.get(&at) {
            edits.at_offset(part, xml, unit, mark.offset, &format!("{start}{end}"))?;
        }
        if let Some(suffix) = &mark.suffix {
            let text = format!("<w:r>{}</w:r>", text_element("w:t", suffix).replacen("<w:t>", "<w:t xml:space=\"preserve\">", 1));
            let nodes = match &mark.tab_url { Some(url) => format!("<w:fldSimple w:instr=\" HYPERLINK &quot;{}&quot; \">{text}</w:fldSimple>", escaped(url)), None => text };
            edits.at_offset(part, xml, unit, mark.offset, &nodes)?;
        }
        if placed.contains(&at) {
            let long = entry_text(&mark.long_name);
            let italic = mark.italic.unwrap_or(0);
            let (lead, short) = (field_text(&utf16_slice(&long, 0, italic)), field_text(&entry_text(if owners[&entry_text(&mark.short_name).to_lowercase()].len() > 1 { &mark.long_name } else { &mark.short_name })));
            let rest = format!("{}\" \\s \"{short}\" \\c {} ", field_text(&utf16_slice(&long, italic, long.encode_utf16().count())), mark.category);
            let runs = if !firsts.contains(&at) { field_runs(&[(&format!(" TA \\s \"{short}\" "), false)]) }
                else if !lead.is_empty() { field_runs(&[(" TA \\l \"", false), (&lead, true), (&rest, false)]) }
                else { field_runs(&[(&format!(" TA \\l \"{rest}"), false)]) };
            edits.at_offset(part, xml, unit, mark.offset, &runs)?;
        }
        // A pinpoint the author already linked keeps that link.
        if let Some(link) = mark.pinpoint_link.as_ref().filter(|link| !linked_ranges(xml, unit).iter().any(|(from, to)| link.start < *to && *from < link.end)) {
            edits.at_offset(part, xml, unit, link.end, "<w:r><w:fldChar w:fldCharType=\"end\"/></w:r>")?;
            edits.at_offset(part, xml, unit, link.start, &format!("<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r><w:r><w:instrText xml:space=\"preserve\"> HYPERLINK &quot;{}&quot; </w:instrText></w:r><w:r><w:fldChar w:fldCharType=\"separate\"/></w:r>", escaped(&link.url)))?;
        }
    }
    // Word's table lists one category per field, under its heading; the brief's own marks are listed with ours, each
    // entry with the pages its citations are on.
    let section = body.children.iter().find(|node| node.name == "w:sectPr");
    let width = text_width(document, section);
    let mut entries = BTreeMap::<u8, HashMap<String, (usize, Vec<usize>)>>::new();
    let category_of = firsts.iter().map(|&at| (entry_text(&marks[at].long_name), marks[at].category)).collect::<HashMap<_, _>>();
    for &at in &placed {
        let long = entry_text(&marks[at].long_name);
        let entry = entries.entry(category_of[&long]).or_default().entry(long).or_insert((marks[at].italic.unwrap_or(0), Vec::new()));
        entry.1.push(page_at(&marks[at].unit_id, marks[at].offset));
    }
    for (unit, found) in &existing {
        for (offset, category, long) in found {
            if let Some(long) = long {
                entries.entry(*category).or_default().entry(entry_text(long)).or_insert((0, Vec::new())).1.push(page_at(unit, *offset));
            }
        }
    }
    let appended = match delivery {
        DocxTableDelivery::NativeAppend => format!("{HEADING}{}", table_fields(&entries, width)),
        DocxTableDelivery::LinkedAppend => linked_table(linked),
        // Word ends the body with a paragraph, never a table.
        DocxTableDelivery::RuledAppend => format!("{HEADING}{}<w:p/>", ruled_table(&linked.iter().map(|entry| (entry,
            entry.authority_id.as_deref().and_then(|id| cited_at.get(id)).cloned().unwrap_or_default())).collect::<Vec<_>>(), width, by_paragraph)),
        DocxTableDelivery::NativeMarks => String::new(),
    };
    edits.insert(0, section.map_or(body.inner_end, |section| section.start), appended);
    edits.apply(&mut parts);
    for (name, text) in [("word/document.xml", &parts[0]), ("word/footnotes.xml", &parts[1])] {
        if !text.is_empty() { crate::docx::replace_file(&mut files, name, text.clone().into_bytes()); }
    }
    if delivery == DocxTableDelivery::NativeAppend {
        if let Some((mut styles, nodes)) = parsed(&files, "word/styles.xml") {
            if let Some(root) = nodes.iter().find(|node| node.name == "w:styles") {
                let present = root.children.iter().filter_map(|style| attribute(&styles, style, "w:styleId")).collect::<HashSet<_>>();
                let added = table_styles(width).into_iter().filter(|(id, _)| !present.contains(*id)).map(|(_, style)| style).collect::<String>();
                styles.insert_str(root.inner_end, &added);
                crate::docx::replace_file(&mut files, "word/styles.xml", styles.into_bytes());
            }
        }
    }
    if delivery == DocxTableDelivery::NativeAppend || !ruled.is_empty() {
        if let Some((mut settings, nodes)) = parsed(&files, "word/settings.xml") {
            if let Some(root) = nodes.iter().find(|node| node.name == "w:settings") {
                let update = "<w:updateFields w:val=\"true\"/>";
                match root.child("w:updateFields") {
                    Some(existing) => settings.replace_range(existing.start..existing.end, update),
                    None => {
                        let at = root.children.iter().find(|node| SETTINGS_AFTER_UPDATE_FIELDS.contains(&node.name.as_str()) || !node.name.starts_with("w:"))
                            .map_or(root.inner_end, |node| node.start);
                        settings.insert_str(at, update);
                    }
                }
                crate::docx::replace_file(&mut files, "word/settings.xml", settings.into_bytes());
            }
        }
    }
    crate::docx::write_docx_files(&files)
}
