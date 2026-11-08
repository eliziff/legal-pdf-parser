use crate::{Error, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::io::{Cursor, Read, Write};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const MAX_DOCX_SUPRA_BYTES: usize = 25 * 1024 * 1024;
const W_NS: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
const MC_NS: &str = "http://schemas.openxmlformats.org/markup-compatibility/2006";

fn file_bytes<'a>(files: &'a [(String, Vec<u8>)], name: &str) -> Option<&'a [u8]> {
    files.iter().find(|(entry, _)| entry == name).map(|(_, bytes)| bytes.as_slice())
}

fn replace_file(files: &mut Vec<(String, Vec<u8>)>, name: &str, bytes: Vec<u8>) {
    match files.iter_mut().find(|(entry, _)| entry == name) {
        Some((_, value)) => *value = bytes,
        None => files.push((name.to_owned(), bytes)),
    }
}

/// What failed, with the error that failed it.
fn failed(what: &str, error: impl std::fmt::Display) -> Error {
    Error::Message(format!("{what}: {error}"))
}

/// A regular expression written out in this file.
fn literal(pattern: &str) -> Regex {
    Regex::new(pattern).expect("literal DOCX regex")
}

#[derive(Clone)]
struct XmlAttribute {
    namespace: Option<String>,
    local: String,
    value: String,
}

#[derive(Clone)]
struct XmlElement {
    namespace: Option<String>,
    local: String,
    attributes: Vec<XmlAttribute>,
    children: Vec<XmlNode>,
    self_closing: bool,
}

#[derive(Clone)]
enum XmlNode {
    Element(XmlElement),
    Raw(Event<'static>),
}

struct XmlDocument {
    nodes: Vec<XmlNode>,
}

fn namespace_value(value: ResolveResult<'_>) -> Option<String> {
    match value {
        ResolveResult::Bound(namespace) => Some(String::from_utf8_lossy(namespace.as_ref()).into_owned()),
        ResolveResult::Unbound | ResolveResult::Unknown(_) => None,
    }
}

fn decode_attribute(raw: &quick_xml::events::attributes::Attribute<'_>, decoder: quick_xml::encoding::Decoder) -> Result<String> {
    #[allow(deprecated)]
    raw.decode_and_unescape_value(decoder).map(|value| value.into_owned()).map_err(|error| failed("XML attribute decoding failed", error))
}

fn parse_element(reader: &NsReader<&[u8]>, start: &BytesStart<'_>, namespace: Option<String>, self_closing: bool) -> Result<XmlElement> {
    let name = |bytes: &[u8], what: &str| std::str::from_utf8(bytes).map(str::to_owned).map_err(|error| failed(what, error));
    let local = name(start.local_name().as_ref(), "XML name is not UTF-8")?;
    let mut attributes = Vec::new();
    for raw in start.attributes().with_checks(false) {
        let raw = raw.map_err(|error| failed("XML attribute failed", error))?;
        let (resolved, local_name) = reader.resolver().resolve_attribute(raw.key);
        let local = name(local_name.as_ref(), "XML attribute name is not UTF-8")?;
        attributes.push(XmlAttribute { namespace: namespace_value(resolved), local, value: decode_attribute(&raw, reader.decoder())? });
    }
    Ok(XmlElement { namespace, local, attributes, children: Vec::new(), self_closing })
}

fn parse_xml(raw: &[u8]) -> Result<XmlDocument> {
    let mut reader = NsReader::from_reader(raw);
    reader.config_mut().trim_text(false);
    let (mut buffer, mut roots, mut stack) = (Vec::new(), Vec::new(), Vec::<XmlElement>::new());
    loop {
        let node = match reader.read_event_into(&mut buffer)? {
            Event::Start(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                stack.push(parse_element(&reader, &start, namespace, false)?);
                None
            }
            Event::Empty(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                Some(XmlNode::Element(parse_element(&reader, &start, namespace, true)?))
            }
            Event::End(_) => Some(XmlNode::Element(stack.pop().ok_or_else(|| Error::Message("XML has an unmatched end tag".to_owned()))?)),
            Event::Eof => break,
            Event::Decl(_) => None,
            other => Some(XmlNode::Raw(other.into_owned())),
        };
        if let Some(node) = node {
            match stack.last_mut() {
                Some(parent) => parent.children.push(node),
                None => roots.push(node),
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() {
        return Err(Error::Message("XML has an unclosed element".to_owned()));
    }
    Ok(XmlDocument { nodes: roots })
}

/// The elements among the nodes.
fn elements(nodes: &[XmlNode]) -> impl Iterator<Item = &XmlElement> {
    nodes.iter().filter_map(|node| match node {
        XmlNode::Element(element) => Some(element),
        XmlNode::Raw(_) => None,
    })
}

impl XmlDocument {
    fn root(&self) -> Result<&XmlElement> {
        elements(&self.nodes).next().ok_or_else(|| Error::Message("XML has no root element".to_owned()))
    }
}

impl XmlElement {
    fn is(&self, namespace: &str, local: &str) -> bool {
        self.namespace.as_deref() == Some(namespace) && self.local == local
    }

    fn attribute(&self, namespace: Option<&str>, local: &str) -> Option<&str> {
        let named = |attribute: &&XmlAttribute| attribute.local == local && namespace.is_none_or(|namespace| attribute.namespace.as_deref() == Some(namespace));
        self.attributes.iter().find(named).map(|attribute| attribute.value.as_str())
    }

    fn direct_elements(&self) -> impl Iterator<Item = &XmlElement> {
        elements(&self.children)
    }

    /// The first Word element of the name among its children.
    fn child(&self, local: &str) -> Option<&XmlElement> {
        self.direct_elements().find(|child| child.is(W_NS, local))
    }
}

fn raw_text(event: &Event<'_>) -> Result<Option<String>> {
    match event {
        Event::Text(text) => {
            let decoded = text.decode().map_err(|error| failed("XML text decoding failed", error))?;
            quick_xml::escape::unescape(&decoded).map(|value| Some(value.into_owned())).map_err(|error| failed("XML text unescaping failed", error))
        }
        Event::CData(text) => text.decode().map(|value| Some(value.into_owned())).map_err(|error| failed("XML text decoding failed", error)),
        Event::GeneralRef(reference) => {
            if let Some(value) = reference.resolve_char_ref().map_err(|error| failed("XML reference failed", error))? {
                return Ok(Some(value.to_string()));
            }
            let name = reference.decode().map_err(|error| failed("XML reference failed", error))?;
            let entity = match name.as_ref() {
                "lt" => "<",
                "gt" => ">",
                "amp" => "&",
                "apos" => "'",
                "quot" => "\"",
                _ => return Ok(Some(format!("&{name};"))),
            };
            Ok(Some(entity.to_owned()))
        }
        _ => Ok(None),
    }
}

fn element_text(element: &XmlElement) -> Result<String> {
    let mut output = String::new();
    for child in &element.children {
        match child {
            XmlNode::Raw(event) => output.extend(raw_text(event)?),
            XmlNode::Element(child) => output.push_str(&element_text(child)?),
        }
    }
    Ok(output)
}

fn walk_elements<'a>(element: &'a XmlElement, output: &mut Vec<&'a XmlElement>) {
    output.push(element);
    for child in element.direct_elements() {
        walk_elements(child, output);
    }
}

fn docx_paragraph_text(element: &XmlElement, output: &mut String) -> Result<()> {
    if element.is(W_NS, "t") {
        output.push_str(&element_text(element)?);
    } else if !element.is(W_NS, "del") {
        for child in element.direct_elements() {
            docx_paragraph_text(child, output)?;
        }
    }
    Ok(())
}

fn normalize_docx_text(text: &str) -> String {
    let text = text
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{00a0}', '\u{2007}', '\u{202f}'], " ");
    legal_structure_model::normalize_javascript_whitespace(&text)
}

fn normalized_docx_paragraph(paragraph: &XmlElement) -> Result<String> {
    let mut text = String::new();
    docx_paragraph_text(paragraph, &mut text)?;
    Ok(normalize_docx_text(&text))
}

fn tolerant_docx_paragraphs(xml: &str) -> Result<Vec<String>> {
    static PARAGRAPHS: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:p\b[^>]*>.*?</w:p>"));
    static DELETIONS: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:del\b[^>]*>.*?</w:del>"));
    PARAGRAPHS
        .find_iter(xml)
        .map(|paragraph| {
            let accepted = DELETIONS.replace_all(paragraph.as_str(), "");
            let mut texts = TEXT.captures_iter(&accepted).map(|captures| quick_xml::escape::unescape(&captures[1]).map(|text| text.into_owned()));
            let value = texts.try_fold(String::new(), |value, text| text.map(|text| value + &text)).map_err(|error| Error::Message(error.to_string()))?;
            Ok(normalize_docx_text(&value))
        })
        .collect()
}

static SUPRA: LazyLock<Regex> = LazyLock::new(|| {
    let space = legal_structure_model::JS_WHITESPACE_CLASS;
    Regex::new(&[r"(?i)supra,?", space, r"{1,4}(?:note|nn?\.?)", space, r"{1,4}([0-9]+)"].concat()).expect("literal DOCX supra regex")
});
static NUMBERING_RESTART: LazyLock<Regex> = LazyLock::new(|| literal(r"(?i)<w:numRestart\b"));
static PARAGRAPH: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:p\b.*?</w:p>"));
static RUN: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:r\b([^>]*)>(.*?)</w:r>"));
static TEXT: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:t\b[^>]*>(.*?)</w:t>"));
static FIELD_MARKER: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:fldChar\b[^>]*\bw:fldCharType=(?:"(begin|end)"|'(begin|end)')[^>]*/?>"#));
static NOTEREF: LazyLock<Regex> = LazyLock::new(|| literal(r"(?i)NOTEREF"));
static RUN_PROPERTIES: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:rPr\b.*?</w:rPr>"));
static FOOTNOTE_REFERENCE: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:footnoteReference\b[^>]*\bw:id=(?:"(-?[0-9]+)"|'(-?[0-9]+)')[^>]*/?>"#));
static CUSTOM_MARK: LazyLock<Regex> = LazyLock::new(|| literal(r"(?-u:\b)w:customMarkFollows="));
static BOOKMARK_ID: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:bookmark(?:Start|End)\b[^>]*\bw:id=(?:"([0-9]+)"|'([0-9]+)')"#));
static BOOKMARK_NAME: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:bookmarkStart\b[^>]*\bw:name=(?:"([^"]*)"|'([^']*)')"#));

#[derive(Debug)]
pub struct DocxSupraCleanup {
    pub bytes: Vec<u8>,
    pub detected: usize,
    pub converted: usize,
    pub already_linked: usize,
    pub review_required: usize,
    pub bookmarks_added: usize,
    pub restarted_numbering: bool,
    pub unsafe_or_split_fields: usize,
}

struct SupraAnalysis {
    detected: usize,
    already_linked: usize,
    ordinals: BTreeSet<usize>,
}

#[derive(Debug)]
struct ParagraphTextNode {
    text: String,
    visible_start: usize,
    visible_end: usize,
    xml_start: usize,
    run_start: usize,
    run_end: usize,
    run_attributes: String,
    run_properties: String,
    safe_to_replace: bool,
}

fn xml_text(value: &str) -> Result<String> {
    quick_xml::escape::unescape(value).map(|value| value.into_owned()).map_err(|error| failed("XML text unescaping failed", error))
}

fn escape_xml_text(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn element_is_open(xml: &str, offset: usize, tag: &str) -> bool {
    let prior = &xml[..offset];
    let open = prior.rfind(&format!("<{tag} ")).max(prior.rfind(&format!("<{tag}>")));
    open.is_some_and(|open| prior.rfind(&format!("</{tag}>")).is_none_or(|close| open > close))
}

fn javascript_iu_word(character: Option<char>) -> bool {
    character.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_' || matches!(character, '\u{017f}' | '\u{212a}'))
}

fn javascript_iu_word_bounded(text: &str, start: usize, end: usize) -> bool {
    let boundary = |at: usize| javascript_iu_word(text[..at].chars().next_back()) != javascript_iu_word(text[at..].chars().next());
    boundary(start) && boundary(end)
}

fn field_spans(paragraph: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut spans = Vec::new();
    for marker in FIELD_MARKER.captures_iter(paragraph) {
        let whole = marker.get(0).unwrap();
        if marker.get(1).or_else(|| marker.get(2)).unwrap().as_str() == "begin" {
            stack.push(whole.start());
        } else if let Some(start) = stack.pop() {
            let field = &paragraph[start..whole.start()];
            if NOTEREF
                .find_iter(field)
                .any(|value| javascript_iu_word_bounded(field, value.start(), value.end()))
            {
                spans.push((start, whole.end()));
            }
        }
    }
    spans
}

fn paragraph_text_nodes(xml: &str, paragraph: &str, paragraph_offset: usize) -> Result<Vec<ParagraphTextNode>> {
    let (mut nodes, mut visible) = (Vec::new(), 0);
    for run in RUN.captures_iter(paragraph) {
        let (whole, body) = (run.get(0).unwrap(), run.get(2).unwrap());
        let texts = TEXT.captures_iter(body.as_str()).collect::<Vec<_>>();
        let properties = RUN_PROPERTIES.find(body.as_str()).map_or("", |value| value.as_str());
        let rest = || body.as_str().replacen(properties, "", 1).replacen(texts[0].get(0).unwrap().as_str(), "", 1);
        let only_text = texts.len() == 1 && legal_structure_model::normalize_javascript_whitespace(&rest()).is_empty();
        let safe = only_text && ["w:hyperlink", "w:fldSimple", "w:ins", "w:del"].iter().all(|tag| !element_is_open(xml, paragraph_offset + whole.start(), tag));
        for text in texts {
            let value = xml_text(text.get(1).unwrap().as_str())?;
            let node = ParagraphTextNode {
                visible_start: visible,
                visible_end: visible + value.len(),
                xml_start: body.start() + text.get(0).unwrap().start(),
                run_start: whole.start(),
                run_end: whole.end(),
                run_attributes: run.get(1).unwrap().as_str().to_owned(),
                run_properties: properties.to_owned(),
                text: value,
                safe_to_replace: safe,
            };
            visible = node.visible_end;
            nodes.push(node);
        }
    }
    Ok(nodes)
}

fn analyze_supras(xml: &str) -> Result<SupraAnalysis> {
    let mut analysis = SupraAnalysis { detected: 0, already_linked: 0, ordinals: BTreeSet::new() };
    for paragraph in PARAGRAPH.find_iter(xml) {
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes.iter().map(|node| node.text.as_str()).collect::<String>();
        let fields = field_spans(paragraph.as_str());
        for matched in SUPRA.captures_iter(&visible) {
            let whole = matched.get(0).unwrap();
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            analysis.detected += 1;
            let number = matched.get(1).unwrap();
            let node = nodes.iter().find(|node| node.visible_start <= number.start() && node.visible_end >= number.end());
            if node.is_some_and(|node| fields.iter().any(|&(start, end)| start <= node.xml_start && node.xml_start < end)) {
                analysis.already_linked += 1;
            } else if let Some(ordinal) = number.as_str().parse::<usize>().ok().filter(|ordinal| *ordinal > 0) {
                analysis.ordinals.insert(ordinal);
            }
        }
    }
    Ok(analysis)
}

fn read_docx_files(bytes: &[u8], wanted: Option<&[&str]>) -> Result<Vec<(String, Vec<u8>)>> {
    const MAX_EXPANDED_BYTES: u64 = 96 * 1024 * 1024;
    const MAX_XML_PART_BYTES: u64 = 16 * 1024 * 1024;
    const MAX_XML_TOTAL_BYTES: u64 = 32 * 1024 * 1024;
    let refused = |message: &str| Error::Message(message.to_owned());
    if bytes.is_empty() || bytes.len() > MAX_DOCX_SUPRA_BYTES {
        return Err(refused("DOCX is empty or exceeds the read limit"));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    let mut files = Vec::with_capacity(wanted.map_or(archive.len(), |names| names.len()));
    let mut seen = HashSet::with_capacity(archive.len());
    let (mut expanded, mut xml_expanded, mut declared_expanded, mut declared_xml_expanded, mut file_count) = (0_u64, 0_u64, 0_u64, 0_u64, 0);
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().replace('\\', "/");
        if name.contains('\0') || name.starts_with('/') || name.as_bytes().get(1) == Some(&b':') || name.split('/').any(|component| component == "..") {
            return Err(refused("DOCX contains an unsafe package path"));
        }
        let lower_name = name.to_ascii_lowercase();
        let xml = lower_name.ends_with(".xml") || lower_name.ends_with(".xml.rels");
        if !entry.is_dir() {
            file_count += 1;
            if file_count > 2_048 {
                return Err(refused("DOCX has too many package entries"));
            }
            declared_expanded = declared_expanded
                .checked_add(entry.size())
                .filter(|&size| size <= MAX_EXPANDED_BYTES)
                .ok_or_else(|| refused("DOCX exceeds the expanded read limit"))?;
            if xml && entry.size() > MAX_XML_PART_BYTES {
                return Err(refused("DOCX XML part exceeds the read limit"));
            }
            if xml {
                declared_xml_expanded = declared_xml_expanded
                    .checked_add(entry.size())
                    .filter(|&size| size <= MAX_XML_TOTAL_BYTES)
                    .ok_or_else(|| refused("DOCX XML parts exceed the read limit"))?;
            }
        }
        if !seen.insert(name.clone()) {
            return Err(Error::Message(format!("DOCX contains duplicate package part {name}")));
        }
        if wanted.is_some_and(|wanted| !wanted.contains(&name.as_str())) {
            continue;
        }
        let remaining = MAX_EXPANDED_BYTES.saturating_sub(expanded);
        let limit = if xml { remaining.min(MAX_XML_PART_BYTES).min(MAX_XML_TOTAL_BYTES.saturating_sub(xml_expanded)) } else { remaining };
        let mut value = Vec::with_capacity(entry.size().min(limit).min(1024 * 1024) as usize);
        entry.by_ref().take(limit + 1).read_to_end(&mut value).map_err(|error| Error::io(&name, error))?;
        if value.len() as u64 > limit {
            return Err(refused(if xml { "DOCX XML part exceeds the read limit" } else { "DOCX exceeds the expanded read limit" }));
        }
        expanded += value.len() as u64;
        xml_expanded += if xml { value.len() as u64 } else { 0 };
        files.push((name, value));
    }
    Ok(files)
}

fn write_docx_files(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, value) in files {
        if name.ends_with('/') {
            archive.add_directory(name, options)?;
        } else {
            archive.start_file(name, options)?;
            archive.write_all(value).map_err(|error| Error::io("DOCX output", error))?;
        }
    }
    Ok(archive.finish()?.into_inner())
}

fn docx_part(files: &[(String, Vec<u8>)], name: &str) -> Option<String> {
    file_bytes(files, name).map(|bytes| String::from_utf8_lossy(bytes).into_owned())
}

/// The body's paragraphs in reading order, each with the indices of the elements that lead to it
/// from the body: a text box's paragraphs (a cover page's) after the paragraph that holds the box,
/// read once, as Word shows it, not again from its fallback for older readers. Citation review and
/// the Word operations that mark it number the body's paragraphs alike by this order.
fn body_paragraphs<'a>(
    element: &'a XmlElement,
    path: &mut Vec<usize>,
    output: &mut Vec<(&'a XmlElement, Vec<usize>)>,
) {
    for (index, child) in element.direct_elements().enumerate() {
        if child.is(W_NS, "del") || child.is(MC_NS, "Fallback") {
            continue;
        }
        path.push(index);
        if child.is(W_NS, "p") {
            output.push((child, path.clone()));
        }
        body_paragraphs(child, path, output);
        path.pop();
    }
}

/// A paragraph's own text: a text box it holds is read as paragraphs of its own.
/// The author's links in a note, as the Python app reads them: a `w:hyperlink`'s relationship target, or a simple
/// `HYPERLINK` field's, over the text it holds (UTF-16 offsets).
#[derive(Default)]
struct AuthorLinks {
    targets: HashMap<String, String>,
    target: Option<String>,
    spans: Vec<(usize, usize, String)>,
}

fn authority_text(element: &XmlElement, text: &mut String, references: &mut Vec<(i64, usize, Option<String>)>, links: &mut AuthorLinks) -> Result<()> {
    if element.is(W_NS, "del") || element.is(W_NS, "txbxContent") || element.is(MC_NS, "Fallback") {
        return Ok(());
    }
    if element.is(W_NS, "t") {
        let start = legal_structure_model::utf16_len(text);
        text.push_str(&element_text(element)?);
        if let Some(target) = &links.target {
            let end = legal_structure_model::utf16_len(text);
            match links.spans.last_mut() {
                Some(span) if span.1 == start && &span.2 == target => span.1 = end,
                _ => links.spans.push((start, end, target.clone())),
            }
        }
        return Ok(());
    }
    let target = if element.is(W_NS, "hyperlink") {
        element.attribute(None, "id").and_then(|id| links.targets.get(id).cloned())
    } else if element.is(W_NS, "fldSimple") {
        element.attribute(None, "instr").and_then(hyperlink_field)
    } else {
        None
    };
    if target.is_some() {
        let outer = std::mem::replace(&mut links.target, target);
        for child in element.direct_elements() {
            authority_text(child, text, references, links)?;
        }
        links.target = outer;
        return Ok(());
    }
    if element.is(W_NS, "tab") || element.is(W_NS, "br") || element.is(W_NS, "cr") {
        text.push(if element.is(W_NS, "tab") { '\t' } else { '\n' });
        return Ok(());
    }
    if element.is(W_NS, "footnoteReference") {
        if let Some(id) = element.attribute(None, "id") {
            let id = id.parse().map_err(|_| Error::Message(format!("Invalid DOCX footnote id {id}")))?;
            let custom = matches!(element.attribute(None, "customMarkFollows"), Some("1" | "true" | "on"));
            references.push((id, legal_structure_model::utf16_len(text), custom.then(String::new)));
        }
        return Ok(());
    }
    // A custom mark is the text or symbol that follows its reference in the same run.
    let mut marked = None;
    for child in element.direct_elements() {
        let before = references.len();
        authority_text(child, text, references, links)?;
        if element.is(W_NS, "r") {
            if references.len() > before && references[before].2.is_some() {
                marked = Some(before);
            } else if let Some(mark) = marked.and_then(|at| references[at].2.as_mut()) {
                if child.is(W_NS, "t") {
                    mark.push_str(&element_text(child)?);
                } else if child.is(W_NS, "sym") {
                    mark.extend(symbol_char(child));
                }
            }
        }
    }
    Ok(())
}

/// The address a `HYPERLINK "…"` field names.
fn hyperlink_field(instruction: &str) -> Option<String> {
    let rest = instruction.trim_start().strip_prefix("HYPERLINK")?.trim_start();
    let quote = rest.chars().next().filter(|quote| *quote == '"' || *quote == '\'')?;
    let rest = &rest[1..];
    Some(rest[..rest.find(quote)?].to_owned()).filter(|target| !target.is_empty())
}

/// A part's relationship targets by id.
fn relationship_targets(files: &[(String, Vec<u8>)], name: &str) -> Result<HashMap<String, String>> {
    let Some(raw) = file_bytes(files, name) else {
        return Ok(HashMap::new());
    };
    let document = parse_xml(raw)?;
    let relationship = |element: &XmlElement| Some((element.attribute(None, "Id")?.to_owned(), element.attribute(None, "Target")?.to_owned()));
    Ok(document.root()?.direct_elements().filter(|element| element.local == "Relationship").filter_map(relationship).collect())
}

/// A `w:sym` character: a symbol font's private-use code (U+F020 to U+F07E) is the ASCII it stands for.
fn symbol_char(element: &XmlElement) -> Option<char> {
    let code = u32::from_str_radix(element.attribute(None, "char")?, 16).ok()?;
    char::from_u32(if (0xF020..=0xF07E).contains(&code) { code - 0xF000 } else { code })
}

#[derive(Clone, Copy, PartialEq)]
enum FootnoteRestart {
    Continuous,
    Section,
    Page,
    Unknown,
}

#[derive(Clone, Copy)]
struct FootnoteNumbering {
    start: Option<u32>,
    restart: FootnoteRestart,
    format: NoteFormat,
}

/// How Word prints a footnote's number (`w:numFmt`). Whatever it prints, Word counts its notes, and a
/// writer refers to a note by that count ("supra note 3" for the note printed ‡).
#[derive(Clone, Copy, PartialEq)]
enum NoteFormat {
    Decimal,
    Chicago,
    LowerRoman,
    UpperRoman,
    LowerLetter,
    UpperLetter,
    Other,
}

impl NoteFormat {
    fn of(value: &str) -> Self {
        match value {
            "decimal" | "decimalZero" => Self::Decimal,
            "chicago" => Self::Chicago,
            "lowerRoman" => Self::LowerRoman,
            "upperRoman" => Self::UpperRoman,
            "lowerLetter" => Self::LowerLetter,
            "upperLetter" => Self::UpperLetter,
            _ => Self::Other,
        }
    }

    /// The mark Word prints for the note it counts as `number`, where it prints no numeral.
    fn mark(self, number: u32) -> Option<String> {
        if number == 0 { return None; }
        let roman = || {
            let mut left = number;
            let mut out = String::new();
            for (value, letters) in [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")] {
                while left >= value { out.push_str(letters); left -= value; }
            }
            out
        };
        let letter = || {
            let index = (number - 1) % 26;
            char::from(b'a' + index as u8).to_string().repeat(((number - 1) / 26 + 1) as usize)
        };
        match self {
            Self::Decimal | Self::Other => None,
            Self::Chicago => {
                let symbols = ['*', '†', '‡', '§'];
                let index = (number - 1) as usize;
                Some(symbols[index % 4].to_string().repeat(index / 4 + 1))
            }
            Self::LowerRoman => Some(roman()),
            Self::UpperRoman => Some(roman().to_uppercase()),
            Self::LowerLetter => Some(letter()),
            Self::UpperLetter => Some(letter().to_uppercase()),
        }
    }
}

fn footnote_numbering(parent: &XmlElement, inherited: FootnoteNumbering) -> FootnoteNumbering {
    let Some(properties) = parent.child("footnotePr") else {
        return inherited;
    };
    let value = |name| properties.child(name).and_then(|child| child.attribute(None, "val"));
    FootnoteNumbering {
        start: value("numStart").map_or(inherited.start, |start| start.parse().ok()),
        restart: match value("numRestart") {
            Some("continuous") => FootnoteRestart::Continuous,
            Some("eachSect") => FootnoteRestart::Section,
            Some("eachPage") => FootnoteRestart::Page,
            Some(_) => FootnoteRestart::Unknown,
            None => inherited.restart,
        },
        format: value("numFmt").map_or(inherited.format, NoteFormat::of),
    }
}

/// Return body paragraphs followed by Word footnotes in citation-review order.
pub fn docx_to_toa_text_units(bytes: &[u8]) -> Result<Vec<serde_json::Value>> {
    let files = read_docx_files(bytes, Some(&["word/document.xml", "word/footnotes.xml", "word/_rels/footnotes.xml.rels", "word/settings.xml"]))?;
    let document = parse_xml(file_bytes(&files, "word/document.xml").ok_or_else(|| Error::Message("DOCX has no word/document.xml".to_owned()))?)?;
    let body = document.root()?.child("body").ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;
    let mut paragraphs = Vec::new();
    body_paragraphs(body, &mut Vec::new(), &mut paragraphs);
    let (mut body_units, mut sections, mut section) = (Vec::new(), Vec::new(), 0);
    for (ordinal, (paragraph, path)) in paragraphs.into_iter().enumerate() {
        let (mut text, mut references) = (String::new(), Vec::new());
        authority_text(paragraph, &mut text, &mut references, &mut AuthorLinks::default())?;
        if !text.trim().is_empty() || !references.is_empty() {
            body_units.push((ordinal, text, references, section, path));
        }
        if let Some(properties) = paragraph.child("pPr").and_then(|properties| properties.child("sectPr")) {
            sections.push(Some(properties));
            section += 1;
        }
    }
    sections.push(body.child("sectPr"));
    let defaults = FootnoteNumbering { start: Some(1), restart: FootnoteRestart::Continuous, format: NoteFormat::Decimal };
    let settings = file_bytes(&files, "word/settings.xml").map(parse_xml).transpose()?;
    let global = settings.as_ref().map(|settings| settings.root()).transpose()?.map_or(defaults, |settings| footnote_numbering(settings, defaults));
    // A section prints its notes in the format its own properties name, decimal when they name none: an author
    // note's chicago section leaves the body's notes decimal, whatever settings.xml mirrors from that first section.
    let decimal = FootnoteNumbering { format: NoteFormat::Decimal, ..global };
    let numbering = sections.iter().map(|properties| properties.map_or(global, |properties| footnote_numbering(properties, decimal))).collect::<Vec<_>>();
    let (mut note_numbers, mut seen, mut current, mut previous_section, mut sequence) = (HashMap::new(), HashSet::new(), global.start, None, 0u32);
    for (_, _, references, section, _) in &body_units {
        for (id, _, mark) in references {
            let (id, custom) = (*id, mark.is_some());
            if !seen.insert(id) {
                continue;
            }
            let properties = numbering[*section];
            if previous_section != Some(*section) {
                // A section printing another format counts its notes afresh, as writers' "supra note 5" in the body
                // after a starred author note shows.
                if previous_section.is_none_or(|previous: usize| numbering[previous].format != properties.format) || properties.restart == FootnoteRestart::Section {
                    sequence += u32::from(previous_section.is_some());
                    current = properties.start;
                }
                previous_section = Some(*section);
            }
            // Word counts a note whatever it prints for it: a note printed "‡" or "iii" is the note a
            // writer's "supra note 3" names, and its printed mark is kept beside the count.
            let counted = matches!(properties.restart, FootnoteRestart::Continuous | FootnoteRestart::Section);
            let number = if custom || !counted { None } else { current };
            let printed = if custom { mark.clone().filter(|mark| !mark.trim().is_empty()) } else { number.and_then(|number| properties.format.mark(number)) };
            note_numbers.insert(id, (number, sequence, printed));
            if !custom {
                current = if counted { current.and_then(|number| number.checked_add(1)) } else { None };
            }
        }
    }
    let (mut footnotes, mut footnote_numbers) = (Vec::new(), HashMap::new());
    if let Some(xml) = file_bytes(&files, "word/footnotes.xml") {
        let targets = relationship_targets(&files, "word/_rels/footnotes.xml.rels")?;
        let document = parse_xml(xml)?;
        let mut elements = Vec::new();
        walk_elements(document.root()?, &mut elements);
        for footnote in elements.into_iter().filter(|element| element.is(W_NS, "footnote")) {
            let Some(raw_id) = footnote.attribute(None, "id") else {
                continue;
            };
            let raw_id: i64 = raw_id.parse().map_err(|_| Error::Message(format!("Invalid DOCX footnote id {raw_id}")))?;
            if raw_id <= 0 {
                continue;
            }
            let (mut text, mut links) = (String::new(), AuthorLinks { targets: targets.clone(), ..AuthorLinks::default() });
            authority_text(footnote, &mut text, &mut Vec::new(), &mut links)?;
            if !text.trim().is_empty() {
                let ordinal = footnotes.len() + 1;
                footnote_numbers.insert(raw_id, ordinal);
                footnotes.push((raw_id, ordinal, text, links.spans));
            }
        }
    }
    let mut units = Vec::with_capacity(body_units.len() + footnotes.len());
    units.extend(body_units.into_iter().map(|(ordinal, text, references, _, path)| {
        let references = references.into_iter().map(|(id, offset, _)| serde_json::json!([footnote_numbers.get(&id).copied().map_or(id, |id| id as i64), offset])).collect::<Vec<_>>();
        serde_json::json!({
            "key": format!("body:{ordinal}"), "kind": "body", "ordinal": ordinal, "footnote_id": serde_json::Value::Null,
            "page_numbers": [], "text": text, "footnote_refs": references, "path": path,
        })
    }));
    units.extend(footnotes.into_iter().map(|(raw_id, ordinal, text, links)| {
        let (number, sequence, mark) = note_numbers.get(&raw_id).cloned().unwrap_or((None, 0, None));
        let mut unit = serde_json::json!({
            "key": format!("footnote:{raw_id}"), "kind": "footnote", "ordinal": ordinal, "footnote_id": ordinal, "note_number": number,
            "note_mark": mark, "restart_sequence": sequence, "page_numbers": [], "text": text, "footnote_refs": [],
        });
        if !links.is_empty() {
            unit["links"] = serde_json::json!(links);
        }
        unit
    }));
    Ok(units)
}

/// The first of a pattern's two alternative groups that matched: a value quoted either way.
fn alternative<'a>(matched: &regex::Captures<'a>) -> Option<regex::Match<'a>> {
    matched.get(1).or_else(|| matched.get(2))
}

fn footnote_reference_ids(document: &str) -> Vec<usize> {
    let custom = |matched: &regex::Captures<'_>| CUSTOM_MARK.is_match(matched.get(0).unwrap().as_str());
    FOOTNOTE_REFERENCE.captures_iter(document).filter(|matched| !custom(matched)).filter_map(|matched| alternative(&matched)?.as_str().parse().ok()).filter(|id| *id > 0).collect()
}

fn add_target_bookmarks(mut xml: String, reference_ids: &[usize], ordinals: &BTreeSet<usize>) -> (String, HashMap<usize, String>, usize) {
    let mut bookmark_id = BOOKMARK_ID.captures_iter(&xml).filter_map(|matched| alternative(&matched)?.as_str().parse().ok()).max().unwrap_or(0) + 1;
    let existing = BOOKMARK_NAME.captures_iter(&xml).filter_map(|matched| alternative(&matched)).map(|name| name.as_str().to_owned()).collect::<HashSet<_>>();
    let mut targets = HashMap::<usize, (usize, usize)>::new();
    for run in RUN.find_iter(&xml) {
        for matched in FOOTNOTE_REFERENCE.captures_iter(run.as_str()) {
            if let Some(reference_id) = alternative(&matched).and_then(|id| id.as_str().parse().ok()) {
                targets.entry(reference_id).or_insert((run.start(), run.end()));
            }
        }
    }
    let (mut names, mut edits, mut added) = (HashMap::new(), BTreeMap::<(usize, usize), Vec<(usize, String)>>::new(), 0);
    for &ordinal in ordinals {
        let Some(&reference_id) = ordinal.checked_sub(1).and_then(|index| reference_ids.get(index)) else {
            continue;
        };
        let name = format!("MikeSupraNote{ordinal}");
        names.insert(ordinal, name.clone());
        if existing.contains(&name) {
            continue;
        }
        let Some(&target) = targets.get(&reference_id) else {
            names.remove(&ordinal);
            continue;
        };
        edits.entry(target).or_default().push((bookmark_id, name));
        bookmark_id += 1;
        added += 1;
    }
    for ((start, end), bookmarks) in edits.into_iter().rev() {
        let opened = bookmarks.iter().map(|(id, name)| format!(r#"<w:bookmarkStart w:id="{id}" w:name="{name}"/>"#));
        let closed = bookmarks.iter().rev().map(|(id, _)| format!(r#"<w:bookmarkEnd w:id="{id}"/>"#));
        let replacement = opened.chain([xml[start..end].to_owned()]).chain(closed).collect::<String>();
        xml.replace_range(start..end, &replacement);
    }
    (xml, names, added)
}

fn plain_run(attributes: &str, properties: &str, text: &str) -> String {
    if text.is_empty() { String::new() } else { format!(r#"<w:r{attributes}>{properties}<w:t xml:space="preserve">{}</w:t></w:r>"#, escape_xml_text(text)) }
}

fn noteref_field(attributes: &str, properties: &str, name: &str, number: &str) -> String {
    let run = |inner: &str| format!("<w:r{attributes}>{properties}{inner}</w:r>");
    let instruction = format!(r#"<w:instrText xml:space="preserve"> NOTEREF {name} \h </w:instrText>"#);
    let (begin, separate, end) = (r#"<w:fldChar w:fldCharType="begin"/>"#, r#"<w:fldChar w:fldCharType="separate"/>"#, r#"<w:fldChar w:fldCharType="end"/>"#);
    [run(begin), run(&instruction), run(separate), plain_run(attributes, properties, number), run(end)].concat()
}

fn convert_safe_paragraphs(xml: &str, names: &HashMap<usize, String>) -> Result<(String, usize)> {
    let (mut output, mut cursor, mut converted) = (String::with_capacity(xml.len()), 0, 0);
    for paragraph in PARAGRAPH.find_iter(xml) {
        output.push_str(&xml[cursor..paragraph.start()]);
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes.iter().map(|node| node.text.as_str()).collect::<String>();
        let fields = field_spans(paragraph.as_str());
        let mut candidates = BTreeMap::<usize, Vec<(&ParagraphTextNode, usize, &str, &str)>>::new();
        for matched in SUPRA.captures_iter(&visible) {
            let (whole, number) = (matched.get(0).unwrap(), matched.get(1).unwrap());
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            let Some(name) = number.as_str().parse::<usize>().ok().and_then(|ordinal| names.get(&ordinal)) else {
                continue;
            };
            let Some(node) = nodes.iter().find(|node| node.visible_start <= number.start() && node.visible_end >= number.end()) else {
                continue;
            };
            if node.safe_to_replace && !fields.iter().any(|&(start, end)| start <= node.xml_start && node.xml_start < end) {
                candidates.entry(node.run_start).or_default().push((node, number.start(), number.as_str(), name));
            }
        }
        let mut edits = Vec::new();
        for rows in candidates.values_mut() {
            rows.sort_by_key(|row| row.1);
            let node = rows[0].0;
            let plain = |text: &str| plain_run(&node.run_attributes, &node.run_properties, text);
            let (mut replacement, mut text_cursor) = (String::new(), 0);
            for (_, start, number, name) in rows {
                let local = *start - node.visible_start;
                replacement.push_str(&plain(&node.text[text_cursor..local]));
                replacement.push_str(&noteref_field(&node.run_attributes, &node.run_properties, name, number));
                text_cursor = local + number.len();
                converted += 1;
            }
            replacement.push_str(&plain(&node.text[text_cursor..]));
            edits.push((node.run_start, node.run_end, replacement));
        }
        let mut next = paragraph.as_str().to_owned();
        edits.sort_by_key(|edit| std::cmp::Reverse(edit.0));
        for (start, end, replacement) in edits {
            next.replace_range(start..end, &replacement);
        }
        output.push_str(&next);
        cursor = paragraph.end();
    }
    output.push_str(&xml[cursor..]);
    Ok((output, converted))
}

pub fn fix_docx_supra_cross_references(bytes: &[u8]) -> Result<DocxSupraCleanup> {
    let files = read_docx_files(bytes, Some(&["word/document.xml", "word/footnotes.xml", "word/settings.xml"]))?;
    let part = |name| docx_part(&files, name).ok_or_else(|| Error::Message("DOCX does not contain ordinary Word footnotes".to_owned()));
    let (document, footnotes) = (part("word/document.xml")?, part("word/footnotes.xml")?);
    let (body, notes) = (analyze_supras(&document)?, analyze_supras(&footnotes)?);
    let (detected, already_linked) = (body.detected + notes.detected, body.already_linked + notes.already_linked);
    let restarted = NUMBERING_RESTART.is_match(&document) || docx_part(&files, "word/settings.xml").is_some_and(|settings| NUMBERING_RESTART.is_match(&settings));
    let unchanged = |review_required, restarted_numbering| DocxSupraCleanup {
        bytes: bytes.to_vec(),
        detected,
        converted: 0,
        already_linked,
        review_required,
        bookmarks_added: 0,
        restarted_numbering,
        unsafe_or_split_fields: review_required,
    };
    if detected == 0 || restarted {
        return Ok(unchanged(detected.saturating_sub(already_linked), restarted));
    }
    let ordinals = body.ordinals.union(&notes.ordinals).copied().collect();
    let reference_ids = footnote_reference_ids(&document);
    let (bookmarked, names, bookmarks_added) = add_target_bookmarks(document, &reference_ids, &ordinals);
    let (next_document, body_converted) = convert_safe_paragraphs(&bookmarked, &names)?;
    let (next_footnotes, note_converted) = convert_safe_paragraphs(&footnotes, &names)?;
    let converted = body_converted + note_converted;
    let review_required = detected.saturating_sub(converted + already_linked);
    if converted == 0 {
        return Ok(unchanged(review_required, false));
    }
    let mut files = read_docx_files(bytes, None)?;
    replace_file(&mut files, "word/document.xml", next_document.into_bytes());
    replace_file(&mut files, "word/footnotes.xml", next_footnotes.into_bytes());
    let bytes = write_docx_files(&files)?;
    Ok(DocxSupraCleanup { bytes, detected, converted, already_linked, review_required, bookmarks_added, restarted_numbering: false, unsafe_or_split_fields: review_required })
}

fn nested_word_elements<'a>(element: &'a XmlElement, wanted: &str) -> Vec<&'a XmlElement> {
    let mut found = Vec::new();
    for child in element.direct_elements() {
        if child.is(W_NS, wanted) {
            found.push(child);
        } else if child.is(W_NS, "sdt") || child.is(W_NS, "sdtContent") {
            found.extend(nested_word_elements(child, wanted));
        }
    }
    found
}

fn paragraphs_under(element: &XmlElement, indexed: &HashMap<*const XmlElement, (usize, usize)>) -> Vec<(usize, usize)> {
    let (mut found, mut pending) = (Vec::new(), vec![element]);
    while let Some(current) = pending.pop() {
        match indexed.get(&(current as *const XmlElement)) {
            Some(index) => found.push(*index),
            None => pending.extend(current.direct_elements().collect::<Vec<_>>().into_iter().rev()),
        }
    }
    found
}

fn word_int(element: Option<&XmlElement>, name: &str) -> Option<usize> {
    element?.child(name)?.attribute(None, "val")?.parse().ok()
}

static OUTLINE: LazyLock<Regex> = LazyLock::new(|| literal(r"<w:outlineLvl\b"));

fn strip_heading_numbering(xml: &str) -> String {
    static PARAGRAPH_PROPERTIES: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:pPr>(.*?)</w:pPr>"));
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:pStyle\b[^>]*w:val="Heading(\d+)""#));
    static NUMBERING: LazyLock<Regex> = LazyLock::new(|| literal(r"(?s)<w:numPr\b.*?</w:numPr>"));
    let paragraph = |captures: &regex::Captures<'_>| {
        let inner = &captures[1];
        let Some(level) = HEADING_STYLE.captures(inner).and_then(|style| style.get(1)).filter(|_| NUMBERING.is_match(inner)) else {
            return captures[0].to_owned();
        };
        let mut inner = NUMBERING.replace_all(inner, "").into_owned();
        let level = level.as_str().parse::<u8>().ok().filter(|level| (1..=6).contains(level)).map(|level| level - 1);
        if let Some(level) = level.filter(|_| !OUTLINE.is_match(&inner)) {
            inner.insert_str(0, &format!(r#"<w:outlineLvl w:val="{level}"/>"#));
        }
        format!("<w:pPr>{inner}</w:pPr>")
    };
    PARAGRAPH_PROPERTIES.replace_all(xml, paragraph).into_owned()
}

fn normalize_heading_styles(xml: &str) -> String {
    static DEFAULT_STYLE: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:style\b[^>]*\bw:default="1""#));
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| literal(r#"(?s)<w:style\b[^>]*\bw:styleId="Heading\d+".*?</w:style>"#));
    static HEADING_NAME: LazyLock<Regex> = LazyLock::new(|| literal(r#"(?i)(<w:name\b[^>]*w:val=")Heading ([0-9])(")"#));
    static STYLE_LEVEL: LazyLock<Regex> = LazyLock::new(|| literal(r#"w:styleId="Heading([1-6])""#));
    static PARAGRAPH_PROPERTIES: LazyLock<Regex> = LazyLock::new(|| literal(r"(<w:pPr[\s>][^<]*)"));
    let normal = r#"<w:style w:default="1" w:styleId="Normal" w:type="paragraph"><w:name w:val="Normal"/><w:qFormat/></w:style></w:styles>"#;
    let xml = if DEFAULT_STYLE.is_match(xml) { xml.to_owned() } else { xml.replacen("</w:styles>", normal, 1) };
    let heading = |captures: &regex::Captures<'_>| {
        let mut style = HEADING_NAME.replace(&captures[0], "${1}heading ${2}${3}").into_owned();
        if let Some(level) = STYLE_LEVEL.captures(&style).and_then(|capture| capture.get(1)).filter(|_| !OUTLINE.is_match(&style)) {
            let level = level.as_str().parse::<u8>().unwrap_or(1) - 1;
            style = PARAGRAPH_PROPERTIES.replacen(&style, 1, format!("$1<w:outlineLvl w:val=\"{level}\"/>").as_str()).into_owned();
        }
        style
    };
    HEADING_STYLE.replace_all(&xml, heading).into_owned()
}

fn split_drafting_image_runs(xml: &str) -> Result<String> {
    fn write(writer: &mut quick_xml::Writer<Vec<u8>>, event: Event<'_>) -> Result<()> {
        writer.write_event(event).map_err(|error| Error::io("word/document.xml", error))
    }
    let word = |reader: &NsReader<&[u8]>, start: &BytesStart<'_>| namespace_value(reader.resolver().resolve_element(start.name()).0).as_deref() == Some(W_NS);
    let mut reader = NsReader::from_reader(xml.as_bytes());
    let mut writer = quick_xml::Writer::new(Vec::new());
    let mut changed = false;
    loop {
        let event = reader.read_event()?.into_owned();
        let is_run = matches!(&event, Event::Start(start) if start.local_name().as_ref() == b"r" && word(&reader, start));
        if matches!(event, Event::Eof) {
            break;
        }
        if !is_run {
            write(&mut writer, event)?;
            continue;
        }
        let start = event;
        let (mut children, mut depth, mut has_image, mut has_text) = (Vec::<(bool, Vec<Event<'static>>)>::new(), 0, false, false);
        let end = loop {
            let event = reader.read_event()?.into_owned();
            if matches!(event, Event::Eof) {
                return Err(Error::Message("DOCX has an unclosed run".to_owned()));
            }
            if matches!(event, Event::End(_)) && depth == 0 {
                break event;
            }
            if depth == 0 {
                let mut properties = false;
                if let Event::Start(child) | Event::Empty(child) = &event {
                    if word(&reader, child) {
                        let local = child.local_name();
                        properties = local.as_ref() == b"rPr";
                        has_image |= matches!(local.as_ref(), b"drawing" | b"pict");
                        has_text |= local.as_ref() == b"t";
                    }
                }
                children.push((properties, Vec::new()));
            }
            match event {
                Event::Start(_) => depth += 1,
                Event::End(_) => depth -= 1,
                _ => {}
            }
            children.last_mut().expect("run child").1.push(event);
        };
        if has_image && has_text {
            // Pandoc treats a run containing an image as image-only. Give it
            // separate runs without changing child order or run formatting.
            changed = true;
            let formatting = || children.iter().filter(|(properties, _)| *properties).flat_map(|(_, events)| events);
            for (_, content) in children.iter().filter(|(properties, _)| !properties) {
                write(&mut writer, start.clone())?;
                for event in formatting().chain(content) {
                    write(&mut writer, event.clone())?;
                }
                write(&mut writer, end.clone())?;
            }
        } else {
            for event in std::iter::once(start).chain(children.into_iter().flat_map(|(_, events)| events)).chain(std::iter::once(end)) {
                write(&mut writer, event)?;
            }
        }
    }
    if !changed {
        return Ok(xml.to_owned());
    }
    String::from_utf8(writer.into_inner()).map_err(|error| failed("DOCX XML is not UTF-8", error))
}

fn drafting_docx_input(bytes: &[u8]) -> Result<Vec<u8>> {
    static METADATA_STYLE: LazyLock<Regex> = LazyLock::new(|| literal(r#"(?i)(<w:name\b[^>]*w:val=")(Title|Subtitle)(")"#));
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| literal(r#"<w:style\b[^>]*\bw:styleId="Heading\d+""#));
    let inspected = read_docx_files(bytes, Some(&["word/document.xml", "word/styles.xml"]))?;
    let document = docx_part(&inspected, "word/document.xml").ok_or_else(|| Error::Message("Drafting mode requires a valid DOCX".to_owned()))?;
    let stripped = split_drafting_image_runs(&strip_heading_numbering(&document)).map_err(|error| failed("DOCX contains malformed XML in word/document.xml", error))?;
    // Pandoc consumes these style names as metadata and its GFM writer
    // omits that metadata. Keep their text in the temporary input's body.
    let styles = docx_part(&inspected, "word/styles.xml").map(|styles| {
        let readable = METADATA_STYLE.replace_all(&styles, "${1}Readable ${2}${3}");
        let normalized = if HEADING_STYLE.is_match(&readable) { normalize_heading_styles(&readable) } else { readable.into_owned() };
        (normalized != styles, normalized)
    });
    if stripped == document && !styles.as_ref().is_some_and(|(changed, _)| *changed) {
        return Ok(bytes.to_vec());
    }
    let mut files = read_docx_files(bytes, None)?;
    if stripped != document {
        replace_file(&mut files, "word/document.xml", stripped.into_bytes());
    }
    if let Some((_, styles)) = styles {
        replace_file(&mut files, "word/styles.xml", styles.into_bytes());
    }
    write_docx_files(&files)
}

fn clean_process_error(bytes: &[u8]) -> String {
    legal_structure_model::normalize_javascript_whitespace(&String::from_utf8_lossy(bytes)).chars().take(500).collect()
}

fn pandoc_drafting_markdown(bytes: Vec<u8>) -> Result<String> {
    const MAX_OUTPUT: usize = MAX_DOCX_SUPRA_BYTES;
    const SYSTEM_ENV: [&str; 20] = [
        "APPDATA", "COMSPEC", "HOME", "LANG", "LC_ALL", "LOCALAPPDATA", "PATH", "PATHEXT", "PROGRAMFILES", "PROGRAMFILES(X86)",
        "SHELL", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "USERPROFILE", "WINDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    ];
    let mut command = Command::new("pandoc");
    command.args(["-f", "docx", "-t", "gfm", "--sandbox", "--wrap=none", "-o", "-"]).env_clear();
    for (name, value) in env::vars_os() {
        if SYSTEM_ENV.iter().any(|allowed| name.to_string_lossy().eq_ignore_ascii_case(allowed)) {
            command.env(name, value);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let output = crate::process::run(command, bytes, Duration::from_secs(120), MAX_OUTPUT, 8_192).map_err(|error| match error {
        crate::process::RunError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
            Error::Message("Pandoc is required for drafting mode but was not found on PATH".to_owned())
        }
        crate::process::RunError::Io(source) => Error::Message(format!("Pandoc conversion failed: {}", clean_process_error(source.to_string().as_bytes()))),
        crate::process::RunError::Timeout => Error::Message("Pandoc conversion timed out".to_owned()),
    })?;
    if output.stdout_exceeded {
        return Err(Error::Message("Pandoc conversion output exceeded 25 MiB".to_owned()));
    }
    if !output.status.success() {
        let code = output.status.code().map_or_else(|| "unknown".to_owned(), |code| code.to_string());
        return Err(Error::Message(format!("Pandoc conversion failed (exit {code}): {}", clean_process_error(&output.stderr))));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn clean_drafting_markdown(markdown: String) -> String {
    static HTML_IMAGE: LazyLock<Regex> = LazyLock::new(|| literal(r"(?i)<img\b[^>]*\/?>"));
    static MARKDOWN_IMAGE: LazyLock<Regex> = LazyLock::new(|| literal(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?"));
    static EMPTY_LINK: LazyLock<Regex> = LazyLock::new(|| literal(&format!(r"(?m)^\[\]\([^)]*\){}*$", legal_structure_model::JS_WHITESPACE_CLASS)));
    static UNSAFE_LINK: LazyLock<Regex> = LazyLock::new(|| literal(r"(?i)\[[^\]]*\]\((?:data|javascript):[^)]*\)"));
    static ESCAPED_BRACKET: LazyLock<Regex> = LazyLock::new(|| literal(r"\\([\[\]])"));
    let markdown = markdown.replace("\r\n", "\n").replace('\r', "\n");
    let replaced = [(&*HTML_IMAGE, "[Image omitted]"), (&*MARKDOWN_IMAGE, "[Image omitted]"), (&*EMPTY_LINK, ""), (&*UNSAFE_LINK, ""), (&*ESCAPED_BRACKET, "$1")]
        .into_iter()
        .fold(markdown, |markdown, (pattern, with)| pattern.replace_all(&markdown, with).into_owned());
    legal_structure_model::trim_javascript_whitespace(&replaced).to_owned()
}

fn docx_document_xml(bytes: &[u8]) -> Result<Vec<u8>> {
    const MAX_DOCX_BYTES: usize = 50 * 1024 * 1024;
    const MAX_XML_BYTES: u64 = 32 * 1024 * 1024;
    let refused = |message: &str| Error::Message(message.to_owned());
    if bytes.is_empty() || bytes.len() > MAX_DOCX_BYTES {
        return Err(refused("DOCX is empty or exceeds the read limit"));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    if archive.len() > 10_000 {
        return Err(refused("DOCX has too many package entries"));
    }
    let mut document = archive.by_name("word/document.xml").map_err(|_| refused("DOCX has no word/document.xml"))?;
    if document.size() > MAX_XML_BYTES {
        return Err(refused("DOCX document XML exceeds the read limit"));
    }
    let mut xml = Vec::with_capacity(document.size() as usize);
    document.read_to_end(&mut xml).map_err(|source| Error::io("word/document.xml", source))?;
    Ok(xml)
}

fn docx_structure_input(bytes: &[u8], include_tables: bool) -> Result<(Vec<String>, Vec<legal_structure_model::AuthoritativeTableCell>)> {
    use legal_structure_model::{AuthoritativeTableCell, ScalarRange};
    let xml = docx_document_xml(bytes)?;
    let Ok(document) = parse_xml(&xml) else {
        let xml = std::str::from_utf8(&xml).map_err(|error| failed("DOCX XML is not UTF-8", error))?;
        return Ok((tolerant_docx_paragraphs(xml)?, Vec::new()));
    };
    let body = document.root()?.child("body").ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;
    let mut all = Vec::new();
    walk_elements(body, &mut all);
    let canonical_elements = all.iter().copied().filter(|element| element.is(W_NS, "p") && !element.self_closing).collect::<Vec<_>>();
    let paragraphs = canonical_elements.iter().map(|paragraph| normalized_docx_paragraph(paragraph)).collect::<Result<Vec<_>>>()?;
    if !include_tables {
        return Ok((paragraphs, Vec::new()));
    }
    let (mut starts, mut text_length) = (Vec::with_capacity(paragraphs.len()), 0);
    for (index, paragraph) in paragraphs.iter().enumerate() {
        text_length += usize::from(index > 0);
        starts.push(text_length);
        text_length += legal_structure_model::utf16_len(paragraph);
    }
    let end_of = |index: usize| starts[index] + legal_structure_model::utf16_len(&paragraphs[index]);
    let mut by_element = canonical_elements.iter().enumerate().map(|(index, paragraph)| (*paragraph as *const XmlElement, (starts[index], end_of(index)))).collect::<HashMap<_, _>>();
    let (mut entry_paragraph, mut preceding) = (HashMap::new(), 0);
    for element in &all {
        entry_paragraph.insert(*element as *const XmlElement, preceding);
        if element.is(W_NS, "p") && element.self_closing {
            let offset = starts.get(preceding).copied().unwrap_or(text_length);
            by_element.insert(*element as *const XmlElement, (offset, offset));
        } else if element.is(W_NS, "p") {
            preceding += 1;
        }
    }
    let mut table_cells = Vec::new();
    for (table_index, table) in all.iter().copied().filter(|element| element.is(W_NS, "tbl")).enumerate() {
        let mut vertical_anchors = HashMap::<usize, usize>::new();
        for (row_index, row) in nested_word_elements(table, "tr").into_iter().enumerate() {
            let mut column = 1 + word_int(row.child("trPr"), "gridBefore").unwrap_or(0);
            let (mut next_vertical_anchors, mut horizontal_anchor) = (HashMap::new(), None);
            for cell in nested_word_elements(row, "tc") {
                let cell_properties = cell.child("tcPr");
                let column_span = word_int(cell_properties, "gridSpan").filter(|value| *value > 0).unwrap_or(1);
                let column_end = column + column_span;
                let merge = |name| {
                    let merged = cell_properties.and_then(|properties| properties.child(name));
                    merged.map(|element| element.attribute(None, "val").is_some_and(|value| value.eq_ignore_ascii_case("restart")))
                };
                let (vertical_merge, horizontal_merge) = (merge("vMerge"), merge("hMerge"));
                let (vertical_continuation, horizontal_continuation) = (vertical_merge == Some(false), horizontal_merge == Some(false));
                let vertical_anchor = vertical_continuation
                    .then(|| vertical_anchors.get(&column).copied().filter(|anchor| (column..column_end).all(|covered| vertical_anchors.get(&covered) == Some(anchor))))
                    .flatten();
                let agree = (!vertical_continuation || vertical_anchor.is_some())
                    && (!horizontal_continuation || horizontal_anchor.is_some())
                    && (!vertical_continuation || !horizontal_continuation || vertical_anchor == horizontal_anchor);
                let anchor = if vertical_continuation || horizontal_continuation {
                    vertical_anchor.or(horizontal_anchor).filter(|_| agree)
                } else {
                    let contents = paragraphs_under(cell, &by_element);
                    let empty_at = || entry_paragraph.get(&(cell as *const XmlElement)).copied().unwrap_or(paragraphs.len()).checked_sub(1).map_or(0, end_of);
                    let start = contents.first().map_or_else(empty_at, |(start, _)| *start);
                    let end = contents.last().map_or_else(empty_at, |(_, end)| *end);
                    let ranges = if contents.is_empty() { vec![ScalarRange { start, end }] } else { contents.iter().map(|&(start, end)| ScalarRange { start, end }).collect() };
                    table_cells.push(AuthoritativeTableCell {
                        ranges,
                        table: table_index + 1,
                        table_name: None,
                        row: row_index + 1,
                        column,
                        row_span: None,
                        column_span: (column_span > 1).then_some(column_span),
                        address: None,
                        display_value: None,
                        start,
                        end,
                    });
                    Some(table_cells.len() - 1)
                };
                if let Some(anchor) = anchor {
                    let merged = &mut table_cells[anchor];
                    if vertical_continuation {
                        let span = (row_index + 2 - merged.row).max(merged.row_span.unwrap_or(1));
                        merged.row_span = (span > 1).then_some(span);
                    }
                    if horizontal_continuation {
                        let span = (column_end - merged.column).max(merged.column_span.unwrap_or(1));
                        merged.column_span = (span > 1).then_some(span);
                    }
                    if vertical_merge.is_some() {
                        next_vertical_anchors.extend((column..column_end).map(|covered| (covered, anchor)));
                    }
                }
                horizontal_anchor = if horizontal_merge.is_some() { anchor } else { None };
                column = column_end;
            }
            vertical_anchors = next_vertical_anchors;
        }
    }
    Ok((paragraphs, table_cells))
}

/// Return the accepted model-visible DOCX text without running structure
/// detection. Drafting mode uses the same Pandoc adaptation as full legal_structure_model::structure_analysis().
pub fn docx_text(bytes: &[u8], drafting: bool) -> Result<String> {
    if drafting {
        drafting_docx_text(bytes)
    } else {
        docx_structure_input(bytes, false).map(|(paragraphs, _)| paragraphs.join("\n"))
    }
}

/// Parse the accepted DOCX text and authoritative table coordinates once, then
/// feed the canonical Rust detector directly.
pub fn analyze_docx_bytes(bytes: &[u8], document_id: String) -> Result<legal_structure_model::DocumentStructure> {
    let (paragraphs, table_cells) = docx_structure_input(bytes, true)?;
    legal_structure_model::structure_analysis().analyze_docx(document_id, paragraphs, &table_cells).map_err(|error| Error::Message(error.to_string()))
}

fn drafting_docx_text(bytes: &[u8]) -> Result<String> {
    if bytes.is_empty() || bytes.len() > MAX_DOCX_SUPRA_BYTES {
        return Err(Error::Message("Precedent DOCX exceeds the drafting read limit".to_owned()));
    }
    let input = drafting_docx_input(bytes).map_err(|error| match error {
        Error::Zip(_) => Error::Message("Precedent DOCX is corrupted or truncated".to_owned()),
        Error::Message(message) if message.contains("XML part exceeds") => Error::Message("Precedent DOCX contains an oversized XML part".to_owned()),
        Error::Message(message) => Error::Message(message.strip_prefix("DOCX ").map_or(message.clone(), |detail| format!("Precedent DOCX {detail}"))),
        error => error,
    })?;
    let markdown = pandoc_drafting_markdown(input).map_err(|error| {
        if error.to_string().contains("was not found on PATH") {
            error
        } else {
            failed("Precedent DOCX contains malformed XML in word/document.xml", error)
        }
    })?;
    let markdown = clean_drafting_markdown(markdown);
    if !markdown.is_empty() {
        return Ok(markdown);
    }
    let text = docx_text(bytes, false)?;
    if text.trim().is_empty() { Err(Error::Message("Precedent DOCX has no readable drafting structure".to_owned())) } else { Ok(text) }
}

/// Build the model-visible drafting document in one Rust operation. OOXML and
/// Pandoc adaptation stay here; the resulting text uses the canonical detector.
pub fn analyze_docx_drafting_bytes(bytes: &[u8], document_id: String) -> Result<legal_structure_model::DocumentStructure> {
    legal_structure_model::structure_analysis().analyze_instrument(drafting_docx_text(bytes)?, document_id, &[], true).map_err(|error| Error::Message(error.to_string()))
}
