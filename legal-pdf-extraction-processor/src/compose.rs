//! Writes PDFs: pages drawn from text, rules and boxes, pages copied from other PDFs, bookmarks, links, page labels,
//! Acrobat's own page-number artifacts and fillable text fields. Beaver's books of authorities, rebuilt statutes and
//! court records are laid out on it, so a PDF is written one way.
use crate::annotate::{disarm_dict, inert};
use crate::outline::destination_array;
use crate::page_labels::label_rules;
use legal_pdf_core::{Error, Result};
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, SaveOptions, Stream, StringFormat};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::OnceLock;

pub type Rgb = [f64; 3];
fn fail(message: impl Into<String>) -> Error { Error::Message(message.into()) }
fn lo(error: lopdf::Error) -> Error { fail(format!("PDF could not be assembled: {error}")) }

/// The faces a page is set in: the standard Times, Helvetica and Courier, and Arial (its own file subset into the PDF
/// where the machine has one, else named with Arial's widths, which a reader shows in its own Arial).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Face { Times, TimesBold, TimesItalic, TimesBoldItalic, Helvetica, HelveticaBold, Courier, Arial, ArialBold, ArialItalic, ArialBoldItalic }

// Advance widths for WinAnsi codes 32 to 255, in thousandths of an em (the Adobe core AFMs; Arial's own, its italics
// sharing them). Courier's are all 600.
const WIDTHS: [&str; 8] = [
    "250,333,408,500,500,833,778,180,333,333,500,564,250,333,250,278,500,500,500,500,500,500,500,500,500,500,278,278,564,564,564,444,921,722,667,667,722,611,556,722,722,333,389,722,611,889,722,722,556,722,667,556,611,722,722,944,722,722,611,333,278,333,469,500,333,444,500,444,500,444,333,500,500,278,278,500,278,778,500,500,500,500,333,389,278,500,500,722,500,500,444,480,200,480,541,0,500,0,333,500,444,1000,500,500,333,1000,556,333,889,0,611,0,0,333,333,444,444,350,500,1000,333,980,389,333,722,0,444,500,250,333,500,500,500,500,200,500,333,760,276,500,564,333,760,333,400,564,300,300,333,500,453,250,333,300,310,500,750,750,750,444,722,722,722,722,722,722,889,667,611,611,611,611,333,333,333,333,722,722,722,722,722,722,722,564,722,722,722,722,722,722,556,500,444,444,444,444,444,444,667,444,444,444,444,444,278,278,278,278,500,500,500,500,500,500,500,564,500,500,500,500,500,500,500,500",
    "250,333,555,500,500,1000,833,278,333,333,500,570,250,333,250,278,500,500,500,500,500,500,500,500,500,500,333,333,570,570,570,500,930,722,667,722,722,667,611,778,778,389,500,778,667,944,722,778,611,778,722,556,667,722,722,1000,722,722,667,333,278,333,581,500,333,500,556,444,556,444,333,500,556,278,333,556,278,833,556,500,556,556,444,389,333,556,500,722,500,500,444,394,220,394,520,0,500,0,333,500,500,1000,500,500,333,1000,556,333,1000,0,667,0,0,333,333,500,500,350,500,1000,333,1000,389,333,722,0,444,500,250,333,500,500,500,500,220,500,333,747,300,500,570,333,747,333,400,570,300,300,333,556,540,250,333,300,330,500,750,750,750,500,722,722,722,722,722,722,1000,722,667,667,667,667,389,389,389,389,722,722,778,778,778,778,778,570,778,722,722,722,722,722,611,556,500,500,500,500,500,500,722,444,444,444,444,444,278,278,278,278,500,556,500,500,500,500,500,570,500,556,556,556,556,500,556,500",
    "250,333,420,500,500,833,778,214,333,333,500,675,250,333,250,278,500,500,500,500,500,500,500,500,500,500,333,333,675,675,675,500,920,611,611,667,722,611,611,722,722,333,444,667,556,833,667,722,611,722,611,500,556,722,611,833,611,556,556,389,278,389,422,500,333,500,500,444,500,444,278,500,500,278,278,444,278,722,500,500,500,500,389,389,278,500,444,667,444,444,389,400,275,400,541,0,500,0,333,500,556,889,500,500,333,1000,500,333,944,0,556,0,0,333,333,556,556,350,500,889,333,980,389,333,667,0,389,444,250,389,500,500,500,500,275,500,333,760,276,500,675,333,760,333,400,675,300,300,333,500,523,250,333,300,310,500,750,750,750,500,611,611,611,611,611,611,889,667,611,611,611,611,333,333,333,333,722,667,722,722,722,722,722,675,722,722,722,722,722,556,611,500,500,500,500,500,500,500,667,444,444,444,444,444,278,278,278,278,500,500,500,500,500,500,500,675,500,500,500,500,500,444,500,444",
    "250,389,555,500,500,833,778,278,333,333,500,570,250,333,250,278,500,500,500,500,500,500,500,500,500,500,333,333,570,570,570,500,832,667,667,667,722,667,667,722,778,389,500,667,611,889,722,722,611,722,667,556,611,722,667,889,667,611,611,333,278,333,570,500,333,500,500,444,500,444,333,500,556,278,278,500,278,778,556,500,500,500,389,389,278,556,444,667,500,444,389,348,220,348,570,0,500,0,333,500,500,1000,500,500,333,1000,556,333,944,0,611,0,0,333,333,500,500,350,500,1000,333,1000,389,333,722,0,389,444,250,389,500,500,500,500,220,500,333,747,266,500,606,333,747,333,400,570,300,300,333,576,500,250,333,300,300,500,750,750,750,500,667,667,667,667,667,667,944,667,667,667,667,667,389,389,389,389,722,722,722,722,722,722,722,570,722,722,722,722,722,611,611,500,500,500,500,500,500,500,722,444,444,444,444,444,278,278,278,278,500,556,500,500,500,500,500,570,500,556,556,556,556,444,500,444",
    "278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,334,260,334,584,0,556,0,222,556,333,1000,556,556,333,1000,667,333,1000,0,611,0,0,222,222,333,333,350,556,1000,333,1000,500,333,944,0,500,500,278,333,556,556,556,556,260,556,333,737,370,556,584,333,737,333,400,584,333,333,333,556,537,278,333,333,365,556,834,834,834,611,667,667,667,667,667,667,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,778,778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,500,556,556,556,556,278,278,278,278,556,556,556,556,556,556,556,584,611,556,556,556,556,500,556,500",
    "278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,333,333,584,584,584,611,975,722,722,722,722,667,611,778,722,278,556,722,611,833,722,778,667,778,722,667,611,722,667,944,667,667,611,333,278,333,584,556,333,556,611,556,611,556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,611,556,778,556,556,500,389,280,389,584,0,556,0,278,556,500,1000,556,556,333,1000,667,333,1000,0,611,0,0,278,278,500,500,350,556,1000,333,1000,556,333,944,0,500,556,278,333,556,556,556,556,280,556,333,737,370,556,584,333,737,333,400,584,333,333,333,611,556,278,333,333,365,556,834,834,834,611,722,722,722,722,722,722,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,778,778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,556,556,556,556,556,278,278,278,278,611,611,611,611,611,611,611,584,611,611,611,611,611,556,611,556",
    "278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,334,260,334,584,0,556,0,222,556,333,1000,556,556,333,1000,667,333,1000,0,611,0,0,222,222,333,333,350,556,1000,333,1000,500,333,944,0,500,667,278,333,556,556,556,556,260,556,333,737,370,556,584,333,737,552,400,549,333,333,333,576,537,333,333,333,365,556,834,834,834,611,667,667,667,667,667,667,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,778,778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,500,556,556,556,556,278,278,278,278,556,556,556,556,556,556,556,549,611,556,556,556,556,500,556,500",
    "278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,333,333,584,584,584,611,975,722,722,722,722,667,611,778,722,278,556,722,611,833,722,778,667,778,722,667,611,722,667,944,667,667,611,333,278,333,584,556,333,556,611,556,611,556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,611,556,778,556,556,500,389,280,389,584,0,556,0,278,556,500,1000,556,556,333,1000,667,333,1000,0,611,0,0,278,278,500,500,350,556,1000,333,1000,556,333,944,0,500,667,278,333,556,556,556,556,280,556,333,737,370,556,584,333,737,552,400,549,333,333,333,576,556,333,333,333,365,556,834,834,834,611,722,722,722,722,722,722,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,778,778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,556,556,556,556,556,278,278,278,278,611,611,611,611,611,611,611,549,611,611,611,611,611,556,611,556",
];
/// WinAnsi's codes 128-159, from 128.
const WIN_ANSI_HIGH: &str = "€\0‚ƒ„…†‡ˆ‰Š‹Œ\0Ž\0\0‘’“”•–—˜™š›œ\0žŸ";

impl Face {
    fn table(self) -> Option<&'static [u16]> {
        static TABLES: OnceLock<Vec<Vec<u16>>> = OnceLock::new();
        let tables = TABLES.get_or_init(|| WIDTHS.iter().map(|row| row.split(',').map(|v| v.parse().unwrap_or(0)).collect()).collect());
        use Face::*;
        let index = match self { Times => 0, TimesBold => 1, TimesItalic => 2, TimesBoldItalic => 3, Helvetica => 4, HelveticaBold => 5,
            Arial | ArialItalic => 6, ArialBold | ArialBoldItalic => 7, Courier => return None };
        Some(&tables[index])
    }
    fn base_name(self) -> &'static str {
        use Face::*;
        match self { Times => "Times-Roman", TimesBold => "Times-Bold", TimesItalic => "Times-Italic", TimesBoldItalic => "Times-BoldItalic",
            Helvetica => "Helvetica", HelveticaBold => "Helvetica-Bold", Courier => "Courier", Arial => "ArialMT",
            ArialBold => "Arial-BoldMT", ArialItalic => "Arial-ItalicMT", ArialBoldItalic => "Arial-BoldItalicMT" }
    }
    fn arial(self) -> bool { matches!(self, Face::Arial | Face::ArialBold | Face::ArialItalic | Face::ArialBoldItalic) }
    /// The width of `text` set at `size`, each character measured as it is drawn.
    pub fn width(self, text: &str, size: f64) -> f64 {
        let em: u32 = text.chars().map(|c| {
            let code = win_ansi(c).unwrap_or(b'?');
            match self.table() { None => if code >= 32 { 600 } else { 0 }, Some(t) => if code < 32 { 0 } else { u32::from(t[usize::from(code - 32)]) } }
        }).sum();
        f64::from(em) * size / 1000.0
    }
}

/// The WinAnsi code of a character, where it has one.
pub fn win_ansi(c: char) -> Option<u8> {
    match c as u32 {
        0x20..=0x7e | 0xa0..=0xff => Some(c as u8),
        0x0a | 0x0d | 0x09 => Some(c as u8),
        _ => WIN_ANSI_HIGH.chars().position(|h| h == c && h != '\0').map(|i| 128 + i as u8),
    }
}

/// The machine's own Arial files (never shipped with Beaver), by face, where it has them.
pub fn system_arial(face: Face) -> Option<&'static [u8]> {
    static FACES: OnceLock<HashMap<Face, Vec<u8>>> = OnceLock::new();
    let faces = FACES.get_or_init(|| {
        let directories: Vec<std::path::PathBuf> = if cfg!(windows) {
            vec![std::path::Path::new(&std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into())).join("Fonts")]
        } else if cfg!(target_os = "macos") {
            vec!["/System/Library/Fonts/Supplemental".into(), "/Library/Fonts".into()]
        } else {
            vec!["/usr/share/fonts/truetype/msttcorefonts".into(), "/usr/share/fonts/TTF".into(), "/usr/share/fonts/truetype".into()]
        };
        let files: [(Face, &[&str]); 4] = [(Face::Arial, &["arial.ttf"]), (Face::ArialBold, &["arialbd.ttf", "arial bold.ttf", "arial_bold.ttf"]),
            (Face::ArialItalic, &["ariali.ttf", "arial italic.ttf", "arial_italic.ttf"]),
            (Face::ArialBoldItalic, &["arialbi.ttf", "arial bold italic.ttf", "arial_bold_italic.ttf"])];
        let mut found = HashMap::new();
        for directory in directories {
            let Ok(entries) = std::fs::read_dir(&directory) else { continue };
            let names: Vec<String> = entries.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect();
            for (face, wanted) in files {
                if found.contains_key(&face) { continue; }
                if let Some(name) = names.iter().find(|n| wanted.contains(&n.to_lowercase().as_str())) {
                    if let Ok(bytes) = std::fs::read(directory.join(name)) { found.insert(face, bytes); }
                }
            }
        }
        found
    });
    faces.get(&face).map(Vec::as_slice)
}

/// A bookmark: its title, the page it opens (with the view its source gave it) or the inert action it takes, and
/// the bookmarks under it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Outline {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Outline>,
    #[serde(skip)]
    pub view: Option<Vec<Object>>,
    #[serde(skip)]
    pub action: Option<Dictionary>,
}

impl Outline {
    pub fn page(title: impl Into<String>, page_index: usize, children: Vec<Outline>) -> Self {
        Self { title: title.into(), page_index: Some(page_index), children, ..Self::default() }
    }
    fn flat(items: &[Outline]) -> Vec<&Outline> { items.iter().flat_map(|i| std::iter::once(i).chain(Self::flat(&i.children))).collect() }
}

/// A flat outline (title, level, page) nested: each entry under the last shallower one.
pub fn nested_outline(entries: impl IntoIterator<Item = (String, usize, Option<usize>)>) -> Vec<Outline> {
    let mut roots: Vec<Outline> = Vec::new();
    let mut open: Vec<usize> = Vec::new(); // levels of the open path, innermost last
    for (title, level, page) in entries {
        let Some(page) = page else { continue };
        while open.last().is_some_and(|l| *l >= level) { open.pop(); }
        let mut siblings = &mut roots;
        for _ in 0..open.len() { siblings = &mut siblings.last_mut().expect("open entry").children; }
        siblings.push(Outline::page(title, page, Vec::new()));
        open.push(level);
    }
    roots
}

/// An outline moved onto other pages: an entry whose page is gone gives way to its children.
pub fn map_outline(outline: &[Outline], page: &dyn Fn(usize) -> Option<usize>) -> Vec<Outline> {
    outline.iter().flat_map(|entry| {
        let children = map_outline(&entry.children, page);
        match entry.page_index {
            None => vec![Outline { children, ..entry.clone() }],
            Some(at) => match page(at) { Some(at) => vec![Outline { page_index: Some(at), children, ..entry.clone() }], None => children },
        }
    }).collect()
}

/// A source's outline: its publisher's bookmarks, with the headings read from its text where those bookmarks are
/// only a frame. Each heading the bookmarks do not name goes under the last bookmark that opens at or before its
/// page. A lone bookmark ("Blank Page") is no outline.
pub fn source_outline(own: Vec<Outline>, read: Vec<Outline>) -> Vec<Outline> {
    fn external(items: &[Outline]) -> Vec<Outline> {
        items.iter().flat_map(|i| if i.action.is_some() { vec![i.clone()] } else { external(&i.children) }).collect()
    }
    let local = Outline::flat(&own).into_iter().filter(|i| i.action.is_none()).count();
    if local < 2 { return external(&own).into_iter().chain(read).collect(); }
    let key = |title: &str| {
        let lower = title.to_lowercase();
        let trimmed = lower.trim_start();
        let marker = trimmed.split_once(char::is_whitespace).filter(|(head, _)| {
            head.chars().all(|c| c.is_numeric() || ".()".contains(c))
                || head.ends_with('.') && (head[..head.len() - 1].chars().all(|c| "ivxlcdm".contains(c)) && head.len() > 1
                    || head.chars().count() == 2 && head.chars().next().is_some_and(char::is_alphabetic))
        });
        let rest = marker.map_or(trimmed, |(_, rest)| rest.trim_start());
        rest.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
    };
    let named: BTreeSet<String> = Outline::flat(&own).iter().filter(|i| i.action.is_none()).map(|i| key(&i.title)).collect();
    fn unnamed(items: Vec<Outline>, named: &BTreeSet<String>, key: &dyn Fn(&str) -> String) -> Vec<Outline> {
        items.into_iter().flat_map(|mut item| {
            let kept = unnamed(std::mem::take(&mut item.children), named, key);
            if named.contains(&key(&item.title)) { kept } else { vec![Outline { children: kept, ..item }] }
        }).collect()
    }
    let headings = unnamed(read, &named, &key);
    if Outline::flat(&headings).len() <= local { return own; }
    // Headings go under the bookmarks alone, never under one another; appending keeps the bookmarks' paths.
    let frame = own.clone();
    let mut roots = own;
    for heading in headings {
        // The path to the last bookmark (in outline order) opening at or before the heading's page.
        fn find(items: &[Outline], at: usize, path: &mut Vec<usize>, best: &mut Option<Vec<usize>>) {
            for (index, item) in items.iter().enumerate() {
                path.push(index);
                if item.page_index.is_some_and(|p| p <= at) { *best = Some(path.clone()); }
                find(&item.children, at, path, best);
                path.pop();
            }
        }
        let mut best = None;
        if let Some(at) = heading.page_index { find(&frame, at, &mut Vec::new(), &mut best); }
        match best {
            Some(path) => {
                let mut node = &mut roots[path[0]];
                for index in &path[1..] { node = &mut node.children[*index]; }
                node.children.push(heading);
            }
            None => roots.push(heading),
        }
    }
    fn by_page(items: &mut [Outline]) {
        if items.iter().all(|i| i.page_index.is_some()) { items.sort_by_key(|i| i.page_index); }
        for item in items { by_page(&mut item.children); }
    }
    by_page(&mut roots);
    roots
}

/// A document's own bookmarks, titled, with the pages they open (shifted by `offset`): each keeps its view unless
/// it fits the page; a bookmark to an inert place outside the document keeps its action; any other gives way to
/// its children.
pub fn read_outline(bytes: &[u8], offset: usize) -> Result<Vec<Outline>> {
    let document = Document::load_mem(bytes).map_err(lo)?;
    Ok(outline_of(&document, offset))
}

fn outline_of(document: &Document, offset: usize) -> Vec<Outline> {
    let pages: HashMap<ObjectId, usize> = document.get_pages().into_iter().map(|(n, id)| (id, n as usize - 1)).collect();
    let first = document.catalog().ok().and_then(|c| document.dereference(c.get(b"Outlines").ok()?).ok())
        .and_then(|(_, o)| o.as_dict().ok()).and_then(|o| o.get(b"First").ok());
    fn branch(document: &Document, first: Option<&Object>, pages: &HashMap<ObjectId, usize>, offset: usize, seen: &mut BTreeSet<ObjectId>) -> Vec<Outline> {
        let mut result = Vec::new();
        let mut next = first;
        while let Some(object) = next {
            if object.as_reference().is_ok_and(|id| !seen.insert(id)) || seen.len() > 10_000 { break; }
            let Some(node) = document.dereference(object).ok().and_then(|(_, o)| o.as_dict().ok()) else { break };
            next = node.get(b"Next").ok();
            let children = branch(document, node.get(b"First").ok(), pages, offset, seen);
            let title = node.get(b"Title").ok().and_then(|t| document.dereference(t).ok()).and_then(|(_, t)| lopdf::decode_text_string(t).ok());
            let action = node.get(b"A").ok().and_then(|a| document.dereference(a).ok()).and_then(|(_, a)| a.as_dict().ok());
            let goto = action.is_some_and(|a| a.get(b"S").and_then(Object::as_name).is_ok_and(|s| s == b"GoTo"));
            let dest = node.get(b"Dest").ok().or_else(|| action.filter(|_| goto).and_then(|a| a.get(b"D").ok()))
                .and_then(|d| destination_array(document, d));
            let page = dest.and_then(|d| d.first()?.as_reference().ok()).and_then(|id| pages.get(&id));
            match (title, page) {
                (Some(title), Some(page)) => {
                    let view = dest.map(|d| d[1..].to_vec()).filter(|v| !(v.len() == 1 && v[0].as_name().is_ok_and(|n| n == b"Fit")));
                    result.push(Outline { title, page_index: Some(offset + page), children, view, action: None });
                }
                (Some(title), None) if action.is_some_and(|a| !goto && inert(document, a)) => {
                    let mut action = action.expect("checked").clone();
                    action.remove(b"Next");
                    result.push(Outline { title, page_index: None, children, view: None, action: Some(action) });
                }
                _ => result.extend(children),
            }
        }
        result
    }
    branch(document, first, &pages, offset, &mut BTreeSet::new())
}

/// A page as links are placed on it: its crop box ([x, y, width, height]), its rotation, and its own links to web
/// addresses, each by its place in the page's annotations, with its address and rect ([x1, y1, x2, y2]).
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct PageFrame { pub crop: [f64; 4], pub rotation: i64, pub links: Vec<(usize, String, [f64; 4])> }

/// Each page of a PDF as links are placed on it: how many pages it has, how big, and the links it carries.
pub fn page_frames(bytes: &[u8]) -> Result<Vec<PageFrame>> {
    let document = Document::load_mem(bytes).map_err(lo)?;
    if document.is_encrypted() { return Err(fail("PDF is encrypted.")); }
    fn deref<'a>(document: &'a Document, o: &'a Object) -> Option<&'a Object> { document.dereference(o).ok().map(|(_, o)| o) }
    let get = |o| deref(&document, o);
    document.get_pages().into_values().map(|id| {
        let (crop, rotation) = crate::annotate::page_frame(&document, id)?;
        let annots = document.get_dictionary(id).ok().and_then(|p| get(p.get(b"Annots").ok()?)).and_then(|a| a.as_array().ok());
        let links = annots.into_iter().flatten().enumerate().filter_map(|(index, item)| {
            let annotation = get(item)?.as_dict().ok()?;
            let uri = lopdf::decode_text_string(get(get(annotation.get(b"A").ok()?)?.as_dict().ok()?.get(b"URI").ok()?)?).ok()?;
            let rect: Vec<f64> = get(annotation.get(b"Rect").ok()?)?.as_array().ok()?.iter()
                .map(|v| get(v).and_then(|v| v.as_float().ok().map(f64::from).or_else(|| v.as_i64().ok().map(|n| n as f64)))).collect::<Option<_>>()?;
            (rect.len() == 4).then(|| (index, uri, [rect[0].min(rect[2]), rect[1].min(rect[3]), rect[0].max(rect[2]), rect[1].max(rect[3])]))
        }).collect();
        Ok(PageFrame { crop, rotation, links })
    }).collect()
}

/// A page label rule: its numbering style (D, R, r, A, a; none for a prefix alone), prefix and first number.
#[derive(Clone, Debug, PartialEq)]
pub struct Label { pub style: Option<String>, pub prefix: Option<String>, pub start: i64 }

impl Label {
    pub fn decimal(start: i64) -> Self { Self { style: Some("D".into()), prefix: None, start } }
    pub fn prefix(prefix: impl Into<String>, start: Option<i64>) -> Self {
        Self { style: start.map(|_| "D".into()), prefix: Some(prefix.into()), start: start.unwrap_or(1) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Position { TopRight, TopCentre, BottomRight, BottomCentre }

/// Where a link takes the reader: a page shown whole, a page at a height, or a web address.
pub enum Target { Page(usize), At(usize, f64), Uri(String) }

pub struct Meta<'a> { pub title: &'a str, pub subject: &'a str, pub creator: &'a str, pub producer: &'a str, pub lang: Option<&'a str>, pub xmp: bool }

struct FontSlot { face: Face, id: ObjectId, name: String, base: String, embedded: Option<(&'static [u8], BTreeMap<u16, char>)> }

#[derive(Default)]
struct Page { id: ObjectId, ops: Vec<u8>, fonts: BTreeSet<usize>, xobjects: Vec<(String, ObjectId)>, states: Vec<(String, f64)>, annots: Vec<ObjectId>, frame: ([f64; 4], i64) }

struct StampSet { header: bool, right: bool, base: String, size: f64, inset: f64, offset: f64, text: Option<String>, settings: ObjectId, pages: Vec<(usize, i64)> }

/// A PDF being written.
pub struct Composer {
    doc: Document,
    root: ObjectId,
    pages: Vec<Page>,
    fonts: Vec<FontSlot>,
    labels: Vec<Label>,
    labels_written: bool,
    fields: Vec<ObjectId>,
    field_fonts: Dictionary,
    layers: Vec<ObjectId>,
    layer: Option<ObjectId>,
    stamps: Vec<StampSet>,
    catalog: Dictionary,
    outline: Option<(Vec<Outline>, bool)>,
    page_labels: Option<Vec<(usize, Label)>>,
}

fn fmt(value: f64) -> String {
    let rounded = (value * 1_000_000.0).round() / 1_000_000.0;
    let text = format!("{rounded:.6}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" || text.is_empty() { "0".into() } else { text.into() }
}
fn real(value: f64) -> Object { Object::Real(value as f32) }
fn rect_object(r: [f64; 4]) -> Object { Object::Array(r.iter().map(|v| real(*v)).collect()) }
/// A text string as UTF-16BE with a byte order mark.
pub fn text_string(text: &str) -> Object {
    let mut bytes = vec![0xFE, 0xFF];
    bytes.extend(text.encode_utf16().flat_map(u16::to_be_bytes));
    Object::String(bytes, StringFormat::Hexadecimal)
}
fn literal(text: &str) -> Object { Object::String(text.as_bytes().to_vec(), StringFormat::Literal) }
const EPOCH: &str = "D:19700101000000Z";

impl Default for Composer { fn default() -> Self { Self::new() } }

impl Composer {
    pub fn new() -> Self {
        let mut doc = Document::with_version("1.7");
        let root = doc.new_object_id();
        Self { doc, root, pages: Vec::new(), fonts: Vec::new(), labels: Vec::new(), labels_written: false, fields: Vec::new(),
            field_fonts: Dictionary::new(), layers: Vec::new(), layer: None, stamps: Vec::new(), catalog: Dictionary::new(),
            outline: None, page_labels: None }
    }

    /// A composer that starts as the PDF `bytes`: its pages, and what its catalog holds beside them (its tags,
    /// language, viewer preferences); what it would run is taken out.
    pub fn open(bytes: &[u8]) -> Result<Self> {
        let mut composer = Self::new();
        let catalog = composer.append(bytes, None)?.1;
        composer.labels_written = catalog.has(b"PageLabels");
        for (key, value) in catalog.iter() {
            if !matches!(key.as_slice(), b"Type" | b"Pages" | b"Outlines" | b"Names" | b"OpenAction" | b"AA" | b"PageLabels" | b"AcroForm" | b"OCProperties" | b"Metadata" | b"Dests" | b"PageMode") {
                composer.catalog.set(key.clone(), value.clone());
            }
        }
        Ok(composer)
    }

    pub fn page_count(&self) -> usize { self.pages.len() }
    /// The page's crop box ([x, y, width, height]) and its rotation.
    pub fn frame(&self, page: usize) -> ([f64; 4], i64) { self.pages[page].frame }
    /// The top of `rect` (page fractions from the top left of the page as displayed) on the page, in its user space.
    pub fn top(&self, page: usize, rect: [f64; 4]) -> Option<f64> {
        let (crop, rotation) = self.frame(page);
        crate::annotate::quad(rect, crop, rotation).ok().map(|q| q[1].max(q[3]).max(q[5]).max(q[7]))
    }

    /// The font for `face`, embedded once.
    pub fn font(&mut self, face: Face) -> usize {
        if let Some(index) = self.fonts.iter().position(|f| f.face == face) { return index; }
        let id = self.doc.new_object_id();
        let embedded = if face.arial() { system_arial(face).filter(|b| ttf_parser::Face::parse(b, 0).is_ok()).map(|b| (b, BTreeMap::new())) } else { None };
        // Named as pdf-lib names a font in a page's resources: its PostScript name and a number.
        self.fonts.push(FontSlot { face, id, name: format!("{}-{}", face.base_name(), self.fonts.len() + 1), base: face.base_name().into(), embedded });
        self.fonts.len() - 1
    }
    pub fn face(&self, font: usize) -> Face { self.fonts[font].face }
    pub fn width(&self, font: usize, text: &str, size: f64) -> f64 { self.fonts[font].face.width(text, size) }

    /// The text as the font's codes: WinAnsi bytes, or an embedded font's glyphs.
    fn encode(&mut self, font: usize, text: &str) -> String {
        let slot = &mut self.fonts[font];
        let mut hex = String::new();
        match &mut slot.embedded {
            Some((bytes, used)) => {
                let face = ttf_parser::Face::parse(bytes, 0).expect("parsed when embedded");
                for c in text.chars() {
                    let glyph = face.glyph_index(c).map_or(0, |g| g.0);
                    used.entry(glyph).or_insert(c);
                    let _ = write!(hex, "{glyph:04X}");
                }
            }
            None => for c in text.chars() { let _ = write!(hex, "{:02X}", win_ansi(c).unwrap_or(b'?')); },
        }
        format!("<{hex}>")
    }

    pub fn add_page(&mut self, width: f64, height: f64) -> usize {
        let id = self.doc.add_object(dictionary! { "Type" => "Page", "Parent" => self.root,
            "MediaBox" => vec![0.into(), 0.into(), real(width), real(height)], "Resources" => Dictionary::new() });
        self.pages.push(Page { id, frame: ([0.0, 0.0, width, height], 0), ..Page::default() });
        self.pages.len() - 1
    }

    fn ops(&mut self, page: usize, text: &str) { let ops = &mut self.pages[page].ops; ops.extend_from_slice(text.as_bytes()); ops.push(b'\n'); }

    fn show(&mut self, font: usize, text: &str, size: f64, x: f64, y: f64, rgb: Rgb) -> String {
        let codes = self.encode(font, text);
        format!("BT /{} {} Tf {} {} {} rg 1 0 0 1 {} {} Tm {codes} Tj ET", self.fonts[font].name, fmt(size), fmt(rgb[0]), fmt(rgb[1]), fmt(rgb[2]), fmt(x), fmt(y))
    }

    pub fn text(&mut self, page: usize, font: usize, text: &str, x: f64, y: f64, size: f64, rgb: Rgb) {
        if text.is_empty() { return; }
        let shown = self.show(font, text, size, x, y, rgb);
        self.pages[page].fonts.insert(font);
        self.ops(page, &format!("q {shown} Q"));
    }

    pub fn rect(&mut self, page: usize, r: [f64; 4], fill: Option<Rgb>, border: Option<(Rgb, f64)>, opacity: Option<f64>) {
        let mut ops = String::from("q ");
        if let Some(opacity) = opacity {
            let name = format!("BvG{}", self.pages[page].states.len() + 1);
            let _ = write!(ops, "/{name} gs ");
            self.pages[page].states.push((name, opacity));
        }
        if let Some(c) = fill { let _ = write!(ops, "{} {} {} rg ", fmt(c[0]), fmt(c[1]), fmt(c[2])); }
        if let Some((c, width)) = border { let _ = write!(ops, "{} {} {} RG {} w ", fmt(c[0]), fmt(c[1]), fmt(c[2]), fmt(width)); }
        let paint = match (fill.is_some(), border.is_some()) { (true, true) => "B", (true, false) => "f", _ => "S" };
        let _ = write!(ops, "{} {} {} {} re {paint} Q", fmt(r[0]), fmt(r[1]), fmt(r[2]), fmt(r[3]));
        self.ops(page, &ops);
    }

    pub fn line(&mut self, page: usize, from: [f64; 2], to: [f64; 2], width: f64, rgb: Rgb) {
        let ops = format!("q {} {} {} RG {} w {} {} m {} {} l S Q", fmt(rgb[0]), fmt(rgb[1]), fmt(rgb[2]), fmt(width), fmt(from[0]), fmt(from[1]), fmt(to[0]), fmt(to[1]));
        self.ops(page, &ops);
    }

    pub fn link(&mut self, page: usize, rect: [f64; 4], target: Target) {
        let mut annotation = dictionary! { "Type" => "Annot", "Subtype" => "Link", "Rect" => rect_object(rect), "Border" => vec![0.into(), 0.into(), 0.into()] };
        match target {
            Target::Page(at) => annotation.set("Dest", vec![self.pages[at].id.into(), "Fit".into()]),
            Target::At(at, top) => annotation.set("Dest", vec![self.pages[at].id.into(), "XYZ".into(), Object::Null, real(top), Object::Null]),
            // A URI is 7-bit ASCII (ISO 32000 12.6.4.7): its characters past ASCII and the string's delimiters are percent-encoded.
            Target::Uri(uri) => {
                let mut encoded = String::new();
                for c in uri.chars() {
                    if matches!(c, '(' | ')' | '\\') || !('\x21'..='\x7e').contains(&c) {
                        let mut buffer = [0; 4];
                        for byte in c.encode_utf8(&mut buffer).bytes() { let _ = write!(encoded, "%{byte:02X}"); }
                    } else { encoded.push(c); }
                }
                let action = self.doc.add_object(dictionary! { "Type" => "Action", "S" => "URI", "URI" => literal(&encoded) });
                annotation.set("A", action);
            }
        }
        let id = self.doc.add_object(annotation);
        self.pages[page].annots.push(id);
    }

    /// The page's own links to `uri`, taken to a page of this document, or taken out.
    pub fn relink(&mut self, page: usize, uri: &str, target: Option<Target>) -> Result<()> {
        let id = self.pages[page].id;
        let annots = self.doc.get_dictionary(id).map_err(lo)?.get(b"Annots").ok().cloned();
        let list = match annots { Some(Object::Reference(r)) => self.doc.get_object(r).map_err(lo)?.as_array().map_err(lo)?.clone(),
            Some(Object::Array(items)) => items, _ => return Ok(()) };
        let dest: Option<Vec<Object>> = match target {
            None => None,
            Some(Target::Page(at)) => Some(vec![self.pages[at].id.into(), "Fit".into()]),
            Some(Target::At(at, top)) => Some(vec![self.pages[at].id.into(), "XYZ".into(), Object::Null, real(top), Object::Null]),
            Some(Target::Uri(_)) => return Err(fail("A link is retargeted within the document.")),
        };
        let mut kept = Vec::new();
        for item in list {
            let linked = item.as_reference().ok().filter(|r| {
                let action = self.doc.get_dictionary(*r).ok().and_then(|a| self.doc.dereference(a.get(b"A").ok()?).ok()).and_then(|(_, a)| a.as_dict().ok());
                action.and_then(|a| lopdf::decode_text_string(a.get(b"URI").ok()?).ok()).is_some_and(|u| u == uri)
            });
            match (linked, &dest) {
                (Some(_), None) => continue,
                (Some(annotation), Some(dest)) => { let a = self.doc.get_dictionary_mut(annotation).map_err(lo)?; a.remove(b"A"); a.set("Dest", dest.clone()); }
                _ => {}
            }
            kept.push(item);
        }
        self.doc.get_dictionary_mut(id).map_err(lo)?.set("Annots", kept);
        Ok(())
    }
    /// A scan's recognized text as an invisible text layer (render mode 3): found by search and selection, never seen.
    pub fn ocr_text(&mut self, page: usize, font: usize, value: &str) {
        use unicode_normalization::UnicodeNormalization;
        if value.trim().is_empty() { return; }
        let text: String = value.nfc().map(|c| if c == '\t' { ' ' } else { c }).take(60_000).collect();
        let lines: Vec<&str> = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).filter(|l| !l.trim().is_empty()).collect();
        let [x, y, width, height] = self.pages[page].frame.0;
        let longest = lines.iter().map(|l| l.encode_utf16().count()).max().unwrap_or(0) as f64;
        let size = 1f64.min(width / (longest + 2.0)).min(height / (lines.len() as f64 + 2.0));
        let mut ops = format!("q BT /{} {} Tf 3 Tr {} TL 1 0 0 1 {} {} Tm", self.fonts[font].name, fmt(size), fmt(size), fmt(x + size), fmt(y + height - size));
        for line in lines { let codes = self.encode(font, line); let _ = write!(ops, " {codes} Tj T*"); }
        ops.push_str(" ET Q");
        self.pages[page].fonts.insert(font);
        self.ops(page, &ops);
    }

    /// Draws a page number, or a running line, as Acrobat draws its own headers and footers: a pagination artifact
    /// showing a form on the Headers/Footers layer, which carries the settings Acrobat's Header & Footer > Update and
    /// Remove work from. Returns where the line lies on the page ([x1, y1, x2, y2]), for a link over it.
    #[allow(clippy::too_many_arguments)]
    pub fn stamp(&mut self, page: usize, text: &str, number: Option<i64>, font: usize, position: Position, size: f64, inset: f64, offset: f64) -> [f64; 4] {
        let width = self.width(font, text, size);
        let ([cx, cy, cw, ch], rotation) = self.pages[page].frame;
        let angle = rotation.rem_euclid(360);
        let (w, h) = if angle == 90 || angle == 270 { (ch, cw) } else { (cw, ch) };
        let (header, right) = (matches!(position, Position::TopRight | Position::TopCentre), matches!(position, Position::TopRight | Position::BottomRight));
        let x = if right { w - inset - width } else { (w - width) / 2.0 };
        let y = if header { h - offset } else { offset };
        let (px, py) = match angle { 90 => (cw - y, x), 180 => (cw - x, ch - y), 270 => (y, ch - x), _ => (x, y) };
        let base = self.fonts[font].base.clone();
        let set = match self.stamps.iter().position(|s| s.header == header && s.right == right && s.base == base && s.size == size
            && s.inset == inset && s.offset == offset && s.text.as_deref() == number.is_none().then_some(text)) {
            Some(index) => index,
            None => {
                let settings = self.doc.new_object_id();
                self.stamps.push(StampSet { header, right, base, size, inset, offset, text: number.is_none().then(|| text.to_owned()), settings, pages: Vec::new() });
                self.stamps.len() - 1
            }
        };
        self.stamps[set].pages.push((page, number.unwrap_or(0)));
        let settings = self.stamps[set].settings;
        let layer = self.header_footer_layer();
        let shown = self.show(font, text, size, 0.0, 0.0, [0.12, 0.12, 0.12]);
        let kind = if header { "Header" } else { "Footer" };
        let form = self.doc.add_object(Stream::new(dictionary! { "Type" => "XObject", "Subtype" => "Form",
            "BBox" => vec![0.into(), real(-size / 4.0), real(width), real(size)], "Matrix" => vec![1.into(), 0.into(), 0.into(), 1.into(), 0.into(), 0.into()],
            "LastModified" => literal(EPOCH), "Resources" => dictionary! { "Font" => dictionary! { self.fonts[font].name.clone() => self.fonts[font].id } },
            "OC" => dictionary! { "Type" => "OCMD", "OCGs" => layer },
            "PieceInfo" => dictionary! { "ADBE_CompoundType" => dictionary! { "DocSettings" => settings, "LastModified" => literal(EPOCH), "Private" => kind } },
        }, shown.into_bytes()));
        let name = format!("BvX{}", self.pages[page].xobjects.len() + 1);
        self.pages[page].xobjects.push((name.clone(), form));
        let radians = (angle as f64).to_radians();
        let (cos, sin) = (radians.cos().round(), radians.sin().round());
        let contents = text.encode_utf16().fold(String::from("<FEFF"), |mut s, u| { let _ = write!(s, "{u:04X}"); s }) + ">";
        let ops = format!("/Artifact <</Contents {contents} /Type /Pagination /Subtype /{kind} /Attached [/{}]>> BDC q {} {} {} {} {} {} cm /{name} Do Q EMC",
            if header { "Top" } else { "Bottom" }, fmt(cos), fmt(sin), fmt(-sin), fmt(cos), fmt(cx + px), fmt(cy + py));
        self.ops(page, &ops);
        let corners = [(0.0, -size / 4.0), (width, size)].map(|(u, v)| (cx + px + cos * u - sin * v, cy + py + sin * u + cos * v));
        [corners[0].0.min(corners[1].0), corners[0].1.min(corners[1].1), corners[0].0.max(corners[1].0), corners[0].1.max(corners[1].1)]
    }

    fn header_footer_layer(&mut self) -> ObjectId {
        if let Some(layer) = self.layer { return layer; }
        let layer = self.doc.add_object(dictionary! { "Type" => "OCG", "Name" => literal("Headers/Footers"), "Usage" => dictionary! { "PageElement" => dictionary! { "Subtype" => "HF" } } });
        self.layers.push(layer);
        self.layer = Some(layer);
        layer
    }

    /// A value as a text field Acrobat fills in: it shows its `lines` (text and page position) as the page would draw
    /// them, and Acrobat edits it in `rect` ([x, y, width, height]). Flattened, the lines are drawn and nothing is fillable.
    #[allow(clippy::too_many_arguments)]
    pub fn field(&mut self, page: usize, name: &str, lines: &[(String, f64, f64)], rect: [f64; 4], font: usize, size: f64, centre: bool, rgb: Rgb, flatten: bool) {
        if flatten {
            for (text, x, y) in lines { self.text(page, font, text, *x, *y, size, rgb); }
            return;
        }
        let mut shown = String::from("/Tx BMC");
        for (text, x, y) in lines { let s = self.show(font, text, size, x - rect[0], y - rect[1], rgb); let _ = write!(shown, " q {s} Q"); }
        shown.push_str(" EMC");
        let (font_name, font_id) = (self.fonts[font].name.clone(), self.fonts[font].id);
        let appearance = self.doc.add_object(Stream::new(dictionary! { "Type" => "XObject", "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), real(rect[2]), real(rect[3])], "Resources" => dictionary! { "Font" => dictionary! { font_name.clone() => font_id } } }, shown.into_bytes()));
        let multiline = lines.len() > 1 || rect[3] > size * 1.6;
        let value = lines.iter().map(|(t, ..)| t.as_str()).collect::<Vec<_>>().join("\n");
        let id = self.doc.add_object(dictionary! { "Type" => "Annot", "Subtype" => "Widget", "FT" => "Tx", "T" => text_string(name),
            "V" => text_string(&value), "DA" => literal(&format!("/{font_name} {} Tf {} {} {} rg", fmt(size), fmt(rgb[0]), fmt(rgb[1]), fmt(rgb[2]))),
            "Ff" => if multiline { 4096i64 } else { 0 }, "Q" => i64::from(centre), "F" => 4, "P" => self.pages[page].id,
            "Rect" => rect_object([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]), "MK" => Dictionary::new(),
            "AP" => dictionary! { "N" => appearance } });
        self.field_fonts.set(font_name, font_id);
        self.fields.push(id);
        self.pages[page].annots.push(id);
    }

    /// Copies the pages of the PDF `bytes` (`pages`, zero-based, or all) after the document's own; returns their
    /// indices here and the source's catalog. Links between copied pages keep working, a link to a page not copied
    /// goes, form fields stay fillable, Headers/Footers layers stay Acrobat's, page labels carry over, and what the
    /// source would run, launch, submit or fetch is taken out.
    pub fn append(&mut self, bytes: &[u8], pages: Option<&[usize]>) -> Result<(Vec<usize>, Dictionary)> {
        let mut source = Document::load_mem(bytes).map_err(lo)?;
        if source.is_encrypted() { return Err(fail("An encrypted PDF cannot be assembled.")); }
        source.renumber_objects_with(self.doc.max_id + 1);
        let all: Vec<ObjectId> = source.get_pages().into_values().collect();
        let indices: Vec<usize> = pages.map_or_else(|| (0..all.len()).collect(), <[usize]>::to_vec);
        if let Some(bad) = indices.iter().find(|i| **i >= all.len()) { return Err(fail(format!("The PDF has no page {}.", bad + 1))); }
        // The labels the source's pages carry, and the document's own continued to the pages drawn since.
        let rules = label_rules(&source).unwrap_or_default();
        while self.labels.len() < self.pages.len() {
            let next = self.labels.last().map_or(Label::decimal(self.labels.len() as i64 + 1), |l| Label { start: l.start + 1, ..l.clone() });
            self.labels.push(next);
        }
        for index in &indices {
            self.labels.push(match rules.range(..=*index).next_back() {
                None => Label::decimal(*index as i64 + 1),
                Some((first, spec)) => Label {
                    style: spec.get(b"S").and_then(Object::as_name).ok().map(|s| String::from_utf8_lossy(s).into_owned()),
                    prefix: spec.get(b"P").ok().and_then(|p| lopdf::decode_text_string(p).ok()),
                    start: spec.get(b"St").and_then(Object::as_i64).unwrap_or(1) + (*index - first) as i64,
                },
            });
        }
        self.labels_written = true;
        let catalog = source.catalog().cloned().unwrap_or_default();
        let kept: BTreeSet<ObjectId> = indices.iter().map(|i| all[*i]).collect();
        let mut edits: Vec<(ObjectId, Object)> = Vec::new();
        let mut copied = Vec::new();
        for index in &indices {
            let id = all[*index];
            let mut page = source.get_dictionary(id).map_err(lo)?.clone();
            // What a page inherits from its tree is its own once it leaves it.
            let mut parent = page.get(b"Parent").and_then(Object::as_reference).ok();
            for _ in 0..64 {
                let Some(node) = parent.and_then(|p| source.get_dictionary(p).ok()) else { break };
                for key in [&b"Resources"[..], b"MediaBox", b"CropBox", b"Rotate"] {
                    if !page.has(key) { if let Ok(value) = node.get(key) { page.set(key.to_vec(), value.clone()); } }
                }
                parent = node.get(b"Parent").and_then(Object::as_reference).ok();
            }
            page.set("Parent", self.root);
            page.remove(b"AA");
            let resources = page.get(b"Resources").ok().and_then(|r| source.dereference(r).ok()).and_then(|(_, r)| r.as_dict().ok()).cloned().unwrap_or_default();
            // Headers/Footers layers the page's forms are on.
            let xobjects = resources.get(b"XObject").ok().and_then(|x| source.dereference(x).ok()).and_then(|(_, x)| x.as_dict().ok());
            for (_, form) in xobjects.into_iter().flat_map(Dictionary::iter) {
                let layer = form.as_reference().ok().and_then(|f| source.get_object(f).ok()).and_then(|f| f.as_stream().ok())
                    .and_then(|f| f.dict.get(b"OC").ok()).and_then(|oc| source.dereference(oc).ok()).and_then(|(_, oc)| oc.as_dict().ok())
                    .and_then(|oc| oc.get(b"OCGs").and_then(Object::as_reference).ok());
                if let Some(layer) = layer { if !self.layers.contains(&layer) { self.layers.push(layer); } }
            }
            let annots = page.get(b"Annots").ok().and_then(|a| source.dereference(a).ok()).and_then(|(_, a)| a.as_array().ok()).cloned().unwrap_or_default();
            let mut kept_annots = Vec::new();
            for item in annots {
                let Some(annotation_id) = item.as_reference().ok() else { kept_annots.push(item); continue };
                let Ok(mut annotation) = source.get_dictionary(annotation_id).cloned() else { continue };
                let action = annotation.get(b"A").ok().and_then(|a| source.dereference(a).ok()).and_then(|(_, a)| a.as_dict().ok()).cloned();
                let goto = action.as_ref().is_some_and(|a| a.get(b"S").and_then(Object::as_name).is_ok_and(|s| s == b"GoTo"));
                if annotation.has(b"Dest") || goto {
                    let dest = annotation.get(b"Dest").ok().or_else(|| action.as_ref().and_then(|a| a.get(b"D").ok()))
                        .and_then(|d| destination_array(&source, d)).map(<[Object]>::to_vec);
                    let Some(dest) = dest.filter(|d| d.first().and_then(|p| p.as_reference().ok()).is_some_and(|p| kept.contains(&p))) else { continue };
                    if annotation.has(b"Dest") { annotation.set("Dest", dest); } else if let Some(mut action) = action { action.set("D", dest); annotation.set("A", action); }
                }
                // A widget's field, and the fields above it, carry triggers of their own.
                disarm_dict(&source, &mut annotation, b"A");
                if annotation.get(b"Subtype").and_then(Object::as_name).is_ok_and(|s| s == b"Widget") {
                    let mut field = annotation_id;
                    while let Some(up) = source.get_dictionary(field).ok().and_then(|f| f.get(b"Parent").and_then(Object::as_reference).ok()) {
                        if let Ok(mut node) = source.get_dictionary(up).cloned() { disarm_dict(&source, &mut node, b"A"); edits.push((up, Object::Dictionary(node))); }
                        field = up;
                        if edits.len() > 100_000 { break; }
                    }
                    if !self.fields.contains(&field) { self.fields.push(field); }
                    let fonts = catalog.get(b"AcroForm").ok().and_then(|f| source.dereference(f).ok()).and_then(|(_, f)| f.as_dict().ok())
                        .and_then(|f| f.get(b"DR").ok()).and_then(|r| source.dereference(r).ok()).and_then(|(_, r)| r.as_dict().ok())
                        .and_then(|r| r.get(b"Font").ok()).and_then(|f| source.dereference(f).ok()).and_then(|(_, f)| f.as_dict().ok()).cloned();
                    for (name, font) in fonts.iter().flat_map(Dictionary::iter) { if !self.field_fonts.has(name) { self.field_fonts.set(name.clone(), font.clone()); } }
                }
                edits.push((annotation_id, Object::Dictionary(annotation)));
                kept_annots.push(item);
            }
            if page.has(b"Annots") { page.set("Annots", kept_annots); }
            edits.push((id, Object::Dictionary(page.clone())));
            let frame = page_frame(&page, &source);
            copied.push(Page { id, frame, ..Page::default() });
        }
        for (id, object) in edits { source.objects.insert(id, object); }
        self.doc.max_id = self.doc.max_id.max(source.max_id);
        self.doc.objects.extend(source.objects);
        let start = self.pages.len();
        self.pages.extend(copied);
        Ok(((start..self.pages.len()).collect(), catalog))
    }

    /// The outline the document saves with, in place of any other; `open` shows the bookmarks as it opens.
    pub fn set_outline(&mut self, outline: Vec<Outline>, open: bool) { self.outline = Some((outline, open)); }
    /// The document's page labels, each from the page it starts at, in place of those its pages carried.
    pub fn set_labels(&mut self, labels: Vec<(usize, Label)>) { self.page_labels = Some(labels); }

    fn write_outline(&mut self, items: &[Outline], parent: ObjectId) -> Option<(ObjectId, ObjectId, i64)> {
        let valid: Vec<&Outline> = items.iter().filter(|i| i.action.is_some() || i.page_index.is_some_and(|p| p < self.pages.len())).collect();
        if valid.is_empty() { return None; }
        let ids: Vec<ObjectId> = valid.iter().map(|_| self.doc.new_object_id()).collect();
        let mut count = 0;
        for (index, item) in valid.iter().enumerate() {
            let mut node = dictionary! { "Title" => text_string(&item.title), "Parent" => parent };
            match (&item.action, item.page_index) {
                (Some(action), _) => node.set("A", action.clone()),
                (None, Some(page)) => {
                    let mut dest = vec![Object::Reference(self.pages[page].id)];
                    dest.extend(item.view.clone().unwrap_or_else(|| vec!["Fit".into()]));
                    node.set("Dest", dest);
                }
                _ => {}
            }
            if index > 0 { node.set("Prev", ids[index - 1]); }
            if index + 1 < ids.len() { node.set("Next", ids[index + 1]); }
            count += 1;
            if let Some((first, last, below)) = self.write_outline(&item.children, ids[index]) {
                node.set("First", first); node.set("Last", last); node.set("Count", below);
                count += below;
            }
            self.doc.objects.insert(ids[index], Object::Dictionary(node));
        }
        Some((ids[0], *ids.last().expect("valid"), count))
    }

    /// The PDF's bytes: its catalog, information and metadata saying `meta`, and nothing in it that nothing refers to.
    pub fn save(mut self, meta: &Meta, object_streams: bool) -> Result<Vec<u8>> {
        let mut catalog = std::mem::take(&mut self.catalog);
        catalog.set("Type", "Catalog");
        catalog.set("Pages", self.root);
        // Each page's drawing, after its own (kept as it was).
        for index in 0..self.pages.len() {
            let page = std::mem::take(&mut self.pages[index]);
            let mut dict = self.doc.get_dictionary(page.id).map_err(lo)?.clone();
            if !page.ops.is_empty() {
                let mut resources = dict.get(b"Resources").ok().and_then(|r| self.doc.dereference(r).ok()).and_then(|(_, r)| r.as_dict().ok()).cloned().unwrap_or_default();
                let mut merge = |kind: &[u8], entries: Vec<(String, Object)>, doc: &Document| {
                    let mut inner = resources.get(kind).ok().and_then(|r| doc.dereference(r).ok()).and_then(|(_, r)| r.as_dict().ok()).cloned().unwrap_or_default();
                    for (name, value) in entries { inner.set(name, value); }
                    if !inner.is_empty() { resources.set(kind.to_vec(), inner); }
                };
                merge(b"Font", page.fonts.iter().map(|f| (self.fonts[*f].name.clone(), self.fonts[*f].id.into())).collect(), &self.doc);
                merge(b"XObject", page.xobjects.iter().map(|(n, id)| (n.clone(), (*id).into())).collect(), &self.doc);
                merge(b"ExtGState", page.states.iter().map(|(n, o)| (n.clone(), Object::Dictionary(dictionary! { "Type" => "ExtGState", "ca" => real(*o), "CA" => 1 }))).collect(), &self.doc);
                dict.set("Resources", resources);
                let drawn = self.doc.add_object(Stream::new(Dictionary::new(), page.ops).with_compression(true));
                let contents = match dict.get(b"Contents").ok().cloned() {
                    None => vec![drawn.into()],
                    // What the page drew stays as it drew it: its state is restored before ours begins.
                    Some(own) => {
                        let open = self.doc.add_object(Stream::new(Dictionary::new(), b"q".to_vec()));
                        let close = self.doc.add_object(Stream::new(Dictionary::new(), b"Q".to_vec()));
                        let own = match self.doc.dereference(&own) { Ok((_, Object::Array(items))) => items.clone(), _ => vec![own] };
                        std::iter::once(open.into()).chain(own).chain([close.into(), drawn.into()]).collect()
                    }
                };
                dict.set("Contents", contents);
            }
            if !page.annots.is_empty() {
                let mut annots = dict.get(b"Annots").ok().and_then(|a| self.doc.dereference(a).ok()).and_then(|(_, a)| a.as_array().ok()).cloned().unwrap_or_default();
                annots.extend(page.annots.iter().map(|id| Object::Reference(*id)));
                dict.set("Annots", annots);
            }
            self.doc.objects.insert(page.id, Object::Dictionary(dict));
            self.pages[index].id = page.id;
        }
        let kids: Vec<Object> = self.pages.iter().map(|p| p.id.into()).collect();
        self.doc.objects.insert(self.root, Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => self.pages.len() as i64 }));
        self.write_fonts()?;
        // Each header and footer's settings, as Acrobat's Header & Footer dialog saves them.
        for set in std::mem::take(&mut self.stamps) {
            let mut placed = set.pages.clone();
            placed.sort();
            let (first, last) = (placed[0], placed[placed.len() - 1]);
            let escape = |v: &str| v.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
            let slot = set.text.as_deref().map_or_else(|| format!("<Page offset = \"{}\"><PageIndex format=\"1\"/></Page>", first.1 - 1), escape);
            let slots = |header: bool| ["Left", "Center", "Right"].map(|name| {
                let here = header == set.header && name == if set.right { "Right" } else { "Center" };
                format!("<{name}>{}</{name}>", if here { slot.as_str() } else { "" })
            }).concat();
            let margin = if set.header { (set.offset - set.size).max(0.0) } else { set.offset };
            let xml = format!("\u{feff}<?xml version = \"1.0\" encoding = \"UTF-8\" ?><HeaderFooterSettings version = \"8.0\"><Font underline=\"false\" type=\"Type1\" size=\"{:.1}\" name=\"{}\"/><Color r=\"0.12\" b=\"0.12\" g=\"0.12\"/><Margin left=\"{}\" right=\"{}\" top=\"{}\" bottom=\"{}\"/><Appearance shrink=\"0\" fixedprint=\"0\"/><PageRange end=\"{}\" start=\"{}\" even=\"1\" odd=\"1\"/><Page offset = \"0\"><PageIndex format=\"1\"/></Page><Date><Month format=\"1\"/>/<Day format=\"1\"/><Year format=\"0\"/></Date><Header>{}</Header><Footer>{}</Footer></HeaderFooterSettings>",
                set.size, set.base, fmt(set.inset), fmt(set.inset), fmt(if set.header { margin } else { 36.0 }), fmt(if set.header { 36.0 } else { margin }), last.0, first.0, slots(true), slots(false));
            self.doc.objects.insert(set.settings, Object::Stream(Stream::new(Dictionary::new(), xml.into_bytes())));
        }
        if !self.fields.is_empty() {
            let fields: Vec<Object> = self.fields.iter().map(|id| Object::Reference(*id)).collect();
            catalog.set("AcroForm", dictionary! { "Fields" => fields, "DR" => dictionary! { "Font" => std::mem::take(&mut self.field_fonts) } });
        }
        if !self.layers.is_empty() {
            let layers: Vec<Object> = self.layers.iter().map(|id| Object::Reference(*id)).collect();
            catalog.set("OCProperties", dictionary! { "OCGs" => layers, "D" => dictionary! { "Order" => Vec::<Object>::new(), "RBGroups" => Vec::<Object>::new() } });
        }
        let labels = self.page_labels.take().or_else(|| self.labels_written.then(|| {
            let mut runs: Vec<(usize, Label)> = Vec::new();
            for (index, label) in self.labels.iter().enumerate() {
                let continues = runs.last().is_some_and(|(at, run)| run.style == label.style && run.prefix == label.prefix
                    && (label.style.is_none() || label.start == run.start + (index - at) as i64));
                if !continues { runs.push((index, label.clone())); }
            }
            runs
        }));
        if let Some(labels) = labels {
            let nums: Vec<Object> = labels.into_iter().flat_map(|(at, label)| {
                let mut spec = Dictionary::new();
                if let Some(style) = &label.style { spec.set("S", Object::Name(style.clone().into_bytes())); spec.set("St", label.start); }
                if let Some(prefix) = &label.prefix { spec.set("P", text_string(prefix)); }
                [Object::Integer(at as i64), Object::Dictionary(spec)]
            }).collect();
            let id = self.doc.add_object(dictionary! { "Nums" => nums });
            catalog.set("PageLabels", id);
        }
        if let Some((outline, open)) = self.outline.take() {
            let root = self.doc.new_object_id();
            if let Some((first, last, count)) = self.write_outline(&outline, root) {
                self.doc.objects.insert(root, Object::Dictionary(dictionary! { "Type" => "Outlines", "First" => first, "Last" => last, "Count" => count }));
                catalog.set("Outlines", root);
                if open { catalog.set("PageMode", "UseOutlines"); }
            }
        }
        // Named as Acrobat shows it: its title in the title bar, and in both the information and the XMP metadata.
        // Dated the epoch, so a build of the same input is the same file.
        if meta.xmp {
            let xml = |v: &str| v.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
            let alt = |v: &str| format!("<rdf:Alt><rdf:li xml:lang=\"x-default\">{}</rdf:li></rdf:Alt>", xml(v));
            let packet = format!("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\" xmlns:pdf=\"http://ns.adobe.com/pdf/1.3/\"><dc:format>application/pdf</dc:format><dc:title>{}</dc:title><dc:description>{}</dc:description><xmp:CreatorTool>{}</xmp:CreatorTool><xmp:CreateDate>1970-01-01T00:00:00Z</xmp:CreateDate><xmp:ModifyDate>1970-01-01T00:00:00Z</xmp:ModifyDate><pdf:Producer>{}</pdf:Producer></rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>",
                alt(meta.title), alt(meta.subject), xml(meta.creator), xml(meta.producer));
            let id = self.doc.add_object(Stream::new(dictionary! { "Type" => "Metadata", "Subtype" => "XML" }, packet.into_bytes()));
            catalog.set("Metadata", id);
            let mut preferences = catalog.get(b"ViewerPreferences").ok().and_then(|v| self.doc.dereference(v).ok()).and_then(|(_, v)| v.as_dict().ok()).cloned().unwrap_or_default();
            preferences.set("DisplayDocTitle", true);
            catalog.set("ViewerPreferences", preferences);
        }
        if let Some(lang) = meta.lang { catalog.set("Lang", text_string(lang)); }
        let info = self.doc.add_object(dictionary! { "Title" => text_string(meta.title), "Subject" => text_string(meta.subject),
            "Creator" => text_string(meta.creator), "Producer" => text_string(meta.producer), "CreationDate" => literal(EPOCH), "ModDate" => literal(EPOCH) });
        let catalog_id = self.doc.add_object(catalog);
        self.doc.trailer = dictionary! { "Root" => catalog_id, "Info" => info };
        self.doc.prune_objects();
        self.doc.compress();
        let mut out = Vec::new();
        let options = SaveOptions::builder().use_object_streams(object_streams).use_xref_streams(object_streams).build();
        self.doc.save_with_options(&mut out, options).map_err(|e| fail(format!("PDF could not be saved: {e}")))?;
        Ok(out)
    }

    /// Each font's dictionary: a standard font by name, Arial named with its widths, or Arial's own file with only
    /// the glyphs the document shows.
    fn write_fonts(&mut self) -> Result<()> {
        for slot in std::mem::take(&mut self.fonts) {
            let face = slot.face;
            let (italic, bold) = (matches!(face, Face::ArialItalic | Face::ArialBoldItalic), matches!(face, Face::ArialBold | Face::ArialBoldItalic));
            let object = match slot.embedded {
                None if !face.arial() => dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => slot.base.clone(), "Encoding" => "WinAnsiEncoding" },
                None => {
                    let boxes = match face { Face::ArialBold => [-628, -376, 2000, 1056], Face::ArialItalic => [-517, -325, 1359, 998], Face::ArialBoldItalic => [-560, -376, 1390, 1018], _ => [-665, -325, 2000, 1040] };
                    let descriptor = self.doc.add_object(dictionary! { "Type" => "FontDescriptor", "FontName" => slot.base.clone(), "Flags" => 32 + if italic { 64 } else { 0 },
                        "FontBBox" => boxes.iter().map(|v| Object::Integer(*v)).collect::<Vec<_>>(), "ItalicAngle" => if italic { -12 } else { 0 },
                        "Ascent" => 905, "Descent" => -212, "CapHeight" => 716, "XHeight" => 519, "StemV" => if bold { 153 } else { 80 } });
                    let widths: Vec<Object> = face.table().expect("Arial has widths").iter().map(|w| Object::Integer(i64::from(*w))).collect();
                    dictionary! { "Type" => "Font", "Subtype" => "TrueType", "BaseFont" => slot.base.clone(), "FirstChar" => 32, "LastChar" => 255,
                        "Widths" => widths, "Encoding" => "WinAnsiEncoding", "FontDescriptor" => descriptor }
                }
                Some((bytes, used)) => {
                    let parsed = ttf_parser::Face::parse(bytes, 0).map_err(|e| fail(format!("Arial could not be read: {e}")))?;
                    let scale = 1000.0 / f64::from(parsed.units_per_em());
                    let mut glyphs: Vec<u16> = used.keys().copied().collect();
                    glyphs.insert(0, 0);
                    // A subset is named by a tag of six capitals before the font's own name, here drawn from its glyphs.
                    let tag: String = glyphs.iter().fold(5381u64, |h, g| h.wrapping_mul(33) ^ u64::from(*g)).to_string().bytes().rev().take(6).map(|b| char::from(b'A' + (b - b'0') * 2)).collect();
                    let name = format!("{tag}+{}", slot.base);
                    let file = self.doc.add_object(Stream::new(Dictionary::new(), subset_truetype(&parsed, &glyphs)?).with_compression(true));
                    let bbox = parsed.global_bounding_box();
                    let descriptor = self.doc.add_object(dictionary! { "Type" => "FontDescriptor", "FontName" => name.clone(), "Flags" => 32 + if italic { 64 } else { 0 },
                        "FontBBox" => [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max].iter().map(|v| Object::Integer((f64::from(*v) * scale).round() as i64)).collect::<Vec<_>>(),
                        "ItalicAngle" => real(f64::from(parsed.italic_angle())), "Ascent" => (f64::from(parsed.ascender()) * scale).round() as i64,
                        "Descent" => (f64::from(parsed.descender()) * scale).round() as i64, "CapHeight" => (f64::from(parsed.capital_height().unwrap_or(0)) * scale).round() as i64,
                        "StemV" => if bold { 153 } else { 80 }, "FontFile2" => file });
                    let widths: Vec<Object> = used.keys().flat_map(|g| [Object::Integer(i64::from(*g)),
                        Object::Array(vec![Object::Integer((f64::from(parsed.glyph_hor_advance(ttf_parser::GlyphId(*g)).unwrap_or(0)) * scale).round() as i64)])]).collect();
                    let cid = self.doc.add_object(dictionary! { "Type" => "Font", "Subtype" => "CIDFontType2", "BaseFont" => name.clone(),
                        "CIDSystemInfo" => dictionary! { "Registry" => literal("Adobe"), "Ordering" => literal("Identity"), "Supplement" => 0 },
                        "FontDescriptor" => descriptor, "W" => widths, "CIDToGIDMap" => "Identity" });
                    let mut cmap = String::from("/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /Adobe-Identity-UCS def /CMapType 2 def 1 begincodespacerange <0000> <FFFF> endcodespacerange\n");
                    for chunk in used.iter().collect::<Vec<_>>().chunks(100) {
                        let _ = writeln!(cmap, "{} beginbfchar", chunk.len());
                        for (glyph, c) in chunk {
                            let unicode: String = c.encode_utf16(&mut [0; 2]).iter().map(|u| format!("{u:04X}")).collect();
                            let _ = writeln!(cmap, "<{glyph:04X}> <{unicode}>");
                        }
                        cmap.push_str("endbfchar\n");
                    }
                    cmap.push_str("endcmap CMapName currentdict /CMap defineresource pop end end");
                    let unicode = self.doc.add_object(Stream::new(Dictionary::new(), cmap.into_bytes()).with_compression(true));
                    dictionary! { "Type" => "Font", "Subtype" => "Type0", "BaseFont" => name, "Encoding" => "Identity-H", "DescendantFonts" => vec![cid.into()], "ToUnicode" => unicode }
                }
            };
            self.doc.objects.insert(slot.id, Object::Dictionary(object));
        }
        Ok(())
    }
}

/// The page's crop box (falling back to its media box), as [x, y, width, height], and its rotation.
fn page_frame(page: &Dictionary, document: &Document) -> ([f64; 4], i64) {
    let number = |o: &Object| o.as_float().ok().map(f64::from).or_else(|| o.as_i64().ok().map(|v| v as f64));
    let rect = |key: &[u8]| page.get(key).ok().and_then(|o| document.dereference(o).ok()).and_then(|(_, o)| o.as_array().ok())
        .filter(|a| a.len() == 4).and_then(|a| a.iter().map(|v| document.dereference(v).ok().and_then(|(_, v)| number(v))).collect::<Option<Vec<_>>>());
    let r = rect(b"CropBox").or_else(|| rect(b"MediaBox")).unwrap_or_else(|| vec![0.0, 0.0, 612.0, 792.0]);
    let rotation = page.get(b"Rotate").ok().and_then(|o| document.dereference(o).ok()).and_then(|(_, o)| o.as_i64().ok()).unwrap_or(0);
    ([r[0].min(r[2]), r[1].min(r[3]), (r[2] - r[0]).abs(), (r[3] - r[1]).abs()], rotation)
}

/// A TrueType font with only `glyphs` (and the glyphs they are composed of) drawn: every other glyph is left empty, so
/// glyph ids stay as they are, and only the tables a PDF reader draws with are kept (ISO 32000 9.9).
fn subset_truetype(face: &ttf_parser::Face, glyphs: &[u16]) -> Result<Vec<u8>> {
    let raw = face.raw_face();
    let table = |tag: &[u8; 4]| raw.table(ttf_parser::Tag::from_bytes(tag));
    let (head, loca, glyf) = match (table(b"head"), table(b"loca"), table(b"glyf")) {
        (Some(h), Some(l), Some(g)) if h.len() >= 54 => (h, l, g),
        _ => return Err(fail("Arial's file has no TrueType outlines.")),
    };
    let long = i16::from_be_bytes([head[50], head[51]]) == 1;
    let count = usize::from(face.number_of_glyphs());
    let offset = |g: usize| -> Option<usize> {
        if long { loca.get(g * 4..g * 4 + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize) }
        else { loca.get(g * 2..g * 2 + 2).map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])) * 2) }
    };
    let data = |g: usize| -> &[u8] { match (offset(g), offset(g + 1)) { (Some(a), Some(b)) if a <= b && b <= glyf.len() => &glyf[a..b], _ => &[] } };
    let mut wanted: BTreeSet<usize> = glyphs.iter().map(|g| usize::from(*g)).filter(|g| *g < count).collect();
    let mut pending: Vec<usize> = wanted.iter().copied().collect();
    while let Some(g) = pending.pop() {
        let d = data(g);
        if d.len() < 10 || i16::from_be_bytes([d[0], d[1]]) >= 0 { continue; }
        let mut at = 10;
        while at + 4 <= d.len() {
            let flags = u16::from_be_bytes([d[at], d[at + 1]]);
            let component = usize::from(u16::from_be_bytes([d[at + 2], d[at + 3]]));
            if component < count && wanted.insert(component) { pending.push(component); }
            at += 4 + if flags & 1 != 0 { 4 } else { 2 } + if flags & 8 != 0 { 2 } else if flags & 0x40 != 0 { 4 } else if flags & 0x80 != 0 { 8 } else { 0 };
            if flags & 0x20 == 0 { break; }
        }
    }
    let (mut new_glyf, mut new_loca) = (Vec::new(), Vec::with_capacity((count + 1) * 4));
    for g in 0..count {
        new_loca.extend((new_glyf.len() as u32).to_be_bytes());
        if wanted.contains(&g) { new_glyf.extend_from_slice(data(g)); while new_glyf.len() % 4 != 0 { new_glyf.push(0); } }
    }
    new_loca.extend((new_glyf.len() as u32).to_be_bytes());
    let mut new_head = head.to_vec();
    new_head[8..12].fill(0);
    new_head[50..52].copy_from_slice(&1i16.to_be_bytes());
    let mut tables: Vec<([u8; 4], Vec<u8>)> = vec![(*b"glyf", new_glyf), (*b"head", new_head), (*b"loca", new_loca)];
    for tag in [b"cvt ", b"fpgm", b"hhea", b"hmtx", b"maxp", b"prep"] { if let Some(t) = table(tag) { tables.push((*tag, t.to_vec())); } }
    tables.sort_by(|a, b| a.0.cmp(&b.0));
    let checksum = |bytes: &[u8]| bytes.chunks(4).fold(0u32, |sum, c| { let mut w = [0; 4]; w[..c.len()].copy_from_slice(c); sum.wrapping_add(u32::from_be_bytes(w)) });
    let n = tables.len() as u16;
    let power = 1u16 << (15 - n.leading_zeros());
    let mut out = Vec::new();
    out.extend(0x0001_0000u32.to_be_bytes());
    for v in [n, power * 16, power.trailing_zeros() as u16, n * 16 - power * 16] { out.extend(v.to_be_bytes()); }
    let mut at = 12 + 16 * tables.len();
    for (tag, bytes) in &tables {
        out.extend(tag); out.extend(checksum(bytes).to_be_bytes()); out.extend((at as u32).to_be_bytes()); out.extend((bytes.len() as u32).to_be_bytes());
        at += bytes.len().div_ceil(4) * 4;
    }
    for (_, bytes) in &tables { out.extend(bytes); while out.len() % 4 != 0 { out.push(0); } }
    Ok(out)
}
