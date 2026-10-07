//! The document outline (bookmarks) as PageIndex's `read_bookmarks` reads it through PDFium.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/embedded_toc.py` (`read_bookmarks`) at
//! 6d23caf416858f2ca136840305d1f479a86f6ef7, together with the PDFium and pypdfium2 behaviour it
//! relies on (`PdfDocument.get_toc`, `FPDFBookmark_GetTitle`, `FPDFBookmark_GetDest`,
//! `FPDFDest_GetDestPageIndex`). Copyright (c) 2025 Vectify AI; MIT License, reproduced at the end of this file.

use legal_pdf_core::model::PdfOutlineEntry;
use lopdf::{decode_text_string, Dictionary, Document, Object, ObjectId};
use std::collections::{HashMap, HashSet};

/// pypdfium2 walks fifteen outline levels and skips deeper subtrees.
const MAX_DEPTH: usize = 15;
const MAX_VISITS: usize = 100_000;

/// Each bookmark with a resolvable target page and a non-empty title, in outline order. A
/// document without a readable outline has none.
pub fn embedded_outline(document: &Document) -> Vec<PdfOutlineEntry> {
    let Some(first) = document
        .catalog()
        .ok()
        .and_then(|catalog| dictionary(document, catalog.get(b"Outlines").ok()?))
        .and_then(|outlines| outlines.get(b"First").ok())
    else {
        return Vec::new();
    };
    let pages = document
        .get_pages()
        .into_iter()
        .map(|(number, id)| (id, number as usize - 1))
        .collect::<HashMap<ObjectId, usize>>();
    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    // The siblings still to visit at each depth, innermost last: a bookmark is read, then its
    // children, then its next sibling.
    let mut stack = vec![Some(first)];
    while let Some(slot) = stack.last_mut() {
        let Some(object) = slot.take() else {
            stack.pop();
            continue;
        };
        let level = stack.len() - 1;
        // A circular reference ends the sibling chain it was found in.
        let id = object.as_reference().ok();
        let Some(node) = dictionary(document, object)
            .filter(|_| id.is_none_or(|id| seen.insert(id)) && seen.len() <= MAX_VISITS)
        else {
            stack.pop();
            continue;
        };
        *stack.last_mut().expect("current level") = node.get(b"Next").ok();
        if let Some(page_index) = destination_page(document, node, &pages) {
            let title = title(document, node);
            if !title.is_empty() {
                entries.push(PdfOutlineEntry {
                    title,
                    level: level + 1,
                    page_index,
                });
            }
        }
        if level + 1 < MAX_DEPTH {
            if let Ok(child) = node.get(b"First") {
                stack.push(Some(child));
            }
        }
    }
    entries
}

fn dictionary<'a>(document: &'a Document, object: &'a Object) -> Option<&'a Dictionary> {
    document.dereference(object).ok()?.1.as_dict().ok()
}

/// PDFium blanks every control character of a title; the caller trims it.
fn title(document: &Document, node: &Dictionary) -> String {
    let Some(text) = node
        .get(b"Title")
        .ok()
        .and_then(|title| document.dereference(title).ok())
        .and_then(|(_, title)| decode_text_string(title).ok())
    else {
        return String::new();
    };
    text.trim_start_matches('\u{feff}')
        .chars()
        .map(|character| if character < ' ' { ' ' } else { character })
        .collect::<String>()
        .trim()
        .to_owned()
}

/// The bookmark's own destination, or else its GoTo action's. A destination names its page by
/// reference, or by a page index of its own.
fn destination_page(
    document: &Document,
    node: &Dictionary,
    pages: &HashMap<ObjectId, usize>,
) -> Option<usize> {
    let array = node
        .get(b"Dest")
        .ok()
        .and_then(|destination| destination_array(document, destination))
        .or_else(|| {
            let action = dictionary(document, node.get(b"A").ok()?)?;
            (action.get(b"S").ok()?.as_name().ok()? == b"GoTo")
                .then(|| destination_array(document, action.get(b"D").ok()?))
                .flatten()
        })?;
    let target = array.first()?;
    if let Ok(id) = target.as_reference() {
        return pages.get(&id).copied();
    }
    usize::try_from(target.as_i64().ok()?).ok()
}

fn destination_array<'a>(document: &'a Document, object: &'a Object) -> Option<&'a [Object]> {
    let object = document.dereference(object).ok()?.1;
    match object {
        Object::Array(values) => Some(values),
        Object::Name(name) | Object::String(name, _) => named_destination(document, name),
        _ => None,
    }
}

/// A named destination from the catalog's name tree, else from its older `Dests` dictionary.
fn named_destination<'a>(document: &'a Document, name: &[u8]) -> Option<&'a [Object]> {
    let catalog = document.catalog().ok()?;
    let resolve = |value: &'a Object| -> Option<&'a [Object]> {
        let value = document.dereference(value).ok()?.1;
        match value {
            Object::Array(values) => Some(values.as_slice()),
            Object::Dictionary(dict) => match document.dereference(dict.get(b"D").ok()?).ok()?.1 {
                Object::Array(values) => Some(values.as_slice()),
                _ => None,
            },
            _ => None,
        }
    };
    let tree = catalog
        .get(b"Names")
        .ok()
        .and_then(|names| dictionary(document, names))
        .and_then(|names| names.get(b"Dests").ok());
    let mut pending = tree.into_iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    while let Some(object) = pending.pop() {
        if let Ok(id) = object.as_reference() {
            if !seen.insert(id) || seen.len() > MAX_VISITS {
                break;
            }
        }
        let Some(node) = dictionary(document, object) else {
            continue;
        };
        if let Some(values) = node
            .get(b"Names")
            .ok()
            .and_then(|names| document.dereference(names).ok())
            .and_then(|(_, names)| names.as_array().ok())
        {
            for pair in values.chunks_exact(2) {
                let key = document.dereference(&pair[0]).ok().map(|(_, key)| key);
                if key.and_then(|key| key.as_str().ok()) == Some(name) {
                    return resolve(&pair[1]);
                }
            }
        }
        if let Some(kids) = node
            .get(b"Kids")
            .ok()
            .and_then(|kids| document.dereference(kids).ok())
            .and_then(|(_, kids)| kids.as_array().ok())
        {
            pending.extend(kids.iter().rev());
        }
    }
    catalog
        .get(b"Dests")
        .ok()
        .and_then(|dests| dictionary(document, dests))
        .and_then(|dests| dests.get(name).ok())
        .and_then(resolve)
}

// PageIndex license, for the code ported above:
//
// MIT License
//
// Copyright (c) 2025 Vectify AI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
