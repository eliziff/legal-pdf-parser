//! Resource-aware Form XObject expansion.
//!
//! Forms are content streams, not a second text format. Resolve their local
//! resources, inline their operations behind the implicit graphics-state
//! save/restore and Form matrix, then let the page interpreter handle every
//! operator. This keeps one text-state machine for pages and nested Forms.

use super::content_stream::strip_pdf_comments;
use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) const MAX_EXPANDED_OPERATIONS: usize = 1_000_000;

#[derive(Clone, Copy)]
enum XObject<'a> {
    Image,
    Form {
        id: Option<ObjectId>,
        stream: &'a Stream,
    },
}

#[derive(Clone, Default)]
struct Resources<'a> {
    fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    xobjects: HashMap<Vec<u8>, XObject<'a>>,
    properties: HashMap<Vec<u8>, &'a Object>,
}

pub(super) struct ExpandedContent<'a> {
    pub operations: Vec<Operation>,
    pub fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    pub font_names: HashMap<String, String>,
    pub images: HashMap<String, String>,
    pub exceeded_limit: bool,
}

/// Inline every reachable Form XObject into one resource-qualified operation
/// sequence. Referenced-form cycles are skipped; repeated non-cyclic uses are
/// expanded independently because each invocation has its own CTM.
pub(super) fn expand_page_content<'a>(
    doc: &'a Document,
    page_id: ObjectId,
    mut operations: Vec<Operation>,
    page_fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
) -> ExpandedContent<'a> {
    let page_resources = page_resources(doc, page_id, page_fonts);
    let mut expander = Expander::new(doc, &page_resources);
    if !page_resources
        .xobjects
        .values()
        .any(|xobject| matches!(xobject, XObject::Form { .. }))
    {
        if operations.len() > MAX_EXPANDED_OPERATIONS {
            expander.exceeded_limit = true;
        } else {
            if !page_resources.properties.is_empty() {
                for operation in &mut operations {
                    if operation.operator == "BDC" {
                        if let Some(Object::Name(name)) = operation.operands.get(1) {
                            if let Some(property) = page_resources.properties.get(name) {
                                operation.operands[1] = (*property).clone();
                            }
                        }
                    }
                }
            }
            expander.operations = operations;
        }
        return expander.finish();
    }
    expander.expand_stream(operations, &page_resources, false);
    expander.finish()
}

struct Expander<'a> {
    doc: &'a Document,
    operations: Vec<Operation>,
    fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    font_names: HashMap<String, String>,
    images: HashMap<String, String>,
    active_forms: HashSet<ObjectId>,
    next_name: u64,
    exceeded_limit: bool,
}

impl<'a> Expander<'a> {
    fn new(doc: &'a Document, page_resources: &Resources<'a>) -> Self {
        let fonts = page_resources.fonts.clone();
        let images = page_resources
            .xobjects
            .iter()
            .filter_map(|(name, xobject)| {
                matches!(xobject, XObject::Image).then(|| {
                    let name = String::from_utf8_lossy(name).into_owned();
                    (name.clone(), name)
                })
            })
            .collect();
        Self {
            doc,
            operations: Vec::new(),
            fonts,
            font_names: HashMap::new(),
            images,
            active_forms: HashSet::new(),
            next_name: 0,
            exceeded_limit: false,
        }
    }

    fn finish(self) -> ExpandedContent<'a> {
        ExpandedContent {
            operations: self.operations,
            fonts: self.fonts,
            font_names: self.font_names,
            images: self.images,
            exceeded_limit: self.exceeded_limit,
        }
    }

    fn push(&mut self, operation: Operation) {
        if self.operations.len() < MAX_EXPANDED_OPERATIONS {
            self.operations.push(operation);
        } else {
            self.exceeded_limit = true;
        }
    }

    fn unique_name(&mut self, kind: &str) -> Vec<u8> {
        loop {
            self.next_name += 1;
            let name = format!("__pdf_inspector_{kind}_{}", self.next_name).into_bytes();
            let text = String::from_utf8_lossy(&name);
            if !self.fonts.contains_key(&name) && !self.images.contains_key(text.as_ref()) {
                return name;
            }
        }
    }

    fn font_alias(
        &mut self,
        source_name: &[u8],
        resources: &Resources<'a>,
        aliases: &mut HashMap<Vec<u8>, Vec<u8>>,
        qualify: bool,
    ) -> Option<Vec<u8>> {
        if let Some(alias) = aliases.get(source_name) {
            return Some(alias.clone());
        }
        let font = *resources.fonts.get(source_name)?;
        let alias = if !qualify
            || self
                .fonts
                .get(source_name)
                .is_some_and(|known| std::ptr::eq(*known, font))
        {
            source_name.to_vec()
        } else {
            self.unique_name("font")
        };
        self.fonts.insert(alias.clone(), font);
        self.font_names.insert(
            String::from_utf8_lossy(&alias).into_owned(),
            String::from_utf8_lossy(source_name).into_owned(),
        );
        aliases.insert(source_name.to_vec(), alias.clone());
        Some(alias)
    }

    fn image_alias(
        &mut self,
        source_name: &[u8],
        aliases: &mut HashMap<Vec<u8>, Vec<u8>>,
        qualify: bool,
    ) -> Vec<u8> {
        if let Some(alias) = aliases.get(source_name) {
            return alias.clone();
        }
        let source_text = String::from_utf8_lossy(source_name).into_owned();
        let alias = if !qualify && self.images.contains_key(&source_text) {
            source_name.to_vec()
        } else {
            self.unique_name("image")
        };
        self.images
            .insert(String::from_utf8_lossy(&alias).into_owned(), source_text);
        aliases.insert(source_name.to_vec(), alias.clone());
        alias
    }

    fn expand_stream(
        &mut self,
        operations: Vec<Operation>,
        resources: &Resources<'a>,
        qualify: bool,
    ) {
        let mut font_aliases = HashMap::new();
        let mut image_aliases = HashMap::new();

        for mut operation in operations {
            if self.exceeded_limit {
                return;
            }
            match operation.operator.as_str() {
                "Tf" => {
                    if let Some(Object::Name(name)) = operation.operands.first_mut() {
                        if let Some(alias) =
                            self.font_alias(name, resources, &mut font_aliases, qualify)
                        {
                            *name = alias;
                        }
                    }
                    self.push(operation);
                }
                "Do" => {
                    let Some(name) = operation
                        .operands
                        .first()
                        .and_then(|value| value.as_name().ok())
                    else {
                        self.push(operation);
                        continue;
                    };
                    match resources.xobjects.get(name).copied() {
                        Some(XObject::Image) => {
                            operation.operands[0] =
                                Object::Name(self.image_alias(name, &mut image_aliases, qualify));
                            self.push(operation);
                        }
                        Some(XObject::Form { id, stream }) => {
                            self.expand_form(id, stream, resources);
                        }
                        None => self.push(operation),
                    }
                }
                "BDC" => {
                    if let Some(Object::Name(name)) = operation.operands.get(1) {
                        if let Some(property) = resources.properties.get(name) {
                            operation.operands[1] = (*property).clone();
                        }
                    }
                    self.push(operation);
                }
                _ => self.push(operation),
            }
        }
    }

    fn expand_form(
        &mut self,
        id: Option<ObjectId>,
        stream: &'a Stream,
        parent_resources: &Resources<'a>,
    ) {
        if id.is_some_and(|id| !self.active_forms.insert(id)) {
            log::warn!("skipping recursive Form XObject {id:?}");
            return;
        }

        let raw_data = stream
            .decompressed_content()
            .unwrap_or_else(|_| stream.content.clone());
        let data = strip_pdf_comments(&raw_data);
        let content = Content::decode(&data);
        drop(data);
        drop(raw_data);
        if let Ok(content) = content {
            let resources = form_resources(self.doc, stream, parent_resources);
            self.push(Operation::new("q", Vec::new()));
            if let Some(matrix) = form_matrix(self.doc, stream) {
                self.push(Operation::new("cm", matrix));
            }
            self.expand_stream(content.operations, &resources, true);
            self.push(Operation::new("Q", Vec::new()));
        } else {
            log::warn!("skipping undecodable Form XObject {id:?}");
        }

        if let Some(id) = id {
            self.active_forms.remove(&id);
        }
    }
}

fn page_resources<'a>(
    doc: &'a Document,
    page_id: ObjectId,
    fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
) -> Resources<'a> {
    let mut resources = Resources {
        fonts,
        ..Resources::default()
    };
    if let Ok((inline, ids)) = doc.get_page_resources(page_id) {
        if let Some(dictionary) = inline {
            collect_nonfont_resources(doc, dictionary, &mut resources, false);
        }
        for id in ids {
            if let Ok(dictionary) = doc.get_dictionary(id) {
                collect_nonfont_resources(doc, dictionary, &mut resources, false);
            }
        }
    }
    resources
}

fn form_resources<'a>(
    doc: &'a Document,
    stream: &'a Stream,
    parent: &Resources<'a>,
) -> Resources<'a> {
    let Ok(value) = stream.dict.get(b"Resources") else {
        return parent.clone();
    };
    let Some(dictionary) = object_dictionary(doc, value) else {
        return Resources::default();
    };
    let mut resources = Resources::default();
    collect_fonts(doc, dictionary, &mut resources.fonts);
    collect_nonfont_resources(doc, dictionary, &mut resources, true);
    resources
}

fn collect_fonts<'a>(
    doc: &'a Document,
    resources: &'a Dictionary,
    fonts: &mut BTreeMap<Vec<u8>, &'a Dictionary>,
) {
    let Some(dictionary) = resources
        .get(b"Font")
        .ok()
        .and_then(|value| object_dictionary(doc, value))
    else {
        return;
    };
    for (name, value) in dictionary.iter() {
        if let Some(font) = object_dictionary(doc, value) {
            fonts.insert(name.clone(), font);
        }
    }
}

fn collect_nonfont_resources<'a>(
    doc: &'a Document,
    dictionary: &'a Dictionary,
    resources: &mut Resources<'a>,
    overwrite: bool,
) {
    if let Some(xobjects) = dictionary
        .get(b"XObject")
        .ok()
        .and_then(|value| object_dictionary(doc, value))
    {
        for (name, value) in xobjects.iter() {
            if let Some(xobject) = resolve_xobject(doc, value) {
                if overwrite {
                    resources.xobjects.insert(name.clone(), xobject);
                } else {
                    resources.xobjects.entry(name.clone()).or_insert(xobject);
                }
            }
        }
    }

    if let Some(properties) = dictionary
        .get(b"Properties")
        .ok()
        .and_then(|value| object_dictionary(doc, value))
    {
        for (name, value) in properties.iter() {
            if overwrite {
                resources.properties.insert(name.clone(), value);
            } else {
                resources.properties.entry(name.clone()).or_insert(value);
            }
        }
    }
}

fn resolve_xobject<'a>(doc: &'a Document, value: &'a Object) -> Option<XObject<'a>> {
    let (id, stream) = match value {
        Object::Reference(id) => {
            let stream = doc.get_object(*id).ok()?.as_stream().ok()?;
            (Some(*id), stream)
        }
        Object::Stream(stream) => (None, stream),
        _ => return None,
    };
    match stream.dict.get(b"Subtype").ok()?.as_name().ok()? {
        b"Image" => Some(XObject::Image),
        b"Form" => Some(XObject::Form { id, stream }),
        _ => None,
    }
}

fn object_dictionary<'a>(doc: &'a Document, value: &'a Object) -> Option<&'a Dictionary> {
    match value {
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        Object::Dictionary(dictionary) => Some(dictionary),
        Object::Stream(stream) => Some(&stream.dict),
        _ => None,
    }
}

fn form_matrix(doc: &Document, stream: &Stream) -> Option<Vec<Object>> {
    let value = stream.dict.get(b"Matrix").ok()?;
    let value = match value {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        value => value,
    };
    let matrix = value.as_array().ok()?;
    (matrix.len() >= 6).then(|| matrix[..6].to_vec())
}
