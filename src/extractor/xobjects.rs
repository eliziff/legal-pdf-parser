//! Resource-aware Form XObject expansion.
//!
//! Forms are content streams, not a second text format. Resolve their local
//! resources, inline their operations behind the implicit graphics-state
//! save/restore and Form matrix, then let the page interpreter handle every
//! operator. This keeps one text-state machine for pages and nested Forms.

use super::content_stream::strip_pdf_comments;
use lopdf::content::Operation;
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

pub(super) const MAX_EXPANDED_OPERATIONS: usize = 1_000_000;
const MAX_FORM_XOBJECT_INVOCATIONS: usize = 10_000;

pub(crate) struct FormWalkBudget {
    invocations: usize,
    operations: usize,
    max_invocations: usize,
    max_operations: usize,
    truncated: bool,
}
impl FormWalkBudget {
    pub(crate) fn new() -> Self {
        Self::with_limits(MAX_FORM_XOBJECT_INVOCATIONS, MAX_EXPANDED_OPERATIONS)
    }
    fn with_limits(max_invocations: usize, max_operations: usize) -> Self {
        Self {
            invocations: 0,
            operations: 0,
            max_invocations,
            max_operations,
            truncated: false,
        }
    }
    #[cfg(test)]
    fn was_truncated(&self) -> bool {
        self.truncated
    }
}

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
    spaces: HashMap<Vec<u8>, &'a Object>,
    states: HashMap<Vec<u8>, &'a Object>,
}

pub(super) struct ExpandedContent<'a> {
    pub operations: Vec<Operation>,
    pub fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    pub font_names: HashMap<String, String>,
    pub images: HashMap<String, String>,
    pub exceeded_limit: bool,
    pub form_truncated: bool,
    pub paint_resources: Dictionary,
}

/// Inline every reachable Form XObject into one resource-qualified operation
/// sequence. Referenced-form cycles are skipped; repeated non-cyclic uses are
/// expanded independently because each invocation has its own CTM.
pub(super) fn expand_page_content<'a>(
    doc: &'a Document,
    page_id: ObjectId,
    mut operations: Vec<Operation>,
    page_fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    budget: &mut FormWalkBudget,
) -> ExpandedContent<'a> {
    let page_resources = page_resources(doc, page_id, page_fonts);
    let mut expander = Expander::new(doc, &page_resources);
    expander.form_invocations = budget.invocations;
    expander.prior_operations = budget.operations;
    expander.max_invocations = budget.max_invocations;
    expander.max_operations = budget.max_operations;
    if !page_resources
        .xobjects
        .values()
        .any(|xobject| matches!(xobject, XObject::Form { .. }))
    {
        if operations.len().saturating_add(budget.operations) > budget.max_operations {
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
        return expander.finish(budget);
    }
    expander.expand_stream(operations, &page_resources, false);
    expander.finish(budget)
}

struct Expander<'a> {
    doc: &'a Document,
    operations: Vec<Operation>,
    fonts: BTreeMap<Vec<u8>, &'a Dictionary>,
    font_names: HashMap<String, String>,
    images: HashMap<String, String>,
    spaces: HashMap<Vec<u8>, Object>,
    states: HashMap<Vec<u8>, Object>,
    form_operations: HashMap<ObjectId, Option<Rc<[Operation]>>>,
    active_forms: HashSet<ObjectId>,
    next_name: u64,
    exceeded_limit: bool,
    form_invocations: usize,
    prior_operations: usize,
    max_invocations: usize,
    max_operations: usize,
    form_truncated: bool,
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
            spaces: page_resources
                .spaces
                .iter()
                .map(|(k, v)| (k.clone(), (*v).clone()))
                .collect(),
            states: page_resources
                .states
                .iter()
                .map(|(k, v)| (k.clone(), (*v).clone()))
                .collect(),
            form_operations: HashMap::new(),
            active_forms: HashSet::new(),
            next_name: 0,
            exceeded_limit: false,
            form_invocations: 0,
            prior_operations: 0,
            max_invocations: MAX_FORM_XOBJECT_INVOCATIONS,
            max_operations: MAX_EXPANDED_OPERATIONS,
            form_truncated: false,
        }
    }

    fn finish(self, budget: &mut FormWalkBudget) -> ExpandedContent<'a> {
        budget.truncated |= self.form_truncated || self.exceeded_limit;
        budget.invocations = self.form_invocations;
        budget.operations = self.prior_operations.saturating_add(self.operations.len());
        let mut paint_resources = Dictionary::new();
        let mut spaces = Dictionary::new();
        let mut states = Dictionary::new();
        for (name, value) in self.spaces {
            spaces.set(name, value);
        }
        for (name, value) in self.states {
            states.set(name, value);
        }
        paint_resources.set("ColorSpace", spaces);
        paint_resources.set("ExtGState", states);
        ExpandedContent {
            paint_resources,
            operations: self.operations,
            fonts: self.fonts,
            font_names: self.font_names,
            images: self.images,
            exceeded_limit: self.exceeded_limit,
            form_truncated: self.form_truncated,
        }
    }

    fn push(&mut self, operation: Operation) {
        if self.operations.len().saturating_add(self.prior_operations) < self.max_operations {
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
            if !self.fonts.contains_key(&name)
                && !self.images.contains_key(text.as_ref())
                && !self.spaces.contains_key(&name)
                && !self.states.contains_key(&name)
            {
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
        operations: impl IntoIterator<Item = Operation>,
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
                "cs" | "CS" | "gs" if qualify => {
                    let is_state = operation.operator == "gs";
                    if let Some(Object::Name(name)) = operation.operands.first_mut() {
                        if is_state
                            || !matches!(
                                name.as_slice(),
                                b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK" | b"Pattern"
                            )
                        {
                            let alias = self.unique_name("paint");
                            let source = if is_state {
                                &resources.states
                            } else {
                                &resources.spaces
                            };
                            if let Some(value) = source.get(name) {
                                let target = if is_state {
                                    &mut self.states
                                } else {
                                    &mut self.spaces
                                };
                                target.insert(alias.clone(), (*value).clone());
                            }
                            *name = alias;
                        }
                    }
                    self.push(operation);
                }
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
        if self.form_invocations >= self.max_invocations {
            self.form_truncated = true;
            return;
        }
        self.form_invocations += 1;
        if id.is_some_and(|id| !self.active_forms.insert(id)) {
            log::warn!("skipping recursive Form XObject {id:?}");
            return;
        }

        let operations = id
            .and_then(|id| self.form_operations.get(&id).cloned())
            .unwrap_or_else(|| {
                let raw = stream.get_plain_content_with_limit(64 * 1024 * 1024).ok()?;
                let decoded = super::content_decode::decode_content_bounded(
                    &strip_pdf_comments(&raw),
                    super::content_decode::MAX_PAGE_OPERATIONS,
                )
                .ok()
                .flatten()
                .map(|content| Rc::from(content.operations));
                if let Some(id) = id {
                    self.form_operations.insert(id, decoded.clone());
                }
                decoded
            });
        if let Some(operations) = operations {
            let owned_resources = stream
                .dict
                .get(b"Resources")
                .ok()
                .map(|value| form_resources(self.doc, value));
            let resources = owned_resources.as_ref().unwrap_or(parent_resources);
            self.push(Operation::new("q", Vec::new()));
            if let Some(matrix) = form_matrix(self.doc, stream) {
                self.push(Operation::new("cm", matrix));
            }
            self.expand_stream(operations.iter().cloned(), resources, true);
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

fn form_resources<'a>(doc: &'a Document, value: &'a Object) -> Resources<'a> {
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
    for (key, target) in [
        (b"ColorSpace".as_slice(), &mut resources.spaces),
        (b"ExtGState".as_slice(), &mut resources.states),
    ] {
        if let Some(entries) = dictionary
            .get(key)
            .ok()
            .and_then(|value| object_dictionary(doc, value))
        {
            for (name, value) in entries.iter() {
                if overwrite {
                    target.insert(name.clone(), value);
                } else {
                    target.entry(name.clone()).or_insert(value);
                }
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::content_stream::extract_page_text_items;
    use crate::extractor::content_stream::FontProductCache;
    use crate::tounicode::FontCMaps;
    use crate::types::TextItem;
    use lopdf::{dictionary, Dictionary, Stream};

    /// Build an acyclic Form XObject DAG: `levels` form objects, each non-leaf
    /// invoking the next form `branches` times. The leaf draws a single `(X)`.
    /// Returns `(doc, root_form_id)`.
    fn form_dag(branches: usize, levels: usize) -> (Document, ObjectId) {
        assert!(levels >= 2);
        let mut doc = Document::new();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let ids: Vec<ObjectId> = (0..levels).map(|_| doc.new_object_id()).collect();
        for level in 0..levels {
            let stream = if level + 1 == levels {
                Stream::new(
                    dictionary! {
                        "Type" => "XObject",
                        "Subtype" => "Form",
                        "BBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
                        "Resources" => dictionary! {
                            "Font" => dictionary! {
                                "F1" => Object::Reference(font_id),
                            },
                        },
                    },
                    b"BT /F1 10 Tf 10 10 Td (X) Tj ET\n".to_vec(),
                )
            } else {
                let next_name = format!("Fm{}", level + 1);
                let content = format!("/{next_name} Do\n").repeat(branches);
                let mut xobjects = Dictionary::new();
                xobjects.set(next_name, Object::Reference(ids[level + 1]));
                let mut resources = Dictionary::new();
                resources.set("XObject", Object::Dictionary(xobjects));
                let mut dict = dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Form",
                    "BBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
                };
                dict.set("Resources", Object::Dictionary(resources));
                Stream::new(dict, content.into_bytes())
            };
            doc.set_object(ids[level], Object::Stream(stream));
        }
        (doc, ids[0])
    }

    fn page_invoking_form(mut doc: Document, form_id: ObjectId) -> (Document, ObjectId) {
        let content_id = doc.add_object(Object::Stream(Stream::new(
            dictionary! {},
            b"/Fm0 Do\n".to_vec(),
        )));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content_id),
            "Resources" => dictionary! {
                "XObject" => dictionary! {
                    "Fm0" => Object::Reference(form_id),
                },
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        let pages_id = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));
        (doc, page_id)
    }

    fn extract_form(
        doc: &Document,
        form_id: ObjectId,
        budget: &mut FormWalkBudget,
    ) -> Vec<TextItem> {
        let (doc, page_id) = page_invoking_form(doc.clone(), form_id);
        let ((items, _, _, _), _, _, _, _) =
            crate::extractor::content_stream::extract_page_text_items_raw(
                &doc,
                page_id,
                1,
                &FontCMaps::from_doc(&doc),
                false,
                &mut FontProductCache::new(),
                budget,
            )
            .unwrap();
        items
    }

    #[test]
    fn nested_form_still_extracts_leaf_text() {
        let (doc, root) = form_dag(1, 3);
        let items = extract_form(&doc, root, &mut FormWalkBudget::new());
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "X");
    }

    #[test]
    fn form_items_carry_family_name_and_resource_tag() {
        // Parity with content_stream.rs: `font` is the /BaseFont family
        // name, `font_tag` the raw resource tag, in both parsers.
        let (doc, root) = form_dag(1, 2);
        let items = extract_form(&doc, root, &mut FormWalkBudget::new());
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].font, "Helvetica");
        assert_eq!(items[0].font_tag, "F1");
    }

    #[test]
    fn acyclic_form_dag_within_budget_keeps_all_leaves() {
        // 4 sibling invocations across 4 nested levels → 4^4 leaf drawings.
        // Default budgets are far above 256, so legitimate nesting is intact.
        let (doc, root) = form_dag(4, 5);
        let items = extract_form(&doc, root, &mut FormWalkBudget::new());
        assert_eq!(items.len(), 4usize.pow(4));
        assert!(items.iter().all(|item| item.text == "X"));
    }

    #[test]
    fn acyclic_form_dag_stops_at_invocation_budget() {
        // Same DAG as above would draw 256 leaves; a tiny invocation cap must
        // stop expansion rather than walking the full tree.
        let (doc, root) = form_dag(4, 5);
        let mut budget = FormWalkBudget::with_limits(20, MAX_EXPANDED_OPERATIONS);
        let items = extract_form(&doc, root, &mut budget);
        assert!(
            items.len() < 4usize.pow(4),
            "invocation budget must truncate DAG expansion; got {} items",
            items.len()
        );
        assert!(budget.was_truncated());
    }

    #[test]
    fn form_operations_stop_at_budget() {
        let mut doc = Document::new();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let mut content = b"q Q\n".repeat(50);
        content.extend_from_slice(b"BT /F1 10 Tf 10 10 Td (X) Tj ET\n");
        let form_id = doc.add_object(Object::Stream(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Form",
                "BBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
                "Resources" => dictionary! {
                    "Font" => dictionary! {
                        "F1" => Object::Reference(font_id),
                    },
                },
            },
            content,
        )));

        let mut budget = FormWalkBudget::with_limits(MAX_FORM_XOBJECT_INVOCATIONS, 10);
        let items = extract_form(&doc, form_id, &mut budget);
        assert!(
            items.is_empty(),
            "operation budget must stop before the trailing text show"
        );
        assert!(budget.was_truncated());
    }

    #[test]
    fn page_level_form_dag_stays_within_production_budget() {
        // A page-level `/Do` of an 8-wide, 6-level Form DAG would expand to
        // 8^5 = 32_768 leaf drawings without a budget. The production
        // invocation cap must keep extraction bounded.
        let (doc, root) = form_dag(8, 6);
        let (doc, page_id) = page_invoking_form(doc, root);

        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _, _), _, _, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        assert!(
            items.len() <= MAX_FORM_XOBJECT_INVOCATIONS,
            "page-level Form expansion must stay within the invocation cap; got {}",
            items.len()
        );
        assert!(
            !items.is_empty(),
            "budget must still allow some nested form text through"
        );
    }

    #[test]
    fn shared_form_budget_spans_two_extraction_passes() {
        // The invisible-layer retry calls extract_page_text_items twice for
        // the same page; both passes must share one budget.
        let (doc, root) = form_dag(1, 2);
        let (doc, page_id) = page_invoking_form(doc, root);
        let font_cmaps = FontCMaps::from_doc(&doc);
        // Root + leaf = 2 invocations on the first pass.
        let mut budget = FormWalkBudget::with_limits(2, MAX_EXPANDED_OPERATIONS);
        let ((first, _, _, _), _, _, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut budget,
        )
        .unwrap();
        assert_eq!(first.iter().filter(|item| item.text == "X").count(), 1);
        assert!(!budget.was_truncated());

        let ((second, _, _, _), _, _, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            true,
            &mut FontProductCache::new(),
            &mut budget,
        )
        .unwrap();
        assert!(
            second.iter().all(|item| item.text != "X"),
            "second pass must not get a fresh invocation budget"
        );
        assert!(budget.was_truncated());
    }

    /// Build a document whose page draws *all* of its content through a single
    /// Form XObject — the shape emitted by print-to-PDF producers like PDFlib,
    /// where the page stream itself is only `q /X1 Do Q`.
    fn doc_with_form_content(form_content: &[u8]) -> (Document, ObjectId) {
        doc_with_page_and_forms(b"q /X1 Do Q", &[form_content])
    }

    /// A page drawing `page_content` with forms `X1`, `X2`, … available to
    /// the page and to each other (so a form can invoke a nested form).
    fn doc_with_page_and_forms(page_content: &[u8], forms: &[&[u8]]) -> (Document, ObjectId) {
        let mut doc = Document::new();
        let widths: Vec<Object> = (0..=255).map(|_| 600.into()).collect();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "FirstChar" => 0,
            "LastChar" => 255,
            "Widths" => Object::Array(widths),
        });
        let form_ids: Vec<ObjectId> = forms.iter().map(|_| doc.new_object_id()).collect();
        let xobjects = || {
            let mut dict = lopdf::Dictionary::new();
            for (index, id) in form_ids.iter().enumerate() {
                dict.set(format!("X{}", index + 1), Object::Reference(*id));
            }
            dict
        };
        for (id, content) in form_ids.iter().zip(forms) {
            doc.set_object(
                *id,
                Object::Stream(Stream::new(
                    dictionary! {
                        "Type" => "XObject",
                        "Subtype" => "Form",
                        "BBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                        "Resources" => dictionary! {
                            "Font" => dictionary! { "F1" => Object::Reference(font_id) },
                            "XObject" => xobjects(),
                        },
                    },
                    content.to_vec(),
                )),
            );
        }
        let content_id = doc.add_object(Object::Stream(Stream::new(
            dictionary! {},
            page_content.to_vec(),
        )));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content_id),
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => Object::Reference(font_id) },
                "XObject" => xobjects(),
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        let pages_id = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));
        (doc, page_id)
    }

    fn form_items(form_content: &[u8]) -> Vec<TextItem> {
        let (doc, page_id) = doc_with_form_content(form_content);
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _, _), _, _, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        items
    }

    fn find<'a>(items: &'a [TextItem], text: &str) -> &'a TextItem {
        items
            .iter()
            .find(|item| item.text == text)
            .unwrap_or_else(|| {
                let found: Vec<&String> = items.iter().map(|i| &i.text).collect();
                panic!("no item {text:?} in {found:?}")
            })
    }

    #[test]
    fn t_star_inside_form_moves_to_next_line() {
        // T* was previously unhandled inside Form XObjects, so every line after
        // the first piled onto the preceding baseline and drifted right.
        let items =
            form_items(b"BT /F1 12 Tf 12 TL 1 0 0 1 100 700 Tm (first) Tj T* (second) Tj ET");

        let first = find(&items, "first");
        let second = find(&items, "second");
        assert!((first.y - 700.0).abs() < 0.1, "first y = {}", first.y);
        assert!((second.y - 688.0).abs() < 0.1, "second y = {}", second.y);
        assert!((second.x - 100.0).abs() < 0.1, "second x = {}", second.x);
    }

    #[test]
    fn td_inside_form_is_relative_to_line_start_not_shown_text() {
        // Td moves relative to the text *line* matrix. Applying it to the
        // matrix already advanced by Tj marched each line off the right edge.
        let items = form_items(b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (AAAAA) Tj 0 -12 Td (B) Tj ET");

        let b = find(&items, "B");
        assert!((b.x - 100.0).abs() < 0.1, "B x = {} (expected 100)", b.x);
        assert!((b.y - 688.0).abs() < 0.1, "B y = {}", b.y);
    }

    #[test]
    fn td_inside_form_sets_leading_for_later_t_star() {
        // `TD` sets the leading to -ty as a side effect; a following T* must
        // reuse it.
        let items = form_items(
            b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (one) Tj 0 -15 TD (two) Tj T* (three) Tj ET",
        );

        assert!((find(&items, "two").y - 685.0).abs() < 0.1);
        let three = find(&items, "three");
        assert!((three.y - 670.0).abs() < 0.1, "three y = {}", three.y);
        assert!((three.x - 100.0).abs() < 0.1, "three x = {}", three.x);
    }

    #[test]
    fn quote_operator_inside_form_moves_to_next_line() {
        let items = form_items(b"BT /F1 12 Tf 12 TL 1 0 0 1 100 700 Tm (first) Tj (second) ' ET");

        let second = find(&items, "second");
        assert!((second.y - 688.0).abs() < 0.1, "second y = {}", second.y);
        assert!((second.x - 100.0).abs() < 0.1, "second x = {}", second.x);
    }

    #[test]
    fn double_quote_operator_inside_form_sets_spacing_and_moves() {
        // `aw ac (string) "` — set word spacing and char spacing, then T* and show.
        let items =
            form_items(b"BT /F1 12 Tf 12 TL 1 0 0 1 100 700 Tm (first) Tj 0 0 (second) \" ET");

        let second = find(&items, "second");
        assert!((second.y - 688.0).abs() < 0.1, "second y = {}", second.y);
        assert!((second.x - 100.0).abs() < 0.1, "second x = {}", second.x);
    }

    #[test]
    fn char_spacing_inside_form_widens_advance() {
        // Tc was hardcoded to 0 in the form parser, so advance widths drifted.
        // 2 glyphs x 600/1000 x 12pt = 14.4, plus 2 x Tc(2.0) = 18.4.
        let items = form_items(b"BT /F1 12 Tf 1 0 0 1 100 700 Tm 2 Tc (AB) Tj ET");

        let ab = find(&items, "AB");
        assert!((ab.width - 18.4).abs() < 0.1, "AB width = {}", ab.width);
    }

    #[test]
    fn q_restores_fill_colour_inside_form() {
        // White letters on a dark background remain visible; Q restores black
        // fill, allowing matching black fill/stroke to supply bold evidence.
        let items = form_items(
            b"0 g 90 680 150 40 re f BT /F1 12 Tf 12 TL 1 0 0 1 100 700 Tm q 1 g (white) Tj Q T* 0 G 0.3 w 2 Tr (black) Tj ET",
        );
        assert!(!find(&items, "white").is_bold);
        assert!(find(&items, "black").is_bold);
    }

    #[test]
    fn q_restores_text_state_inside_form() {
        // Tc/TL live in the graphics state; `Q` must roll them back.
        let items =
            form_items(b"BT /F1 12 Tf 12 TL 1 0 0 1 100 700 Tm q 30 TL (a) Tj Q T* (b) Tj ET");

        let b = find(&items, "b");
        assert!(
            (b.y - 688.0).abs() < 0.1,
            "b y = {} (leading should restore to 12)",
            b.y
        );
    }

    #[test]
    fn form_only_rotated_page_is_turned_like_page_stream_text() {
        // The whole page is one Form XObject whose runs are all 90°: the
        // form runs must vote, so the page is re-based exactly as if the
        // runs had been shown by the page stream itself.
        let (doc, page_id) = doc_with_form_content(
            b"BT /F1 12 Tf 0 1 -1 0 200 100 Tm (HELLO) Tj ET
BT /F1 12 Tf 0 1 -1 0 240 100 Tm (WORLD) Tj ET",
        );
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _, _), _, page_rotation, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        assert_eq!(page_rotation, crate::extractor::geometry::PageRotation::Ccw);
        let hello = find(&items, "HELLO");
        assert_eq!(hello.rotation, 0.0);
        assert!((hello.x - 100.0).abs() < 0.01, "x = {}", hello.x);
        assert!((hello.y + 200.0).abs() < 0.01, "y = {}", hello.y);
        assert!((hello.width - 36.0).abs() < 0.01, "width = {}", hello.width);
    }

    #[test]
    fn form_only_page_with_a_lone_split_tj_stays_upright() {
        // One rotated TJ that splits at a 6em gap yields two items but is a
        // single show operator: a lone stamp, not a rotated page.
        let (doc, page_id) =
            doc_with_form_content(b"BT /F1 10 Tf 0 1 -1 0 40 100 Tm [(AB) -6000 (CD)] TJ ET");
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _, _), _, page_rotation, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        assert_eq!(items.len(), 2, "{items:?}");
        assert_eq!(
            page_rotation,
            crate::extractor::geometry::PageRotation::Upright
        );
        assert!(items.iter().all(|i| (i.rotation - 90.0).abs() < 1e-3));
    }

    #[test]
    fn invisible_form_text_is_skipped_and_does_not_vote() {
        // An OCR layer drawn inside a form with `3 Tr`: two rotated hidden
        // runs next to one visible upright caption. The hidden runs must
        // neither appear nor turn the page, and the page must report that
        // it skipped them so the `include_invisible` retry recovers them,
        // exactly as for page-stream text.
        let content = b"BT /F1 12 Tf 72 700 Td (Caption) Tj ET
BT 3 Tr /F1 12 Tf 0 1 -1 0 200 100 Tm (HIDDEN) Tj ET
BT 3 Tr /F1 12 Tf 0 1 -1 0 240 100 Tm [(ALSO) -3000 (HIDDEN)] TJ ET";
        let (doc, page_id) = doc_with_form_content(content);
        let font_cmaps = FontCMaps::from_doc(&doc);
        let extract = |include_invisible: bool| {
            extract_page_text_items(
                &doc,
                page_id,
                1,
                &font_cmaps,
                include_invisible,
                &mut FontProductCache::new(),
                &mut FormWalkBudget::new(),
            )
            .unwrap()
        };

        let ((items, _, _, _), _, page_rotation, skipped_invisible, _) = extract(false);
        assert_eq!(
            page_rotation,
            crate::extractor::geometry::PageRotation::Upright
        );
        assert!(skipped_invisible);
        let texts: Vec<&str> = items.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, ["Caption"]);

        let ((items, _, _, _), _, page_rotation, _, _) = extract(true);
        assert!(items.iter().any(|i| i.text == "HIDDEN"), "{items:?}");
        assert!(
            items.iter().any(|i| i.text.starts_with("ALSO")),
            "{items:?}"
        );
        // Two rotated operators against one upright: the recovered layer
        // now turns the page like any other rotated text.
        assert_eq!(page_rotation, crate::extractor::geometry::PageRotation::Ccw);
    }

    fn extract_page(
        doc: &Document,
        page_id: ObjectId,
        include_invisible: bool,
    ) -> (Vec<TextItem>, bool) {
        let font_cmaps = FontCMaps::from_doc(doc);
        let ((items, _, _, _), _, _, skipped_invisible, _) = extract_page_text_items(
            doc,
            page_id,
            1,
            &font_cmaps,
            include_invisible,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        (items, skipped_invisible)
    }

    #[test]
    fn form_inherits_the_pages_text_rendering_mode() {
        // `3 Tr` set by the page stream before `Do`: the form's text is an
        // OCR-style hidden layer and must stay hidden on the visible pass.
        let (doc, page_id) = doc_with_page_and_forms(
            b"BT 3 Tr ET q /X1 Do Q",
            &[b"BT /F1 12 Tf 72 700 Td (Hidden) Tj ET"],
        );
        let (items, skipped_invisible) = extract_page(&doc, page_id, false);
        assert!(items.is_empty(), "{items:?}");
        assert!(skipped_invisible);
        let (items, _) = extract_page(&doc, page_id, true);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "Hidden");
    }

    #[test]
    fn form_inherits_fill_stroke_weight_and_restores_it() {
        let (doc, page_id) = doc_with_page_and_forms(
            b"0.3 w 2 Tr BT ET /X1 Do 0 Tr /X2 Do",
            &[
                b"BT /F1 12 Tf 72 700 Td (Lead) Tj ET q 0 Tr BT /F1 12 Tf 72 680 Td (Plain) Tj ET Q BT /F1 12 Tf 72 660 Td (Restored) Tj ET",
                b"BT /F1 12 Tf 72 640 Td (Body) Tj ET",
            ],
        );
        let (items, _) = extract_page(&doc, page_id, false);
        let styles: Vec<_> = items.iter().map(|i| (i.text.as_str(), i.is_bold)).collect();
        assert_eq!(
            styles,
            [
                ("Lead", true),
                ("Plain", false),
                ("Restored", true),
                ("Body", false)
            ]
        );
    }

    #[test]
    fn form_graphics_states_are_scoped_and_restored() {
        let (mut doc, page_id) = doc_with_page_and_forms(
            b"/State gs 0.3 w 2 Tr /X1 Do /X2 Do",
            &[
                b"/State gs BT /F1 12 Tf 72 700 Td (Plain) Tj ET",
                b"/State gs BT /F1 12 Tf 72 680 Td (Lead) Tj ET
                  q /Missing gs BT /F1 12 Tf 72 660 Td (Unknown) Tj ET Q
                  BT /F1 12 Tf 72 640 Td (Restored) Tj ET",
            ],
        );
        doc.get_dictionary_mut(page_id)
            .unwrap()
            .get_mut(b"Resources")
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .set(
                "ExtGState",
                dictionary! { "State" => dictionary! { "SM" => 0.02 } },
            );
        for (id, state) in [
            ((2, 0), dictionary! { "ca" => 0.5 }),
            ((3, 0), dictionary! { "OPM" => 1 }),
        ] {
            doc.get_object_mut(id)
                .unwrap()
                .as_stream_mut()
                .unwrap()
                .dict
                .get_mut(b"Resources")
                .unwrap()
                .as_dict_mut()
                .unwrap()
                .set("ExtGState", dictionary! { "State" => state });
        }
        let (items, _) = extract_page(&doc, page_id, false);
        let styles: Vec<_> = items.iter().map(|i| (i.text.as_str(), i.is_bold)).collect();
        assert_eq!(
            styles,
            [
                ("Plain", false),
                ("Lead", true),
                ("Unknown", false),
                ("Restored", true)
            ]
        );
    }

    #[test]
    fn nested_form_resolves_indirect_paint_resources_without_leaking_state() {
        let (mut doc, page_id) = doc_with_page_and_forms(
            b"0.3 w 2 Tr /X1 Do",
            &[
                b"/Tone cs 0.2 0.3 0.4 sc /Tone CS 0.2 0.3 0.4 SC /State gs
                  BT /F1 12 Tf 72 700 Td (Outer) Tj ET /X2 Do
                  0.2 0.3 0.4 rg BT /F1 12 Tf 72 660 Td (Restored) Tj ET",
                b"/Tone cs 0 sc /Tone CS 0 SC /State gs
                  BT /F1 12 Tf 72 680 Td (Inner) Tj ET",
            ],
        );
        for (id, color_space, state) in [
            ((2, 0), "DeviceRGB", dictionary! { "SM" => 0.02 }),
            ((3, 0), "DeviceGray", dictionary! { "OPM" => 1 }),
        ] {
            let mut resources = doc
                .get_object(id)
                .unwrap()
                .as_stream()
                .unwrap()
                .dict
                .get(b"Resources")
                .unwrap()
                .as_dict()
                .unwrap()
                .clone();
            let space_id = doc.add_object(Object::Name(color_space.as_bytes().to_vec()));
            let state_id = doc.add_object(state);
            let spaces_id = doc.add_object(dictionary! { "Tone" => Object::Reference(space_id) });
            let states_id = doc.add_object(dictionary! { "State" => Object::Reference(state_id) });
            resources.set("ColorSpace", Object::Reference(spaces_id));
            resources.set("ExtGState", Object::Reference(states_id));
            let resources_id = doc.add_object(resources);
            doc.get_object_mut(id)
                .unwrap()
                .as_stream_mut()
                .unwrap()
                .dict
                .set("Resources", Object::Reference(resources_id));
        }
        let (items, _) = extract_page(&doc, page_id, false);
        assert_eq!(
            items.iter().map(|i| i.text.as_str()).collect::<Vec<_>>(),
            ["Outer", "Inner", "Restored"]
        );
        assert!(items.iter().all(|i| i.is_bold));
    }

    #[test]
    fn form_fill_stroke_covers_all_show_operators() {
        for show in [
            "(Styled) Tj",
            "[(Sty) (led)] TJ",
            "(Styled) '",
            "0 0 (Styled) \"",
        ] {
            let content = format!("BT /F1 12 Tf 72 700 Td {show} ET");
            let (doc, page_id) =
                doc_with_page_and_forms(b"0.3 w 2 Tr /X1 Do", &[content.as_bytes()]);
            let (items, _) = extract_page(&doc, page_id, false);
            assert_eq!(items.len(), 1, "{show}: {items:?}");
            assert_eq!(items[0].text, "Styled");
            assert!(items[0].is_bold, "{show}: {items:?}");
        }
    }

    #[test]
    fn unresolved_form_font_does_not_gain_painted_bold() {
        let items = form_items(b"0.3 w 2 Tr BT /Missing 12 Tf 72 700 Td (Alpha) Tj ET");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "Alpha");
        assert!(!items[0].is_bold);
    }

    #[test]
    fn inline_form_fonts_match_referenced_style_and_geometry() {
        for (subtype, name, mode, text, bold, italic) in [
            ("Type1", "Helvetica", 2, "Alpha", true, false),
            ("Type1", "Helvetica-BoldOblique", 0, "Alpha", true, true),
            ("Type1", "Wingdings", 2, "A", false, false),
            ("Type3", "Shape", 2, "A", false, false),
        ] {
            let content = format!("0.3 w {mode} Tr BT /F1 12 Tf 72 700 Td ({text}) Tj ET");
            let (mut referenced_doc, page_id) =
                doc_with_page_and_forms(b"/X1 Do", &[content.as_bytes()]);
            let glyph = referenced_doc.add_object(Stream::new(
                dictionary! {},
                b"600 0 0 0 600 700 d1 0 0 600 700 re f".to_vec(),
            ));
            let font = referenced_doc.get_dictionary_mut((1, 0)).unwrap();
            font.set("Subtype", Object::Name(subtype.as_bytes().to_vec()));
            font.set("BaseFont", Object::Name(name.as_bytes().to_vec()));
            if subtype == "Type3" {
                font.set(
                    "FontMatrix",
                    vec![
                        0.001.into(),
                        0.into(),
                        0.into(),
                        0.001.into(),
                        0.into(),
                        0.into(),
                    ],
                );
                font.set("FontBBox", vec![0.into(), 0.into(), 600.into(), 700.into()]);
                font.set("CharProcs", dictionary! { "A" => Object::Reference(glyph) });
                font.set(
                    "Encoding",
                    dictionary! { "Differences" => vec![65.into(), Object::Name(b"A".to_vec())] },
                );
            }
            let direct_font = font.clone();
            let mut inline_doc = referenced_doc.clone();
            inline_doc
                .get_object_mut((2, 0))
                .unwrap()
                .as_stream_mut()
                .unwrap()
                .dict
                .get_mut(b"Resources")
                .unwrap()
                .as_dict_mut()
                .unwrap()
                .get_mut(b"Font")
                .unwrap()
                .as_dict_mut()
                .unwrap()
                .set("F1", direct_font);
            let (referenced_items, _) = extract_page(&referenced_doc, page_id, false);
            let (inline_items, _) = extract_page(&inline_doc, page_id, false);
            assert_eq!(referenced_items.len(), 1, "{name}");
            assert_eq!(inline_items.len(), 1, "{name}");
            let expected = &referenced_items[0];
            let actual = &inline_items[0];
            assert_eq!(actual.text, text, "{name}");
            assert_eq!((actual.is_bold, actual.is_italic), (bold, italic), "{name}");
            assert_eq!(
                (
                    &actual.font,
                    actual.font_size,
                    actual.width,
                    actual.height,
                    actual.x,
                    actual.y,
                    actual.is_bold,
                    actual.is_italic,
                    actual.advance_known
                ),
                (
                    &expected.font,
                    expected.font_size,
                    expected.width,
                    expected.height,
                    expected.x,
                    expected.y,
                    expected.is_bold,
                    expected.is_italic,
                    expected.advance_known
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn nested_form_inherits_the_outer_forms_text_rendering_mode() {
        // The outer form sets `3 Tr` and invokes the inner form, whose text
        // must stay hidden; an outer run at the default mode stays visible.
        let (doc, page_id) = doc_with_page_and_forms(
            b"q /X1 Do Q",
            &[
                b"BT /F1 12 Tf 72 700 Td (Visible) Tj ET BT 3 Tr ET q /X2 Do Q",
                b"BT /F1 12 Tf 72 650 Td (Hidden) Tj ET",
            ],
        );
        let (items, skipped_invisible) = extract_page(&doc, page_id, false);
        let texts: Vec<&str> = items.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, ["Visible"]);
        assert!(skipped_invisible);
        let (items, _) = extract_page(&doc, page_id, true);
        assert!(items.iter().any(|i| i.text == "Hidden"), "{items:?}");
    }

    #[test]
    fn form_inherits_the_pages_text_rise() {
        // `5 Ts` set by the page stream before `Do`: text state is graphics
        // state, so the form's first run is raised until the form itself
        // resets the rise.
        let (doc, page_id) = doc_with_page_and_forms(
            b"BT 5 Ts ET q /X1 Do Q",
            &[b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (raised) Tj 0 Ts (base) Tj ET"],
        );
        let (items, _) = extract_page(&doc, page_id, false);
        let raised = find(&items, "raised");
        let base = find(&items, "base");
        assert!((raised.y - 505.0).abs() < 0.1, "raised y = {}", raised.y);
        assert!((base.y - 500.0).abs() < 0.1, "base y = {}", base.y);
    }

    #[test]
    fn nested_form_inherits_the_outer_forms_text_rise() {
        // The outer form raises the baseline and invokes the inner form; the
        // inner run is raised, the outer's own run at rise 0 is not.
        let (doc, page_id) = doc_with_page_and_forms(
            b"q /X1 Do Q",
            &[
                b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (outer) Tj ET BT 5 Ts ET q /X2 Do Q",
                b"BT /F1 12 Tf 1 0 0 1 100 400 Tm (inner) Tj ET",
            ],
        );
        let (items, _) = extract_page(&doc, page_id, false);
        let outer = find(&items, "outer");
        let inner = find(&items, "inner");
        assert!((outer.y - 500.0).abs() < 0.1, "outer y = {}", outer.y);
        assert!((inner.y - 405.0).abs() < 0.1, "inner y = {}", inner.y);
    }

    #[test]
    fn form_tj_at_a_negative_size_votes_with_its_own_items() {
        // A form's vertical TJ runs at `-12 Tf` read top-to-bottom: their
        // page-rotation votes must say so, like the items they produce, or
        // the page would be turned against them.
        let (doc, page_id) = doc_with_page_and_forms(
            b"q /X1 Do Q",
            &[b"BT /F1 -12 Tf 0 1 -1 0 100 100 Tm [(UP)] TJ ET BT /F1 -12 Tf 0 1 -1 0 130 100 Tm [(UP)] TJ ET"],
        );
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _, _), _, page_rotation, _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontProductCache::new(),
            &mut FormWalkBudget::new(),
        )
        .unwrap();
        assert_eq!(page_rotation, crate::extractor::geometry::PageRotation::Cw);
        assert_eq!(items.len(), 2, "{items:?}");
        assert!(items.iter().all(|i| i.rotation == 0.0), "{items:?}");
    }

    #[test]
    fn text_rise_inside_form_shifts_the_baseline() {
        // Ts displaces the glyph origin without touching the advance, in a
        // form exactly as in the page stream; the next run at rise 0 returns
        // to the original baseline and follows the raised run horizontally.
        let items = form_items(
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (base) Tj 5 Ts (super) Tj 0 Ts (after) Tj ET",
        );
        let base = find(&items, "base");
        let raised = find(&items, "super");
        let after = find(&items, "after");
        assert!((base.y - 500.0).abs() < 0.1, "base y = {}", base.y);
        assert!((raised.y - 505.0).abs() < 0.1, "raised y = {}", raised.y);
        assert!((after.y - 500.0).abs() < 0.1, "after y = {}", after.y);
        assert!(after.x > raised.x);
    }

    #[test]
    fn rotated_run_inside_form_gets_tall_thin_box() {
        // Same contract as the page-level parser: a 20pt stamp reading
        // bottom-to-top gets its em as width and its advance as height, for
        // both Tj and TJ.
        let items = form_items(
            b"BT /F1 12 Tf 72 700 Td (Body line one) Tj ET
BT /F1 12 Tf 72 686 Td (Body line two) Tj ET
BT /F1 12 Tf 72 672 Td (Body line three) Tj ET
BT /F1 20 Tf 0 1 -1 0 32 200 Tm (arXiv:2301.00001) Tj ET
BT /F1 10 Tf 0 1 -1 0 60 200 Tm [(ABCD)] TJ ET",
        );
        let stamp = find(&items, "arXiv:2301.00001");
        assert!(
            (stamp.rotation - 90.0).abs() < 1e-3,
            "rotation = {}",
            stamp.rotation
        );
        assert!((stamp.x - 12.0).abs() < 0.01, "x = {}", stamp.x);
        assert!((stamp.y - 200.0).abs() < 0.01, "y = {}", stamp.y);
        assert!((stamp.width - 20.0).abs() < 0.01, "width = {}", stamp.width);
        assert!(
            (stamp.height - 192.0).abs() < 0.01,
            "height = {}",
            stamp.height
        );

        let tj = find(&items, "ABCD");
        assert!(
            (tj.rotation - 90.0).abs() < 1e-3,
            "rotation = {}",
            tj.rotation
        );
        assert!((tj.x - 50.0).abs() < 0.01, "x = {}", tj.x);
        assert!((tj.width - 10.0).abs() < 0.01, "width = {}", tj.width);
        assert!((tj.y - 200.0).abs() < 0.01, "y = {}", tj.y);
        assert!((tj.height - 24.0).abs() < 0.01, "height = {}", tj.height);

        let body = find(&items, "Body line one");
        assert_eq!(body.rotation, 0.0);
        assert!(
            (body.width - 13.0 * 7.2).abs() < 0.01,
            "width = {}",
            body.width
        );
        assert_eq!(body.height, 12.0);
    }

    /// The form parser positions `TJ` sub-runs like the page parser: pen
    /// travel ahead of the first glyph moves the box, not just the pen.
    #[test]
    fn tj_positioning_ahead_of_the_first_glyph_inside_form_moves_the_box() {
        // -5400 at 10pt carries the pen 54pt from x=100 to 154, flush against
        // "Intr" (130..154), so the merge pass rejoins the word.
        let items = form_items(
            b"BT /F1 10 Tf 1 0 0 1 130 700 Tm (Intr) Tj 1 0 0 1 100 700 Tm [-5400 (oduction)] TJ ET",
        );
        let word = find(&items, "Introduction");
        assert!((word.x - 130.0).abs() < 0.05, "{items:?}");
        assert!((word.width - 72.0).abs() < 0.05, "{items:?}");

        // A squeezed space run positioned the same way is still the word
        // space of the item before it.
        let items = form_items(
            b"BT /F1 12 Tf 72 700 Td (for) Tj -6 Tc 1 0 0 1 60 700 Tm [-2800 ( )] TJ 0 Tc 1 0 0 1 94.8 700 Tm (the) Tj ET",
        );
        find(&items, "for the");
    }
}
