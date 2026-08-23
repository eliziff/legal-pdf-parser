use crate::{
    javascript_whitespace, locator::normalize_numbered_section_locator,
    text::trim_javascript_whitespace as js_trim, InstrumentCrossReferenceGraph,
    InstrumentCrossReferenceStatus, ScalarText, SourceDoc, SourceDocBlock, SourceDocKind,
    SourceDocOrigin,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

mod text_fragment;
pub use text_fragment::text_fragment_directives;

const JS_WS: &str = r"[\u{0009}-\u{000d}\u{0020}\u{00a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\u{feff}]";

fn regex(pattern: &'static str, cell: &'static OnceLock<Regex>) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("query regex must compile"))
}

fn regex_parts(parts: &[&str], cell: &'static OnceLock<Regex>) -> &'static Regex {
    cell.get_or_init(|| Regex::new(&parts.concat()).expect("query regex must compile"))
}

fn js_regex(pattern: &'static str, cell: &'static OnceLock<Regex>) -> &'static Regex {
    cell.get_or_init(|| {
        Regex::new(&pattern.replace(r"\s", JS_WS)).expect("query regex must compile")
    })
}

fn equal_fold(left: &str, right: &str) -> bool {
    if left.is_ascii() && right.is_ascii() {
        left.eq_ignore_ascii_case(right)
    } else {
        left.to_lowercase() == right.to_lowercase()
    }
}

fn block_matches(block: &SourceDocBlock, label: &str) -> bool {
    equal_fold(&block.label, label) || block.aliases.iter().any(|alias| equal_fold(alias, label))
}

fn context(value: usize) -> usize {
    value.min(2)
}

fn slice_utf16<'a>(text: &ScalarText<'a>, start: usize, end: usize) -> &'a str {
    let Some(start) = text.byte_at_utf16(start) else {
        return "";
    };
    let Some(end) = text.byte_at_utf16(end) else {
        return "";
    };
    text.value.get(start..end).unwrap_or("")
}

#[derive(Clone, Serialize)]
pub struct MaterializedSourceDocBlock {
    #[serde(flatten)]
    pub block: SourceDocBlock,
    pub text: String,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDocLookupStatus {
    Found,
    NotFound,
    Unavailable,
    Ambiguous,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceDocLookup {
    pub status: SourceDocLookupStatus,
    pub requested_label: String,
    pub matches: Vec<String>,
    pub block: Option<MaterializedSourceDocBlock>,
    pub before: Vec<MaterializedSourceDocBlock>,
    pub after: Vec<MaterializedSourceDocBlock>,
}

#[derive(Serialize)]
pub struct SourceDocRangeLookup {
    pub selected: Vec<MaterializedSourceDocBlock>,
    pub before: Vec<MaterializedSourceDocBlock>,
    pub after: Vec<MaterializedSourceDocBlock>,
}

#[derive(Clone, Serialize)]
pub struct SourceDocWordSpan {
    pub word: String,
    pub start: usize,
    pub end: usize,
}

struct SearchIndex {
    tokens: Vec<SourceDocWordSpan>,
    postings: HashMap<String, Vec<usize>>,
}

impl SearchIndex {
    fn new(text: &str) -> Self {
        Self::with_tokens(tokenize_source_text(text))
    }

    fn with_scalar(text: &str, scalar: &ScalarText<'_>) -> Self {
        Self::with_tokens(tokenize_with_scalar(text, scalar))
    }

    fn with_tokens(tokens: Vec<SourceDocWordSpan>) -> Self {
        let mut postings = HashMap::<String, Vec<usize>>::new();
        for (position, token) in tokens.iter().enumerate() {
            if let Some(posting) = postings.get_mut(&token.word) {
                posting.push(position);
            } else {
                postings.insert(token.word.clone(), vec![position]);
            }
        }
        Self { tokens, postings }
    }
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseOptions {
    pub start: Option<usize>,
    pub end: Option<usize>,
    #[serde(default)]
    pub same_line: bool,
    pub limit: Option<usize>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseSpan {
    pub start: usize,
    pub end: usize,
    pub first_word: usize,
    pub last_word: usize,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageSpan {
    pub ordinal: usize,
    pub pdf_page: Option<usize>,
    pub printed_label: Option<String>,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PageMapSource {
    Artifact,
    Markers,
    Unpaginated,
    Unindexed,
}

#[derive(Serialize)]
pub struct PageMap {
    pub pages: Vec<PageSpan>,
    pub source: PageMapSource,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PageSense {
    Pdf,
    Printed,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PageLookup {
    Found {
        page: PageSpan,
        #[serde(rename = "matchedOn")]
        matched_on: PageSense,
        text: String,
    },
    NoPages,
    NotFound {
        requested: String,
        sense: PageSense,
        count: usize,
        first: Option<String>,
        last: Option<String>,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DocumentAddress {
    Section { locator: String },
    Page { spec: String },
    Offset { start: usize },
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FollowDirection {
    None,
    Out,
    In,
    Both,
}

#[derive(Serialize)]
pub struct GraphScope {
    pub seed: MaterializedSourceDocBlock,
    pub nodes: Vec<GraphScopeNode>,
    pub depth: usize,
}

#[derive(Serialize)]
pub struct GraphScopeNode {
    #[serde(flatten)]
    pub block: MaterializedSourceDocBlock,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<Vec<MaterializedSourceDocBlock>>,
}

pub struct SourceDocQuery {
    document: SourceDoc,
    queries: AtomicUsize,
    search: OnceLock<SearchIndex>,
    line_breaks: OnceLock<Vec<usize>>,
}

impl SourceDocQuery {
    pub fn new(document: SourceDoc) -> Self {
        Self {
            document,
            queries: AtomicUsize::new(0),
            search: OnceLock::new(),
            line_breaks: OnceLock::new(),
        }
    }

    pub fn document(&self) -> &SourceDoc {
        &self.document
    }

    pub fn into_document(self) -> SourceDoc {
        self.document
    }

    pub fn tokens(&self) -> &[SourceDocWordSpan] {
        &self.index().tokens
    }

    fn blocks(&self, kind: SourceDocKind) -> Vec<&SourceDocBlock> {
        self.document
            .blocks
            .iter()
            .filter(|block| block.kind == kind)
            .collect()
    }

    fn materialize(
        &self,
        block: &SourceDocBlock,
        text: &ScalarText<'_>,
    ) -> MaterializedSourceDocBlock {
        MaterializedSourceDocBlock {
            block: block.clone(),
            text: js_trim(slice_utf16(text, block.start, block.end)).to_owned(),
        }
    }

    pub fn subtree_labels(&self, seed_label: &str) -> Vec<String> {
        let by_label = self
            .document
            .blocks
            .iter()
            .map(|block| (block.label.as_str(), block))
            .collect::<HashMap<_, _>>();
        let mut labels = Vec::new();
        for block in &self.document.blocks {
            let mut current = Some(block);
            let mut seen = HashSet::new();
            while let Some(candidate) = current {
                if !seen.insert(candidate.label.as_str()) {
                    break;
                }
                if candidate.label == seed_label {
                    labels.push(block.label.clone());
                    break;
                }
                current = candidate
                    .parent_label
                    .as_deref()
                    .and_then(|parent| by_label.get(parent).copied());
            }
        }
        labels
    }

    pub fn has_native_ancestor(&self, kind: SourceDocKind, label: &str) -> bool {
        let mut current = self
            .document
            .index
            .get(label)
            .and_then(|position| self.document.blocks.get(position))
            .filter(|block| block.kind == kind);
        let mut seen = HashSet::new();
        while let Some(block) = current {
            if !seen.insert(block.label.as_str()) {
                return false;
            }
            if block.origin == SourceDocOrigin::Native {
                return true;
            }
            current = block.parent_label.as_deref().and_then(|parent| {
                self.document
                    .index
                    .get(parent)
                    .and_then(|position| self.document.blocks.get(position))
                    .filter(|candidate| candidate.kind == kind)
            });
        }
        false
    }

    pub fn lookup(
        &self,
        kind: SourceDocKind,
        locator: &str,
        context_blocks: usize,
    ) -> SourceDocLookup {
        self.lookup_with_text(kind, locator, context_blocks, None)
    }

    fn lookup_with_text(
        &self,
        kind: SourceDocKind,
        locator: &str,
        context_blocks: usize,
        text: Option<&ScalarText<'_>>,
    ) -> SourceDocLookup {
        let requested = self.requested_label(kind, locator);
        self.lookup_label_with_text(kind, &requested, context_blocks, text)
    }

    fn requested_label(&self, kind: SourceDocKind, locator: &str) -> String {
        let exact = js_trim(locator);
        let exact_label = self
            .document
            .blocks
            .iter()
            .any(|block| block.kind == kind && equal_fold(&block.label, exact));
        let requested = if exact_label {
            exact.to_owned()
        } else {
            let normalized = normalize_source_doc_locator(kind, locator);
            if !normalized.is_empty() {
                normalized
            } else if self.document.blocks.iter().any(|block| {
                block.kind == kind
                    && block
                        .anchor
                        .as_deref()
                        .is_some_and(|anchor| equal_fold(anchor, exact))
                    || block.kind == kind
                        && block.aliases.iter().any(|alias| equal_fold(alias, exact))
            }) {
                exact.to_owned()
            } else {
                String::new()
            }
        };
        requested
    }

    pub fn lookup_label(
        &self,
        kind: SourceDocKind,
        requested_label: &str,
        context_blocks: usize,
    ) -> SourceDocLookup {
        self.lookup_label_with_text(kind, requested_label, context_blocks, None)
    }

    fn lookup_label_with_text(
        &self,
        kind: SourceDocKind,
        requested_label: &str,
        context_blocks: usize,
        text: Option<&ScalarText<'_>>,
    ) -> SourceDocLookup {
        let available = self.blocks(kind);
        let empty = |status| SourceDocLookup {
            status,
            requested_label: requested_label.to_owned(),
            matches: Vec::new(),
            block: None,
            before: Vec::new(),
            after: Vec::new(),
        };
        if requested_label.is_empty() || available.is_empty() {
            return empty(SourceDocLookupStatus::Unavailable);
        }
        let selected = self
            .document
            .index
            .get(requested_label)
            .and_then(|position| self.document.blocks.get(position))
            .filter(|block| block.kind == kind);
        let Some(selected) = selected else {
            let matches = available
                .iter()
                .filter(|block| block_matches(block, requested_label))
                .map(|block| block.label.clone())
                .collect::<Vec<_>>();
            return SourceDocLookup {
                status: if matches.is_empty() {
                    SourceDocLookupStatus::NotFound
                } else {
                    SourceDocLookupStatus::Ambiguous
                },
                matches,
                ..empty(SourceDocLookupStatus::NotFound)
            };
        };
        let order = available
            .iter()
            .position(|block| std::ptr::eq(*block, selected))
            .unwrap_or(0);
        let context = context(context_blocks);
        let owned_text;
        let text = if let Some(text) = text {
            text
        } else {
            owned_text = ScalarText::new(&self.document.text);
            &owned_text
        };
        SourceDocLookup {
            status: SourceDocLookupStatus::Found,
            requested_label: requested_label.to_owned(),
            matches: vec![selected.label.clone()],
            block: Some(self.materialize(selected, &text)),
            before: available[order.saturating_sub(context)..order]
                .iter()
                .map(|block| self.materialize(block, &text))
                .collect(),
            after: available[order + 1..(order + 1 + context).min(available.len())]
                .iter()
                .map(|block| self.materialize(block, &text))
                .collect(),
        }
    }

    pub fn read_range(
        &self,
        kind: SourceDocKind,
        from: &str,
        to: &str,
        context_blocks: usize,
    ) -> Option<SourceDocRangeLookup> {
        let text = ScalarText::new(&self.document.text);
        let available = self.blocks(kind);
        let resolve = |locator: &str| {
            let label = self.requested_label(kind, locator);
            let block = self
                .document
                .index
                .get(&label)
                .and_then(|position| self.document.blocks.get(position))
                .filter(|block| block.kind == kind)?;
            available
                .iter()
                .position(|candidate| std::ptr::eq(*candidate, block))
        };
        let first_index = resolve(from)?;
        let last_index = resolve(to)?;
        let (low, high) = if first_index <= last_index {
            (first_index, last_index)
        } else {
            (last_index, first_index)
        };
        let context = context(context_blocks);
        Some(SourceDocRangeLookup {
            selected: self.materialize_leaf_blocks(&available[low..=high], &text),
            before: self
                .materialize_leaf_blocks(&available[low.saturating_sub(context)..low], &text),
            after: self.materialize_leaf_blocks(
                &available[high + 1..(high + 1 + context).min(available.len())],
                &text,
            ),
        })
    }

    fn contained_leaf_blocks<'a>(&'a self, blocks: &[&SourceDocBlock]) -> Vec<&'a SourceDocBlock> {
        let Some(kind) = blocks.first().map(|block| block.kind) else {
            return Vec::new();
        };
        let mut ranges = blocks
            .iter()
            .map(|block| (block.start, block.end))
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        let mut maximum_ends = Vec::<usize>::with_capacity(ranges.len());
        for &(_, end) in &ranges {
            maximum_ends.push(maximum_ends.last().copied().unwrap_or(0).max(end));
        }
        let contained = self
            .document
            .blocks
            .iter()
            .filter(|block| {
                if block.kind != kind {
                    return false;
                }
                let count = ranges.partition_point(|(start, _)| *start <= block.start);
                count > 0 && maximum_ends[count - 1] >= block.end
            })
            .collect::<Vec<_>>();
        if kind != SourceDocKind::Section || contained.len() <= 1 {
            return contained;
        }
        let mut ordered = (0..contained.len()).collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|&index| {
            (
                contained[index].start,
                std::cmp::Reverse(contained[index].end),
            )
        });
        let mut stack = Vec::<usize>::new();
        let mut parents = HashSet::new();
        for index in ordered {
            let candidate = contained[index];
            while stack
                .last()
                .is_some_and(|&parent| contained[parent].end < candidate.end)
            {
                stack.pop();
            }
            for &parent in &stack {
                let parent = contained[parent];
                if parent.end >= candidate.end
                    && (parent.start < candidate.start || parent.end > candidate.end)
                {
                    parents.insert(parent as *const SourceDocBlock);
                }
            }
            stack.push(index);
        }
        contained
            .into_iter()
            .filter(|candidate| !parents.contains(&(*candidate as *const SourceDocBlock)))
            .collect()
    }

    fn materialize_leaf_blocks(
        &self,
        blocks: &[&SourceDocBlock],
        text: &ScalarText<'_>,
    ) -> Vec<MaterializedSourceDocBlock> {
        self.contained_leaf_blocks(blocks)
            .into_iter()
            .map(|unit| self.materialize(unit, text))
            .filter(|unit| !unit.text.is_empty())
            .collect()
    }

    pub fn smallest_containing_block(
        &self,
        start: usize,
        end: usize,
    ) -> Option<MaterializedSourceDocBlock> {
        let block = self
            .document
            .blocks
            .iter()
            .filter(|block| block.start <= start && block.end >= end)
            .min_by_key(|block| block.end - block.start)?;
        Some(self.materialize(block, &ScalarText::new(&self.document.text)))
    }

    fn index(&self) -> &SearchIndex {
        self.search
            .get_or_init(|| SearchIndex::new(&self.document.text))
    }

    fn index_with_text(&self, text: &ScalarText<'_>) -> &SearchIndex {
        self.search
            .get_or_init(|| SearchIndex::with_scalar(&self.document.text, text))
    }

    fn line_breaks(&self) -> &[usize] {
        self.line_breaks.get_or_init(|| {
            let text = ScalarText::new(&self.document.text);
            collect_line_breaks(&self.document.text, &text)
        })
    }

    fn line_breaks_with_text(&self, text: &ScalarText<'_>) -> &[usize] {
        self.line_breaks
            .get_or_init(|| collect_line_breaks(&self.document.text, text))
    }

    pub fn phrase_spans(&self, words: &[String], options: PhraseOptions) -> Vec<PhraseSpan> {
        if words.is_empty() {
            return Vec::new();
        }
        let query = self.queries.fetch_add(1, Ordering::Relaxed) + 1;
        if query == 1
            && self.search.get().is_none()
            && options.start.is_none()
            && options.end.is_none()
        {
            return scan_phrase_spans(&self.document.text, words, options);
        }
        indexed_phrase_spans(self.index(), self.line_breaks(), words, options)
    }

    fn phrase_spans_with_text(
        &self,
        words: &[String],
        options: PhraseOptions,
        text: &ScalarText<'_>,
    ) -> Vec<PhraseSpan> {
        if words.is_empty() {
            return Vec::new();
        }
        let query = self.queries.fetch_add(1, Ordering::Relaxed) + 1;
        if query == 1
            && self.search.get().is_none()
            && options.start.is_none()
            && options.end.is_none()
        {
            return scan_phrase_spans_with_text(&self.document.text, text, words, options);
        }
        indexed_phrase_spans(
            self.index_with_text(text),
            self.line_breaks_with_text(text),
            words,
            options,
        )
    }

    pub fn contains_quote(&self, quote: &str, start: Option<usize>, end: Option<usize>) -> bool {
        !self
            .phrase_spans(
                &quote_words(quote),
                PhraseOptions {
                    start,
                    end,
                    limit: Some(1),
                    same_line: false,
                },
            )
            .is_empty()
    }

    pub fn page_map(&self) -> PageMap {
        let mut pages = Vec::new();
        for block in self
            .document
            .blocks
            .iter()
            .filter(|block| block.kind == SourceDocKind::Page)
        {
            let pdf_page: Option<usize> = block
                .anchor
                .as_deref()
                .and_then(|anchor| anchor.strip_prefix("page="))
                .filter(|value| {
                    !value.is_empty()
                        && value.len() <= 6
                        && value.bytes().all(|byte| byte.is_ascii_digit())
                })
                .and_then(|value| value.parse().ok());
            let printed_label = block
                .aliases
                .iter()
                .map(|alias| js_trim(alias))
                .find(|alias| {
                    !alias.is_empty()
                        && *alias
                            != pdf_page.map_or_else(|| "null".to_owned(), |page| page.to_string())
                })
                .map(str::to_owned);
            pages.push(PageSpan {
                ordinal: pages.len() + 1,
                pdf_page,
                printed_label,
                start: block.start,
                end: block.end,
            });
        }
        pages.sort_by_key(|page| page.start);
        for (index, page) in pages.iter_mut().enumerate() {
            page.ordinal = index + 1;
        }
        if pages.is_empty() {
            return page_map_from_markers(&self.document.text);
        }
        PageMap {
            source: PageMapSource::Artifact,
            pages,
        }
    }

    pub fn resolve_page(&self, requested: &str) -> PageLookup {
        resolve_page(&self.page_map(), &self.document.text, requested)
    }

    pub fn structure_block(&self, locator: &str, context_blocks: usize) -> SourceDocLookup {
        let direct = js_trim(locator).to_lowercase();
        if direct.starts_with("table:") {
            let kind = if direct.contains("/col:") {
                SourceDocKind::Cell
            } else if direct.contains("/row:") {
                SourceDocKind::Row
            } else {
                SourceDocKind::Table
            };
            return self.lookup_label(kind, &direct, context_blocks);
        }
        let normalized = normalize_source_doc_locator(SourceDocKind::Section, locator);
        if !normalized.is_empty() {
            let found = self.lookup_label(SourceDocKind::Section, &normalized, context_blocks);
            if !matches!(found.status, SourceDocLookupStatus::NotFound) {
                return found;
            }
        }
        self.lookup_label(SourceDocKind::Section, &direct, context_blocks)
    }

    pub fn graph_scope(
        &self,
        graph: &InstrumentCrossReferenceGraph,
        seed_label: &str,
        follow: FollowDirection,
        depth: usize,
        include_descendants: bool,
        include_units: bool,
    ) -> Option<GraphScope> {
        let wanted = js_trim(seed_label).to_lowercase();
        let seed = self
            .document
            .blocks
            .iter()
            .find(|block| block.label.to_lowercase() == wanted)?;
        let limit = depth.min(3);
        let mut by_label = HashMap::new();
        for block in &self.document.blocks {
            by_label.entry(block.label.as_str()).or_insert(block);
        }
        let initial = if include_descendants {
            self.subtree_labels(&seed.label)
        } else {
            vec![seed.label.clone()]
        };
        let initial = initial.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut reached = HashMap::from([(seed.label.as_str(), seed)]);
        let mut frontier = initial.iter().copied().collect::<Vec<_>>();
        let mut hops = 0;
        while follow != FollowDirection::None && hops < limit && !frontier.is_empty() {
            let in_frontier = frontier.iter().copied().collect::<HashSet<_>>();
            let mut next = Vec::new();
            for edge in &graph.edges {
                if edge.status != InstrumentCrossReferenceStatus::Resolved || edge.self_loop {
                    continue;
                }
                let forward = matches!(follow, FollowDirection::Out | FollowDirection::Both)
                    && edge
                        .source_label
                        .as_deref()
                        .is_some_and(|label| in_frontier.contains(label));
                let backward = matches!(follow, FollowDirection::In | FollowDirection::Both)
                    && edge
                        .target_label
                        .as_deref()
                        .is_some_and(|label| in_frontier.contains(label));
                let other = if forward {
                    edge.target_label.as_deref()
                } else if backward {
                    edge.source_label.as_deref()
                } else {
                    None
                };
                let Some(other) = other else { continue };
                if initial.contains(other) || reached.contains_key(other) {
                    continue;
                }
                let Some(node) = by_label.get(other).copied() else {
                    continue;
                };
                reached.insert(other, node);
                next.push(other);
            }
            frontier = next;
            hops += 1;
            if frontier.is_empty() {
                break;
            }
        }
        let mut rest = reached
            .values()
            .copied()
            .filter(|block| block.label != seed.label)
            .collect::<Vec<_>>();
        rest.sort_by_key(|block| block.start);
        let text = ScalarText::new(&self.document.text);
        let seed = self.materialize(seed, &text);
        Some(GraphScope {
            seed,
            nodes: rest
                .into_iter()
                .map(|block| {
                    let materialized = self.materialize(block, &text);
                    let units =
                        include_units.then(|| self.materialize_leaf_blocks(&[block], &text));
                    let units = units.filter(|units| {
                        units.len() != 1
                            || units[0].block.start != block.start
                            || units[0].block.end != block.end
                    });
                    GraphScopeNode {
                        block: materialized,
                        units,
                    }
                })
                .collect(),
            depth: hops.min(limit),
        })
    }
}

pub fn normalize_source_doc_locator(kind: SourceDocKind, locator: &str) -> String {
    static FOOTNOTE: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH: OnceLock<Regex> = OnceLock::new();
    static PAGE: OnceLock<Regex> = OnceLock::new();
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    static CANONICAL_PREFIX: OnceLock<Regex> = OnceLock::new();
    static HEADING: OnceLock<Regex> = OnceLock::new();
    static NON_TITLE: OnceLock<Regex> = OnceLock::new();
    let value = js_trim(locator);
    let numbered = |capture: &regex::Captures<'_>, prefix: &str| {
        format!("{prefix}{}", capture[1].parse::<usize>().unwrap_or(0))
    };
    match kind {
        SourceDocKind::Footnote => {
            return regex_parts(
                &[
                    r"(?iu)^(?:fn|footnotes?|notes?)?(?:",
                    JS_WS,
                    r"|[#.])*(\d{1,5})$",
                ],
                &FOOTNOTE,
            )
            .captures(value)
            .map_or_else(String::new, |capture| numbered(&capture, "fn"))
        }
        SourceDocKind::Paragraph => {
            return regex_parts(
                &[
                    r"(?iu)^(?:\[",
                    JS_WS,
                    r"*)?(?:paras?\.?|paragraphs?)?",
                    JS_WS,
                    r"*(\d{1,4})(?:",
                    JS_WS,
                    r"*\])?$",
                ],
                &PARAGRAPH,
            )
            .captures(value)
            .map_or_else(String::new, |capture| numbered(&capture, "par"))
        }
        SourceDocKind::Page => {
            return regex_parts(&[r"(?iu)^(?:pages?|pp?\.)?", JS_WS, r"*(\d{1,4})$"], &PAGE)
                .captures(value)
                .map_or_else(String::new, |capture| numbered(&capture, "page"))
        }
        SourceDocKind::Section => {}
        _ => return String::new(),
    }
    let without_prefix =
        regex_parts(&[r"(?iu)^(?:sections?|ss?\.?)", JS_WS, "+"], &PREFIX).replace(value, "");
    let without_prefix =
        regex(r"(?iu)^sec([\p{L}\p{N}])", &CANONICAL_PREFIX).replace(&without_prefix, "$1");
    let numbered = normalize_numbered_section_locator(&without_prefix);
    if !numbered.is_empty() {
        return numbered;
    }
    let heading = without_prefix
        .trim_end_matches(|character| character == '.' || javascript_whitespace(character));
    if regex(r"^(?:[IVXLCDM]+|[A-Z])$", &HEADING).is_match(heading) {
        return format!("sec{heading}");
    }
    let lowercase = heading.to_lowercase();
    let normalized = regex(r"[^\p{L}\p{N}]+", &NON_TITLE).replace_all(&lowercase, " ");
    let title = js_trim(&normalized);
    if title.is_empty() {
        String::new()
    } else {
        format!("sectitle:{title}")
    }
}

fn tokenize_with_scalar(text: &str, scalar: &ScalarText<'_>) -> Vec<SourceDocWordSpan> {
    static WORD: OnceLock<Regex> = OnceLock::new();
    regex(r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*", &WORD)
        .find_iter(text)
        .map(|found| SourceDocWordSpan {
            word: found.as_str().to_lowercase(),
            start: scalar.utf16_at_byte(found.start()).expect("token boundary"),
            end: scalar.utf16_at_byte(found.end()).expect("token boundary"),
        })
        .collect()
}

pub fn tokenize_source_text(text: &str) -> Vec<SourceDocWordSpan> {
    tokenize_with_scalar(text, &ScalarText::new(text))
}

pub fn quote_text(value: &str) -> String {
    static BRACKET_LETTER: OnceLock<Regex> = OnceLock::new();
    static BRACKETS: OnceLock<Regex> = OnceLock::new();
    static ELISION: OnceLock<Regex> = OnceLock::new();
    let value =
        js_trim(value).trim_matches(|character| matches!(character, '"' | '\'' | '“' | '”'));
    let value = regex(r"\[([A-Za-z])\]([A-Za-z])", &BRACKET_LETTER).replace_all(value, "$1$2");
    let value = regex(r"\[([^\]]+)\]", &BRACKETS).replace_all(&value, "$1");
    let value = regex(r"\.{3}|…", &ELISION).replace_all(&value, " ");
    let mut normalized = String::with_capacity(value.len());
    let mut separating = false;
    for character in value.chars() {
        if matches!(
            character,
            ' ' | '\t' | '\r' | '\n' | '\u{000c}' | '\u{000b}'
        ) {
            separating = !normalized.is_empty();
        } else {
            if separating {
                normalized.push(' ');
            }
            normalized.push(character);
            separating = false;
        }
    }
    normalized
}

pub fn quote_words(quote: &str) -> Vec<String> {
    tokenize_source_text(&quote_text(quote))
        .into_iter()
        .map(|token| token.word)
        .collect()
}

fn token_index_at_or_after(tokens: &[SourceDocWordSpan], offset: usize) -> usize {
    tokens.partition_point(|token| token.start < offset)
}

fn collect_line_breaks(text: &str, scalar: &ScalarText<'_>) -> Vec<usize> {
    text.match_indices('\n')
        .map(|(byte, _)| {
            scalar
                .utf16_at_byte(byte)
                .expect("line break is a scalar boundary")
        })
        .collect()
}

fn crosses_line_break(line_breaks: &[usize], start: usize, end: usize) -> bool {
    let at = line_breaks.partition_point(|offset| *offset < start);
    at < line_breaks.len() && line_breaks[at] < end
}

fn indexed_phrase_spans(
    index: &SearchIndex,
    line_breaks: &[usize],
    words: &[String],
    options: PhraseOptions,
) -> Vec<PhraseSpan> {
    let from = options
        .start
        .map_or(0, |offset| token_index_at_or_after(&index.tokens, offset));
    let until = options.end.map_or(index.tokens.len(), |offset| {
        token_index_at_or_after(&index.tokens, offset)
    });
    let limit = options.limit.unwrap_or(usize::MAX);
    let Some((anchor, _)) = words
        .iter()
        .enumerate()
        .map(|(offset, word)| (offset, index.postings.get(word).map_or(0, Vec::len)))
        .filter(|(_, size)| *size > 0)
        .min_by_key(|(_, size)| *size)
    else {
        return Vec::new();
    };
    if words.iter().any(|word| !index.postings.contains_key(word)) {
        return Vec::new();
    }
    let mut spans = Vec::new();
    for &position in &index.postings[&words[anchor]] {
        let Some(start) = position.checked_sub(anchor) else {
            continue;
        };
        if start < from {
            continue;
        }
        if start + words.len() > until {
            break;
        }
        if index.tokens[start..start + words.len()]
            .iter()
            .zip(words)
            .any(|(token, word)| token.word != *word)
        {
            continue;
        }
        let first = &index.tokens[start];
        let last = &index.tokens[start + words.len() - 1];
        if options.same_line && crosses_line_break(line_breaks, first.start, last.end) {
            continue;
        }
        spans.push(PhraseSpan {
            start: first.start,
            end: last.end,
            first_word: start,
            last_word: start + words.len() - 1,
        });
        if spans.len() >= limit {
            break;
        }
    }
    spans
}

fn scan_phrase_spans_with_text(
    text: &str,
    scalar: &ScalarText<'_>,
    words: &[String],
    options: PhraseOptions,
) -> Vec<PhraseSpan> {
    static WORD: OnceLock<Regex> = OnceLock::new();
    let size = words.len();
    let limit = options.limit.unwrap_or(usize::MAX);
    let mut ring = vec![
        SourceDocWordSpan {
            word: String::new(),
            start: 0,
            end: 0
        };
        size
    ];
    let mut spans = Vec::new();
    let mut seen = 0;
    for found in regex(r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*", &WORD).find_iter(text) {
        let slot = &mut ring[seen % size];
        slot.word = found.as_str().to_lowercase();
        slot.start = scalar.utf16_at_byte(found.start()).expect("token boundary");
        slot.end = scalar.utf16_at_byte(found.end()).expect("token boundary");
        seen += 1;
        if seen < size
            || (0..size).any(|offset| ring[(seen - size + offset) % size].word != words[offset])
        {
            continue;
        }
        let first = &ring[(seen - size) % size];
        let last = &ring[(seen - 1) % size];
        if options.start.is_some_and(|start| first.start < start)
            || options.end.is_some_and(|end| last.start >= end)
        {
            continue;
        }
        if options.same_line {
            let start_byte = scalar.byte_at_utf16(first.start).expect("token boundary");
            let end_byte = scalar.byte_at_utf16(last.end).expect("token boundary");
            if text[start_byte..end_byte].contains('\n') {
                continue;
            }
        }
        spans.push(PhraseSpan {
            start: first.start,
            end: last.end,
            first_word: seen - size,
            last_word: seen - 1,
        });
        if spans.len() >= limit {
            break;
        }
    }
    spans
}

fn scan_phrase_spans(text: &str, words: &[String], options: PhraseOptions) -> Vec<PhraseSpan> {
    scan_phrase_spans_with_text(text, &ScalarText::new(text), words, options)
}

pub fn phrase_spans(text: &str, words: &[String], options: PhraseOptions) -> Vec<PhraseSpan> {
    if words.is_empty() {
        Vec::new()
    } else {
        scan_phrase_spans(text, words, options)
    }
}

pub fn page_map_from_markers(text: &str) -> PageMap {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    let scalar = ScalarText::new(text);
    let mut pages = Vec::<PageSpan>::new();
    for capture in regex(r"(?m)^\[page ([^\]\n]{1,40})\]$", &MARKER).captures_iter(text) {
        let label = js_trim(&capture[1]);
        if label.is_empty() {
            continue;
        }
        let start = scalar
            .utf16_at_byte(capture.get(0).expect("full match").start())
            .expect("marker boundary");
        if let Some(previous) = pages.last_mut() {
            previous.end = start;
        }
        let pdf_page = (label.len() <= 6 && label.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| label.parse().ok())
            .flatten();
        pages.push(PageSpan {
            ordinal: pages.len() + 1,
            pdf_page,
            printed_label: Some(label.to_owned()),
            start,
            end: scalar.utf16_len(),
        });
    }
    PageMap {
        source: if pages.is_empty() {
            PageMapSource::Unpaginated
        } else {
            PageMapSource::Markers
        },
        pages,
    }
}

pub fn resolve_page(map: &PageMap, text: &str, requested: &str) -> PageLookup {
    static QUALIFIED: OnceLock<Regex> = OnceLock::new();
    if map.pages.is_empty() {
        return PageLookup::NoPages;
    }
    let raw = js_trim(requested);
    let qualified = regex_parts(
        &[r"(?iu)^(pdf|printed)", JS_WS, r"*[:=]", JS_WS, r"*(.+)$"],
        &QUALIFIED,
    )
    .captures(raw);
    let wanted = js_trim(qualified.as_ref().map_or(raw, |capture| &capture[2]));
    let sense = qualified
        .as_ref()
        .map(|capture| capture[1].to_lowercase())
        .map_or_else(
            || {
                if !wanted.is_empty()
                    && wanted.len() <= 6
                    && wanted.bytes().all(|byte| byte.is_ascii_digit())
                {
                    PageSense::Pdf
                } else {
                    PageSense::Printed
                }
            },
            |sense| {
                if sense == "pdf" {
                    PageSense::Pdf
                } else {
                    PageSense::Printed
                }
            },
        );
    let page = map.pages.iter().find(|page| match sense {
        PageSense::Pdf => {
            page.pdf_page
                .map_or_else(|| "null".to_owned(), |value| value.to_string())
                == wanted
        }
        PageSense::Printed => page
            .printed_label
            .as_deref()
            .is_some_and(|label| equal_fold(label, wanted)),
    });
    if let Some(page) = page {
        let scalar = ScalarText::new(text);
        return PageLookup::Found {
            page: page.clone(),
            matched_on: sense,
            text: slice_utf16(&scalar, page.start, page.end).to_owned(),
        };
    }
    let describe = |page: &PageSpan| {
        let number = page.pdf_page.unwrap_or(page.ordinal);
        if page.printed_label.as_deref().is_some_and(|label| {
            label
                != page
                    .pdf_page
                    .map_or_else(|| "null".to_owned(), |value| value.to_string())
        }) {
            format!(
                "PDF page {number} (printed \"{}\")",
                page.printed_label.as_deref().unwrap_or_default()
            )
        } else {
            format!("PDF page {number}")
        }
    };
    PageLookup::NotFound {
        requested: raw.to_owned(),
        sense,
        count: map.pages.len(),
        first: map.pages.first().map(&describe),
        last: map.pages.last().map(describe),
    }
}

pub fn parse_address(spec: &str) -> Option<DocumentAddress> {
    static PAGE: OnceLock<Regex> = OnceLock::new();
    static OFFSET: OnceLock<Regex> = OnceLock::new();
    static SECTION: OnceLock<Regex> = OnceLock::new();
    let raw = js_trim(spec);
    if raw.is_empty() {
        return None;
    }
    if let Some(capture) = regex_parts(
        &[
            r"(?iu)^(printed|pdf|page|pg|p)(?-u:\b)",
            JS_WS,
            r"*[:.]?",
            JS_WS,
            r"*(.+)$",
        ],
        &PAGE,
    )
    .captures(raw)
    {
        let qualifier = capture[1].to_lowercase();
        let value = js_trim(&capture[2]);
        return Some(DocumentAddress::Page {
            spec: if matches!(qualifier.as_str(), "pdf" | "printed") {
                format!("{qualifier}:{value}")
            } else {
                value.to_owned()
            },
        });
    }
    if let Some(capture) = regex_parts(
        &[
            r"(?iu)^(?:off|offset)",
            JS_WS,
            r"*[:.]?",
            JS_WS,
            r"*(\d{1,9})$",
        ],
        &OFFSET,
    )
    .captures(raw)
    {
        return Some(DocumentAddress::Offset {
            start: capture[1].parse().expect("bounded digits"),
        });
    }
    let locator = regex_parts(
        &[r"(?iu)^(?:sec|art|sched)", JS_WS, r"*[:.]", JS_WS, "*"],
        &SECTION,
    )
    .replace(raw, "")
    .into_owned();
    Some(DocumentAddress::Section { locator })
}
