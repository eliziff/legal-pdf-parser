#[cfg(feature = "recovery")]
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::io::{BufRead, Write};
#[cfg(feature = "recovery")]
use std::sync::OnceLock;

#[cfg(feature = "a2aj")]
mod a2aj;
#[cfg(feature = "journal")]
mod journal;
#[cfg(all(feature = "recovery", feature = "source-doc"))]
mod native_markup;
#[cfg(feature = "source-doc")]
mod source_doc;
#[cfg(feature = "a2aj")]
pub use a2aj::{a2aj_source_doc, A2ajInput, A2ajSectionMap, A2ajSourceKind};
#[cfg(feature = "journal")]
pub use journal::{journal_source_doc, journal_text_source_doc, JournalPageLabel};
#[cfg(all(feature = "recovery", feature = "source-doc"))]
pub use native_markup::{native_markup_source_doc, NativeMarkupInput};
#[cfg(feature = "source-doc")]
pub use source_doc::{
    create_source_doc, ProjectionOrder, SourceDoc, SourceDocBlock, SourceDocIndex, SourceDocKind,
    SourceDocOrigin, SourceDocProvider, SourceDocType,
};

pub const EVIDENCE_SCHEMA: &str = "legalpdf.structure-evidence.v1";
pub const RESULT_SCHEMA: &str = "legalpdf.structure-graph.v1";
pub const SIDECAR_PROTOCOL: &str = "legalpdf.structure-sidecar.v1";
pub const SOURCE_DOC_VERSION: u32 = 1;
const ENGINE_ORIGIN: &str = "legalpdf.structure-engine";
const MAX_DOCUMENTS: usize = 25;
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy)]
pub struct NumericSequenceCandidate {
    pub index: usize,
    pub value: u32,
    pub position: (usize, usize),
    pub page: u32,
    pub score: f64,
    pub start_supported: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum NumericSequencePolicy {
    RootedConsecutive,
    FootnoteBackbone,
}

#[derive(Debug, PartialEq)]
pub struct NumericSequenceSelection {
    pub indices: Vec<usize>,
    pub score: f64,
}

pub fn select_numeric_sequence(
    mut candidates: Vec<NumericSequenceCandidate>,
    policy: NumericSequencePolicy,
) -> NumericSequenceSelection {
    candidates.sort_by_key(|candidate| (candidate.position, candidate.value));
    if candidates.is_empty() {
        return NumericSequenceSelection {
            indices: Vec::new(),
            score: 0.0,
        };
    }
    let mut best = vec![f64::NEG_INFINITY; candidates.len()];
    let mut parent = vec![None; candidates.len()];
    let mut prior_page_best = HashMap::<u32, usize>::new();
    let mut same_page_best = HashMap::<u32, usize>::new();
    let mut current_page = None;
    let mut group = 0;
    while group < candidates.len() {
        let end = (group + 1..candidates.len())
            .find(|index| candidates[*index].position != candidates[group].position)
            .unwrap_or(candidates.len());
        let page = candidates[group].page;
        if current_page != Some(page) {
            for (value, index) in same_page_best.drain() {
                if prior_page_best
                    .get(&value)
                    .is_none_or(|prior| best[index] > best[*prior] + 1e-9)
                {
                    prior_page_best.insert(value, index);
                }
            }
            current_page = Some(page);
        }
        for index in group..end {
            let candidate = candidates[index];
            match policy {
                NumericSequencePolicy::RootedConsecutive if candidate.value == 1 => {
                    best[index] = candidate.score
                }
                NumericSequencePolicy::FootnoteBackbone => {
                    best[index] = if candidate.start_supported {
                        candidate.score
                    } else {
                        candidate.score
                            + (-0.25 * f64::from(candidate.value.saturating_sub(1))).max(-4.0)
                    }
                }
                NumericSequencePolicy::RootedConsecutive => {}
            }
            let first = match policy {
                NumericSequencePolicy::RootedConsecutive => candidate.value.saturating_sub(1),
                NumericSequencePolicy::FootnoteBackbone => {
                    candidate.value.saturating_sub(201).max(1)
                }
            };
            let mut options = (first..candidate.value)
                .flat_map(|value| {
                    [
                        prior_page_best
                            .get(&value)
                            .copied()
                            .map(|index| (index, false)),
                        same_page_best
                            .get(&value)
                            .copied()
                            .map(|index| (index, true)),
                    ]
                    .into_iter()
                    .flatten()
                })
                .collect::<Vec<_>>();
            options.sort_unstable();
            for (previous, same_page) in options {
                let gap = candidate.value - candidates[previous].value - 1;
                let penalty = match policy {
                    NumericSequencePolicy::RootedConsecutive => 0.0,
                    NumericSequencePolicy::FootnoteBackbone => {
                        ((if same_page { -0.4 } else { -0.12 }) * f64::from(gap)).max(-4.0)
                    }
                };
                let score =
                    best[previous] + candidate.score + penalty + if gap == 0 { 0.3 } else { 0.0 };
                if score > best[index] + 1e-9 {
                    best[index] = score;
                    parent[index] = Some(previous);
                }
            }
        }
        for index in group..end {
            let value = candidates[index].value;
            if same_page_best
                .get(&value)
                .is_none_or(|prior| best[index] > best[*prior] + 1e-9)
            {
                same_page_best.insert(value, index);
            }
        }
        group = end;
    }
    let tail = match policy {
        NumericSequencePolicy::RootedConsecutive => (0..candidates.len())
            .filter(|index| best[*index].is_finite())
            .reduce(|left, right| {
                if best[right] > best[left] + 1e-9 {
                    right
                } else {
                    left
                }
            }),
        NumericSequencePolicy::FootnoteBackbone => (0..candidates.len()).max_by(|left, right| {
            best[*left]
                .total_cmp(&best[*right])
                .then_with(|| candidates[*right].position.cmp(&candidates[*left].position))
        }),
    };
    let Some(mut tail) = tail else {
        return NumericSequenceSelection {
            indices: Vec::new(),
            score: 0.0,
        };
    };
    let score = best[tail];
    let mut indices = Vec::new();
    loop {
        indices.push(candidates[tail].index);
        if let Some(previous) = parent[tail] {
            tail = previous;
        } else {
            break;
        }
    }
    indices.reverse();
    NumericSequenceSelection { indices, score }
}

#[derive(Debug)]
pub struct EngineError {
    pub code: &'static str,
    pub message: String,
}

impl EngineError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_evidence",
            message: message.into(),
        }
    }

    #[cfg(feature = "source-doc")]
    fn source(message: impl Display) -> Self {
        Self {
            code: "invalid_source",
            message: message.to_string(),
        }
    }
}

impl Display for EngineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for EngineError {}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarRange {
    pub start: usize,
    pub end: usize,
}

impl ScalarRange {
    fn valid(self, length: usize) -> bool {
        self.start <= self.end && self.end <= length
    }
}

#[derive(Clone, Copy, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Paragraph,
    Prose,
    Page,
    Section,
    Heading,
    Footnote,
    Endnote,
}

#[derive(Clone, Copy, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum UnitRole {
    Page,
    Region,
    Line,
    Word,
    Span,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CoverageState {
    Absent,
    Augment,
    Complete,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ScopeKind {
    Complete,
    Excerpt,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DetectionProfile {
    CaseRootedComplete,
    CaseContiguousComplete,
    CaseLossy,
    Legislation,
    Instrument,
    Journal,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    kind: ScopeKind,
    excerpt_of: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Origin {
    id: String,
    producer: String,
    representation: String,
    revision: String,
    authority: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Geometry {
    coordinate_space: String,
    page_width: f64,
    page_height: f64,
    bbox: [f64; 4],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PageLayout {
    column_separator: Option<f64>,
    source: Option<String>,
    text_quality: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionLayout {
    kind: Option<String>,
    member_line_ids: Option<Vec<String>>,
    reading_order: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DetachedReference {
    note_id: Option<String>,
    range: Option<ScalarRange>,
    selected_text: Option<String>,
    source_line_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LineLayout {
    source_index: Option<usize>,
    reading_order: Option<usize>,
    block_index: Option<usize>,
    source: Option<String>,
    exclude_from_body: Option<bool>,
    region_id: Option<String>,
    region_type: Option<String>,
    note_region_mode: Option<String>,
    suppress_footnote_label: Option<bool>,
    detached_references: Option<Vec<DetachedReference>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpanStyle {
    font: Option<String>,
    size: Option<f64>,
    flags: Option<i64>,
    superscript: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unit {
    id: String,
    role: UnitRole,
    source_order: usize,
    range: ScalarRange,
    origin_id: String,
    parent_id: Option<String>,
    provider_order: Option<usize>,
    page_index: Option<usize>,
    page_number: Option<usize>,
    flow_id: Option<String>,
    raw_geometry: Option<Geometry>,
    page_layout: Option<PageLayout>,
    region_layout: Option<RegionLayout>,
    line_layout: Option<LineLayout>,
    span_style: Option<SpanStyle>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeClaim {
    id: String,
    kind: EvidenceKind,
    label: Option<String>,
    aliases: Vec<String>,
    range: ScalarRange,
    provider_order: usize,
    origin_id: String,
    parent_label: Option<String>,
    anchor: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Coverage {
    kind: EvidenceKind,
    range: ScalarRange,
    state: CoverageState,
    reason: String,
    origin_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Exclusion {
    range: ScalarRange,
    applies_to: Vec<String>,
    reason: String,
    origin_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParagraphBreak {
    at: usize,
    origin_id: String,
    strength: String,
    before_unit: Option<String>,
    after_unit: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentInput {
    schema_version: String,
    document_id: String,
    provider: String,
    #[cfg(feature = "source-doc")]
    url: Option<String>,
    #[cfg(feature = "source-doc")]
    doc_type: Option<SourceDocType>,
    provider_revision: String,
    profile: DetectionProfile,
    report_start_page: Option<u32>,
    require_report_start: bool,
    allow_hyphenated_sections: bool,
    text: String,
    text_sha256: String,
    source_sha256: Option<String>,
    offset_unit: String,
    scope: Scope,
    origins: Vec<Origin>,
    units: Vec<Unit>,
    native_claims: Vec<NativeClaim>,
    coverage: Vec<Coverage>,
    exclusions: Vec<Exclusion>,
    paragraph_breaks: Vec<ParagraphBreak>,
    #[cfg(feature = "source-doc")]
    #[serde(skip)]
    original_claims: HashMap<String, SourceDocBlock>,
}

fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(values: impl IntoIterator<Item = impl AsRef<str>>) -> bool {
    values.into_iter().all(|value| !value.as_ref().is_empty())
}

impl EvidenceKind {
    fn name(self) -> &'static str {
        match self {
            Self::Paragraph => "paragraph",
            Self::Prose => "prose",
            Self::Page => "page",
            Self::Section => "section",
            Self::Heading => "heading",
            Self::Footnote => "footnote",
            Self::Endnote => "endnote",
        }
    }
}

impl DocumentInput {
    #[cfg(feature = "source-doc")]
    pub fn set_original_claims(&mut self, claims: HashMap<String, SourceDocBlock>) {
        self.original_claims = claims;
    }

    fn validate(&self) -> Result<(), EngineError> {
        let length = self.text.chars().count();
        if self.schema_version != EVIDENCE_SCHEMA
            || self.offset_unit != "unicode-scalar"
            || !nonempty([
                &self.document_id,
                &self.provider,
                &self.provider_revision,
                &self.text_sha256,
            ])
            || !hash(&self.text_sha256)
            || format!("{:x}", Sha256::digest(self.text.as_bytes())) != self.text_sha256
            || self
                .source_sha256
                .as_deref()
                .is_some_and(|value| !hash(value))
            || match self.scope.kind {
                ScopeKind::Complete => self.scope.excerpt_of.is_some(),
                ScopeKind::Excerpt => self.scope.excerpt_of.as_deref().is_none_or(str::is_empty),
            }
        {
            return Err(EngineError::invalid("invalid evidence identity or schema"));
        }
        if self.profile != DetectionProfile::Legislation && self.allow_hyphenated_sections {
            return Err(EngineError::invalid(
                "hyphenated-section option requires legislation profile",
            ));
        }
        if matches!(
            self.profile,
            DetectionProfile::CaseRootedComplete | DetectionProfile::CaseContiguousComplete
        ) && self.scope.kind != ScopeKind::Complete
        {
            return Err(EngineError::invalid(
                "complete case profile requires complete document scope",
            ));
        }
        if matches!(
            self.profile,
            DetectionProfile::Legislation
                | DetectionProfile::Instrument
                | DetectionProfile::Journal
        ) && (self.report_start_page.is_some() || self.require_report_start)
        {
            return Err(EngineError::invalid(
                "report-page options require a case profile",
            ));
        }
        let origins = self
            .origins
            .iter()
            .map(|value| value.id.as_str())
            .collect::<HashSet<_>>();
        if origins.len() != self.origins.len()
            || self.origins.iter().any(|value| {
                !nonempty([
                    &value.id,
                    &value.producer,
                    &value.representation,
                    &value.revision,
                    &value.authority,
                ])
            })
        {
            return Err(EngineError::invalid("origins are invalid or duplicated"));
        }
        let unit_ids = self
            .units
            .iter()
            .map(|value| value.id.as_str())
            .collect::<HashSet<_>>();
        let mut orders = BTreeMap::<UnitRole, Vec<usize>>::new();
        for unit in &self.units {
            let geometry = unit.raw_geometry.as_ref().is_none_or(|value| {
                !value.coordinate_space.is_empty()
                    && value.page_width.is_finite()
                    && value.page_height.is_finite()
                    && value.bbox.iter().all(|number| number.is_finite())
            });
            let layout = unit.page_layout.as_ref().is_none_or(|value| {
                unit.role == UnitRole::Page
                    && value.column_separator.is_none_or(f64::is_finite)
                    && value.text_quality.is_none_or(f64::is_finite)
            }) && unit
                .region_layout
                .as_ref()
                .is_none_or(|_| unit.role == UnitRole::Region)
                && unit.line_layout.as_ref().is_none_or(|value| {
                    unit.role == UnitRole::Line
                        && value.detached_references.as_ref().is_none_or(|refs| {
                            refs.iter().all(|reference| {
                                reference.range.is_none_or(|range| range.valid(length))
                            })
                        })
                })
                && unit.span_style.as_ref().is_none_or(|value| {
                    unit.role == UnitRole::Span && value.size.is_none_or(f64::is_finite)
                });
            if unit.id.is_empty()
                || !unit.range.valid(length)
                || !origins.contains(unit.origin_id.as_str())
                || unit
                    .parent_id
                    .as_deref()
                    .is_some_and(|id| !unit_ids.contains(id))
                || !geometry
                || !layout
            {
                return Err(EngineError::invalid(
                    "unit identity, range, or layout is invalid",
                ));
            }
            orders.entry(unit.role).or_default().push(unit.source_order);
        }
        if unit_ids.len() != self.units.len()
            || orders.values_mut().any(|values| {
                values.sort_unstable();
                values.iter().copied().ne(1..=values.len())
            })
        {
            return Err(EngineError::invalid("unit IDs or source order are invalid"));
        }
        let claims = self
            .native_claims
            .iter()
            .map(|value| value.id.as_str())
            .collect::<HashSet<_>>();
        if claims.len() != self.native_claims.len()
            || self.native_claims.iter().any(|value| {
                value.id.is_empty()
                    || !value.range.valid(length)
                    || !origins.contains(value.origin_id.as_str())
                    || !nonempty(value.aliases.iter())
                    || value.label.as_deref().is_some_and(str::is_empty)
                    || value.anchor.as_deref().is_some_and(str::is_empty)
            })
        {
            return Err(EngineError::invalid(
                "native claims are invalid or duplicated",
            ));
        }
        let mut coverage = BTreeMap::<EvidenceKind, Vec<ScalarRange>>::new();
        for value in &self.coverage {
            if !value.range.valid(length)
                || value.reason.is_empty()
                || value
                    .origin_id
                    .as_deref()
                    .is_some_and(|id| !origins.contains(id))
            {
                return Err(EngineError::invalid("coverage is invalid"));
            }
            coverage.entry(value.kind).or_default().push(value.range);
        }
        for kind in [
            EvidenceKind::Paragraph,
            EvidenceKind::Prose,
            EvidenceKind::Page,
            EvidenceKind::Section,
            EvidenceKind::Heading,
            EvidenceKind::Footnote,
            EvidenceKind::Endnote,
        ] {
            let Some(rows) = coverage.get_mut(&kind) else {
                return Err(EngineError::invalid("coverage kind is missing"));
            };
            rows.sort_by_key(|value| value.start);
            let mut cursor = 0;
            for range in rows {
                if range.start != cursor {
                    return Err(EngineError::invalid("coverage has a gap or overlap"));
                }
                cursor = range.end;
            }
            if cursor != length {
                return Err(EngineError::invalid("coverage does not span text"));
            }
        }
        if self.exclusions.iter().any(|value| {
            !value.range.valid(length)
                || value.reason.is_empty()
                || value.applies_to.is_empty()
                || !nonempty(value.applies_to.iter())
                || !origins.contains(value.origin_id.as_str())
        }) || self.paragraph_breaks.iter().any(|value| {
            value.at > length
                || value.strength.is_empty()
                || !origins.contains(value.origin_id.as_str())
                || value
                    .before_unit
                    .as_deref()
                    .is_some_and(|id| !unit_ids.contains(id))
                || value
                    .after_unit
                    .as_deref()
                    .is_some_and(|id| !unit_ids.contains(id))
        }) {
            return Err(EngineError::invalid(
                "exclusion or paragraph break is invalid",
            ));
        }
        Ok(())
    }

    fn clip_inference(&self, kind: EvidenceKind, range: ScalarRange) -> Option<ScalarRange> {
        let mut end = range.end;
        for value in self
            .coverage
            .iter()
            .filter(|value| value.kind == kind && value.state == CoverageState::Complete)
        {
            if value.range.start <= range.start && range.start < value.range.end {
                return None;
            }
            if value.range.start > range.start {
                end = end.min(value.range.start);
            }
        }
        for value in self
            .exclusions
            .iter()
            .filter(|value| value.applies_to.iter().any(|name| name == kind.name()))
        {
            if value.range.start <= range.start && range.start < value.range.end {
                return None;
            }
            if value.range.start > range.start {
                end = end.min(value.range.start);
            }
        }
        (end > range.start).then_some(ScalarRange {
            start: range.start,
            end,
        })
    }

    fn needs_recovery(&self) -> bool {
        self.coverage
            .iter()
            .any(|value| value.state != CoverageState::Complete)
    }
}

impl TryFrom<Value> for DocumentInput {
    type Error = EngineError;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let evidence: DocumentInput = serde_json::from_value(value)
            .map_err(|error| EngineError::invalid(error.to_string()))?;
        evidence.validate()?;
        Ok(evidence)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Paragraph,
    Page,
    Section,
    Heading,
    Footnote,
    Endnote,
    Prose,
}

impl NodeKind {
    fn evidence(self) -> EvidenceKind {
        match self {
            Self::Paragraph => EvidenceKind::Paragraph,
            Self::Prose => EvidenceKind::Prose,
            Self::Page => EvidenceKind::Page,
            Self::Section => EvidenceKind::Section,
            Self::Heading => EvidenceKind::Heading,
            Self::Footnote => EvidenceKind::Footnote,
            Self::Endnote => EvidenceKind::Endnote,
        }
    }
    fn name(self) -> &'static str {
        self.evidence().name()
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStatus {
    Complete,
    Partial,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Derivation {
    Native,
    Heuristic,
    Model,
}

#[derive(Serialize)]
pub struct StructureNodeV1 {
    pub id: String,
    pub kind: NodeKind,
    pub range: ScalarRange,
    pub origin_id: String,
    pub source: Derivation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_start: Option<usize>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    Paragraph,
    Prose,
}

#[derive(Serialize)]
pub struct StructureBoundaryV1 {
    pub kind: BoundaryKind,
    pub at: usize,
    pub origin_id: String,
    pub source: Derivation,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Contains,
    Precedes,
    References,
    FootnoteFor,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum RelationEndpointV1 {
    Node { node_id: String },
    Range { range: ScalarRange },
}

#[derive(Serialize)]
pub struct StructureRelationV1 {
    pub id: String,
    pub kind: RelationKind,
    pub from: RelationEndpointV1,
    pub to: RelationEndpointV1,
    pub origin_id: String,
    pub source: Derivation,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Serialize)]
pub struct StructureDiagnosticV1 {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub ranges: Vec<ScalarRange>,
    pub node_ids: Vec<String>,
}

#[derive(Serialize)]
pub struct StructureGraphV1 {
    pub schema_version: &'static str,
    pub document_id: String,
    pub text_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    pub status: GraphStatus,
    pub nodes: Vec<StructureNodeV1>,
    pub boundaries: Vec<StructureBoundaryV1>,
    pub relations: Vec<StructureRelationV1>,
    pub diagnostics: Vec<StructureDiagnosticV1>,
}

#[cfg(feature = "recovery")]
#[derive(Clone, Copy)]
struct OffsetCheckpoint {
    scalar: usize,
    byte: usize,
    utf16: usize,
}

#[cfg(feature = "recovery")]
struct ScalarText<'a> {
    value: &'a str,
    offsets: Vec<OffsetCheckpoint>,
    scalar_len: usize,
    utf16_len: usize,
    lines: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "recovery")]
impl<'a> ScalarText<'a> {
    fn new(value: &'a str) -> Self {
        if value.is_ascii() {
            let mut lines = Vec::new();
            let mut line_start = 0;
            for (at, byte) in value.bytes().enumerate() {
                if byte != b'\n' {
                    continue;
                }
                let end = at - usize::from(at > line_start && value.as_bytes()[at - 1] == b'\r');
                lines.push((line_start, end, line_start));
                line_start = at + 1;
            }
            let end = value.len() - usize::from(value.as_bytes().last() == Some(&b'\r'));
            lines.push((line_start, end, line_start));
            return Self {
                value,
                offsets: Vec::new(),
                scalar_len: value.len(),
                utf16_len: value.len(),
                lines,
            };
        }
        const STRIDE: usize = 256;
        let mut offsets = Vec::new();
        let mut lines = Vec::new();
        let mut utf16_len = 0;
        let mut scalar_len = 0;
        let mut line_start = (0, 0);
        for (scalar, (at, character)) in value.char_indices().enumerate() {
            if scalar % STRIDE == 0 {
                offsets.push(OffsetCheckpoint {
                    scalar,
                    byte: at,
                    utf16: utf16_len,
                });
            }
            if character == '\n' {
                let end = at - usize::from(at > line_start.0 && value.as_bytes()[at - 1] == b'\r');
                lines.push((line_start.0, end, line_start.1));
                line_start = (at + 1, scalar + 1);
            }
            scalar_len = scalar + 1;
            utf16_len += character.len_utf16();
        }
        if offsets
            .last()
            .is_none_or(|offset| offset.scalar != scalar_len)
        {
            offsets.push(OffsetCheckpoint {
                scalar: scalar_len,
                byte: value.len(),
                utf16: utf16_len,
            });
        }
        let end = value.len() - usize::from(value.as_bytes().last() == Some(&b'\r'));
        lines.push((line_start.0, end, line_start.1));
        Self {
            value,
            offsets,
            scalar_len,
            utf16_len,
            lines,
        }
    }
    fn len(&self) -> usize {
        self.scalar_len
    }
    fn checkpoint_for_scalar(&self, scalar: usize) -> OffsetCheckpoint {
        self.offsets[self
            .offsets
            .partition_point(|offset| offset.scalar <= scalar)
            - 1]
    }
    fn checkpoint_for_byte(&self, byte: usize) -> OffsetCheckpoint {
        self.offsets[self.offsets.partition_point(|offset| offset.byte <= byte) - 1]
    }
    fn scalar(&self, byte: usize) -> usize {
        if self.offsets.is_empty() {
            byte
        } else {
            let offset = self.checkpoint_for_byte(byte);
            offset.scalar + self.value[offset.byte..byte].chars().count()
        }
    }
    fn byte(&self, scalar: usize) -> usize {
        if self.offsets.is_empty() {
            scalar
        } else {
            let offset = self.checkpoint_for_scalar(scalar);
            if offset.scalar == scalar {
                offset.byte
            } else {
                offset.byte
                    + self.value[offset.byte..]
                        .char_indices()
                        .nth(scalar - offset.scalar)
                        .map_or(self.value.len() - offset.byte, |(byte, _)| byte)
            }
        }
    }
    fn utf16(&self, scalar: usize) -> usize {
        if self.offsets.is_empty() {
            scalar
        } else {
            let offset = self.checkpoint_for_scalar(scalar);
            offset.utf16
                + self.value[offset.byte..self.byte(scalar)]
                    .encode_utf16()
                    .count()
        }
    }
    fn utf16_len(&self) -> usize {
        self.utf16_len
    }
    fn slice(&self, range: ScalarRange) -> &'a str {
        &self.value[self.byte(range.start)..self.byte(range.end)]
    }
}

#[cfg(feature = "recovery")]
fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
}

#[cfg(feature = "recovery")]
fn utf16_prefix(value: &str, maximum: usize) -> &str {
    let end = value
        .char_indices()
        .take_while(|(_, character)| character.len_utf16() <= maximum)
        .scan(0, |used, (at, character)| {
            *used += character.len_utf16();
            Some((at, *used))
        })
        .take_while(|(_, used)| *used <= maximum)
        .last()
        .map_or(0, |(at, _)| {
            at + value[at..].chars().next().unwrap().len_utf8()
        });
    &value[..end]
}

#[derive(Clone)]
struct Block {
    kind: NodeKind,
    range: ScalarRange,
    label: Option<String>,
    aliases: Vec<String>,
    parent_label: Option<String>,
    content_start: Option<usize>,
    diagnostic: Option<&'static str>,
}

impl Block {
    fn labelled(kind: NodeKind, label: String, start: usize, end: usize) -> Self {
        Self {
            kind,
            range: ScalarRange { start, end },
            label: Some(label),
            aliases: Vec::new(),
            parent_label: None,
            content_start: None,
            diagnostic: None,
        }
    }
}

#[cfg(feature = "recovery")]
mod recovery {
    use super::*;

    macro_rules! cached_regex {
        ($name:ident, $pattern:expr) => {{
            static $name: OnceLock<Regex> = OnceLock::new();
            $name.get_or_init(|| Regex::new($pattern).unwrap())
        }};
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum MarkerStyle {
        Bracket,
        Dot,
        Bare,
    }

    #[derive(Clone)]
    struct Marker {
        number: u32,
        start: usize,
        style: MarkerStyle,
        score: f64,
        formal: bool,
        sentence: bool,
    }

    #[derive(Clone, Copy)]
    struct Line<'a> {
        byte_start: usize,
        byte_end: usize,
        scalar_start: usize,
        text: &'a str,
    }

    fn lines<'a>(text: &'a ScalarText<'a>) -> impl Iterator<Item = Line<'a>> + 'a {
        text.lines
            .iter()
            .map(|&(byte_start, byte_end, scalar_start)| {
                let text = &text.value[byte_start..byte_end];
                Line {
                    byte_start,
                    byte_end,
                    scalar_start,
                    text,
                }
            })
    }

    fn javascript_lines<'a>(text: &'a ScalarText<'a>) -> Vec<Line<'a>> {
        let mut result = Vec::new();
        let mut chars = text.value.char_indices().peekable();
        let (mut byte_start, mut scalar_start, mut scalar) = (0, 0, 0);
        while let Some((byte, character)) = chars.next() {
            if matches!(character, '\r' | '\n' | '\u{2028}' | '\u{2029}') {
                result.push(Line {
                    byte_start,
                    byte_end: byte,
                    scalar_start,
                    text: &text.value[byte_start..byte],
                });
                scalar += 1;
                let mut next_byte = byte + character.len_utf8();
                if character == '\r' && chars.peek().is_some_and(|(_, next)| *next == '\n') {
                    let (next, _) = chars.next().unwrap();
                    next_byte = next + 1;
                    scalar += 1;
                }
                byte_start = next_byte;
                scalar_start = scalar;
            } else {
                scalar += 1;
            }
        }
        result.push(Line {
            byte_start,
            byte_end: text.value.len(),
            scalar_start,
            text: &text.value[byte_start..],
        });
        result
    }

    fn leading_ascii_space(value: &str) -> usize {
        value
            .bytes()
            .take_while(|byte| matches!(byte, b' ' | b'\t'))
            .count()
    }

    fn javascript_whitespace(value: char) -> bool {
        value == '\u{feff}' || (value != '\u{85}' && value.is_whitespace())
    }

    fn decimal_prefix(value: &str, maximum: usize) -> Option<(&str, usize)> {
        let length = value.bytes().take_while(u8::is_ascii_digit).count();
        (length > 0 && length <= maximum).then(|| (&value[..length], length))
    }

    fn paragraph_markers(text: &ScalarText<'_>, contiguous: bool) -> Vec<Marker> {
        let mut result = Vec::new();
        for line in lines(text) {
            let lead = leading_ascii_space(line.text);
            let value = &line.text[lead..];
            let start = line.scalar_start;
            let basic = if let Some(rest) = value.strip_prefix('[') {
                decimal_prefix(rest, 4).and_then(|(number, length)| {
                    (rest.as_bytes().get(length) == Some(&b']'))
                        .then(|| (number, MarkerStyle::Bracket))
                })
            } else {
                decimal_prefix(value, 4).and_then(|(number, length)| {
                    let rest = &value[length..];
                    if rest.starts_with('.')
                        && (rest[1..].chars().next().is_some_and(char::is_whitespace)
                            || (rest.len() == 1 && line.byte_end < text.value.len()))
                    {
                        Some((number, MarkerStyle::Dot))
                    } else if contiguous
                        && rest.starts_with('.')
                        && rest[1..].chars().next().is_some_and(char::is_uppercase)
                    {
                        Some((number, MarkerStyle::Dot))
                    } else if rest.chars().next().is_some_and(char::is_whitespace)
                        || (rest.is_empty() && line.byte_end < text.value.len())
                    {
                        Some((
                            number,
                            if contiguous {
                                MarkerStyle::Dot
                            } else {
                                MarkerStyle::Bare
                            },
                        ))
                    } else {
                        None
                    }
                })
            };
            if let Some((number, style)) = basic {
                result.push(Marker {
                    number: number.parse().unwrap(),
                    start,
                    style,
                    score: 1.0,
                    formal: false,
                    sentence: false,
                });
            }
            if contiguous {
                let glyph = value
                    .chars()
                    .next()
                    .filter(|value| matches!(value, '¶' | '\u{95}' | '•'));
                if let Some(glyph) = glyph {
                    let rest = value[glyph.len_utf8()..].trim_start_matches([' ', '\t']);
                    if let Some((number, length)) = decimal_prefix(rest, 4) {
                        let after = rest[length..].chars().next();
                        if after
                            .is_none_or(|value| value.is_whitespace() || ".,;:—-".contains(value))
                        {
                            result.push(Marker {
                                number: number.parse().unwrap(),
                                start,
                                style: MarkerStyle::Dot,
                                score: 1.0,
                                formal: false,
                                sentence: false,
                            });
                        }
                    }
                }
            }
        }
        result.sort_by_key(|marker| marker.start);
        result.dedup_by_key(|marker| marker.start);
        result
    }

    fn marker_visible(marker: &Marker, excluded: &[ScalarRange]) -> bool {
        !excluded
            .iter()
            .any(|range| range.start <= marker.start && marker.start < range.end)
    }

    fn word_count(value: &str, letters_only: bool) -> usize {
        let mut count = 0;
        let mut inside = false;
        let mut characters = value.chars().peekable();
        while let Some(character) = characters.next() {
            let member = if letters_only {
                character.is_alphabetic()
            } else {
                character.is_alphanumeric()
            };
            if member {
                count += usize::from(!inside);
                inside = true;
            } else if matches!(character, '\'' | '’')
                && inside
                && characters.peek().is_some_and(|next| {
                    if letters_only {
                        next.is_alphabetic()
                    } else {
                        next.is_alphanumeric()
                    }
                })
            {
                continue;
            } else {
                inside = false;
            }
        }
        count
    }

    fn median(values: &[usize]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut values = values.to_vec();
        values.sort_unstable();
        let middle = values.len() / 2;
        if values.len() % 2 == 1 {
            values[middle] as f64
        } else {
            (values[middle - 1] + values[middle]) as f64 / 2.0
        }
    }

    fn heading_enumerator(value: &str) -> bool {
        cached_regex!(VALUE, r"^(?:\([\p{L}\p{N}]{1,5}\)|\p{L}[.)]|[IVXLCDM]{1,4}[.)]|[ivxlcdm]{1,4}[.)]|\d{1,3}(?:\.\d{1,3})*[.)])$").is_match(value)
    }

    fn level_opens(value: &str) -> bool {
        value
            .chars()
            .next()
            .is_some_and(|character| character.is_uppercase() || character.is_numeric())
    }

    fn heading_level(words: &[&str], enumerated: bool) -> bool {
        if words.is_empty() || words.len() > 12 {
            return false;
        }
        if words.len() == 1 && heading_enumerator(words[0]) {
            return true;
        }
        let text = words.join(" ");
        if !level_opens(&text) || text.ends_with(['.', ',', ';']) {
            return false;
        }
        if text.ends_with(['?', ':']) {
            return true;
        }
        let title = words.iter().all(|word| {
            let letters = word
                .chars()
                .filter(|value| value.is_alphabetic())
                .collect::<String>();
            utf16_len(&letters) < 4 || letters.chars().next().is_some_and(char::is_uppercase)
        });
        title || enumerated || words.len() <= 6
    }

    fn trim_leading_parenthetical(value: &str) -> &str {
        let value = value.trim();
        let Some(rest) = value.strip_prefix('(') else {
            return value;
        };
        let Some(close) = rest.find(')') else {
            return value;
        };
        if !rest[..close].is_empty()
            && rest[..close].chars().all(char::is_alphanumeric)
            && rest[close + 1..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            rest[close + 1..].trim_start()
        } else {
            value
        }
    }

    pub(super) fn formal_heading(value: &str) -> bool {
        let heading = trim_leading_parenthetical(value);
        if heading.is_empty()
            || utf16_len(heading) > 120
            || heading.chars().any(|value| ";![]{}".contains(value))
        {
            return false;
        }
        let words = heading.split_whitespace().collect::<Vec<_>>();
        let mut levels: Vec<(Vec<&str>, bool)> = vec![(Vec::new(), false)];
        for (index, word) in words.iter().enumerate() {
            let opener = words
                .get(index + 1)
                .is_some_and(|next| !heading_enumerator(next) && level_opens(next));
            if heading_enumerator(word) && opener {
                if levels.last().unwrap().0.is_empty() {
                    levels.last_mut().unwrap().1 = true;
                } else {
                    levels.push((Vec::new(), true));
                }
            } else {
                levels.last_mut().unwrap().0.push(word);
            }
        }
        levels
            .iter()
            .all(|(words, enumerated)| heading_level(words, *enumerated))
    }

    fn sentence_heading(value: &str, following: &str) -> bool {
        let heading = trim_leading_parenthetical(value);
        let words = heading.split_whitespace().collect::<Vec<_>>();
        utf16_len(heading) <= 120
            && (4..=18).contains(&words.len())
            && heading.chars().next().is_some_and(char::is_uppercase)
            && words
                .iter()
                .any(|word| word.chars().next().is_some_and(char::is_lowercase))
            && !heading.chars().any(|value| "[].,;:!?".contains(value))
            && following
                .trim_start()
                .chars()
                .next()
                .is_some_and(char::is_uppercase)
    }

    fn heading_joined(
        text: &ScalarText<'_>,
        known: &HashSet<usize>,
        style: MarkerStyle,
    ) -> Vec<Marker> {
        if style == MarkerStyle::Bare {
            return Vec::new();
        }
        let mut result = Vec::new();
        for line in lines(text) {
            let bytes = line.text.as_bytes();
            let mut at = 0;
            while at < bytes.len() {
                let digits = if style == MarkerStyle::Bracket && bytes[at] == b'[' {
                    at + 1
                } else if style == MarkerStyle::Dot && bytes[at].is_ascii_digit() {
                    at
                } else {
                    at += 1;
                    continue;
                };
                let length = bytes[digits..]
                    .iter()
                    .take(4)
                    .take_while(|byte| byte.is_ascii_digit())
                    .count();
                if length == 0 {
                    at += 1;
                    continue;
                }
                let tail = digits + length;
                let end = if style == MarkerStyle::Bracket {
                    (bytes.get(tail) == Some(&b']')).then_some(tail + 1)
                } else if bytes.get(tail) == Some(&b'.') {
                    let after = tail + 1;
                    if after == bytes.len() {
                        Some(after)
                    } else {
                        line.text[after..]
                            .chars()
                            .next()
                            .filter(|character| character.is_whitespace())
                            .map(|character| after + character.len_utf8())
                    }
                } else {
                    None
                };
                let Some(end) = end else {
                    at += 1;
                    continue;
                };
                let start = line.scalar_start + line.text[..at].chars().count();
                if known.contains(&start) {
                    at = end;
                    continue;
                }
                let heading = &line.text[..at];
                let formal = formal_heading(heading)
                    && (style == MarkerStyle::Bracket || !heading.contains('.'));
                let sentence = style == MarkerStyle::Bracket
                    && sentence_heading(heading, &text.value[line.byte_start + end..]);
                if formal || sentence {
                    result.push(Marker {
                        number: line.text[digits..tail].parse().unwrap(),
                        start,
                        style,
                        score: if formal { 0.6 } else { 0.35 },
                        formal,
                        sentence,
                    });
                }
                at = end;
            }
        }
        result
    }

    fn rooted_chain(candidates: Vec<Marker>) -> (Vec<Marker>, f64) {
        let selected = select_numeric_sequence(
            candidates
                .iter()
                .enumerate()
                .map(|(index, marker)| NumericSequenceCandidate {
                    index,
                    value: marker.number,
                    position: (marker.start, 0),
                    page: 0,
                    score: marker.score,
                    start_supported: false,
                })
                .collect(),
            NumericSequencePolicy::RootedConsecutive,
        );
        (
            selected
                .indices
                .into_iter()
                .map(|index| candidates[index].clone())
                .collect(),
            selected.score,
        )
    }

    fn sole_chain(chain: &[Marker], candidates: &[Marker]) -> bool {
        let claimed = chain
            .iter()
            .map(|value| value.start)
            .collect::<HashSet<_>>();
        let last = chain.last().map_or(0, |value| value.number);
        let mut rest = candidates
            .iter()
            .filter(|value| !claimed.contains(&value.start))
            .collect::<Vec<_>>();
        rest.sort_by_key(|value| value.start);
        !rest
            .iter()
            .any(|value| (1..=last + 1).contains(&value.number))
            && rest.windows(2).all(|pair| pair[1].number <= pair[0].number)
    }

    fn endnote_shaped(text: &ScalarText<'_>, chain: &[Marker]) -> bool {
        chain.len() >= 8
            && text.utf16_len() > 0
            && chain
                .iter()
                .filter(|value| text.utf16(value.start) as f64 > text.utf16_len() as f64 * 0.75)
                .count() as f64
                / chain.len() as f64
                >= 0.7
    }

    fn monotone_scopes(markers: &[Marker], max_gap: u32) -> Vec<Vec<Marker>> {
        let mut scopes = Vec::<Vec<Marker>>::new();
        let mut by_last = HashMap::<u32, Vec<usize>>::new();
        for marker in markers {
            let candidates = (marker.number.saturating_sub(max_gap)..marker.number)
                .flat_map(|value| by_last.get(&value).into_iter().flatten().copied())
                .collect::<Vec<_>>();
            let index = candidates
                .into_iter()
                .reduce(|best, current| {
                    let left = scopes[current][0].number;
                    let right = scopes[best][0].number;
                    if left < right || (left == right && current < best) {
                        current
                    } else {
                        best
                    }
                })
                .unwrap_or(scopes.len());
            if index == scopes.len() {
                scopes.push(vec![marker.clone()]);
            } else {
                let previous = scopes[index].last().unwrap().number;
                if let Some(values) = by_last.get_mut(&previous) {
                    values.retain(|value| *value != index);
                }
                scopes[index].push(marker.clone());
            }
            by_last.entry(marker.number).or_default().push(index);
        }
        scopes
    }

    fn contiguous_scopes(markers: &[Marker]) -> Vec<Vec<Marker>> {
        let mut scopes: Vec<Vec<Marker>> = Vec::new();
        for marker in markers {
            if scopes
                .last()
                .and_then(|scope| scope.last())
                .is_some_and(|prior| marker.number == prior.number + 1)
            {
                scopes.last_mut().unwrap().push(marker.clone());
            } else {
                scopes.push(vec![marker.clone()]);
            }
        }
        scopes
    }

    fn recover_contiguous(
        text: &ScalarText<'_>,
        markers: &[Marker],
        style: MarkerStyle,
    ) -> Vec<Marker> {
        let line = markers
            .iter()
            .filter(|value| value.style == style)
            .cloned()
            .collect::<Vec<_>>();
        if line.is_empty() {
            return line;
        }
        let candidates =
            heading_joined(text, &line.iter().map(|value| value.start).collect(), style);
        let within = |number: u32, from: usize, to: usize, formal: bool, sentence: bool| {
            candidates
                .iter()
                .filter(|value| {
                    value.number == number
                        && value.start > from
                        && value.start < to
                        && ((!formal || value.formal) && (!sentence || value.sentence))
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut recovered = HashMap::<usize, Marker>::new();
        for pair in line.windows(2) {
            if pair[0].number >= pair[1].number {
                continue;
            }
            let mut found = Vec::new();
            for number in pair[0].number + 1..pair[1].number {
                let candidates = within(number, pair[0].start, pair[1].start, true, false);
                if candidates.len() != 1 {
                    found.clear();
                    break;
                }
                found.push(candidates[0].clone());
            }
            for marker in found {
                recovered.insert(marker.start, marker);
            }
            if pair[1].number == pair[0].number + 2 {
                let candidates = within(
                    pair[0].number + 1,
                    pair[0].start,
                    pair[1].start,
                    false,
                    true,
                );
                if candidates.len() == 1 {
                    recovered.insert(candidates[0].start, candidates[0].clone());
                }
            }
        }
        if let Some(first) = line.first().filter(|value| value.number > 1) {
            let candidates = candidates
                .iter()
                .filter(|value| {
                    value.number == first.number - 1
                        && value.start < first.start
                        && value.formal
                        && text.utf16(first.start) - text.utf16(value.start) <= 2_000
                })
                .cloned()
                .collect::<Vec<_>>();
            if candidates.len() == 1 {
                recovered.insert(candidates[0].start, candidates[0].clone());
            }
        }
        let mut result = line;
        result.extend(recovered.into_values());
        result.sort_by_key(|value| value.start);
        result
    }

    fn recover_lossy(text: &ScalarText<'_>, spine: &[Marker], style: MarkerStyle) -> Vec<Marker> {
        let candidates = heading_joined(
            text,
            &spine.iter().map(|value| value.start).collect(),
            style,
        );
        let mut recovered = HashMap::<u32, Vec<Marker>>::new();
        for candidate in candidates {
            let before = spine
                .iter()
                .rev()
                .find(|value| value.start < candidate.start);
            let after = spine.iter().find(|value| value.start > candidate.start);
            let between = before.zip(after).is_some_and(|(left, right)| {
                left.number < candidate.number && candidate.number < right.number
            });
            let leading = before.is_none()
                && after.is_some_and(|right| {
                    candidate.number > 0
                        && candidate.number < right.number
                        && right.number - candidate.number <= 2
                        && text.utf16(right.start) - text.utf16(candidate.start) <= 2_000
                });
            let sentence = before.zip(after).is_some_and(|(left, right)| {
                left.number + 1 == candidate.number && candidate.number + 1 == right.number
            }) && candidate.sentence;
            if (between || leading) && (candidate.formal || sentence) {
                recovered
                    .entry(candidate.number)
                    .or_default()
                    .push(candidate);
            }
        }
        let mut result = spine.to_vec();
        result.extend(
            recovered
                .into_values()
                .filter(|values| values.len() == 1)
                .flatten(),
        );
        result.sort_by_key(|value| value.start);
        result
    }

    fn quoted_dot(text: &ScalarText<'_>, marker: &Marker) -> bool {
        if marker.style != MarkerStyle::Dot {
            return false;
        }
        let start = text.byte(marker.start);
        let end = text.value[start..]
            .find('\n')
            .map_or(text.value.len(), |at| start + at);
        let line = &text.value[start..end];
        cached_regex!(OPEN, r"^\d{1,4}\.\s+\(\d{1,4}\)\s+").is_match(line)
            && cached_regex!(WORD, r"(?i)\b(?:Act|Code|Regulations?|Rules?|shall|must)\b")
                .is_match(line)
    }

    struct Hypothesis {
        style: MarkerStyle,
        markers: Vec<Marker>,
        all: Vec<Marker>,
        short: bool,
        score: f64,
    }

    fn paragraph_ranges(
        text: &ScalarText<'_>,
        selected: &[Marker],
        all: &[Marker],
        style: MarkerStyle,
        recover: bool,
        extra: &[usize],
    ) -> Vec<Block> {
        let selected = if recover && style != MarkerStyle::Bare {
            recover_lossy(text, selected, style)
        } else {
            selected.to_vec()
        };
        let mut boundaries = all
            .iter()
            .filter(|value| value.style == style)
            .map(|value| value.start)
            .chain(selected.iter().map(|value| value.start))
            .chain(extra.iter().copied())
            .chain([text.len()])
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        selected
            .into_iter()
            .map(|marker| {
                let end = boundaries
                    .iter()
                    .copied()
                    .find(|value| *value > marker.start)
                    .unwrap_or(text.len());
                Block::labelled(
                    NodeKind::Paragraph,
                    format!("par{}", marker.number),
                    marker.start,
                    end,
                )
            })
            .collect()
    }

    fn detect_paragraphs(
        text: &ScalarText<'_>,
        profile: DetectionProfile,
        excluded: &[ScalarRange],
    ) -> Vec<Block> {
        let strict = profile != DetectionProfile::CaseLossy;
        let contiguous = profile == DetectionProfile::CaseContiguousComplete;
        let markers = paragraph_markers(text, contiguous);
        let visible = markers
            .iter()
            .filter(|value| {
                marker_visible(value, excluded) && (!strict || !quoted_dot(text, value))
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut hypotheses = Vec::<Hypothesis>::new();
        for style in [MarkerStyle::Bracket, MarkerStyle::Dot, MarkerStyle::Bare] {
            if profile == DetectionProfile::CaseRootedComplete {
                let mut candidates = visible
                    .iter()
                    .filter(|value| value.style == style)
                    .cloned()
                    .collect::<Vec<_>>();
                if style != MarkerStyle::Bare {
                    let known = candidates.iter().map(|value| value.start).collect();
                    candidates.extend(heading_joined(text, &known, style));
                }
                let (chain, score) = rooted_chain(candidates.clone());
                if chain.len() < 2 || endnote_shaped(text, &chain) {
                    continue;
                }
                if chain.len() >= 5 {
                    hypotheses.push(Hypothesis {
                        style,
                        markers: chain,
                        all: candidates,
                        short: false,
                        score,
                    });
                } else if style == MarkerStyle::Bracket && sole_chain(&chain, &candidates) {
                    hypotheses.push(Hypothesis {
                        style,
                        markers: chain,
                        all: candidates,
                        short: true,
                        score,
                    });
                }
                continue;
            }
            let style_markers: Vec<Marker> = if contiguous && style != MarkerStyle::Bare {
                recover_contiguous(text, &visible, style)
                    .into_iter()
                    .filter(|value| marker_visible(value, excluded))
                    .collect()
            } else {
                markers
                    .iter()
                    .filter(|value| value.style == style)
                    .cloned()
                    .collect()
            };
            let scopes = if contiguous {
                contiguous_scopes(&style_markers)
            } else {
                monotone_scopes(&style_markers, 8)
            };
            for scope in scopes.iter() {
                if scope.len() >= 5 {
                    hypotheses.push(Hypothesis {
                        style,
                        markers: scope.clone(),
                        all: style_markers.clone(),
                        short: false,
                        score: 0.0,
                    });
                } else if style == MarkerStyle::Bracket
                    && scope.len() >= 2
                    && scope
                        .iter()
                        .enumerate()
                        .all(|(index, value)| value.number == index as u32 + 1)
                    && (!strict
                        || (scopes
                            .iter()
                            .all(|other| std::ptr::eq(other, scope) || other.len() == 1)
                            && style_markers.iter().all(|value| {
                                scope.iter().any(|mark| mark.start == value.start)
                                    || value.number > scope.last().unwrap().number + 1
                            })))
                {
                    hypotheses.push(Hypothesis {
                        style,
                        markers: scope.clone(),
                        all: style_markers.clone(),
                        short: true,
                        score: 0.0,
                    });
                }
            }
        }
        if profile != DetectionProfile::CaseRootedComplete
            && hypotheses
                .iter()
                .any(|value| !value.short && value.markers[0].number <= 5)
        {
            hypotheses.retain(|value| value.short || value.markers[0].number <= 5);
        }
        let rank = |style| match style {
            MarkerStyle::Bracket => 2,
            MarkerStyle::Dot => 1,
            MarkerStyle::Bare => 0,
        };
        hypotheses.sort_by(|left, right| {
            left.short.cmp(&right.short).then_with(|| {
                if profile == DetectionProfile::CaseRootedComplete {
                    right
                        .score
                        .total_cmp(&left.score)
                        .then(rank(right.style).cmp(&rank(left.style)))
                } else {
                    right
                        .markers
                        .len()
                        .cmp(&left.markers.len())
                        .then(rank(right.style).cmp(&rank(left.style)))
                        .then(left.markers[0].number.cmp(&right.markers[0].number))
                }
            })
        });
        for hypothesis in hypotheses {
            let offsets = hypothesis
                .all
                .iter()
                .filter(|value| value.style == hypothesis.style)
                .map(|value| value.start)
                .collect::<Vec<_>>();
            let mut next = HashMap::new();
            for (index, start) in offsets.iter().enumerate() {
                next.insert(
                    *start,
                    offsets.get(index + 1).copied().unwrap_or(text.len()),
                );
            }
            let preliminary = hypothesis
                .markers
                .iter()
                .map(|marker| ScalarRange {
                    start: marker.start,
                    end: next.get(&marker.start).copied().unwrap_or(text.len()),
                })
                .collect::<Vec<_>>();
            let counts = preliminary
                .iter()
                .map(|range| {
                    if range.end >= range.start {
                        word_count(text.slice(*range), contiguous)
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>();
            let bounded = if counts.len() > 1 {
                &counts[..counts.len() - 1]
            } else {
                &counts[..]
            };
            let median = median(bounded);
            let mean = bounded.iter().sum::<usize>() as f64 / bounded.len().max(1) as f64;
            let maximum = bounded.iter().copied().max().unwrap_or(0);
            // SourceDocs scores the unmodified hypothesis. Lossy heading recovery
            // changes only the returned ranges after that hypothesis is accepted.
            let start = text.utf16(preliminary[0].start) as f64 / text.utf16_len().max(1) as f64;
            let span = (text.utf16(preliminary.last().unwrap().start)
                - text.utf16(preliminary[0].start)) as f64
                / text.utf16_len().max(1) as f64;
            if hypothesis.short {
                if text.utf16_len() <= 6_000
                    && (text.utf16(preliminary[0].start) <= 1_200 || start <= 0.5)
                    && counts.iter().copied().max().unwrap_or(0) >= 30
                {
                    return paragraph_ranges(
                        text,
                        &hypothesis.markers,
                        &hypothesis.all,
                        hypothesis.style,
                        !strict,
                        &excluded.iter().map(|value| value.start).collect::<Vec<_>>(),
                    );
                }
                continue;
            }
            let substantive = counts.iter().filter(|value| **value >= 12).count() as f64
                / preliminary.len() as f64;
            if !(median >= 12.0 || mean >= 20.0 || maximum >= 30)
                || span < 0.05
                || (hypothesis.style == MarkerStyle::Bracket
                    && text.utf16_len() > 6_000
                    && start > 0.7
                    && substantive < 0.5)
                || (hypothesis.style != MarkerStyle::Bracket && substantive < 0.7)
                || (hypothesis.style == MarkerStyle::Bare
                    && (median < 20.0 || span < 0.15 || start > 0.7))
            {
                continue;
            }
            return paragraph_ranges(
                text,
                &hypothesis.markers,
                &hypothesis.all,
                hypothesis.style,
                !strict,
                &excluded.iter().map(|value| value.start).collect::<Vec<_>>(),
            );
        }
        Vec::new()
    }

    fn gapped_paragraphs(blocks: &[Block]) -> bool {
        let values = blocks
            .iter()
            .filter(|value| value.kind == NodeKind::Paragraph)
            .filter_map(|value| {
                value
                    .label
                    .as_deref()?
                    .strip_prefix("par")?
                    .parse::<u32>()
                    .ok()
            })
            .collect::<Vec<_>>();
        values.windows(2).any(|pair| pair[1] != pair[0] + 1)
    }

    fn clipped_case_paragraphs(
        mut blocks: Vec<Block>,
        excluded: &[ScalarRange],
        text: &ScalarText<'_>,
    ) -> Vec<Block> {
        blocks.retain_mut(|block| {
            for range in excluded {
                if range.start >= block.range.end {
                    break;
                }
                if range.end <= block.range.start {
                    continue;
                }
                if range.start <= block.range.start {
                    return false;
                }
                block.range.end = range.start;
            }
            block.range.end > block.range.start
                && text.slice(block.range).chars().any(char::is_alphabetic)
        });
        blocks
    }

    #[derive(Clone)]
    struct PageMarker {
        number: u32,
        start: usize,
        content_start: usize,
    }

    fn page_markers(text: &ScalarText<'_>, report_start: Option<u32>) -> Vec<PageMarker> {
        let page_word = cached_regex!(PAGE_WORD, r"(?iu)\bpage\b");
        if !page_word.is_match(text.value) {
            return Vec::new();
        }
        let regex = cached_regex!(
            VALUE,
            r"(?imu)\[[ \t]*pages?[ \t]*[.:,;]?[ \t]*(\d{1,4})[ \t]*[.:,;]?[ \t]*[\]\[)}]?[ \t]*[.,;:]?|^[ \t]*\[?[ \t]*page[ \t]*[.:,;]?[ \t]*(\d{1,4})[ \t]*[\])}]?[ \t]*[.,;:]?[ \t]*$"
        );
        let mut result = Vec::new();
        for line in lines(text).filter(|line| {
            line.text
                .as_bytes()
                .windows(4)
                .any(|word| word.eq_ignore_ascii_case(b"page"))
        }) {
            for capture in regex.captures_iter(line.text) {
                let whole = capture.get(0).unwrap();
                let number = capture
                    .get(1)
                    .or_else(|| capture.get(2))
                    .unwrap()
                    .as_str()
                    .parse::<u32>()
                    .unwrap();
                if report_start.is_some_and(|start| number < start) {
                    continue;
                }
                let start = line.scalar_start + line.text[..whole.start()].chars().count();
                let content_start = line.scalar_start + line.text[..whole.end()].chars().count();
                result.push(PageMarker {
                    number,
                    start,
                    content_start,
                });
            }
        }
        result
    }

    fn detect_pages(
        text: &ScalarText<'_>,
        report_start: Option<u32>,
        require_report_start: bool,
    ) -> Vec<Block> {
        if require_report_start && report_start.is_none() {
            return Vec::new();
        }
        let mut scopes = Vec::<Vec<PageMarker>>::new();
        let mut by_last = HashMap::<u32, Vec<usize>>::new();
        for marker in page_markers(text, report_start) {
            let candidates = marker
                .number
                .checked_sub(1)
                .and_then(|number| by_last.get(&number))
                .cloned()
                .unwrap_or_default();
            let index = candidates
                .into_iter()
                .reduce(|best, current| {
                    if scopes[current].last().unwrap().start > scopes[best].last().unwrap().start {
                        current
                    } else {
                        best
                    }
                })
                .unwrap_or(scopes.len());
            if index == scopes.len() {
                scopes.push(vec![marker.clone()]);
            } else {
                let previous = scopes[index].last().unwrap().number;
                if let Some(values) = by_last.get_mut(&previous) {
                    values.retain(|value| *value != index);
                }
                scopes[index].push(marker.clone());
            }
            by_last.entry(marker.number).or_default().push(index);
        }
        scopes.retain(|scope| scope.len() >= 3);
        scopes.sort_by_key(|scope| std::cmp::Reverse(scope.len()));
        if scopes.is_empty()
            || scopes
                .get(1)
                .is_some_and(|other| other.len() == scopes[0].len())
        {
            return Vec::new();
        }
        let best = &scopes[0];
        let mut blocks = best
            .windows(2)
            .map(|pair| {
                Block::labelled(
                    NodeKind::Page,
                    format!("page{}", pair[0].number),
                    pair[0].content_start,
                    pair[1].start,
                )
            })
            .collect::<Vec<_>>();
        if report_start.is_some_and(|start| best[0].number == start + 1) {
            blocks.insert(
                0,
                Block::labelled(
                    NodeKind::Page,
                    format!("page{}", report_start.unwrap()),
                    0,
                    best[0].start,
                ),
            );
        }
        blocks
    }

    fn detect_case(
        text: &ScalarText<'_>,
        profile: DetectionProfile,
        report_start_page: Option<u32>,
        require_report_start: bool,
        excluded: &[ScalarRange],
    ) -> Vec<Block> {
        let mut paragraphs = match profile {
            DetectionProfile::CaseContiguousComplete => {
                let complete = clipped_case_paragraphs(
                    detect_paragraphs(text, DetectionProfile::CaseLossy, excluded),
                    excluded,
                    text,
                );
                if !complete.is_empty() && !gapped_paragraphs(&complete) {
                    complete
                } else {
                    detect_paragraphs(text, DetectionProfile::CaseContiguousComplete, excluded)
                }
            }
            profile => detect_paragraphs(text, profile, excluded),
        };
        paragraphs.extend(detect_pages(text, report_start_page, require_report_start));
        paragraphs
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    pub(super) enum SectionStyle {
        Integer,
        Dot,
        DotTerm,
        Hyphen,
        Mixed,
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    pub(super) enum SectionFamily {
        Bare,
        DotTerm,
        Markdown,
        Emphasis,
        Range,
    }

    #[derive(Clone)]
    pub(super) struct SectionMark {
        pub(super) label: String,
        pub(super) start: usize,
        pub(super) content_start: usize,
        pub(super) style: SectionStyle,
        pub(super) family: SectionFamily,
        pub(super) aliases: Vec<String>,
    }

    #[derive(Clone)]
    struct LabelPart {
        separator: char,
        digits: Option<String>,
        text: String,
        suffix: u32,
    }

    fn suffix_value(value: &str) -> u32 {
        value
            .to_ascii_uppercase()
            .bytes()
            .fold(0, |total, value| total * 26 + u32::from(value - b'A' + 1))
    }

    fn label_parts(label: &str) -> Vec<LabelPart> {
        let mut separator = '\0';
        label
            .split_inclusive(['.', '-'])
            .filter_map(|piece| {
                let (body, next) = piece
                    .strip_suffix('.')
                    .map(|value| (value, '.'))
                    .or_else(|| piece.strip_suffix('-').map(|value| (value, '-')))
                    .unwrap_or((piece, '\0'));
                if body.is_empty() {
                    separator = next;
                    return None;
                }
                let digits = body.bytes().take_while(u8::is_ascii_digit).count();
                let numeric = (digits > 0
                    && body[digits..]
                        .chars()
                        .all(|value| value.is_ascii_alphabetic()))
                .then(|| body[..digits].to_owned());
                let value = LabelPart {
                    separator,
                    digits: numeric,
                    text: body.to_owned(),
                    suffix: suffix_value(&body[digits..]),
                };
                separator = next;
                Some(value)
            })
            .collect()
    }

    fn compare_parts(
        left: &[LabelPart],
        right: &[LabelPart],
        fraction: bool,
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        for index in 0..left.len().max(right.len()) {
            let (Some(a), Some(b)) = (left.get(index), right.get(index)) else {
                return left.len().cmp(&right.len());
            };
            if a.separator != b.separator {
                return a.separator.cmp(&b.separator);
            }
            match (&a.digits, &b.digits) {
                (Some(a_digits), Some(b_digits)) => {
                    let width = a_digits.len().max(b_digits.len());
                    let ordered = if fraction && a.separator == '.' {
                        format!("{a_digits:0<width$}").cmp(&format!("{b_digits:0<width$}"))
                    } else {
                        format!("{:0>width$}", a_digits.trim_start_matches('0'))
                            .cmp(&format!("{:0>width$}", b_digits.trim_start_matches('0')))
                    };
                    if ordered != Equal {
                        return ordered;
                    }
                    if a_digits.len() != b_digits.len() {
                        return a_digits.len().cmp(&b_digits.len());
                    }
                    if a.suffix != b.suffix {
                        return a.suffix.cmp(&b.suffix);
                    }
                }
                (Some(_), None) => return Less,
                (None, Some(_)) => return Greater,
                (None, None) => {
                    let ordered = a
                        .text
                        .to_ascii_uppercase()
                        .cmp(&b.text.to_ascii_uppercase());
                    if ordered != Equal {
                        return ordered;
                    }
                }
            }
        }
        Equal
    }

    pub(super) fn compare_labels(left: &str, right: &str, fraction: bool) -> std::cmp::Ordering {
        compare_parts(&label_parts(left), &label_parts(right), fraction)
    }

    fn numeric_label(value: &str, markdown: bool) -> Option<(&str, usize)> {
        let matched =
            cached_regex!(VALUE, r"^\d{1,8}(?:[.-]\d{1,8}){0,3}[A-Z]{0,2}").find(value)?;
        let label = matched.as_str();
        (!markdown || label.contains(['.', '-'])).then_some((label, matched.end()))
    }

    fn provision_label(value: &str) -> Option<(&str, usize)> {
        let matched = cached_regex!(VALUE,
        r"^(?:\d{1,8}[A-Za-z]{0,3}(?:[.-]\d{1,8}[A-Za-z]{0,3}){0,3}|[A-Za-z]{1,3}(?:[.-][0-9A-Za-z]{1,8}){1,3})"
    ).find(value)?;
        Some((matched.as_str(), matched.end()))
    }

    fn section_style(label: &str, trailing: bool) -> SectionStyle {
        if trailing {
            SectionStyle::DotTerm
        } else if label.contains('.') && label.contains('-') {
            SectionStyle::Mixed
        } else if label.contains('-') {
            SectionStyle::Hyphen
        } else if label.contains('.') {
            SectionStyle::Dot
        } else {
            SectionStyle::Integer
        }
    }

    fn bare_content_starts(value: &str) -> bool {
        value.chars().next().is_some_and(|character| {
            character.is_alphabetic() || character.is_ascii_digit() || "([*“\"«".contains(character)
        })
    }

    fn dotterm_content_starts(value: &str) -> bool {
        value
            .chars()
            .next()
            .is_some_and(|character| character.is_alphanumeric() || "\"'“«(".contains(character))
    }

    fn markdown_range_continuation(value: &str) -> bool {
        cached_regex!(
            VALUE,
            r"(?iu)^[ \t]*#{1,6}[ \t]+.*(?:[ \t](?:to|à)|[-–—])[ \t]*$"
        )
        .is_match(value)
    }

    fn previous_nonblank<'a>(source: &'a [Line<'a>], index: usize) -> Option<&'a str> {
        source[..index]
            .iter()
            .rev()
            .map(|line| line.text)
            .find(|value| !value.trim().is_empty())
    }

    fn collect_sections(text: &ScalarText<'_>, family: SectionFamily) -> Vec<SectionMark> {
        let source = lines(text).collect::<Vec<_>>();
        let mut result = Vec::new();
        for (index, line) in source.iter().enumerate() {
            let lead = leading_ascii_space(line.text);
            let mut value = &line.text[lead..];
            if family == SectionFamily::Markdown {
                let hashes = value.bytes().take_while(|byte| *byte == b'#').count();
                if !(1..=6).contains(&hashes) || !value[hashes..].starts_with([' ', '\t']) {
                    continue;
                }
                value = value[hashes..].trim_start_matches([' ', '\t']);
            }
            let bold = value.starts_with("**");
            if bold {
                value = &value[2..];
            }
            let Some((label, length)) = numeric_label(value, family == SectionFamily::Markdown)
            else {
                continue;
            };
            let mut after = length;
            if bold {
                if !value[after..].starts_with("**") {
                    continue;
                }
                after += 2;
            }
            let mut trailing = false;
            if family == SectionFamily::DotTerm {
                let Some(punctuation) = value[after..]
                    .chars()
                    .next()
                    .filter(|value| matches!(value, '.' | ')'))
                else {
                    continue;
                };
                after += punctuation.len_utf8();
                trailing = true;
            } else if family == SectionFamily::Markdown && value[after..].starts_with('.') {
                after += 1;
                trailing = true;
            }
            let rest = &value[after..];
            let spaces = leading_ascii_space(rest);
            let content = &rest[spaces..];
            let accepted = match family {
                SectionFamily::Bare => {
                    content.is_empty()
                        || (spaces > 0 && bare_content_starts(content))
                        || (spaces == 0 && content.starts_with('('))
                }
                SectionFamily::DotTerm => {
                    !content.is_empty()
                        && dotterm_content_starts(content)
                        && (spaces > 0 || content.starts_with('('))
                }
                SectionFamily::Markdown => {
                    content.is_empty() || (spaces > 0 && !content.is_empty())
                }
                _ => false,
            };
            if !accepted {
                continue;
            }
            let start_byte = line.byte_start + lead;
            let content_byte = line.byte_end - content.len();
            if family == SectionFamily::Bare
                && content.is_empty()
                && previous_nonblank(&source, index).is_some_and(markdown_range_continuation)
            {
                continue;
            }
            result.push(SectionMark {
                label: label.to_owned(),
                start: text.scalar(start_byte),
                content_start: text.scalar(content_byte),
                style: section_style(label, trailing),
                family,
                aliases: Vec::new(),
            });
        }
        result
    }

    fn section_key(label: &str) -> Vec<u64> {
        label
            .split(['.', '-'])
            .filter_map(|value| {
                value
                    .bytes()
                    .take_while(u8::is_ascii_digit)
                    .fold(None, |total, digit| {
                        Some(total.unwrap_or(0) * 10 + u64::from(digit - b'0'))
                    })
            })
            .collect()
    }

    fn section_scopes(
        marks: &[SectionMark],
        styles: &[SectionStyle],
        root: bool,
        fraction: bool,
    ) -> Vec<Vec<SectionMark>> {
        let mut scopes = Vec::<Vec<SectionMark>>::new();
        for mark in marks.iter().filter(|value| styles.contains(&value.style)) {
            let parts = label_parts(&mark.label);
            let best = scopes
                .iter()
                .enumerate()
                .filter(|(_, scope)| {
                    let last = scope.last().unwrap();
                    let prior = label_parts(&last.label);
                    parts.len() == prior.len() && compare_parts(&parts, &prior, fraction).is_gt()
                })
                .reduce(|best, candidate| {
                    if compare_labels(
                        &candidate.1.last().unwrap().label,
                        &best.1.last().unwrap().label,
                        fraction,
                    )
                    .is_gt()
                    {
                        candidate
                    } else {
                        best
                    }
                })
                .map(|value| value.0);
            if let Some(best) = best {
                scopes[best].push(mark.clone());
            } else {
                scopes.push(vec![mark.clone()]);
            }
            if scopes.len() > 8 {
                let smallest = (0..scopes.len())
                    .min_by_key(|index| scopes[*index].len())
                    .unwrap();
                scopes.remove(smallest);
            }
        }
        scopes
            .into_iter()
            .filter(|scope| {
                scope.len() >= 3
                    && (!root || section_key(&scope[0].label).iter().all(|value| *value == 1))
            })
            .collect()
    }

    fn expand_descendants(
        scope: Vec<SectionMark>,
        marks: &[SectionMark],
        length: usize,
    ) -> Vec<SectionMark> {
        if scope
            .first()
            .is_none_or(|value| section_key(&value.label).len() != 1)
        {
            return scope;
        }
        let mut result = Vec::new();
        for (index, parent) in scope.iter().enumerate() {
            let end = scope.get(index + 1).map_or(length, |value| value.start);
            let descendants = marks
                .iter()
                .filter(|mark| {
                    mark.start > parent.start
                        && mark.start < end
                        && matches!(mark.style, SectionStyle::Dot | SectionStyle::DotTerm)
                        && mark.label.contains('.')
                        && section_key(&mark.label).first() == section_key(&parent.label).first()
                })
                .cloned()
                .collect::<Vec<_>>();
            let counts =
                descendants
                    .iter()
                    .fold(HashMap::<String, usize>::new(), |mut counts, value| {
                        *counts.entry(value.label.clone()).or_default() += 1;
                        counts
                    });
            result.push(parent.clone());
            result.extend(
                descendants
                    .into_iter()
                    .filter(|value| counts.get(value.label.as_str()) == Some(&1)),
            );
        }
        result
    }

    fn same_labels(left: &[SectionMark], right: &[SectionMark]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.label == right.label)
    }

    fn choose_sections(
        left: Option<Vec<SectionMark>>,
        right: Option<Vec<SectionMark>>,
    ) -> Option<Vec<SectionMark>> {
        match (left, right) {
            (None, value) | (value, None) => value,
            (Some(left), Some(right)) if same_labels(&left, &right) => Some(left),
            (Some(left), Some(right)) if left[0].start != right[0].start => {
                Some(if left[0].start < right[0].start {
                    left
                } else {
                    right
                })
            }
            (Some(left), Some(right)) if left.len() != right.len() => {
                Some(if left.len() > right.len() {
                    left
                } else {
                    right
                })
            }
            _ => None,
        }
    }

    fn section_guard(value: &[SectionMark], text: &ScalarText<'_>) -> bool {
        !value.is_empty()
            && text.utf16_len() > 0
            && text.utf16(value[0].start) as f64 / text.utf16_len() as f64 <= 0.7
    }

    fn scope_winner(
        scopes: Vec<Vec<SectionMark>>,
        marks: &[SectionMark],
        text: &ScalarText<'_>,
    ) -> Option<Vec<SectionMark>> {
        let mut values = scopes
            .into_iter()
            .map(|scope| {
                if scope.first().is_some_and(|value| {
                    value.style != SectionStyle::DotTerm && section_key(&value.label).len() == 1
                }) {
                    expand_descendants(scope, marks, text.len())
                } else {
                    scope
                }
            })
            .filter(|scope| section_guard(scope, text))
            .collect::<Vec<_>>();
        values.sort_by(|left, right| {
            right
                .len()
                .cmp(&left.len())
                .then(left[0].start.cmp(&right[0].start))
        });
        let best = values.first()?.clone();
        (!values.iter().skip(1).any(|value| {
            value.len() == best.len()
                && value[0].start == best[0].start
                && !same_labels(value, &best)
        }))
        .then_some(best)
    }

    fn statute_winner(
        marks: &[SectionMark],
        text: &ScalarText<'_>,
        allow_hyphen: bool,
    ) -> Option<Vec<SectionMark>> {
        if marks.len() < 3 {
            return None;
        }
        let mut component = section_scopes(
            marks,
            &[
                SectionStyle::Integer,
                SectionStyle::Dot,
                SectionStyle::DotTerm,
            ],
            false,
            false,
        );
        if allow_hyphen {
            component.extend(section_scopes(marks, &[SectionStyle::Hyphen], true, false));
            component.extend(section_scopes(marks, &[SectionStyle::Mixed], true, false));
        }
        choose_sections(
            scope_winner(component, marks, text),
            scope_winner(
                section_scopes(marks, &[SectionStyle::Dot], false, true),
                marks,
                text,
            ),
        )
    }

    fn inline_section(text: &ScalarText<'_>, mark: &SectionMark) -> bool {
        let start = text.byte(mark.content_start);
        let end = text.value[start..]
            .find('\n')
            .map_or(text.value.len(), |value| start + value);
        !text.value[start..end].trim().is_empty()
    }

    fn next_nonblank<'a>(source: &'a [Line<'a>], start: usize) -> Option<&'a Line<'a>> {
        source
            .iter()
            .find(|line| line.scalar_start > start && !line.text.trim().is_empty())
    }

    fn short_root(text: &ScalarText<'_>) -> Vec<SectionMark> {
        let status = cached_regex!(
            STATUS,
            r"(?iu)^(?:\[\s*)?(?:repealed|revoked|abrog(?:ated|é|ée|és|ées)|renumbered|spent|not (?:yet )?in force|omitted)\b"
        );
        let heading = cached_regex!(HEADING, r#"^(?:(?:["'“«]\s*)?\p{Lu}|\(\d+\))"#);
        let source = lines(text).collect::<Vec<_>>();
        let mut candidates = collect_sections(text, SectionFamily::Bare);
        candidates.extend(collect_sections(text, SectionFamily::DotTerm));
        candidates.extend(collect_sections(text, SectionFamily::Markdown));
        for line in &source {
            let value = line.text.trim_matches([' ', '\t']);
            if matches!(value, "1" | "2") {
                candidates.push(SectionMark {
                    label: value.to_owned(),
                    start: line.scalar_start + leading_ascii_space(line.text),
                    content_start: line.scalar_start + line.text.chars().count(),
                    style: SectionStyle::Integer,
                    family: SectionFamily::Bare,
                    aliases: Vec::new(),
                });
            }
        }
        candidates.retain(|value| matches!(value.label.as_str(), "1" | "2"));
        let mut invalid = false;
        for marker in &mut candidates {
            let start = text.byte(marker.content_start);
            let end = text.value[start..]
                .find('\n')
                .map_or(text.value.len(), |value| start + value);
            // Preserve the source grammar's raw offset check: on CRLF input a
            // label ending immediately before `\r` is not "at" the `\n` line
            // end, even though the intervening scalar is whitespace.
            if start >= end {
                let Some(next) = next_nonblank(&source, marker.start) else {
                    invalid = true;
                    continue;
                };
                let parenthetical = next.text.trim_start().starts_with('(')
                    && next.text.trim_start()[1..]
                        .chars()
                        .next()
                        .is_some_and(char::is_numeric);
                if !heading.is_match(next.text.trim_start())
                    && !parenthetical
                    && !status.is_match(next.text.trim_start())
                {
                    invalid = true;
                } else {
                    marker.content_start = next.scalar_start + leading_ascii_space(next.text);
                }
            }
        }
        if invalid {
            return Vec::new();
        }
        candidates.sort_by_key(|value| value.start);
        candidates.dedup_by(|left, right| left.label == right.label && left.start == right.start);
        let ones = candidates
            .iter()
            .filter(|value| value.label == "1")
            .cloned()
            .collect::<Vec<_>>();
        let twos = candidates
            .iter()
            .filter(|value| value.label == "2")
            .cloned()
            .collect::<Vec<_>>();
        if ones.len() != 1
            || twos.len() > 1
            || twos.first().is_some_and(|two| two.start <= ones[0].start)
        {
            return Vec::new();
        }
        let result = if twos.is_empty() {
            vec![ones[0].clone()]
        } else {
            vec![ones[0].clone(), twos[0].clone()]
        };
        if text.utf16(result[0].start) as f64 / text.utf16_len().max(1) as f64 <= 0.7 {
            result
        } else {
            Vec::new()
        }
    }

    fn statute_spine_over(
        text: &ScalarText<'_>,
        allow_hyphen: bool,
        inline_only: bool,
    ) -> Vec<SectionMark> {
        let families = [
            SectionFamily::Bare,
            SectionFamily::DotTerm,
            SectionFamily::Markdown,
        ]
        .map(|family| {
            collect_sections(text, family)
                .into_iter()
                .filter(|mark| !inline_only || inline_section(text, mark))
                .collect::<Vec<_>>()
        });
        let mut candidates = families
            .iter()
            .filter_map(|marks| statute_winner(marks, text, allow_hyphen))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|value| value[0].start);
        if candidates.is_empty() {
            return short_root(text);
        }
        let mut best = candidates[0].clone();
        let first_start = best[0].start;
        for candidate in candidates
            .into_iter()
            .skip(1)
            .take_while(|value| value[0].start == first_start)
        {
            let Some(chosen) = choose_sections(Some(best), Some(candidate)) else {
                return Vec::new();
            };
            best = chosen;
        }
        if best[0].family == SectionFamily::DotTerm {
            let mut all = [families[0].clone(), families[1].clone()].concat();
            all.sort_by_key(|value| value.start);
            expand_descendants(best, &all, text.len())
        } else {
            best
        }
    }

    pub(super) fn statute_spine(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
        let result = statute_spine_over(text, allow_hyphen, false);
        if result.is_empty() || result.iter().any(|value| inline_section(text, value)) {
            result
        } else {
            statute_spine_over(text, allow_hyphen, true)
        }
    }

    fn dotted_order(marks: &[SectionMark]) -> Option<bool> {
        let dotted = marks
            .iter()
            .map(|value| &value.label)
            .filter(|value| value.contains('.') && !value.contains('-'))
            .collect::<Vec<_>>();
        let inversions = |fraction| {
            dotted
                .windows(2)
                .filter(|pair| compare_labels(pair[0], pair[1], fraction).is_gt())
                .count()
        };
        let component = inversions(false);
        let fraction = inversions(true);
        if component != fraction {
            return Some(fraction < component);
        }
        if dotted.windows(2).any(|pair| {
            compare_labels(pair[0], pair[1], false) != compare_labels(pair[0], pair[1], true)
        }) {
            None
        } else {
            Some(false)
        }
    }

    fn emphasis_sections(text: &ScalarText<'_>) -> Vec<SectionMark> {
        let mut candidates = Vec::new();
        for line in lines(text) {
            let lead = leading_ascii_space(line.text);
            let value = &line.text[lead..];
            let Some(value) = value.strip_prefix("**") else {
                continue;
            };
            let Some((label, length)) = provision_label(value) else {
                continue;
            };
            if !value[length..].starts_with("**") {
                continue;
            }
            let rest = &value[length + 2..];
            if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
                continue;
            }
            candidates.push(SectionMark {
                label: label.to_owned(),
                start: line.scalar_start + lead,
                content_start: line.scalar_start
                    + lead
                    + 2
                    + label.chars().count()
                    + 2
                    + leading_ascii_space(rest),
                style: section_style(label, false),
                family: SectionFamily::Emphasis,
                aliases: Vec::new(),
            });
        }
        let Some(first) = candidates.first() else {
            return candidates;
        };
        let numeric = first
            .label
            .starts_with(|value: char| value.is_ascii_digit());
        candidates.retain(|value| {
            value
                .label
                .starts_with(|character: char| character.is_ascii_digit())
                == numeric
        });
        let Some(fraction) = dotted_order(&candidates) else {
            return Vec::new();
        };
        let mut result = Vec::<SectionMark>::new();
        for marker in candidates {
            if result
                .last()
                .is_none_or(|prior| compare_labels(&marker.label, &prior.label, fraction).is_gt())
            {
                result.push(marker);
            }
        }
        if result.is_empty()
            || text.utf16(result[0].start) as f64 / text.utf16_len().max(1) as f64 > 0.7
            || (text.utf16_len() - text.utf16(result[0].start)) as f64
                / (text.utf16_len().max(1) as f64)
                < 0.1
        {
            Vec::new()
        } else {
            result
        }
    }

    fn status_sections(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
        let regex = cached_regex!(
            VALUE,
            r"(?imu)^[ \t]*(?:\*\*)?(\d{1,4})(?:[ \t]+(?:to|through|and|à|a|et)[ \t]+|[ \t]*([-–—])[ \t]*)(\d{1,4})(?:\*\*)?[ \t]*[,;:]?[ \t]*(?:\[[ \t]*)?(?:repealed|revoked|abrog(?:ated|é|ée|és|ées)|renumbered|spent|not (?:yet )?in force|omitted)\b"
        );
        regex
            .captures_iter(text.value)
            .filter_map(|capture| {
                if allow_hyphen && capture.get(2).is_some() {
                    return None;
                }
                let from = capture[1].parse::<u32>().ok()?;
                let to = capture[3].parse::<u32>().ok()?;
                if from >= to || to > from + 400 {
                    return None;
                }
                let whole = capture.get(0).unwrap();
                Some(SectionMark {
                    label: from.to_string(),
                    start: text.scalar(whole.start() + leading_ascii_space(whole.as_str())),
                    content_start: text.scalar(whole.end()),
                    style: SectionStyle::Integer,
                    family: SectionFamily::Range,
                    aliases: (from + 1..=to).map(|value| value.to_string()).collect(),
                })
            })
            .collect()
    }

    fn coherent_sections(marks: &[SectionMark]) -> bool {
        let Some(fraction) = dotted_order(marks) else {
            return false;
        };
        marks
            .windows(2)
            .all(|pair| compare_labels(&pair[0].label, &pair[1].label, fraction).is_lt())
    }

    fn selected_sections(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
        let emphasis = emphasis_sections(text);
        let flat = statute_spine(text, allow_hyphen);
        let mut selected = if emphasis.is_empty() {
            flat
        } else if flat.is_empty() {
            emphasis
        } else {
            let occurrences = emphasis
                .iter()
                .map(|value| (value.label.to_ascii_lowercase(), value.content_start))
                .collect::<HashSet<_>>();
            if !flat.iter().any(|value| {
                occurrences.contains(&(value.label.to_ascii_lowercase(), value.content_start))
            }) {
                emphasis
            } else {
                let mut labels = flat
                    .into_iter()
                    .map(|value| (value.label.to_ascii_lowercase(), value))
                    .collect::<HashMap<_, _>>();
                for marker in emphasis.iter().cloned() {
                    let key = marker.label.to_ascii_lowercase();
                    if labels
                        .get(&key)
                        .is_none_or(|value| value.content_start == marker.content_start)
                    {
                        labels.insert(key, marker);
                    }
                }
                let mut combined = labels.into_values().collect::<Vec<_>>();
                combined.sort_by_key(|value| value.start);
                if coherent_sections(&combined) {
                    combined
                } else {
                    emphasis
                }
            }
        };
        let ranges = status_sections(text, allow_hyphen);
        if !ranges.is_empty() {
            let mut labels = selected
                .iter()
                .cloned()
                .map(|value| (value.label.to_ascii_lowercase(), value))
                .collect::<HashMap<_, _>>();
            for marker in ranges {
                for alias in &marker.aliases {
                    labels.remove(&alias.to_ascii_lowercase());
                }
                labels.insert(marker.label.to_ascii_lowercase(), marker);
            }
            let mut combined = labels.into_values().collect::<Vec<_>>();
            combined.sort_by_key(|value| value.start);
            if coherent_sections(&combined) {
                selected = combined;
            }
        }
        selected
    }

    fn roman_value(value: &str) -> Option<u32> {
        let mut total = 0i32;
        let mut prior = 0;
        for character in value.to_ascii_lowercase().bytes().rev() {
            let value = match character {
                b'i' => 1,
                b'v' => 5,
                b'x' => 10,
                b'l' => 50,
                b'c' => 100,
                b'd' => 500,
                b'm' => 1000,
                _ => return None,
            };
            total += if value < prior { -value } else { value };
            prior = prior.max(value);
        }
        (total > 0).then_some(total as u32)
    }

    struct EnumFrame {
        family: u8,
        value: String,
        label: String,
    }

    struct ChildMark<'a> {
        token: &'a str,
        start: usize,
        content_start: usize,
    }

    #[derive(Default)]
    struct StructureState {
        nodes: Vec<(Block, usize)>,
        container: Option<String>,
        section: Option<(String, usize)>,
        stack: Vec<EnumFrame>,
        used: HashMap<String, usize>,
    }

    fn enum_readings(token: &str) -> [Option<(u8, String)>; 2] {
        if token
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return [Some((0, token.to_owned())), None];
        }
        let lower = token.to_ascii_lowercase();
        let alpha = match lower.as_bytes() {
            [value @ b'a'..=b'z'] => Some(u32::from(*value - b'a' + 1)),
            [left @ b'a'..=b'z', right] if left == right => Some(26 + u32::from(*left - b'a' + 1)),
            _ => None,
        };
        let upper = token != lower;
        let alpha = alpha.map(|value| (if upper { 3 } else { 1 }, value.to_string()));
        let roman = roman_value(&lower).map(|value| (if upper { 4 } else { 2 }, value.to_string()));
        if lower.len() > 1 {
            [
                roman
                    .clone()
                    .filter(|(_, value)| value.parse::<u32>().unwrap() <= 50)
                    .or(alpha)
                    .or(roman),
                None,
            ]
        } else {
            [alpha, roman]
        }
    }

    fn instrument_marker(value: &str, tail: bool, dot: bool) -> Option<(&str, usize)> {
        let found = cached_regex!(MARKER,
            r"^(?:\((\d{1,3}(?:\.\d{1,3})?|[a-z]{1,2}|[ivxlcdm]{1,6}|[A-Z]{1,2}|[IVXLCDM]{1,6})\)[ \t]*(.*)|([a-z]{1,2}|[ivxlcdm]{1,6})\)[ \t]+(\S.*)|([a-z]{1,2}|[ivxlcdm]{1,6})\.[ \t]+(\S.*))$"
        ).captures(value)?;
        [(1, 2, true), (3, 4, tail), (5, 6, dot)]
            .into_iter()
            .find_map(|(token, rest, enabled)| {
                enabled.then_some((found.get(token)?.as_str(), found.get(rest)?.start()))
            })
    }

    fn instrument_space(character: char) -> bool {
        character.is_whitespace() || character == '\u{feff}'
    }

    fn legislation_marker(value: &str, followed_by_newline: bool) -> Option<(&str, usize, usize)> {
        let found = cached_regex!(
            MARKER,
            r"^\((\d+(?:\.\d+)?|[A-Za-z](?:\.\d+)?|[ivxlcdmIVXLCDM]+)\)"
        )
        .captures(value)?;
        let whole = found.get(0).unwrap();
        let rest = &value[whole.end()..];
        (rest.chars().next().is_some_and(char::is_whitespace)
            || (rest.is_empty() && followed_by_newline))
            .then(|| {
                (
                    found.get(1).unwrap().as_str(),
                    whole.end() + leading_ascii_space(rest),
                    whole.end() - 1,
                )
            })
    }

    fn compare_child_values(left: &str, right: &str) -> std::cmp::Ordering {
        let left = left.split('.').collect::<Vec<_>>();
        let right = right.split('.').collect::<Vec<_>>();
        for index in 0..left.len().max(right.len()) {
            let (left, right) = (
                left.get(index).copied().unwrap_or_default(),
                right.get(index).copied().unwrap_or_default(),
            );
            if left == right {
                continue;
            }
            let ordered = if index == 0 {
                left.parse::<f64>()
                    .unwrap_or_default()
                    .partial_cmp(&right.parse::<f64>().unwrap_or_default())
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                let width = left.len().max(right.len());
                format!("{left:0<width$}").cmp(&format!("{right:0<width$}"))
            };
            if !ordered.is_eq() {
                return ordered;
            }
        }
        std::cmp::Ordering::Equal
    }

    fn admitted_dialects(text: &ScalarText<'_>) -> (bool, bool) {
        let mut live = [HashMap::<u8, (String, usize)>::new(), HashMap::new()];
        let mut best = [0, 0];
        for line in lines(text) {
            let value = line.text.trim_matches(instrument_space);
            if instrument_marker(value, false, false).is_some() {
                continue;
            }
            for (index, dialect) in [(true, false), (false, true)].into_iter().enumerate() {
                if let Some((token, _)) = instrument_marker(value, dialect.0, dialect.1) {
                    for (family, value) in enum_readings(token).into_iter().flatten() {
                        let state = live[index].entry(family).or_default();
                        if value == "1" {
                            *state = (value, 1);
                        } else if state.1 > 0 && compare_labels(&value, &state.0, true).is_gt() {
                            *state = (value, state.1 + 1);
                        }
                        best[index] = best[index].max(state.1);
                    }
                }
            }
        }
        (best[0] >= 3, best[1] >= 3)
    }

    impl StructureState {
        fn emit_child(
            &mut self,
            token: &str,
            start: usize,
            content_start: usize,
            parent: String,
            depth: usize,
            code: &'static str,
        ) -> String {
            let base = format!("{parent}({token})");
            let occurrence = self.used.entry(base.clone()).or_insert(1);
            let label = (*occurrence > 1)
                .then(|| format!("{base}@{occurrence}"))
                .unwrap_or(base);
            *occurrence += 1;
            let mut block = Block::labelled(NodeKind::Section, label.clone(), start, usize::MAX);
            block.parent_label = Some(parent);
            block.content_start = Some(content_start);
            block.diagnostic = Some(code);
            self.nodes.push((block, depth));
            label
        }

        fn child(&mut self, token: &str, start: usize, content_start: usize) {
            let (root, root_depth) = self.section.clone().unwrap();
            let readings = enum_readings(token);
            let selected = (0..4).find_map(|pass| {
                readings.iter().flatten().find_map(|(family, value)| {
                    let at = self.stack.iter().rposition(|frame| frame.family == *family);
                    match (pass, at) {
                        (0, Some(index)) => {
                            let prior = &self.stack[index].value;
                            match (prior.parse::<u32>(), value.parse::<u32>()) {
                                (Ok(prior), Ok(value)) => prior + 1 == value,
                                _ => compare_labels(value, prior, true).is_gt(),
                            }
                        }
                        (1, None) => value == "1" && self.stack.len() < 6,
                        (2, Some(index)) => {
                            value == "1"
                                || compare_labels(value, &self.stack[index].value, true).is_gt()
                        }
                        (3, None) => self.stack.len() < 6,
                        _ => false,
                    }
                    .then(|| (pass, *family, value.clone(), at))
                })
            });
            let (pass, family, value, at) = selected.unwrap_or_else(|| (4, 0, String::new(), None));
            let (parent, depth) = if pass == 4 {
                (root.clone(), root_depth + 1)
            } else if let Some(index) = at {
                self.stack.truncate(index + 1);
                self.stack[index].value = value.clone();
                (
                    index
                        .checked_sub(1)
                        .map_or_else(|| root.clone(), |parent| self.stack[parent].label.clone()),
                    root_depth + index + 1,
                )
            } else {
                let parent = self
                    .stack
                    .last()
                    .map_or_else(|| root.clone(), |frame| frame.label.clone());
                let depth = root_depth + self.stack.len() + 1;
                self.stack.push(EnumFrame {
                    family,
                    value: value.clone(),
                    label: String::new(),
                });
                (parent, depth)
            };
            let code = match (pass, value.as_str()) {
                (0, _) => "instrument_ladder_increment",
                (1, _) => "instrument_ladder_level_open",
                (2, "1") => "instrument_ladder_restart",
                (2, _) => "instrument_ladder_forward_jump",
                (3, _) => "instrument_ladder_midcounter_open",
                _ => "instrument_ladder_violation",
            };
            let label = self.emit_child(token, start, content_start, parent, depth, code);
            if pass < 4 {
                self.stack.last_mut().unwrap().label = label.clone();
            }
        }

        fn legislation_child(
            &mut self,
            token: &str,
            next: Option<&str>,
            start: usize,
            content_start: usize,
        ) {
            let (head, suffix) = token.split_once('.').unwrap_or((token, ""));
            let numeric = token
                .as_bytes()
                .first()
                .is_some_and(|value| value.is_ascii_digit());
            let roman = roman_value(head);
            let upper = head != head.to_ascii_lowercase();
            let alpha_level = if upper { 4 } else { 2 };
            let alpha_value = head
                .as_bytes()
                .first()
                .filter(|value| value.is_ascii_alphabetic())
                .map(|value| u32::from(value.to_ascii_lowercase() - b'a' + 1));
            let prior = |family| {
                self.stack
                    .iter()
                    .find(|frame| frame.family == family)
                    .map(|frame| frame.value.as_str())
            };
            let roman_preferred = if head.len() > 1 {
                true
            } else if let Some(value) = roman {
                if prior(3)
                    .and_then(|prior| prior.parse::<u32>().ok())
                    .is_some_and(|prior| prior + 1 == value)
                {
                    true
                } else if alpha_value.is_some_and(|value| {
                    prior(alpha_level)
                        .and_then(|prior| prior.parse::<u32>().ok())
                        .is_some_and(|prior| prior + 1 == value)
                }) {
                    head.eq_ignore_ascii_case("i")
                        && next.is_some_and(|next| next.eq_ignore_ascii_case("ii"))
                } else {
                    !upper && head == "i" && self.stack.iter().any(|frame| frame.family == 2)
                }
            } else {
                false
            };
            let (family, value) = if numeric {
                (1, token.to_owned())
            } else if roman_preferred {
                (3, roman.unwrap().to_string())
            } else {
                let Some(alpha) = alpha_value else { return };
                (
                    alpha_level,
                    if suffix.is_empty() {
                        alpha.to_string()
                    } else {
                        format!("{alpha}.{suffix}")
                    },
                )
            };
            if prior(family).is_some_and(|prior| !compare_child_values(&value, prior).is_gt()) {
                return;
            }

            self.stack.retain(|frame| frame.family <= family);
            let at = self.stack.iter().position(|frame| frame.family == family);
            let (root, root_depth) = self.section.clone().unwrap();
            let parent = at
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| self.stack.get(index))
                .or_else(|| self.stack.iter().rev().find(|frame| frame.family < family))
                .map_or_else(|| root.clone(), |frame| frame.label.clone());
            if let Some(index) = at {
                self.stack[index].value = value.clone();
            } else {
                self.stack.push(EnumFrame {
                    family,
                    value,
                    label: String::new(),
                });
            }
            let label = self.emit_child(
                token,
                start,
                content_start,
                parent,
                root_depth + usize::from(family),
                "legislation_child",
            );
            self.stack
                .iter_mut()
                .find(|frame| frame.family == family)
                .unwrap()
                .label = label;
        }
    }

    fn enumerated_children(
        value: &str,
        offset: usize,
        root: String,
        root_depth: usize,
        content_start: usize,
        inline_at_root: bool,
        leading_label: Option<&str>,
    ) -> Vec<Block> {
        let text = ScalarText::new(value);
        let public_parent = root.clone();
        let mut state = StructureState {
            section: Some((root, root_depth)),
            ..Default::default()
        };
        let mut markers = Vec::new();
        let leading = leading_label.and_then(|label| {
            let lead = leading_ascii_space(value);
            let rest = value[lead..].strip_prefix(label)?;
            let gap = leading_ascii_space(rest);
            let prefix = lead + label.len() + gap;
            let line = value[prefix..].lines().next().unwrap_or_default();
            let newline = line.len() < value[prefix..].len();
            let (token, content, close) = legislation_marker(line, newline)?;
            Some(ChildMark {
                token,
                start: value[..prefix + close].chars().count(),
                content_start: value[..prefix + content].chars().count(),
            })
        });
        if let Some(marker) = leading {
            markers.push(marker);
        } else {
            let inline = &text.value[text.byte(content_start)..];
            let inline_line = inline.lines().next().unwrap_or_default();
            if let Some((token, at, _)) = legislation_marker(inline_line, inline.contains('\n')) {
                markers.push(ChildMark {
                    token,
                    start: if inline_at_root { 0 } else { content_start },
                    content_start: content_start + inline_line[..at].chars().count(),
                });
            }
        }
        for line in lines(&text) {
            let value = line.text.trim_start_matches(instrument_space);
            if let Some((token, at, _)) =
                legislation_marker(value, line.byte_end < text.value.len())
            {
                if !markers
                    .iter()
                    .any(|marker| marker.start == line.scalar_start)
                {
                    markers.push(ChildMark {
                        token,
                        start: line.scalar_start,
                        content_start: line.scalar_start
                            + line.text[..line.text.len() - value.len()].chars().count()
                            + value[..at].chars().count(),
                    });
                }
            }
        }
        markers.sort_by_key(|marker| marker.start);
        for index in 0..markers.len() {
            let marker = &markers[index];
            state.legislation_child(
                marker.token,
                markers.get(index + 1).map(|next| next.token),
                marker.start,
                marker.content_start,
            );
        }
        for index in 0..state.nodes.len() {
            let start = state.nodes[index].0.range.start;
            let end = markers
                .iter()
                .find(|marker| marker.start > start)
                .map_or(text.len(), |marker| marker.start);
            state.nodes[index].0.range.end = end;
            state.nodes[index].0.range.start += offset;
            state.nodes[index].0.range.end += offset;
            if let Some(at) = &mut state.nodes[index].0.content_start {
                *at += offset;
            }
            state.nodes[index].0.parent_label = Some(public_parent.clone());
        }
        state.nodes.into_iter().map(|(block, _)| block).collect()
    }

    #[cfg(all(feature = "a2aj", feature = "source-doc"))]
    pub(super) fn source_doc_children(
        label: &str,
        value: &str,
        offset: usize,
    ) -> Vec<SourceDocBlock> {
        let text = ScalarText::new(value);
        enumerated_children(value, 0, format!("sec{label}"), 0, 0, false, Some(label))
            .into_iter()
            .map(|block| {
                let mut projected = SourceDocBlock::new(
                    SourceDocKind::Section,
                    block.label.unwrap(),
                    offset + text.utf16(block.range.start),
                    offset + text.utf16(block.range.end),
                    SourceDocOrigin::Heuristic,
                );
                projected.parent_label = block.parent_label;
                projected
            })
            .collect()
    }

    fn direct_section(value: &str) -> Option<(String, usize)> {
        let found = cached_regex!(SECTION,
            r#"^(?:(?:Section|SECTION)\s+(\d{1,3}(?:\.\d{1,3})*[A-Za-z]?)[.)]?\s*[—–\-:]?\s*(["'“(A-Z].*|)|(\d{1,3}\.\d{1,3}(?:\.\d{1,3})*)\s+(["'“(A-Z].*)|((?:[0-4]?\d{1,2}|500))[.)]\s+(["'“(A-Z].*)|(\d{1,3}(?:\.\d{1,3}){0,3})[ \t]+(\(\d.*))$"#
        ).captures(value)?;
        let label = [1, 3, 5, 7]
            .into_iter()
            .find(|index| found.get(*index).is_some())?;
        Some((found[label].to_owned(), found.get(label + 1)?.start()))
    }

    fn instrument_top(value: &str, direct: bool) -> Option<(String, usize, bool)> {
        if let Some(found) = cached_regex!(TOP,
            r"^(?:(ARTICLE|Article|PART|Part|DIVISION|Division)\s+([IVXLCDM]+|\d{1,3})\b\s*[—–\-.:]?\s*(.*)|(SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)\s+([A-Z0-9][\w.\-]*)\s*[—–\-.:]?\s*(.*))$"
        ).captures(value) {
            let container = found.get(1).is_some();
            let (word, token, rest) = if container { (1, 2, 3) } else { (4, 5, 6) };
            let heading = &found[rest];
            if !container
                && !(heading.is_empty()
                    || heading.starts_with(['"', '\'', '“', '('])
                    || heading
                        .chars()
                        .next()
                        .is_some_and(|character| character.is_ascii_uppercase()))
            {
                return None;
            }
            let word = found[word].to_ascii_lowercase();
            let prefix = match word.as_str() {
                "part" | "annex" => word.as_str(),
                "schedule" => &word[..5],
                _ => &word[..3],
            };
            let suffix = if container {
                found[token].parse().ok()
                    .or_else(|| roman_value(&found[token]))?
                    .to_string()
            } else {
                found[token].to_ascii_lowercase()
            };
            return Some((format!("{prefix}{suffix}"), found.get(rest)?.start(), true));
        }
        direct
            .then_some(value)
            .and_then(direct_section)
            .map(|(label, at)| (format!("sec{label}"), at, false))
    }

    fn detect_instrument(text: &ScalarText<'_>) -> Vec<Block> {
        let mut spine = statute_spine(text, false).into_iter().peekable();
        let direct = spine.peek().is_none();
        let dialects = admitted_dialects(text);
        let mut state = StructureState::default();
        for line in lines(text) {
            let trimmed_start = line.text.trim_start_matches(instrument_space);
            let value = trimmed_start.trim_end_matches(instrument_space);
            if value.is_empty() {
                continue;
            }
            let start = line.scalar_start
                + line.text[..line.text.len() - trimmed_start.len()]
                    .chars()
                    .count();
            let selected = spine
                .next_if(|mark| mark.start == start)
                .map(|mark| (format!("sec{}", mark.label), mark.content_start, false))
                .or_else(|| {
                    instrument_top(value, direct).map(|(label, at, container)| {
                        (label, start + value[..at].chars().count(), container)
                    })
                });
            if let Some((label, content_start, container)) = selected {
                let depth = usize::from(!container && state.container.is_some());
                let mut block =
                    Block::labelled(NodeKind::Section, label.clone(), start, usize::MAX);
                block.parent_label = (!container).then(|| state.container.clone()).flatten();
                block.content_start = Some(content_start);
                state.nodes.push((block, depth));
                state.stack.clear();
                if container {
                    state.container = Some(label);
                    state.section = None;
                } else {
                    state.section = Some((label, depth));
                    let inline = &text.value[text.byte(content_start)..line.byte_end];
                    if let Some((token, at)) = instrument_marker(inline, false, false) {
                        state.child(
                            token,
                            content_start,
                            content_start + inline[..at].chars().count(),
                        );
                    }
                }
                continue;
            }
            if let (Some((token, at)), Some(_)) = (
                instrument_marker(value, dialects.0, dialects.1),
                state.section.as_ref(),
            ) {
                state.child(token, start, start + value[..at].chars().count());
            }
        }
        for index in 0..state.nodes.len() {
            let depth = state.nodes[index].1;
            let end = state
                .nodes
                .iter()
                .skip(index + 1)
                .find(|(_, candidate)| *candidate <= depth)
                .map_or(text.len(), |(block, _)| block.range.start);
            state.nodes[index].0.range.end = end;
        }
        state.nodes.into_iter().map(|(block, _)| block).collect()
    }

    fn detect_legislation(
        text: &ScalarText<'_>,
        allow_hyphenated_sections: bool,
        native_claims: &[NativeClaim],
    ) -> Vec<Block> {
        let sections = selected_sections(text, allow_hyphenated_sections);
        let mut result = Vec::new();
        for (index, section) in sections.iter().enumerate() {
            let end = sections
                .get(index + 1)
                .map_or(text.len(), |value| value.start);
            let label = format!("sec{}", section.label);
            let mut top = Block::labelled(NodeKind::Section, label, section.start, end);
            top.aliases = section
                .aliases
                .iter()
                .map(|value| format!("sec{value}"))
                .collect();
            top.content_start = Some(section.content_start);
            result.push(top);
            let value = text.slice(ScalarRange {
                start: section.start,
                end,
            });
            result.extend(enumerated_children(
                value,
                section.start,
                format!("sec{}", section.label),
                0,
                section.content_start - section.start,
                matches!(section.family, SectionFamily::Bare | SectionFamily::DotTerm),
                None,
            ));
        }
        for claim in native_claims
            .iter()
            .filter(|claim| claim.kind == EvidenceKind::Section && claim.parent_label.is_none())
        {
            let Some(label) = claim
                .label
                .as_deref()
                .and_then(|value| value.strip_prefix("sec"))
            else {
                continue;
            };
            if !provision_label(label)
                .is_some_and(|(value, end)| value == label && end == label.len())
            {
                continue;
            }
            let value = text.slice(claim.range);
            let lead = leading_ascii_space(value);
            let content_start = value[lead..]
                .strip_prefix(label)
                .map_or(claim.range.start, |_| {
                    claim.range.start + value[..lead].chars().count() + label.chars().count()
                });
            let children = enumerated_children(
                value,
                claim.range.start,
                format!("sec{label}"),
                0,
                content_start - claim.range.start,
                false,
                Some(label),
            );
            result.retain(|block| {
                !block
                    .parent_label
                    .as_deref()
                    .is_some_and(|parent| parent.eq_ignore_ascii_case(&format!("sec{label}")))
                    || block.range.start < claim.range.start
                    || block.range.end > claim.range.end
            });
            result.extend(children);
        }
        result
    }

    fn add_ranges(mut blocks: Vec<Block>, length: usize) -> Vec<Block> {
        for index in 0..blocks.len() {
            blocks[index].range.end = blocks
                .get(index + 1)
                .map_or(length, |value| value.range.start);
        }
        blocks
    }

    fn detect_journal(text: &ScalarText<'_>) -> Vec<Block> {
        let source = javascript_lines(text);
        let mut result = add_ranges(
            source
                .iter()
                .filter_map(|line| {
                    cached_regex!(
                        SECTION,
                        r"(?u)^[ \t]*([IVXLCDM]+|[A-Z])\.[ \t]+([^\x08]{3,180})$"
                    )
                    .captures(line.text)
                    .map(|capture| (line, capture))
                })
                .map(|(line, capture)| {
                    let whole = capture.get(0).unwrap();
                    let title = capture[2].split_whitespace().collect::<Vec<_>>().join(" ");
                    let alias = title
                        .to_lowercase()
                        .chars()
                        .map(|value| if value.is_alphanumeric() { value } else { ' ' })
                        .collect::<String>()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    let mut block = Block::labelled(
                        NodeKind::Section,
                        format!("sec{}", &capture[1]),
                        line.scalar_start + line.text[..whole.start()].chars().count(),
                        0,
                    );
                    block.aliases = std::iter::once(capture[1].to_owned())
                        .chain((!alias.is_empty()).then(|| format!("sectitle:{alias}")))
                        .collect();
                    block
                })
                .collect(),
            text.len(),
        );
        result.extend(add_ranges(
            source
                .iter()
                .filter_map(|line| {
                    cached_regex!(NOTE, r"(?u)^[ \t]*(\d{1,5})\t[ \t]*$")
                        .captures(line.text)
                        .map(|capture| (line, capture))
                })
                .map(|(line, capture)| {
                    let whole = capture.get(0).unwrap();
                    let mut block = Block::labelled(
                        NodeKind::Footnote,
                        format!("fn{}", capture[1].parse::<u32>().unwrap()),
                        line.scalar_start + line.text[..whole.start()].chars().count(),
                        0,
                    );
                    block.aliases.push(capture[1].to_owned());
                    block
                })
                .collect(),
            text.len(),
        ));
        let mut start = None;
        for (index, line) in source.iter().enumerate() {
            let blank = line.text.trim_matches([' ', '\t', '\r']).is_empty();
            if start.is_none() && !blank {
                start = line
                    .text
                    .char_indices()
                    .find(|(_, value)| !javascript_whitespace(*value))
                    .map(|(at, _)| line.scalar_start + line.text[..at].chars().count());
            }
            let next_blank = source
                .get(index + 1)
                .is_some_and(|value| value.text.trim_matches([' ', '\t', '\r']).is_empty());
            if let Some(block_start) = start.filter(|_| next_blank || index + 1 == source.len()) {
                let end = if index + 1 == source.len() {
                    text.len()
                } else {
                    line.scalar_start + line.text.chars().count()
                };
                let value = text.slice(ScalarRange {
                    start: block_start,
                    end,
                });
                let block_start =
                    if cached_regex!(PAGE, r"(?iu)^\[page [^\]\n]{1,40}\]").is_match(value) {
                        value
                            .find('\n')
                            .map_or(end, |at| block_start + value[..=at].chars().count())
                    } else {
                        block_start
                    };
                if block_start < end {
                    result.push(Block {
                        kind: NodeKind::Prose,
                        range: ScalarRange {
                            start: block_start,
                            end,
                        },
                        label: None,
                        aliases: Vec::new(),
                        parent_label: None,
                        content_start: None,
                        diagnostic: None,
                    });
                }
                start = None;
            }
        }
        result
    }

    pub(super) fn inferred_blocks(evidence: &DocumentInput, text: &ScalarText<'_>) -> Vec<Block> {
        let mut paragraph_exclusions = evidence
            .exclusions
            .iter()
            .filter(|value| value.applies_to.iter().any(|name| name == "paragraph"))
            .map(|value| value.range)
            .collect::<Vec<_>>();
        paragraph_exclusions.sort_by_key(|range| (range.start, range.end));
        match evidence.profile {
            DetectionProfile::Legislation => detect_legislation(
                text,
                evidence.allow_hyphenated_sections,
                &evidence.native_claims,
            ),
            DetectionProfile::Instrument => detect_instrument(text),
            DetectionProfile::Journal => detect_journal(text),
            _ => detect_case(
                text,
                evidence.profile,
                evidence.report_start_page,
                evidence.require_report_start,
                &paragraph_exclusions,
            ),
        }
    }

    #[cfg(all(feature = "a2aj", feature = "source-doc"))]
    pub(super) fn a2aj_source_doc_blocks(
        text: &str,
        profile: DetectionProfile,
        report_start_page: Option<u32>,
        require_report_start: bool,
        allow_hyphenated_sections: bool,
    ) -> Vec<SourceDocBlock> {
        let scalar = ScalarText::new(text);
        let inferred = if profile == DetectionProfile::Legislation {
            detect_legislation(&scalar, allow_hyphenated_sections, &[])
        } else {
            detect_case(
                &scalar,
                profile,
                report_start_page,
                require_report_start,
                &[],
            )
        };
        let mut prose = 0;
        let mut blocks = inferred
            .into_iter()
            .filter_map(|block| {
                let kind = match block.kind {
                    NodeKind::Paragraph | NodeKind::Prose => SourceDocKind::Paragraph,
                    NodeKind::Page => SourceDocKind::Page,
                    NodeKind::Section => SourceDocKind::Section,
                    NodeKind::Footnote => SourceDocKind::Footnote,
                    NodeKind::Heading | NodeKind::Endnote => return None,
                };
                let label = if block.kind == NodeKind::Prose {
                    prose += 1;
                    format!("par{prose}")
                } else {
                    block.label?
                };
                let mut projected = SourceDocBlock::new(
                    kind,
                    label,
                    scalar.utf16(block.range.start),
                    scalar.utf16(block.range.end),
                    SourceDocOrigin::Heuristic,
                );
                projected.aliases = block.aliases;
                projected.parent_label = block.parent_label;
                Some(projected)
            })
            .collect::<Vec<_>>();
        if profile == DetectionProfile::Legislation {
            blocks.sort_by(|left, right| {
                left.start
                    .cmp(&right.start)
                    .then_with(|| right.end.cmp(&left.end))
                    .then_with(|| left.label.cmp(&right.label))
            });
        }
        blocks
    }
}

fn native_kind(kind: EvidenceKind) -> NodeKind {
    match kind {
        EvidenceKind::Paragraph => NodeKind::Paragraph,
        EvidenceKind::Prose => NodeKind::Prose,
        EvidenceKind::Page => NodeKind::Page,
        EvidenceKind::Section => NodeKind::Section,
        EvidenceKind::Heading => NodeKind::Heading,
        EvidenceKind::Footnote => NodeKind::Footnote,
        EvidenceKind::Endnote => NodeKind::Endnote,
    }
}

fn infer_graph(
    evidence: DocumentInput,
    inferred: Vec<Block>,
    recovery_available: bool,
) -> StructureGraphV1 {
    let complete = evidence.scope.kind == ScopeKind::Complete
        && (recovery_available || !evidence.needs_recovery());
    let mut nodes = evidence
        .native_claims
        .iter()
        .map(|claim| StructureNodeV1 {
            id: claim.id.clone(),
            kind: native_kind(claim.kind),
            range: claim.range,
            origin_id: claim.origin_id.clone(),
            source: Derivation::Native,
            label: claim.label.clone(),
            aliases: (!claim.aliases.is_empty()).then(|| claim.aliases.clone()),
            parent_id: None,
            anchor: claim.anchor.clone(),
            content_start: None,
        })
        .collect::<Vec<_>>();
    let native = nodes
        .iter()
        .map(|node| {
            (
                node.kind,
                node.label.as_deref(),
                node.aliases.as_deref().unwrap_or(&[]),
            )
        })
        .collect::<Vec<_>>();
    let mut counters = HashMap::<NodeKind, usize>::new();
    let generated = inferred
        .into_iter()
        .filter_map(|mut block| {
            block.range = evidence.clip_inference(block.kind.evidence(), block.range)?;
            if block.content_start.is_some_and(|at| {
                block.kind != NodeKind::Section || at < block.range.start || at > block.range.end
            }) {
                return None;
            }
            (!native.iter().any(|(kind, label, aliases)| {
                *kind == block.kind
                    && block.label.as_deref().is_some_and(|candidate| {
                        label.is_some_and(|value| value.eq_ignore_ascii_case(candidate))
                            || aliases
                                .iter()
                                .any(|value| value.eq_ignore_ascii_case(candidate))
                    })
            }))
            .then_some(block)
        })
        .map(|block| {
            let ordinal = counters.entry(block.kind).or_default();
            *ordinal += 1;
            let id = format!("heuristic-{}-{:06}", block.kind.name(), ordinal);
            (block, id)
        })
        .collect::<Vec<_>>();
    let mut labels = nodes
        .iter()
        .flat_map(|node| {
            node.label
                .iter()
                .chain(node.aliases.iter().flatten())
                .map(|label| (label.to_ascii_lowercase(), node.id.clone()))
        })
        .collect::<HashMap<_, _>>();
    for (block, id) in &generated {
        if let Some(label) = &block.label {
            labels.insert(label.to_ascii_lowercase(), id.clone());
        }
        for alias in &block.aliases {
            labels.insert(alias.to_ascii_lowercase(), id.clone());
        }
    }
    for (claim, node) in evidence.native_claims.iter().zip(nodes.iter_mut()) {
        node.parent_id = claim
            .parent_label
            .as_ref()
            .and_then(|label| labels.get(&label.to_ascii_lowercase()))
            .cloned();
    }
    let mut relations = Vec::new();
    let diagnostics = generated
        .iter()
        .filter_map(|(block, id)| {
            block.diagnostic.map(|code| StructureDiagnosticV1 {
                code: code.to_owned(),
                severity: if code.ends_with("violation") {
                    DiagnosticSeverity::Warning
                } else {
                    DiagnosticSeverity::Info
                },
                ranges: vec![block.range],
                node_ids: vec![id.clone()],
            })
        })
        .collect();
    for (block, id) in generated {
        let parent_id = block
            .parent_label
            .as_ref()
            .and_then(|label| labels.get(&label.to_ascii_lowercase()))
            .cloned();
        if let Some(parent) = &parent_id {
            relations.push(StructureRelationV1 {
                id: format!("heuristic-contains-{:06}", relations.len() + 1),
                kind: RelationKind::Contains,
                from: RelationEndpointV1::Node {
                    node_id: parent.clone(),
                },
                to: RelationEndpointV1::Node {
                    node_id: id.clone(),
                },
                origin_id: ENGINE_ORIGIN.to_owned(),
                source: Derivation::Heuristic,
            });
        }
        nodes.push(StructureNodeV1 {
            id,
            kind: block.kind,
            range: block.range,
            origin_id: ENGINE_ORIGIN.to_owned(),
            source: Derivation::Heuristic,
            label: block.label,
            aliases: (!block.aliases.is_empty()).then_some(block.aliases),
            parent_id,
            anchor: None,
            content_start: block.content_start,
        });
    }
    let mut boundaries = evidence
        .paragraph_breaks
        .iter()
        .map(|value| StructureBoundaryV1 {
            kind: BoundaryKind::Prose,
            at: value.at,
            origin_id: value.origin_id.clone(),
            source: Derivation::Native,
        })
        .collect::<Vec<_>>();
    boundaries.extend(
        nodes
            .iter()
            .filter(|node| {
                node.kind == NodeKind::Prose && matches!(node.source, Derivation::Heuristic)
            })
            .map(|node| StructureBoundaryV1 {
                kind: BoundaryKind::Prose,
                at: node.range.end,
                origin_id: ENGINE_ORIGIN.to_owned(),
                source: Derivation::Heuristic,
            }),
    );
    StructureGraphV1 {
        schema_version: RESULT_SCHEMA,
        document_id: evidence.document_id,
        text_sha256: evidence.text_sha256,
        source_sha256: evidence.source_sha256,
        status: if complete {
            GraphStatus::Complete
        } else {
            GraphStatus::Partial
        },
        nodes,
        boundaries,
        relations,
        diagnostics,
    }
}

#[cfg(feature = "recovery")]
pub fn derive_structure_evidence(evidence: DocumentInput) -> Result<StructureGraphV1, EngineError> {
    let inferred = if evidence.needs_recovery() {
        let text = ScalarText::new(&evidence.text);
        recovery::inferred_blocks(&evidence, &text)
    } else {
        Vec::new()
    };
    Ok(infer_graph(evidence, inferred, true))
}

pub fn derive_native_structure_evidence(
    evidence: DocumentInput,
) -> Result<StructureGraphV1, EngineError> {
    Ok(infer_graph(evidence, Vec::new(), false))
}

#[cfg(feature = "source-doc")]
fn projection(profile: DetectionProfile) -> (ProjectionOrder, Option<SourceDocType>) {
    match profile {
        DetectionProfile::CaseRootedComplete => (ProjectionOrder::Case, Some(SourceDocType::Cases)),
        DetectionProfile::CaseContiguousComplete | DetectionProfile::CaseLossy => {
            (ProjectionOrder::Position, Some(SourceDocType::Cases))
        }
        DetectionProfile::Legislation => (ProjectionOrder::Legislation, Some(SourceDocType::Laws)),
        DetectionProfile::Instrument => (ProjectionOrder::Legislation, None),
        DetectionProfile::Journal => (ProjectionOrder::StablePosition, None),
    }
}

#[cfg(feature = "source-doc")]
fn compose_with(
    mut input: DocumentInput,
    recovery: bool,
    validate: bool,
) -> Result<SourceDoc, EngineError> {
    if validate {
        input.validate()?;
    }
    let (order, inferred_type) = projection(input.profile);
    let provider = SourceDocProvider::from_name(&input.provider);
    let id = input.document_id.clone();
    let url = input.url.take();
    let doc_type = input.doc_type.or(inferred_type);
    let revision = input.text_sha256.clone();
    let originals = source_doc::native_blocks(
        &input.text,
        &input.native_claims,
        std::mem::take(&mut input.original_claims),
    );
    #[cfg(feature = "recovery")]
    let inferred = if recovery && input.needs_recovery() {
        let text = ScalarText::new(&input.text);
        recovery::inferred_blocks(&input, &text)
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "recovery"))]
    let inferred = Vec::new();
    let text = std::mem::take(&mut input.text);
    let graph = infer_graph(input, inferred, recovery);
    Ok(source_doc::project_graph(
        provider, id, url, doc_type, text, revision, &originals, graph, order,
    ))
}

#[cfg(feature = "source-doc")]
pub fn compose_native(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, false, true)
}

#[cfg(all(feature = "recovery", feature = "source-doc"))]
pub fn compose(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, true, true)
}

#[cfg(all(feature = "recovery", feature = "source-doc"))]
pub(crate) fn compose_trusted(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, true, false)
}

#[cfg(all(feature = "journal", feature = "recovery", feature = "source-doc"))]
pub(crate) fn compose_journal_source_doc(
    article_id: usize,
    url: Option<String>,
    text: String,
    blocks: Vec<SourceDocBlock>,
) -> Result<SourceDoc, EngineError> {
    let offsets = source_doc::utf16_offsets(&text);
    let scalar = |offset: usize| {
        offsets
            .binary_search(&offset)
            .map_err(|_| EngineError::source("journal range splits a Unicode scalar"))
    };
    let mut originals = HashMap::new();
    let mut claims = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let id = format!("native-{:06}", index + 1);
        claims.push(NativeClaim {
            id: id.clone(),
            kind: match block.kind {
                SourceDocKind::Paragraph => EvidenceKind::Paragraph,
                SourceDocKind::Page => EvidenceKind::Page,
                SourceDocKind::Section => EvidenceKind::Section,
                SourceDocKind::Footnote => EvidenceKind::Footnote,
            },
            label: Some(block.label.clone()),
            aliases: block.aliases.clone(),
            parent_label: block.parent_label.clone(),
            anchor: block.anchor.clone(),
            range: ScalarRange {
                start: scalar(block.start)?,
                end: scalar(block.end)?,
            },
            provider_order: index,
            origin_id: "journal".to_owned(),
        });
        originals.insert(id, block.clone());
    }
    let has = |kind| {
        blocks.iter().any(|block| match kind {
            EvidenceKind::Paragraph | EvidenceKind::Prose => block.kind == SourceDocKind::Paragraph,
            EvidenceKind::Page => block.kind == SourceDocKind::Page,
            EvidenceKind::Section => block.kind == SourceDocKind::Section,
            EvidenceKind::Footnote => block.kind == SourceDocKind::Footnote,
            EvidenceKind::Heading | EvidenceKind::Endnote => false,
        })
    };
    let length = text.chars().count();
    let text_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    compose_trusted(DocumentInput {
        schema_version: EVIDENCE_SCHEMA.to_owned(),
        document_id: article_id.to_string(),
        provider: "journal".to_owned(),
        url,
        doc_type: None,
        provider_revision: "journal-adapter-v1".to_owned(),
        profile: DetectionProfile::Journal,
        report_start_page: None,
        require_report_start: false,
        allow_hyphenated_sections: false,
        text,
        text_sha256,
        source_sha256: None,
        offset_unit: "unicode-scalar".to_owned(),
        scope: Scope {
            kind: ScopeKind::Complete,
            excerpt_of: None,
        },
        origins: vec![Origin {
            id: "journal".to_owned(),
            producer: "journal".to_owned(),
            representation: "provider-rendered-text".to_owned(),
            revision: "journal-adapter-v1".to_owned(),
            authority: "provider-native-claims".to_owned(),
        }],
        units: Vec::new(),
        native_claims: claims,
        coverage: [
            EvidenceKind::Paragraph,
            EvidenceKind::Prose,
            EvidenceKind::Page,
            EvidenceKind::Section,
            EvidenceKind::Heading,
            EvidenceKind::Footnote,
            EvidenceKind::Endnote,
        ]
        .into_iter()
        .map(|kind| Coverage {
            kind,
            range: ScalarRange {
                start: 0,
                end: length,
            },
            state: if has(kind) {
                CoverageState::Complete
            } else {
                CoverageState::Absent
            },
            reason: "journal native coverage".to_owned(),
            origin_id: has(kind).then(|| "journal".to_owned()),
        })
        .collect(),
        exclusions: Vec::new(),
        paragraph_breaks: Vec::new(),
        original_claims: originals,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeriveBatch {
    #[serde(rename = "type")]
    kind: String,
    request_id: String,
    documents: Vec<Value>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
struct ItemError<'a> {
    id: &'a str,
    ok: bool,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ItemResult<'a> {
    id: &'a str,
    ok: bool,
    result: StructureGraphV1,
}

fn io<T>(value: std::io::Result<T>) -> Result<T, EngineError> {
    value.map_err(|error| EngineError {
        code: "sidecar_io",
        message: error.to_string(),
    })
}

fn json_error(error: serde_json::Error) -> EngineError {
    EngineError {
        code: "sidecar_json",
        message: error.to_string(),
    }
}

fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> Result<usize, EngineError> {
    loop {
        let buffer = io(reader.fill_buf())?;
        if buffer.is_empty() {
            return Ok(line.len());
        }
        let used = buffer
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(buffer.len(), |index| index + 1);
        if line.len() + used > MAX_BYTES + 1 {
            return Err(EngineError::invalid("oversized sidecar line"));
        }
        line.extend_from_slice(&buffer[..used]);
        reader.consume(used);
        if line.last() == Some(&b'\n') {
            return Ok(line.len());
        }
    }
}

fn sidecar_with(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    derive: fn(DocumentInput) -> Result<StructureGraphV1, EngineError>,
    capabilities: &[&str],
) -> Result<(), EngineError> {
    let executable = std::env::current_exe().map_err(|error| EngineError {
        code: "sidecar_identity",
        message: error.to_string(),
    })?;
    let engine_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(executable).map_err(|error| EngineError {
            code: "sidecar_identity",
            message: error.to_string()
        })?)
    );
    serde_json::to_writer(&mut *writer, &json!({ "type": "hello", "protocol": SIDECAR_PROTOCOL, "evidence_schema": EVIDENCE_SCHEMA,
        "result_schema": RESULT_SCHEMA, "engine_sha256": engine_sha256, "capabilities": capabilities,
        "max_documents": MAX_DOCUMENTS, "max_bytes": MAX_BYTES })).map_err(json_error)?;
    io(writer.write_all(b"\n"))?;
    io(writer.flush())?;
    loop {
        let mut line = Vec::new();
        if read_line(reader, &mut line)? == 0 {
            return Err(EngineError::invalid("sidecar received unexpected EOF"));
        }
        if line.last() != Some(&b'\n')
            || line.len() - 1 > MAX_BYTES
            || line[..line.len() - 1].contains(&b'\r')
        {
            return Err(EngineError::invalid("invalid sidecar line"));
        }
        line.pop();
        let batch: DeriveBatch = serde_json::from_slice(&line).map_err(json_error)?;
        if batch.kind != "derive_batch"
            || batch.request_id.is_empty()
            || batch.documents.is_empty()
            || batch.documents.len() > MAX_DOCUMENTS
        {
            return Err(EngineError::invalid("invalid derive_batch envelope"));
        }
        let ids = batch
            .documents
            .iter()
            .map(|value| {
                value
                    .get("document_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| EngineError::invalid("derive_batch document ID is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids.iter().collect::<HashSet<_>>().len() != ids.len() {
            return Err(EngineError::invalid(
                "derive_batch document IDs are duplicated",
            ));
        }
        io(writer.write_all(b"{\"type\":\"result_batch\",\"request_id\":"))?;
        serde_json::to_writer(&mut *writer, &batch.request_id).map_err(json_error)?;
        io(writer.write_all(b",\"items\":["))?;
        for (index, (id, value)) in ids.iter().zip(batch.documents).enumerate() {
            if index > 0 {
                io(writer.write_all(b","))?;
            }
            match DocumentInput::try_from(value).and_then(derive) {
                Ok(result) => serde_json::to_writer(
                    &mut *writer,
                    &ItemResult {
                        id,
                        ok: true,
                        result,
                    },
                )
                .map_err(json_error)?,
                Err(error) => serde_json::to_writer(
                    &mut *writer,
                    &ItemError {
                        id,
                        ok: false,
                        error: ErrorBody {
                            code: error.code,
                            message: &error.message,
                        },
                    },
                )
                .map_err(json_error)?,
            }
        }
        io(writer.write_all(b"]}\n"))?;
        io(writer.flush())?;
    }
}

#[cfg(feature = "recovery")]
pub fn sidecar(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<(), EngineError> {
    sidecar_with(
        reader,
        writer,
        derive_structure_evidence,
        &["native_claims", "raw_recovery"],
    )
}

pub fn native_sidecar(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<(), EngineError> {
    sidecar_with(
        reader,
        writer,
        derive_native_structure_evidence,
        &["native_claims"],
    )
}

#[cfg(feature = "recovery")]
pub fn stdio_sidecar() -> Result<(), EngineError> {
    sidecar(&mut std::io::stdin().lock(), &mut std::io::stdout().lock())
}

pub fn native_stdio_sidecar() -> Result<(), EngineError> {
    native_sidecar(&mut std::io::stdin().lock(), &mut std::io::stdout().lock())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "recovery")]
    use super::recovery::{formal_heading, statute_spine};
    use super::*;

    fn evidence(text: &str, profile: DetectionProfile) -> DocumentInput {
        let range = ScalarRange {
            start: 0,
            end: text.chars().count(),
        };
        DocumentInput {
            schema_version: EVIDENCE_SCHEMA.into(),
            document_id: "test".into(),
            provider: "test".into(),
            #[cfg(feature = "source-doc")]
            url: None,
            #[cfg(feature = "source-doc")]
            doc_type: None,
            provider_revision: "1".into(),
            profile,
            report_start_page: None,
            require_report_start: false,
            allow_hyphenated_sections: false,
            text: text.into(),
            text_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            source_sha256: None,
            offset_unit: "unicode-scalar".into(),
            scope: Scope {
                kind: ScopeKind::Complete,
                excerpt_of: None,
            },
            origins: vec![Origin {
                id: "native".into(),
                producer: "test".into(),
                representation: "text".into(),
                revision: "1".into(),
                authority: "test".into(),
            }],
            units: Vec::new(),
            native_claims: Vec::new(),
            coverage: [
                EvidenceKind::Paragraph,
                EvidenceKind::Prose,
                EvidenceKind::Page,
                EvidenceKind::Section,
                EvidenceKind::Heading,
                EvidenceKind::Footnote,
                EvidenceKind::Endnote,
            ]
            .into_iter()
            .map(|kind| Coverage {
                kind,
                range,
                state: CoverageState::Absent,
                reason: "test".into(),
                origin_id: None,
            })
            .collect(),
            exclusions: Vec::new(),
            paragraph_breaks: Vec::new(),
            #[cfg(feature = "source-doc")]
            original_claims: HashMap::new(),
        }
    }

    #[test]
    fn validates_scope_and_ranges() {
        let mut value = evidence("abc", DetectionProfile::CaseRootedComplete);
        value.scope = Scope {
            kind: ScopeKind::Excerpt,
            excerpt_of: Some("whole".into()),
        };
        assert!(value.validate().is_err());
        value.profile = DetectionProfile::CaseLossy;
        assert!(value.validate().is_ok());
        value.native_claims.push(NativeClaim {
            id: "bad".into(),
            kind: EvidenceKind::Page,
            label: Some("page1".into()),
            aliases: Vec::new(),
            range: ScalarRange { start: 0, end: 4 },
            provider_order: 0,
            origin_id: "native".into(),
            parent_label: None,
            anchor: None,
        });
        assert!(value.validate().is_err());
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn weighted_numeric_sequence_policies_preserve_lane_rules() {
        let candidate =
            |index, value, position, page, score, start_supported| NumericSequenceCandidate {
                index,
                value,
                position: (position, 0),
                page,
                score,
                start_supported,
            };
        let rooted = select_numeric_sequence(
            vec![
                candidate(0, 1, 0, 0, 1.0, false),
                candidate(1, 2, 0, 0, 100.0, false),
                candidate(2, 2, 1, 0, 1.0, false),
                candidate(3, 3, 2, 0, 1.0, false),
            ],
            NumericSequencePolicy::RootedConsecutive,
        );
        assert_eq!(rooted.indices, [0, 2, 3]);
        assert!((rooted.score - 3.6).abs() < 1e-9);

        let footnotes = select_numeric_sequence(
            vec![
                candidate(10, 1, 0, 1, 1.0, false),
                candidate(11, 3, 1, 1, 1.0, false),
                candidate(12, 4, 2, 2, 1.0, false),
            ],
            NumericSequencePolicy::FootnoteBackbone,
        );
        assert_eq!(footnotes.indices, [10, 11, 12]);
        assert!((footnotes.score - 2.9).abs() < 1e-9);
        assert_eq!(
            select_numeric_sequence(
                vec![candidate(20, 50, 0, 1, 5.0, true)],
                NumericSequencePolicy::FootnoteBackbone
            ),
            NumericSequenceSelection {
                indices: vec![20],
                score: 5.0
            }
        );
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn joined_page_tokens_are_not_reporter_pages() {
        let graph = derive_structure_evidence(evidence(
            "Quoted text [page624] continues.\nMore [page625] text.\nLast [page626] text.",
            DetectionProfile::CaseRootedComplete,
        ))
        .unwrap();
        assert!(graph.nodes.iter().all(|node| node.kind != NodeKind::Page));
    }

    #[test]
    fn clips_at_complete_native_coverage() {
        let mut value = evidence("0123456789", DetectionProfile::CaseLossy);
        value
            .coverage
            .retain(|row| row.kind != EvidenceKind::Section);
        value.coverage.extend(
            [
                (0, 3, CoverageState::Absent),
                (3, 7, CoverageState::Complete),
                (7, 10, CoverageState::Augment),
            ]
            .map(|(start, end, state)| Coverage {
                kind: EvidenceKind::Section,
                range: ScalarRange { start, end },
                state,
                reason: "test".into(),
                origin_id: None,
            }),
        );
        assert_eq!(
            value
                .clip_inference(EvidenceKind::Section, ScalarRange { start: 1, end: 9 })
                .unwrap()
                .end,
            3
        );
        assert!(value
            .clip_inference(EvidenceKind::Section, ScalarRange { start: 4, end: 9 })
            .is_none());
        assert_eq!(
            value
                .clip_inference(EvidenceKind::Section, ScalarRange { start: 7, end: 10 })
                .unwrap()
                .end,
            10
        );
    }

    #[test]
    fn native_projection_preserves_claims_without_recovery() {
        let text = "1 Native provision";
        let mut value = evidence(text, DetectionProfile::Legislation);
        value
            .coverage
            .iter_mut()
            .for_each(|row| row.state = CoverageState::Complete);
        value.native_claims.push(NativeClaim {
            id: "native-section".into(),
            kind: EvidenceKind::Section,
            label: Some("sec1".into()),
            aliases: vec!["s1".into()],
            range: ScalarRange {
                start: 0,
                end: text.chars().count(),
            },
            provider_order: 1,
            origin_id: "native".into(),
            parent_label: None,
            anchor: Some("section-1".into()),
        });
        let graph = derive_native_structure_evidence(value).expect("valid native evidence");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].label.as_deref(), Some("sec1"));
        assert!(matches!(graph.nodes[0].source, Derivation::Native));
        assert!(matches!(graph.status, GraphStatus::Complete));

        let incomplete =
            derive_native_structure_evidence(evidence(text, DetectionProfile::Legislation))
                .expect("valid incomplete native evidence");
        assert!(matches!(incomplete.status, GraphStatus::Partial));
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn native_parent_wins_and_children_survive() {
        let derive = |value| derive_structure_evidence(value).expect("valid fixture evidence");
        let flat = derive(evidence(
            "1 First provision\n2 Second provision",
            DetectionProfile::Legislation,
        ));
        assert!(flat
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Section)
            .all(|node| node
                .content_start
                .is_some_and(|at| node.range.start <= at && at <= node.range.end)));
        let text = "1 (1) Parent words\n(2) Child words";
        let mut value = evidence(text, DetectionProfile::Legislation);
        value
            .coverage
            .iter_mut()
            .find(|row| row.kind == EvidenceKind::Section)
            .unwrap()
            .state = CoverageState::Augment;
        value.native_claims.push(NativeClaim {
            id: "native-section".into(),
            kind: EvidenceKind::Section,
            label: Some("sec1".into()),
            aliases: Vec::new(),
            range: ScalarRange {
                start: 0,
                end: text.chars().count(),
            },
            provider_order: 0,
            origin_id: "native".into(),
            parent_label: None,
            anchor: None,
        });
        let graph = derive(value);
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.label.as_deref() == Some("sec1"))
                .count(),
            1
        );
        assert!(["sec1(1)", "sec1(2)"].into_iter().all(|label| graph
            .nodes
            .iter()
            .any(|node| node.label.as_deref() == Some(label))));
        let text = "1 (1) Parent provision.\n(a) First paragraph.\n(i) First subparagraph.\n(ii) Second subparagraph.\n(a) Duplicate paragraph marker.\n(b) Second paragraph.\n(2) Sibling subsection.\n2 Next provision.";
        let law = derive(evidence(text, DetectionProfile::Legislation));
        let node = |label| {
            law.nodes
                .iter()
                .find(|node| node.label.as_deref() == Some(label))
                .unwrap()
        };
        assert_eq!(node("sec1(1)").range.start, 0);
        assert_eq!(
            node("sec1(1)").range.end,
            text[..text.find("(a) First").unwrap()].chars().count()
        );
        assert_eq!(
            node("sec1(1)(a)").parent_id.as_deref(),
            Some(node("sec1").id.as_str())
        );
        assert_eq!(
            law.nodes
                .iter()
                .filter(|node| node.label.as_deref() == Some("sec1(1)(a)"))
                .count(),
            1
        );

        let criminal = "**231** (4) Parent subsection.\n(a) First paragraph.\n(b) Second paragraph.\n(c) Third paragraph.\n(5) Sibling subsection.";
        let criminal = derive(evidence(criminal, DetectionProfile::Legislation));
        let section = criminal
            .nodes
            .iter()
            .find(|node| node.label.as_deref() == Some("sec231"))
            .unwrap();
        let subsection = criminal
            .nodes
            .iter()
            .find(|node| node.label.as_deref() == Some("sec231(4)"))
            .unwrap();
        assert_eq!(
            subsection.range.end,
            "**231** (4) Parent subsection.\n".chars().count()
        );
        assert!(["a", "b", "c"].into_iter().all(|label| {
            let label = format!("sec231(4)({label})");
            criminal.nodes.iter().any(|node| {
                node.label.as_deref() == Some(label.as_str())
                    && node.parent_id.as_deref() == Some(section.id.as_str())
            })
        }));

        let section_map = "**22.1** (a) Parent paragraph.\n(i) First subparagraph.\n(ii) Second subparagraph.\n(b) Sibling paragraph.";
        let section_map = derive(evidence(section_map, DetectionProfile::Legislation));
        let section = section_map
            .nodes
            .iter()
            .find(|node| node.label.as_deref() == Some("sec22.1"))
            .unwrap();
        let paragraph = section_map
            .nodes
            .iter()
            .find(|node| node.label.as_deref() == Some("sec22.1(a)"))
            .unwrap();
        assert_eq!(
            paragraph.range.end,
            "**22.1** (a) Parent paragraph.\n".chars().count()
        );
        assert!(["i", "ii"].into_iter().all(|label| {
            let label = format!("sec22.1(a)({label})");
            section_map.nodes.iter().any(|node| {
                node.label.as_deref() == Some(label.as_str())
                    && node.parent_id.as_deref() == Some(section.id.as_str())
            })
        }));
        let instrument = derive(evidence(
            "Section 1.01 Nested.\n(a) alpha\n(i) roman\n(A) upper\n(I) roman\n(1) digit\nSection 1.02 Doubled.\n(a) alpha\n(z) jump\n(aa) double\n(bb) double",
            DetectionProfile::Instrument,
        ));
        assert!(instrument.nodes.iter().any(|node| {
            node.label.as_deref() == Some("sec1.01(a)(i)(A)(I)(1)")
                && node.content_start.is_some()
                && node.parent_id.is_some()
        }));
        assert!(instrument
            .nodes
            .iter()
            .any(|node| node.label.as_deref() == Some("sec1.02(bb)")));
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn instrument_heads_keep_token_and_heading_boundaries() {
        let text = concat!(
            "PART 4.20(a) of the Disclosure Schedule\n",
            "EXHIBIT IV   45\n",
            "SCHEDULE 14D-9 37\n",
            "EXHIBIT III Valid Heading\n",
        );
        let graph = derive_structure_evidence(evidence(text, DetectionProfile::Instrument))
            .expect("valid instrument evidence");
        let labels = graph
            .nodes
            .iter()
            .filter_map(|node| node.label.as_deref())
            .collect::<Vec<_>>();
        assert!(labels.contains(&"part4"));
        assert!(labels.contains(&"exhiii"));
        assert!(!labels
            .iter()
            .any(|label| matches!(*label, "exhi" | "sched14")));
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn short_root_preserves_crlf_offset_semantics() {
        let labels = statute_spine(
            &ScalarText::new("1\r\n\r\n________________\r\n2\r\n\r\n________________\r\n"),
            false,
        )
        .into_iter()
        .map(|mark| mark.label)
        .collect::<Vec<_>>();
        assert_eq!(labels, ["1", "2"]);
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn heading_and_utf16_edges_match_javascript() {
        assert!(formal_heading("Qualified Privilege"));
        assert!(!formal_heading("*429"));
        assert!(!formal_heading("() Heading"));
        let case = "[1] This opening paragraph contains enough ordinary words to establish substantive reasons for decision.\nAll Canadian people affected by the breach [1]\nClass Period\n[2] This second paragraph contains enough ordinary words to establish substantive reasons for decision.\n[3] This third paragraph contains enough ordinary words to establish substantive reasons for decision.\n[4] This fourth paragraph contains enough ordinary words to establish substantive reasons for decision.\n[5] This fifth paragraph contains enough ordinary words to establish substantive reasons for decision.";
        let case_graph =
            derive_structure_evidence(evidence(case, DetectionProfile::CaseRootedComplete))
                .expect("valid case evidence");
        assert_eq!(
            case_graph
                .nodes
                .iter()
                .find(|node| node.label.as_deref() == Some("par1"))
                .unwrap()
                .range
                .end,
            case[..case.find("[1]\nClass Period").unwrap()]
                .chars()
                .count()
        );
        assert_eq!(utf16_prefix("😀ab", 1), "");
        assert_eq!(utf16_prefix("😀ab", 3), "😀a");
        let journal = derive_structure_evidence(evidence(
            "\u{85}alpha\n\n\u{feff}beta",
            DetectionProfile::Journal,
        ))
        .expect("valid journal evidence");
        assert_eq!(
            journal
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Prose)
                .map(|node| node.range.start)
                .collect::<Vec<_>>(),
            [0, 9]
        );
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn bare_label_alone_extends_substantive_statute_spines() {
        let labels = |text: &str| {
            statute_spine(&ScalarText::new(text), false)
                .into_iter()
                .map(|mark| mark.label)
                .collect::<Vec<_>>()
        };
        assert_eq!(labels("### Interpretation\n85F\nProvision text continues on the next line.\n85G Next provision.\n86 Final provision."), ["85F", "85G", "86"]);
        assert_eq!(labels("1 This Act may be cited as the Example Act.\n2 The following definitions apply in this Act.\n3\nApplication\nThis Act applies to every person in the territory.\n4 The Minister may make regulations."), ["1", "2", "3", "4"]);
    }

    #[test]
    #[cfg(feature = "recovery")]
    fn dotterm_inline_child_preserves_bare_spine_precedence() {
        let body = "This section provides for the administration of the enactment in force across the territory.";
        let text = format!("1. There is established a board. {body}\n2.(1) A term has the prescribed meaning. {body}\n2.1. This inserted provision governs administration. {body}\n3. The Minister may make regulations. {body}\n4. This Act comes into force. {body}");
        let graph = derive_structure_evidence(evidence(&text, DetectionProfile::Legislation))
            .expect("valid fixture evidence");
        let top = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Section && node.parent_id.is_none())
            .filter_map(|node| node.label.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(top, ["sec1", "sec2", "sec2.1", "sec3", "sec4"]);
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.label.as_deref() == Some("sec2(1)") && node.parent_id.is_some()));

        let ontario = "1 A person is exempted if the person satisfies the following:\n1. The person is registered with a regulatory authority.\n2. A regulatory authority has not refused the person.\n3. A finding of misconduct has not been made.\n4. The person is not the subject of any proceeding.\n5. The person has submitted an application.\n2 A person who is exempted must notify the College.\n3 Omitted (provides for coming into force).";
        assert_eq!(
            statute_spine(&ScalarText::new(ontario), false)
                .into_iter()
                .map(|mark| mark.label)
                .collect::<Vec<_>>(),
            ["1", "2", "3"]
        );
    }
}
