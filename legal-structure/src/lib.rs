#[cfg(feature = "structure-inference")]
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
#[cfg(feature = "structure-inference")]
use std::sync::OnceLock;

#[cfg(feature = "a2aj")]
mod a2aj;
#[cfg(feature = "journal")]
mod journal;
mod instrument;
#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
mod native_markup;
mod numeric_sequence;
mod sidecar;
#[cfg(feature = "source-doc")]
mod source_doc;
#[cfg(feature = "a2aj")]
pub use a2aj::{a2aj_source_doc, A2ajInput, A2ajSectionMap, A2ajSourceKind};
#[cfg(feature = "journal")]
pub use journal::{journal_source_doc, journal_text_source_doc, JournalPageLabel};
pub use instrument::*;
#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub use native_markup::{native_markup_source_doc, NativeMarkupInput};
pub use numeric_sequence::*;
pub use sidecar::{native_sidecar, native_stdio_sidecar};
#[cfg(feature = "structure-inference")]
pub use sidecar::{sidecar, stdio_sidecar};
#[cfg(feature = "source-doc")]
pub use source_doc::{
    create_source_doc, ProjectionOrder, SourceDoc, SourceDocBlock, SourceDocIndex, SourceDocKind,
    SourceDocOrigin, SourceDocProvider, SourceDocType,
};

pub const EVIDENCE_SCHEMA: &str = "legalpdf.structure-evidence.v1";
pub const RESULT_SCHEMA: &str = "legalpdf.structure-graph.v2";
pub const SIDECAR_PROTOCOL: &str = "legalpdf.structure-sidecar.v1";
pub const SOURCE_DOC_VERSION: u32 = 1;
const ENGINE_ORIGIN: &str = "legalpdf.structure-engine";
const MAX_DOCUMENTS: usize = 25;
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[cfg(feature = "structure-inference")]
fn javascript_whitespace(character: char) -> bool {
    character == '\u{feff}' || (character != '\u{85}' && character.is_whitespace())
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
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
    List,
    Navigation,
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
            Self::List => "list",
            Self::Navigation => "navigation",
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

    fn needs_inference(&self) -> bool {
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Paragraph,
    Page,
    Section,
    Heading,
    Footnote,
    Endnote,
    Prose,
    List,
    ListItem,
    Navigation,
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
            Self::List | Self::ListItem => EvidenceKind::List,
            Self::Navigation => EvidenceKind::Navigation,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::ListItem => "list_item",
            _ => self.evidence().name(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStatus {
    Complete,
    Partial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Derivation {
    Native,
    Heuristic,
    Model,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteKindV2 {
    Footnote,
    Endnote,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureNodeV2 {
    pub id: String,
    pub kind: NodeKind,
    pub range: ScalarRange,
    pub origin_id: String,
    pub source: Derivation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locator_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marker_range: Option<ScalarRange>,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<ResolutionProofV2>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    Paragraph,
    Prose,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureBoundaryV2 {
    pub kind: BoundaryKind,
    pub at: usize,
    pub origin_id: String,
    pub source: Derivation,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Contains,
    Precedes,
    References,
    FootnoteFor,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RelationEndpointV2 {
    Node { node_id: String },
    Range { range: ScalarRange },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureRelationV2 {
    pub id: String,
    pub kind: RelationKind,
    pub from: RelationEndpointV2,
    pub to: RelationEndpointV2,
    pub origin_id: String,
    pub source: Derivation,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureDiagnosticV2 {
    pub code: String,
    pub severity: DiagnosticSeverity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub candidate_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub rules: Vec<ResolutionRuleV2>,
    pub ranges: Vec<ScalarRange>,
    pub node_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructureGraphV2 {
    pub schema_version: String,
    pub document_id: String,
    pub text_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    pub status: GraphStatus,
    pub nodes: Vec<StructureNodeV2>,
    pub boundaries: Vec<StructureBoundaryV2>,
    pub relations: Vec<StructureRelationV2>,
    pub diagnostics: Vec<StructureDiagnosticV2>,
}

impl StructureGraphV2 {
    pub fn from_parts(
        document_id: String,
        text: &str,
        source_sha256: Option<String>,
        status: GraphStatus,
        nodes: Vec<StructureNodeV2>,
        boundaries: Vec<StructureBoundaryV2>,
        relations: Vec<StructureRelationV2>,
        diagnostics: Vec<StructureDiagnosticV2>,
    ) -> Self {
        Self {
            schema_version: RESULT_SCHEMA.to_owned(),
            document_id,
            text_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            source_sha256,
            status,
            nodes,
            boundaries,
            relations,
            diagnostics,
        }
    }
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CandidateGrammar {
    Numeric,
    Hierarchy,
    Enumerator,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructureMarkerCandidate {
    pub id: String,
    pub range: ScalarRange,
    pub marker_range: ScalarRange,
    pub label: String,
    pub grammar_value: String,
    pub parent_candidate_id: Option<String>,
    pub level: usize,
    pub content_start: usize,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructureCandidateRun {
    pub id: String,
    pub grammar: CandidateGrammar,
    pub range: ScalarRange,
    pub rooted: bool,
    pub consecutive: bool,
    pub markers: Vec<StructureMarkerCandidate>,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEvidenceV2 {
    pub candidate_id: String,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
    pub observations: Vec<CandidateObservationV2>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateObservationV2 {
    BodyProseFlow,
    SectionHeading,
    ListItemLayout,
    CrossReference,
    Furniture,
    TableOrForm,
    ContentsRow,
    IndexRow,
    TranscriptLineNumber,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextAnchorV2 {
    pub range: ScalarRange,
    pub page_index: usize,
    pub line_id: String,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NoteBodyV2 {
    pub range: ScalarRange,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotePairClaimV2 {
    pub pair_id: String,
    pub kind: NoteKindV2,
    pub label: TextAnchorV2,
    pub body: NoteBodyV2,
    pub references: Vec<TextAnchorV2>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionRuleV2 {
    RootedNumericProse,
    HierarchySection,
    ListItemLayout,
    PairedNote,
    DirectExclusion,
    ConflictingRoles,
    InsufficientEvidence,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolutionProofV2 {
    pub rule: ResolutionRuleV2,
    pub observations: Vec<CandidateObservationV2>,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedRole {
    NumberedParagraph,
    Section,
    ListItem,
}

#[cfg(feature = "structure-inference")]
impl ResolvedRole {
    pub fn node_kind(self) -> NodeKind {
        match self {
            Self::NumberedParagraph => NodeKind::Paragraph,
            Self::Section => NodeKind::Section,
            Self::ListItem => NodeKind::ListItem,
        }
    }
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedCandidate {
    pub candidate: StructureMarkerCandidate,
    pub role: Option<ResolvedRole>,
    pub proof: ResolutionProofV2,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Copy)]
struct OffsetCheckpoint {
    scalar: usize,
    byte: usize,
    utf16: usize,
}

#[cfg(feature = "structure-inference")]
struct ScalarText<'a> {
    value: &'a str,
    offsets: Vec<OffsetCheckpoint>,
    scalar_len: usize,
    utf16_len: usize,
    lines: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "structure-inference")]
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

#[cfg(feature = "structure-inference")]
fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
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

#[cfg(feature = "structure-inference")]
mod inference;

#[cfg(feature = "structure-inference")]
mod candidates;
mod derive;

#[cfg(feature = "structure-inference")]
pub use candidates::{
    detect_structure_candidate_runs, resolve_structure_candidates, resolve_structure_graph,
};
pub use derive::derive_native_structure_evidence;
#[cfg(feature = "structure-inference")]
pub use derive::derive_structure_evidence;
#[cfg(feature = "source-doc")]
pub use derive::compose_native;
#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub use derive::compose;
#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(crate) use derive::compose_trusted;

#[cfg(test)]
mod tests {
    #[cfg(feature = "structure-inference")]
    use super::inference::{formal_heading, statute_spine};
    use super::*;

    #[test]
    #[cfg(feature = "structure-inference")]
    fn instrument_lineation_hypotheses_preserve_the_typescript_contract() {
        let source = "AGREEMENT  ARTICLE I DEFINITIONS. Section 1.01 Terms.\t( a ) stays prose.";
        assert_eq!(
            instrument_lineation_hypotheses(source),
            [
                source,
                "AGREEMENT\n ARTICLE I DEFINITIONS. Section 1.01 Terms.\t( a ) stays prose.",
                "AGREEMENT  ARTICLE I DEFINITIONS.\nSection 1.01 Terms.\t( a ) stays prose.",
                "AGREEMENT\n ARTICLE I DEFINITIONS.\nSection 1.01 Terms.\t( a ) stays prose.",
            ]
        );
        assert_eq!(
            instrument_lineation_hypotheses("already\nlineated"),
            ["already\nlineated"]
        );
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn instrument_lineation_selection_uses_endorsed_spans_and_source_ties() {
        let text = "x".repeat(200);
        let graph = |labels: &[(&str, usize)]| StructureGraphV2 {
            schema_version: RESULT_SCHEMA.to_owned(),
            document_id: "test".to_owned(),
            text_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            source_sha256: None,
            status: GraphStatus::Complete,
            nodes: labels
                .iter()
                .enumerate()
                .map(|(index, (label, start))| StructureNodeV2 {
                    id: format!("node-{index}"),
                    kind: NodeKind::Section,
                    range: ScalarRange {
                        start: *start,
                        end: text.len(),
                    },
                    origin_id: ENGINE_ORIGIN.to_owned(),
                    source: Derivation::Heuristic,
                    label: Some((*label).to_owned()),
                    locator_kind: None,
                    aliases: None,
                    parent_id: None,
                    anchor: None,
                    content_start: Some(*start),
                    marker_range: None,
                    page_indexes: Vec::new(),
                    line_ids: Vec::new(),
                    level: None,
                    grammar: None,
                    proof: None,
                })
                .collect(),
            boundaries: Vec::new(),
            relations: Vec::new(),
            diagnostics: Vec::new(),
        };
        let graphs = [graph(&[("sec1", 0)]), graph(&[("sec1", 0), ("sec2", 150)])];
        assert_eq!(
            select_instrument_lineation(
                &text,
                &graphs,
                &[InstrumentReferenceEvidence {
                    key: "sec2".to_owned(),
                    start: 20,
                    end: 24,
                }],
            )
            .unwrap(),
            1
        );
        assert_eq!(select_instrument_lineation(&text, &graphs, &[]).unwrap(), 0);
    }

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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
    fn typed_hierarchy_candidates_use_the_production_detector() {
        let runs = detect_structure_candidate_runs(
            "Section 1.01 Opening.\n(a) First clause.\n(b) Second clause.\nSection 1.02 Closing.",
        );
        let sections = runs
            .iter()
            .filter(|run| run.grammar == CandidateGrammar::Hierarchy)
            .flat_map(|run| &run.markers)
            .collect::<Vec<_>>();
        assert_eq!(
            sections
                .iter()
                .map(|section| section.grammar_value.as_str())
                .collect::<Vec<_>>(),
            ["sec1.01", "sec1.01(a)", "sec1.01(b)", "sec1.02"]
        );
        assert_eq!(
            sections[1].parent_candidate_id.as_deref(),
            Some(sections[0].id.as_str())
        );
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn bare_section_label_without_inline_content_is_safe() {
        let _ =
            detect_structure_candidate_runs("1\nProvision text.\n2\nMore text.\n3\nFinal text.");
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn raw_numeric_candidates_keep_late_and_gapped_runs_for_resolution() {
        let runs = detect_structure_candidate_runs(
            "7. First excerpt paragraph.\n9. A gap remains visible.\n12. Another item.",
        );
        let run = runs
            .iter()
            .find(|run| run.grammar == CandidateGrammar::Numeric)
            .expect("numeric candidate run");
        assert!(!run.rooted);
        assert!(!run.consecutive);
        assert_eq!(
            run.markers
                .iter()
                .map(|marker| marker.grammar_value.as_str())
                .collect::<Vec<_>>(),
            ["7", "9", "12"]
        );
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn typed_evidence_resolves_numeric_prose_and_reports_each_run() {
        let text = "1. First paragraph.\n2. Second paragraph.";
        let run = detect_structure_candidate_runs(text)
            .into_iter()
            .find(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
            .expect("rooted numeric candidate run");
        let evidence = run
            .markers
            .iter()
            .enumerate()
            .map(|(index, candidate)| CandidateEvidenceV2 {
                candidate_id: candidate.id.clone(),
                page_indexes: vec![0],
                line_ids: vec![format!("line-{index}")],
                observations: vec![CandidateObservationV2::BodyProseFlow],
            })
            .collect::<Vec<_>>();
        let graph = resolve_structure_graph(
            "numeric".to_owned(),
            text,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[run],
            &evidence,
            &[],
            Vec::new(),
        )
        .expect("valid typed evidence");
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Paragraph)
                .count(),
            2
        );
        assert_eq!(graph.diagnostics.len(), 1);
        assert_eq!(graph.diagnostics[0].code, "structure_run_resolved");
        assert_eq!(
            graph.diagnostics[0].rules,
            [ResolutionRuleV2::RootedNumericProse]
        );
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn contents_and_transcript_number_evidence_force_abstention() {
        let text = "1. First paragraph.\n2. Second paragraph.";
        let run = detect_structure_candidate_runs(text)
            .into_iter()
            .find(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
            .expect("rooted numeric candidate run");
        for exclusion in [
            CandidateObservationV2::ContentsRow,
            CandidateObservationV2::IndexRow,
            CandidateObservationV2::TranscriptLineNumber,
        ] {
            let evidence = run
                .markers
                .iter()
                .enumerate()
                .map(|(index, candidate)| CandidateEvidenceV2 {
                    candidate_id: candidate.id.clone(),
                    page_indexes: vec![0],
                    line_ids: vec![format!("line-{index}")],
                    observations: vec![CandidateObservationV2::BodyProseFlow, exclusion],
                })
                .collect::<Vec<_>>();
            let resolved = resolve_structure_candidates(std::slice::from_ref(&run), &evidence)
                .expect("valid exclusion evidence");
            assert!(resolved.iter().all(|candidate| candidate.role.is_none()));
            assert!(resolved
                .iter()
                .all(|candidate| candidate.proof.rule == ResolutionRuleV2::DirectExclusion));
        }
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn local_candidate_parent_ids_create_honest_list_items() {
        let text = "Section 1\n(a) item";
        let length = text.chars().count();
        let run = StructureCandidateRun {
            id: "hierarchy-1".to_owned(),
            grammar: CandidateGrammar::Hierarchy,
            range: ScalarRange {
                start: 0,
                end: length,
            },
            rooted: true,
            consecutive: true,
            markers: vec![
                StructureMarkerCandidate {
                    id: "section".to_owned(),
                    range: ScalarRange {
                        start: 0,
                        end: length,
                    },
                    marker_range: ScalarRange { start: 0, end: 9 },
                    label: "Section 1".to_owned(),
                    grammar_value: "sec1".to_owned(),
                    parent_candidate_id: None,
                    level: 0,
                    content_start: 9,
                },
                StructureMarkerCandidate {
                    id: "item".to_owned(),
                    range: ScalarRange {
                        start: 10,
                        end: length,
                    },
                    marker_range: ScalarRange { start: 10, end: 14 },
                    label: "(a)".to_owned(),
                    grammar_value: "1:1".to_owned(),
                    parent_candidate_id: Some("section".to_owned()),
                    level: 1,
                    content_start: 14,
                },
            ],
        };
        let evidence = vec![
            CandidateEvidenceV2 {
                candidate_id: "section".to_owned(),
                page_indexes: vec![0],
                line_ids: vec!["heading".to_owned()],
                observations: vec![CandidateObservationV2::SectionHeading],
            },
            CandidateEvidenceV2 {
                candidate_id: "item".to_owned(),
                page_indexes: vec![0],
                line_ids: vec!["item".to_owned()],
                observations: vec![CandidateObservationV2::ListItemLayout],
            },
        ];
        let graph = resolve_structure_graph(
            "hierarchy".to_owned(),
            text,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[run],
            &evidence,
            &[],
            Vec::new(),
        )
        .expect("valid hierarchy evidence");
        let section = graph
            .nodes
            .iter()
            .find(|node| node.kind == NodeKind::Section)
            .unwrap();
        let item = graph
            .nodes
            .iter()
            .find(|node| node.kind == NodeKind::ListItem)
            .unwrap();
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Section)
                .count(),
            1
        );
        assert_eq!(section.locator_kind.as_deref(), Some("section"));
        assert_eq!(item.parent_id.as_deref(), Some(section.id.as_str()));
        assert!(!graph.nodes.iter().any(|node| node.kind == NodeKind::List));
    }

    #[test]
    #[cfg(feature = "structure-inference")]
    fn paired_note_claim_keeps_every_anchor_and_deduplicates_relations() {
        let text = "Body ref 1.\n1 Note body.";
        let reference_start = text.find('1').unwrap();
        let label_start = text.rfind('1').unwrap();
        let pair = NotePairClaimV2 {
            pair_id: "pair-1".to_owned(),
            kind: NoteKindV2::Footnote,
            label: TextAnchorV2 {
                range: ScalarRange {
                    start: label_start,
                    end: label_start + 1,
                },
                page_index: 1,
                line_id: "note-line".to_owned(),
            },
            body: NoteBodyV2 {
                range: ScalarRange {
                    start: label_start,
                    end: text.chars().count(),
                },
                page_indexes: vec![1],
                line_ids: vec!["note-line".to_owned()],
            },
            references: vec![
                TextAnchorV2 {
                    range: ScalarRange {
                        start: reference_start,
                        end: reference_start + 1,
                    },
                    page_index: 0,
                    line_id: "body-line".to_owned(),
                },
                TextAnchorV2 {
                    range: ScalarRange {
                        start: reference_start,
                        end: reference_start + 1,
                    },
                    page_index: 0,
                    line_id: "body-line".to_owned(),
                },
            ],
        };
        let graph = resolve_structure_graph(
            "notes".to_owned(),
            text,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            &[],
            &[pair],
            Vec::new(),
        )
        .expect("valid note claim");
        let note = graph
            .nodes
            .iter()
            .find(|node| node.kind == NodeKind::Footnote)
            .unwrap();
        assert_eq!(note.anchor.as_deref(), Some("pair-1"));
        assert_eq!(note.page_indexes, [1]);
        assert_eq!(note.line_ids, ["note-line"]);
        assert_eq!(
            graph
                .relations
                .iter()
                .filter(|relation| relation.kind == RelationKind::References)
                .count(),
            1
        );
        assert_eq!(
            graph
                .relations
                .iter()
                .filter(|relation| relation.kind == RelationKind::FootnoteFor)
                .count(),
            1
        );
        assert!(graph
            .relations
            .iter()
            .all(|relation| relation.line_ids == ["body-line"]));
    }

    #[test]
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
    #[cfg(feature = "structure-inference")]
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
