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
mod tests;
