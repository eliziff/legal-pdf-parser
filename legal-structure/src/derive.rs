use super::*;
#[cfg(feature = "source-doc")]
use crate::source_doc::ProjectionOrder;

fn native_kind(kind: EvidenceKind) -> NodeKind {
    match kind {
        EvidenceKind::Paragraph => NodeKind::Paragraph,
        EvidenceKind::Prose => NodeKind::Prose,
        EvidenceKind::Page => NodeKind::Page,
        EvidenceKind::Section => NodeKind::Section,
        EvidenceKind::Heading => NodeKind::Heading,
        EvidenceKind::Footnote => NodeKind::Footnote,
        EvidenceKind::Endnote => NodeKind::Endnote,
        EvidenceKind::List => NodeKind::List,
        EvidenceKind::Table => NodeKind::Table,
        EvidenceKind::Row => NodeKind::Row,
        EvidenceKind::Cell => NodeKind::Cell,
    }
}
fn infer_graph(mut evidence: DocumentInput, inferred: Vec<Block>) -> DocumentStructure {
    let native_labels = evidence
        .native_claims
        .iter()
        .flat_map(|claim| {
            claim
                .label
                .iter()
                .chain(&claim.aliases)
                .map(|label| (claim.kind, label.to_ascii_lowercase()))
        })
        .collect::<HashSet<_>>();
    let mut nodes = evidence
        .native_claims
        .iter()
        .map(|claim| {
            let mut node = StructureNode::new(
                claim.id.clone(),
                native_kind(claim.kind),
                claim.range,
                claim.origin_id.clone(),
                Derivation::Native,
                None,
            );
            node.label.clone_from(&claim.label);
            node.aliases = (!claim.aliases.is_empty()).then(|| claim.aliases.clone());
            node.anchor.clone_from(&claim.anchor);
            node
        })
        .collect::<Vec<_>>();
    let mut counters = HashMap::<NodeKind, usize>::new();
    let mut generated_parents = Vec::new();
    let mut diagnostics = Vec::new();
    for mut block in inferred {
        let Some(range) = evidence.clip_inference(block.kind.evidence(), block.range) else {
            continue;
        };
        block.range = range;
        if block.content_start.is_some_and(|at| {
            block.kind != NodeKind::Section || at < block.range.start || at > block.range.end
        }) || (!native_labels.is_empty()
            && block.label.as_ref().is_some_and(|label| {
                native_labels.contains(&(block.kind.evidence(), label.to_ascii_lowercase()))
            }))
        {
            continue;
        }
        let ordinal = counters.entry(block.kind).or_default();
        *ordinal += 1;
        let source = match block.source {
            Derivation::Native => "native",
            Derivation::Heuristic => "heuristic",
            Derivation::Model => "model",
        };
        let id = format!("{source}-{}-{:06}", block.kind.name(), ordinal);
        if let Some(code) = block.diagnostic {
            diagnostics.push(StructureDiagnostic {
                code: code.to_owned(),
                severity: if code.ends_with("violation") {
                    DiagnosticSeverity::Warning
                } else {
                    DiagnosticSeverity::Info
                },
                ranges: vec![block.range],
                node_ids: vec![id.clone()],
            });
        }
        generated_parents.push(block.parent_label);
        let mut node = StructureNode::new(
            id,
            block.kind,
            block.range,
            block.origin_id,
            block.source,
            None,
        );
        node.label = block.label;
        node.aliases = (!block.aliases.is_empty()).then_some(block.aliases);
        node.content_start = block.content_start;
        nodes.push(node);
    }
    let mut labels = HashMap::with_capacity(nodes.len());
    for (position, node) in nodes.iter().enumerate() {
        for label in node.label.iter().chain(node.aliases.iter().flatten()) {
            labels.insert(label.to_ascii_lowercase(), position);
        }
    }
    for (position, parent) in evidence
        .native_claims
        .iter()
        .map(|claim| claim.parent_label.as_ref())
        .chain(generated_parents.iter().map(Option::as_ref))
        .enumerate()
    {
        nodes[position].parent_id = parent
            .and_then(|label| match labels.get(label) {
                Some(&position) => Some(position),
                None if label.bytes().any(|byte| byte.is_ascii_uppercase()) => {
                    labels.get(&label.to_ascii_lowercase()).copied()
                }
                None => None,
            })
            .map(|parent_position| nodes[parent_position].id.clone());
    }
    let document_id = std::mem::take(&mut evidence.document_id);
    let provider = std::mem::take(&mut evidence.provider);
    let profile = evidence.profile;
    let text_sha256 = std::mem::take(&mut evidence.text_sha256);
    #[cfg(feature = "source-doc")]
    let url = evidence.url.take();
    #[cfg(feature = "source-doc")]
    let doc_type = evidence.doc_type.map(|value| value.as_str().to_owned());
    let text = std::mem::take(&mut evidence.text);
    let source_sha256 = evidence.source_sha256.take();
    let scope = evidence.scope;
    let origins = evidence.origins;
    let mut structure = DocumentStructure::from_scalar_parts(
        document_id,
        text,
        text_sha256,
        source_sha256,
        scope,
        origins,
        nodes,
        Vec::new(),
        diagnostics,
    );
    structure.provider = provider;
    structure.profile = Some(profile);
    #[cfg(feature = "source-doc")]
    {
        structure.url = url;
        structure.doc_type = doc_type;
    }
    structure
}

#[cfg(all(feature = "structure-inference", test))]
pub(crate) fn derive_structure_evidence(
    evidence: DocumentInput,
) -> Result<DocumentStructure, EngineError> {
    let inferred = if evidence.needs_inference() {
        let text = ScalarText::new(&evidence.text);
        inference::inferred_blocks(&evidence, &text)
    } else {
        Vec::new()
    };
    Ok(infer_graph(evidence, inferred))
}

#[cfg(any(feature = "journal", test))]
pub(crate) fn derive_native_structure_evidence(
    evidence: DocumentInput,
) -> Result<DocumentStructure, EngineError> {
    Ok(infer_graph(evidence, Vec::new()))
}

#[cfg(feature = "source-doc")]
fn projection(profile: Option<DetectionProfile>) -> (ProjectionOrder, Option<SourceDocType>) {
    match profile {
        Some(DetectionProfile::CaseRootedComplete) => {
            (ProjectionOrder::Case, Some(SourceDocType::Cases))
        }
        Some(DetectionProfile::CaseContiguousComplete | DetectionProfile::CaseLossy) => {
            (ProjectionOrder::Position, Some(SourceDocType::Cases))
        }
        Some(DetectionProfile::Legislation) => {
            (ProjectionOrder::Legislation, Some(SourceDocType::Laws))
        }
        Some(DetectionProfile::Instrument) => (ProjectionOrder::Native, None),
        Some(DetectionProfile::Journal) => (ProjectionOrder::StablePosition, None),
        None => (ProjectionOrder::Position, None),
    }
}

#[cfg(feature = "source-doc")]
fn compose_with(
    input: DocumentInput,
    validate: bool,
    precomputed: Option<Vec<Block>>,
) -> Result<DocumentStructure, EngineError> {
    if validate {
        input.validate()?;
    }
    #[cfg(feature = "structure-inference")]
    let inferred = precomputed.unwrap_or_else(|| {
        if input.needs_inference() {
            let text = ScalarText::new(&input.text);
            inference::inferred_blocks(&input, &text)
        } else {
            Vec::new()
        }
    });
    #[cfg(not(feature = "structure-inference"))]
    let inferred = {
        let _ = precomputed;
        Vec::new()
    };
    Ok(infer_graph(input, inferred))
}

#[cfg(feature = "source-doc")]
pub fn project_document_structure(structure: DocumentStructure) -> SourceDoc {
    let (order, inferred_type) = projection(structure.profile);
    source_doc::project_graph(structure, order, inferred_type)
}

#[cfg(feature = "source-doc")]
pub fn project_document_structure_view(structure: &DocumentStructure) -> SourceDoc {
    let (order, inferred_type) = projection(structure.profile);
    source_doc::project_graph_view(structure, order, inferred_type)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub fn derive_document_structure(input: DocumentInput) -> Result<DocumentStructure, EngineError> {
    compose_with(input, true, None)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(crate) fn derive_trusted(input: DocumentInput) -> Result<DocumentStructure, EngineError> {
    compose_with(input, false, None)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(crate) fn derive_trusted_inferred(
    input: DocumentInput,
    inferred: Vec<Block>,
) -> Result<DocumentStructure, EngineError> {
    compose_with(input, false, Some(inferred))
}
