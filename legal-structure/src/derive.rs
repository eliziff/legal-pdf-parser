use super::*;

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
    let generated = inferred
        .into_iter()
        .filter_map(|mut block| {
            block.range = evidence.clip_inference(block.kind.evidence(), block.range)?;
            if block.content_start.is_some_and(|at| {
                block.kind != NodeKind::Section || at < block.range.start || at > block.range.end
            }) {
                return None;
            }
            (!evidence.native_claims.iter().any(|claim| {
                claim.kind == block.kind.evidence()
                    && block.label.as_deref().is_some_and(|candidate| {
                        claim
                            .label
                            .as_deref()
                            .is_some_and(|value| value.eq_ignore_ascii_case(candidate))
                            || claim
                                .aliases
                                .iter()
                                .any(|value| value.eq_ignore_ascii_case(candidate))
                    })
            }))
            .then_some(block)
        })
        .map(|block| {
            let ordinal = counters.entry(block.kind).or_default();
            *ordinal += 1;
            let source = match block.source {
                Derivation::Native => "native",
                Derivation::Heuristic => "heuristic",
                Derivation::Model => "model",
            };
            let id = format!("{source}-{}-{:06}", block.kind.name(), ordinal);
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
    let diagnostics = generated
        .iter()
        .filter_map(|(block, id)| {
            block.diagnostic.map(|code| StructureDiagnostic {
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
        let mut node = StructureNode::new(
            id,
            block.kind,
            block.range,
            block.origin_id,
            block.source,
            parent_id,
        );
        node.label = block.label;
        node.aliases = (!block.aliases.is_empty()).then_some(block.aliases);
        node.content_start = block.content_start;
        nodes.push(node);
    }
    let document_id = std::mem::take(&mut evidence.document_id);
    let provider = std::mem::take(&mut evidence.provider);
    let profile = evidence.profile;
    let revision = evidence.text_sha256.clone();
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
        source_sha256,
        scope,
        origins,
        nodes,
        Vec::new(),
        diagnostics,
    );
    structure.provider = provider;
    structure.profile = Some(profile);
    structure.revision = revision;
    #[cfg(feature = "source-doc")]
    {
        structure.url = url;
        structure.doc_type = doc_type;
    }
    structure
}

#[cfg(feature = "structure-inference")]
pub fn derive_structure_evidence(
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

pub fn derive_native_structure_evidence(
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
        Some(DetectionProfile::Instrument) => (ProjectionOrder::Legislation, None),
        Some(DetectionProfile::Journal) => (ProjectionOrder::StablePosition, None),
        None => (ProjectionOrder::Position, None),
    }
}

#[cfg(feature = "source-doc")]
fn compose_with(
    input: DocumentInput,
    inference_enabled: bool,
    validate: bool,
    precomputed: Option<Vec<Block>>,
) -> Result<DocumentStructure, EngineError> {
    if validate {
        input.validate()?;
    }
    #[cfg(feature = "structure-inference")]
    let inferred = precomputed.unwrap_or_else(|| {
        if inference_enabled && input.needs_inference() {
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
fn project_with(
    input: DocumentInput,
    inference_enabled: bool,
    validate: bool,
    precomputed: Option<Vec<Block>>,
) -> Result<SourceDoc, EngineError> {
    let structure = compose_with(input, inference_enabled, validate, precomputed)?;
    Ok(project_document_structure(structure))
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
    compose_with(input, true, true, None)
}

#[cfg(feature = "source-doc")]
pub fn compose_native(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    project_with(input, false, true, None)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub fn compose(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    project_with(input, true, true, None)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(crate) fn derive_trusted(input: DocumentInput) -> Result<DocumentStructure, EngineError> {
    compose_with(input, true, false, None)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(super) fn derive_trusted_inferred(
    input: DocumentInput,
    inferred: Vec<Block>,
) -> Result<DocumentStructure, EngineError> {
    compose_with(input, true, false, Some(inferred))
}
