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
        EvidenceKind::Navigation => NodeKind::Navigation,
    }
}
fn infer_graph(
    evidence: DocumentInput,
    inferred: Vec<Block>,
    inference_available: bool,
) -> StructureGraphV2 {
    let complete = evidence.scope.kind == ScopeKind::Complete
        && (inference_available || !evidence.needs_inference());
    let mut nodes = evidence
        .native_claims
        .iter()
        .map(|claim| StructureNodeV2 {
            id: claim.id.clone(),
            kind: native_kind(claim.kind),
            range: claim.range,
            origin_id: claim.origin_id.clone(),
            source: Derivation::Native,
            label: claim.label.clone(),
            locator_kind: None,
            aliases: (!claim.aliases.is_empty()).then(|| claim.aliases.clone()),
            parent_id: None,
            anchor: claim.anchor.clone(),
            content_start: None,
            marker_range: None,
            page_indexes: Vec::new(),
            line_ids: Vec::new(),
            level: None,
            grammar: None,
            proof: None,
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
            block.diagnostic.map(|code| StructureDiagnosticV2 {
                code: code.to_owned(),
                severity: if code.ends_with("violation") {
                    DiagnosticSeverity::Warning
                } else {
                    DiagnosticSeverity::Info
                },
                run_id: None,
                candidate_ids: Vec::new(),
                rules: Vec::new(),
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
            relations.push(StructureRelationV2 {
                id: format!("heuristic-contains-{:06}", relations.len() + 1),
                kind: RelationKind::Contains,
                from: RelationEndpointV2::Node {
                    node_id: parent.clone(),
                },
                to: RelationEndpointV2::Node {
                    node_id: id.clone(),
                },
                origin_id: ENGINE_ORIGIN.to_owned(),
                source: Derivation::Heuristic,
                page_indexes: Vec::new(),
                line_ids: Vec::new(),
            });
        }
        nodes.push(StructureNodeV2 {
            id,
            kind: block.kind,
            range: block.range,
            origin_id: ENGINE_ORIGIN.to_owned(),
            source: Derivation::Heuristic,
            label: block.label,
            locator_kind: None,
            aliases: (!block.aliases.is_empty()).then_some(block.aliases),
            parent_id,
            anchor: None,
            content_start: block.content_start,
            marker_range: None,
            page_indexes: Vec::new(),
            line_ids: Vec::new(),
            level: None,
            grammar: None,
            proof: None,
        });
    }
    let mut boundaries = evidence
        .paragraph_breaks
        .iter()
        .map(|value| StructureBoundaryV2 {
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
            .map(|node| StructureBoundaryV2 {
                kind: BoundaryKind::Prose,
                at: node.range.end,
                origin_id: ENGINE_ORIGIN.to_owned(),
                source: Derivation::Heuristic,
            }),
    );
    StructureGraphV2::from_parts(
        evidence.document_id,
        &evidence.text,
        evidence.source_sha256,
        if complete {
            GraphStatus::Complete
        } else {
            GraphStatus::Partial
        },
        nodes,
        boundaries,
        relations,
        diagnostics,
    )
}

#[cfg(feature = "structure-inference")]
pub fn derive_structure_evidence(evidence: DocumentInput) -> Result<StructureGraphV2, EngineError> {
    let inferred = if evidence.needs_inference() {
        let text = ScalarText::new(&evidence.text);
        inference::inferred_blocks(&evidence, &text)
    } else {
        Vec::new()
    };
    Ok(infer_graph(evidence, inferred, true))
}

pub fn derive_native_structure_evidence(
    evidence: DocumentInput,
) -> Result<StructureGraphV2, EngineError> {
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
    inference_enabled: bool,
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
    #[cfg(feature = "structure-inference")]
    let inferred = if inference_enabled && input.needs_inference() {
        let text = ScalarText::new(&input.text);
        inference::inferred_blocks(&input, &text)
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "structure-inference"))]
    let inferred = Vec::new();
    let text = std::mem::take(&mut input.text);
    let graph = infer_graph(input, inferred, inference_enabled);
    Ok(source_doc::project_graph(
        provider, id, url, doc_type, text, revision, &originals, graph, order,
    ))
}

#[cfg(feature = "source-doc")]
pub fn compose_native(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, false, true)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub fn compose(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, true, true)
}

#[cfg(all(feature = "structure-inference", feature = "source-doc"))]
pub(crate) fn compose_trusted(input: DocumentInput) -> Result<SourceDoc, EngineError> {
    compose_with(input, true, false)
}
