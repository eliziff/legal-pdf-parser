//! Source-anchored scope openings, identity selection and exact containment.
use crate::witness::{Table, NAMES};
use legal_structure_model::{DocumentStructure, NumberingMarker, ScalarRange, ScalarText, StructureNode,
    StructureOwnership, StructureScopeAnnotation, StructureScopeKind};
use std::collections::{BTreeMap, BTreeSet};

pub const SCOPE_SCHEMA: &str = "source-scope-identity-v8";
pub const SCOPE_OPENING_SCHEMA: &str = "source-scope-opening-v3";
const NAMES_SCOPE: &[&str] = &[
    "document_axis", "quotation_axis", "stream_axis", "document_parent_axis", "quotation_parent_axis", "quotation_document_axis",
    "root", "new", "resume", "units_since_seen", "starts_quote", "ends_quote", "contains_quote",
    "candidate_same_style", "candidate_size_delta", "opening_slot", "candidate_opening_slot",
    "candidate_depth", "candidate_opened_here",
];

#[derive(serde::Serialize)]
pub struct ScopeTrainingRow {
    pub unit: String,
    pub axis: &'static str,
    pub candidate: String,
    pub features: Vec<f32>,
    pub positive: bool,
}
#[derive(serde::Serialize)]
pub struct ScopeTrainingRows {
    pub schema: &'static str,
    pub feature_names: Vec<String>,
    pub rows: Vec<ScopeTrainingRow>,
    pub unavailable_parent_targets: Vec<UnavailableScopeParent>,
    pub unavailable_opening_targets: Vec<UnavailableScopeOpening>,
}
#[derive(Clone, serde::Serialize)]
pub struct UnavailableScopeOpening {
    pub scope: String,
    pub axis: &'static str,
    pub start: Option<usize>,
    pub reason: &'static str,
}
#[derive(serde::Serialize)]
pub struct ScopeOpeningTrainingRow {
    pub unit: String,
    pub start: usize,
    pub axis: &'static str,
    pub features: Vec<f32>,
    pub count: usize,
}
#[derive(serde::Serialize)]
pub struct ScopeOpeningTrainingRows {
    pub schema: &'static str,
    pub feature_names: Vec<String>,
    pub rows: Vec<ScopeOpeningTrainingRow>,
    pub unavailable_targets: Vec<UnavailableScopeOpening>,
}
#[derive(serde::Serialize)]
pub struct UnavailableScopeParent {
    pub scope: String,
    pub axis: &'static str,
    pub parent: Option<String>,
    pub unit: Option<String>,
    pub reason: &'static str,
}
#[derive(Default)]
struct Selection {
    rows: Vec<ScopeTrainingRow>,
    assignments: Vec<(ScalarRange, StructureOwnership)>,
    declarations: Vec<StructureScopeAnnotation>,
    unavailable: Vec<UnavailableScopeParent>,
    unavailable_openings: Vec<UnavailableScopeOpening>,
}
struct Unit {
    id: String,
    boundary: Vec<f32>,
    numbering: Vec<f32>,
    marker: Option<String>,
    opening: ScalarRange,
    body: String,
    values: Vec<f32>,
    ownership: Option<StructureOwnership>,
    trainable: bool,
}
#[derive(Clone)]
struct History {
    position: usize,
    opened: usize,
    slot: usize,
    depth: usize,
    marker: Option<NumberingMarker>,
    values: Vec<f32>,
}
struct GoldOpenings<'a> {
    at: BTreeMap<(usize, u8), Vec<&'a StructureScopeAnnotation>>,
    unavailable: Vec<UnavailableScopeOpening>,
}
const AXES: [&str; 6] = ["document", "quotation", "reading_stream", "document_parent", "quotation_parent", "quotation_document"];
fn flag(value: bool) -> f32 { if value { 1.0 } else { 0.0 } }
fn quote(value: char) -> bool { matches!(value, '"' | '“' | '”' | '‘' | '’' | '«' | '»') }

pub fn feature_names() -> Vec<String> {
    crate::spans::START_NAMES.iter().map(|name| format!("start_{name}"))
        .chain(crate::numbering::SOURCE_NAMES.iter().map(|name| format!("numbering_{name}")))
        .chain(NAMES_SCOPE.iter().map(|name| (*name).to_owned()))
        .chain(crate::numbering::PAIR_NAMES.iter().map(|name| format!("numbering_{name}")))
        .chain(crate::profile::NAMES.iter().map(|name| (*name).to_owned()))
        .chain(NAMES.iter().map(|name| format!("source_{name}")))
        .chain(NAMES.iter().map(|name| format!("candidate_{name}"))).collect()
}
pub(crate) fn opening_feature_names() -> Vec<String> {
    crate::spans::START_NAMES.iter().map(|name| format!("start_{name}"))
        .chain(crate::numbering::SOURCE_NAMES.iter().map(|name| format!("numbering_{name}")))
        .chain(["document_axis", "quotation_axis"].into_iter().map(str::to_owned))
        .chain(crate::profile::NAMES.iter().map(|name| (*name).to_owned()))
        .chain(NAMES.iter().map(|name| format!("source_{name}"))).collect()
}
fn opening_features(axis: u8, unit: &Unit, profile: Option<legal_structure_model::DetectionProfile>) -> Vec<f32> {
    let mut features = unit.boundary.clone();
    features.extend_from_slice(&unit.numbering);
    features.extend([flag(axis == 0), flag(axis == 1)]);
    features.extend(crate::profile::values(profile));
    features.extend_from_slice(&unit.values);
    features
}
fn scope_features(axis: u8, kind: &str, position: usize, slot: usize, unit: &Unit,
    prior: Option<&History>, profile: Option<legal_structure_model::DetectionProfile>) -> Vec<f32> {
    let zeros = vec![0.0; NAMES.len()];
    let previous = prior.map_or(zeros.as_slice(), |prior| prior.values.as_slice());
    let values = &unit.values;
    let at = |name| crate::witness::index(name).unwrap();
    let body = unit.body.trim();
    let mut features = unit.boundary.clone();
    features.extend_from_slice(&unit.numbering);
    features.extend([flag(axis == 0), flag(axis == 1), flag(axis == 2), flag(axis == 3), flag(axis == 4), flag(axis == 5),
        flag(kind == "root"), flag(kind == "new"), flag(kind == "resume"),
        prior.map_or(0.0, |prior| position.saturating_sub(prior.position) as f32),
        flag(body.chars().next().is_some_and(quote)), flag(body.chars().last().is_some_and(quote)),
        flag(body.chars().any(quote)),
        flag(prior.is_some() && values[at("size")] == previous[at("size")] && values[at("bold")] == previous[at("bold")]),
        values[at("size")] - previous[at("size")], slot as f32,
        prior.map_or(0.0, |prior| prior.slot as f32), prior.map_or(0.0, |prior| prior.depth as f32),
        flag(prior.is_some_and(|prior| prior.opened == position))]);
    let previous_marker = prior.and_then(|prior| prior.marker.as_ref());
    let marker = crate::numbering::parse(unit.marker.as_deref(), previous_marker);
    features.extend(crate::numbering::pair_values(marker.as_ref(), previous_marker));
    features.extend(crate::profile::values(profile));
    features.extend_from_slice(values);
    features.extend_from_slice(previous);
    features
}
fn source_units(nodes: &[StructureNode], text: &str, source: &Table,
    fragments: &[legal_structure_model::SourceFragment]) -> Result<Vec<Unit>, String> {
    let candidates = crate::spans::source_candidates(text, source, fragments, None)?;
    let mut table = source.clone();
    crate::witness::bind_source_fragments(&mut table, text, fragments)?;
    table.rows.sort_by_key(|row| row.source_range.map(|range| (range.start, range.end)));
    let coordinates = ScalarText::new(text);
    let mut changes = BTreeMap::<usize, Vec<(usize, bool)>>::new();
    for (index, node) in nodes.iter().enumerate().filter(|(_, node)| node.role.is_some() || crate::outline::outline_unit(node)) {
        let mut parts = if node.ranges.is_empty() { vec![node.range] } else { node.ranges.clone() };
        ScalarRange::normalize_union(&mut parts);
        for part in parts.into_iter().filter(|part| part.start < part.end) {
            changes.entry(part.start).or_default().push((index, true));
            changes.entry(part.end).or_default().push((index, false));
        }
    }
    let boundaries = candidates.iter().map(|candidate| candidate.start).collect::<Vec<_>>();
    let mut changes = changes.into_iter().peekable();
    let mut active = BTreeSet::new();
    let mut intervals = Vec::new();
    for pair in boundaries.windows(2) {
        while changes.peek().is_some_and(|(start, _)| *start <= pair[0]) {
            let (_, events) = changes.next().unwrap();
            for (index, opening) in &events {
                if !*opening { active.remove(index); }
            }
            for (index, opening) in &events {
                if *opening { active.insert(*index); }
            }
        }
        let covering = active.iter().filter(|index| {
            let node = &nodes[**index];
            let parts = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
            parts.iter().any(|part| part.start <= pair[0] && pair[1] <= part.end)
        }).collect::<Vec<_>>();
        let mut ownership = covering.first().and_then(|index| nodes[**index].ownership.clone());
        let conflicting_streams = covering.iter().any(|index| nodes[**index].ownership.as_ref().and_then(|owner| owner.reading_stream_id.as_ref())
            != ownership.as_ref().and_then(|owner| owner.reading_stream_id.as_ref()));
        if conflicting_streams {
            if let Some(owner) = &mut ownership { owner.reading_stream_id = None; }
        }
        intervals.push(((pair[0], pair[1]), ownership, !conflicting_streams));
    }
    intervals.into_iter().map(|((start, end), ownership, trainable)| {
        let row = table.rows.partition_point(|row| row.source_range.is_some_and(|range| range.start <= start))
            .checked_sub(1).and_then(|index| table.rows.get(index))
            .filter(|row| row.source_range.is_some_and(|range| start < range.end));
        let body = coordinates.slice(start..end).ok_or("Scope unit lies outside source")?.to_owned();
        let local = row.and_then(|row| row.source_range).map_or(0, |range| start - range.start);
        let values = crate::witness::local_span_values(row, &body, local, local + end - start);
        let candidate = candidates.binary_search_by_key(&start, |candidate| candidate.start).ok()
            .map(|index| &candidates[index]);
        let boundary = candidate.map(|candidate| candidate.features[..crate::spans::START_NAMES.len()].to_vec())
            .unwrap_or_else(|| vec![0.0; crate::spans::START_NAMES.len()]);
        let numbering = candidate.map(|candidate| candidate.numbering.clone())
            .unwrap_or_else(|| vec![0.0; crate::numbering::SOURCE_NAMES.len()]);
        let marker = candidate.and_then(|candidate| candidate.marker.clone());
        Ok(Unit { id: format!("source@{start}:{end}"), opening: ScalarRange { start, end },
            body, boundary, numbering, marker, values, ownership, trainable })
    }).collect()
}

/// Gold scope membership supervises source intervals independently of role-node cuts.
fn gold_ownership(units: &mut [Unit], scopes: &[StructureScopeAnnotation], text: &str,
    document_id: &str) -> Result<(), String> {
    let coordinates = ScalarText::new(text);
    let by_id = scopes.iter().map(|scope| (scope.id.as_str(), scope)).collect::<BTreeMap<_, _>>();
    let memberships = scopes.iter().map(|scope| {
        let ranges = if scope.kind == StructureScopeKind::Document { &scope.owned_ranges }
            else if scope.kind == StructureScopeKind::ReadingStream && !scope.owned_ranges.is_empty() { &scope.owned_ranges }
            else { &scope.ranges };
        let mut ranges = ranges.iter().map(|range| Ok(ScalarRange {
            start: coordinates.scalar_at_utf16(range.start).ok_or("Invalid gold scope start")?,
            end: coordinates.scalar_at_utf16(range.end).ok_or("Invalid gold scope end")?,
        })).collect::<Result<Vec<_>, String>>()?;
        ScalarRange::normalize_union(&mut ranges);
        Ok((scope.id.as_str(), ranges))
    }).collect::<Result<BTreeMap<_, _>, String>>()?;
    let inside = |ranges: &[ScalarRange], unit: &Unit| {
        ranges.iter().any(|range| range.start <= unit.opening.start && unit.opening.end <= range.end)
    };
    for unit in units {
        let mut annotated = unit.ownership.is_some();
        let mut owner = unit.ownership.clone().unwrap_or(StructureOwnership {
            document_id: document_id.to_owned(), quotation_id: None, reading_stream_id: None });
        if (owner.document_id != document_id && !by_id.contains_key(owner.document_id.as_str()))
            || owner.quotation_id.as_deref().is_some_and(|id| !by_id.contains_key(id)) {
            unit.trainable = false;
            continue;
        }
        let original_document = owner.document_id.clone();
        owner.document_id = document_id.to_owned();
        for kind in [StructureScopeKind::Document, StructureScopeKind::Quotation] {
            let mut matches = Vec::new();
            for scope in scopes.iter().filter(|scope| scope.kind == kind) {
                let ranges = &memberships[scope.id.as_str()];
                if inside(ranges, unit) { annotated = true; matches.push(scope); }
                else {
                    for range in ranges {
                        if range.start < unit.opening.end && unit.opening.start < range.end { unit.trainable = false; }
                    }
                }
            }
            let ancestors = matches.iter().flat_map(|scope| {
                let mut parents = Vec::new();
                let mut parent = scope.parent_scope_id.as_deref();
                while let Some(id) = parent {
                    if parents.contains(&id) { break; }
                    parents.push(id);
                    parent = by_id.get(id).and_then(|scope| scope.parent_scope_id.as_deref());
                }
                parents
            }).collect::<BTreeSet<_>>();
            matches.retain(|scope| !ancestors.contains(scope.id.as_str()));
            if matches.len() > 1 { return Err("Gold source interval has conflicting scope identities".into()); }
            if kind == StructureScopeKind::Document {
                if let Some(scope) = matches.first() { owner.document_id.clone_from(&scope.id); }
            } else {
                owner.quotation_id = matches.first().map(|scope| scope.id.clone());
                if let Some(scope) = matches.first() { owner.document_id.clone_from(&scope.ownership.document_id); }
            }
        }
        if original_document != owner.document_id { owner.reading_stream_id = None; }
        let mut streams = Vec::new();
        for scope in scopes.iter().filter(|scope| scope.kind == StructureScopeKind::ReadingStream) {
            let ranges = &memberships[scope.id.as_str()];
            if inside(ranges, unit) { streams.push(scope); }
            else if ranges.iter().any(|range| range.start < range.end && range.start < unit.opening.end && unit.opening.start < range.end) {
                unit.trainable = false;
            }
        }
        if streams.len() > 1 { return Err("Gold source interval belongs to conflicting reading streams".into()); }
        if let Some(stream) = streams.first() {
            if annotated && owner.document_id != stream.ownership.document_id {
                return Err("Gold reading stream belongs to another document".into());
            }
            if owner.reading_stream_id.as_deref().is_some_and(|id| id != stream.id) {
                return Err("Gold node reading stream conflicts with its source membership".into());
            }
            owner.document_id.clone_from(&stream.ownership.document_id);
            owner.reading_stream_id = Some(stream.id.clone());
            annotated = true;
        } else if owner.reading_stream_id.as_deref().is_some_and(|id| memberships.contains_key(id)) {
            unit.trainable = false;
        }
        unit.trainable &= annotated;
        unit.ownership = Some(owner);
    }
    Ok(())
}
fn gold_openings<'a>(scopes: &'a [StructureScopeAnnotation], units: &[Unit], text: &str,
    document_id: &str) -> Result<GoldOpenings<'a>, String> {
    let coordinates = ScalarText::new(text);
    let by_id = scopes.iter().map(|scope| (scope.id.as_str(), scope)).collect::<BTreeMap<_, _>>();
    if by_id.len() != scopes.len() { return Err("Gold scope identities are duplicated".into()); }
    let mut depths = BTreeMap::new();
    for scope in scopes {
        let mut depth = 0usize;
        let mut parent = scope.parent_scope_id.as_deref();
        let mut seen = BTreeSet::from([scope.id.as_str()]);
        while let Some(id) = parent {
            if !seen.insert(id) { return Err("Gold scope containment cycle".into()); }
            depth += 1;
            if id == document_id { break; }
            parent = by_id.get(id).ok_or("Gold scope parent is absent")?.parent_scope_id.as_deref();
        }
        depths.insert(scope.id.as_str(), depth);
    }
    let mut result = GoldOpenings { at: BTreeMap::new(), unavailable: Vec::new() };
    for scope in scopes.iter().filter(|scope| matches!(scope.kind, StructureScopeKind::Document | StructureScopeKind::Quotation)
        && !(scope.kind == StructureScopeKind::Document && scope.id == document_id)) {
        let axis = if scope.kind == StructureScopeKind::Document { 0 } else { 1 };
        let mut inclusive = scope.ranges.clone();
        let mut owned = scope.owned_ranges.clone();
        ScalarRange::normalize_union(&mut inclusive);
        ScalarRange::normalize_union(&mut owned);
        for range in inclusive.iter().chain(&owned) {
            for endpoint in [range.start, range.end] {
                let endpoint = coordinates.scalar_at_utf16(endpoint).ok_or("Invalid gold scope boundary")?;
                if units.partition_point(|unit| unit.opening.start < endpoint).checked_sub(1)
                    .is_some_and(|index| endpoint < units[index].opening.end) {
                    result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[axis as usize],
                        start: Some(endpoint), reason: "scope_membership_boundary_has_no_aligned_source_unit" });
                }
            }
        }
        let Some(first) = inclusive.iter().min_by_key(|range| (range.start, range.start == range.end, range.end)) else {
            result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[axis as usize],
                start: None, reason: "scope_has_no_source_opening" });
            continue;
        };
        let opening = ScalarRange {
            start: coordinates.scalar_at_utf16(first.start).ok_or("Gold scope opening starts inside a source character")?,
            end: coordinates.scalar_at_utf16(first.end).ok_or("Gold scope opening ends inside a source character")?,
        };
        if opening.start >= opening.end {
            result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[axis as usize],
                start: Some(opening.start), reason: "scope_has_no_positive_source_opening" });
            continue;
        }
        let matched = units.binary_search_by_key(&opening.start, |unit| unit.opening.start).ok()
            .filter(|index| units[*index].opening.end <= opening.end).map(|index| (index, &units[index]));
        if let Some((position, _)) = matched {
            result.at.entry((position, axis)).or_default().push(scope);
        } else {
            result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[axis as usize], start: Some(opening.start),
                reason: "opening_has_no_aligned_source_unit" });
        }
    }
    for scope in scopes.iter().filter(|scope| scope.kind == StructureScopeKind::ReadingStream) {
        let ranges = if scope.owned_ranges.is_empty() { &scope.ranges } else { &scope.owned_ranges };
        let mut ranges = ranges.iter().map(|range| Ok(ScalarRange {
            start: coordinates.scalar_at_utf16(range.start).ok_or("Invalid gold stream start")?,
            end: coordinates.scalar_at_utf16(range.end).ok_or("Invalid gold stream end")?,
        })).collect::<Result<Vec<_>, String>>()?;
        ScalarRange::normalize_union(&mut ranges);
        if !ranges.iter().any(|range| range.start < range.end) {
            result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[2],
                start: ranges.first().map(|range| range.start), reason: "reading_stream_has_no_positive_source_membership" });
        }
        for range in ranges.iter().filter(|range| range.start < range.end) {
            for endpoint in [range.start, range.end] {
                if units.partition_point(|unit| unit.opening.start < endpoint).checked_sub(1)
                    .is_some_and(|index| endpoint < units[index].opening.end) {
                    result.unavailable.push(UnavailableScopeOpening { scope: scope.id.clone(), axis: AXES[2],
                        start: Some(endpoint), reason: "reading_stream_boundary_has_no_aligned_source_unit" });
                }
            }
        }
    }
    let mut missing = BTreeSet::new();
    for unit in units {
        if let Some(owner) = &unit.ownership {
            for (axis, identity) in [(0u8, Some(owner.document_id.as_str())), (1, owner.quotation_id.as_deref())] {
                let Some(identity) = identity else { continue; };
                if (axis == 0 && identity == document_id) || by_id.contains_key(identity) || !missing.insert((axis, identity)) { continue; }
                result.unavailable.push(UnavailableScopeOpening { scope: identity.to_owned(), axis: AXES[axis as usize],
                    start: None, reason: "missing_scope_annotation" });
            }
        }
    }
    for scopes in result.at.values_mut() {
        scopes.sort_by_key(|scope| (depths[scope.id.as_str()], scope.id.as_str()));
    }
    Ok(result)
}

fn choose(axis: u8, position: usize, slot: usize, unit: &Unit, query: &str, candidates: Vec<(String, &'static str)>,
    history: &BTreeMap<String, History>, expected: Option<&str>, model: Option<&crate::model::Model>,
    profile: Option<legal_structure_model::DetectionProfile>, rows: &mut Vec<ScopeTrainingRow>) -> Result<Option<String>, String> {
    if expected.is_some_and(|expected| !candidates.iter().any(|(id, _)| id == expected)) { return Ok(None); }
    let mut best = None::<(String, f64)>;
    for (identity, kind) in candidates {
        let features = scope_features(axis, kind, position, slot, unit, history.get(&identity), profile);
        if let Some(expected) = expected {
            rows.push(ScopeTrainingRow { unit: query.to_owned(), axis: AXES[axis as usize],
                candidate: format!("{kind}:{identity}"), features, positive: identity == expected });
        } else if let Some(model) = model {
            let score = model.scores(&features)[0];
            if !score.is_finite() { return Err("Scope model supplied a non-finite score".into()); }
            if best.as_ref().is_none_or(|(_, prior)| score > *prior) { best = Some((identity, score)); }
        }
    }
    if let Some(expected) = expected { Ok(Some(expected.to_owned())) }
    else { Ok(Some(best.ok_or("Scope model supplied no candidate")?.0)) }
}
fn opening_count(model: &crate::model::Model, axis: u8, unit: &Unit,
    profile: Option<legal_structure_model::DetectionProfile>) -> Result<usize, String> {
    let score = model.scores(&opening_features(axis, unit, profile))[0];
    if !score.is_finite() { return Err("Scope opening model supplied a non-finite count".into()); }
    let count = score.max(0.0).round();
    if count >= usize::MAX as f64 { return Err("Scope opening count is not representable".into()); }
    Ok(count as usize)
}

fn fresh_identity(base: String, occupied: &mut BTreeSet<String>) -> Result<String, String> {
    let mut identity = base.clone();
    let mut suffix = 0usize;
    while !occupied.insert(identity.clone()) {
        suffix = suffix.checked_add(1).ok_or("Scope identity suffix is not representable")?;
        identity = format!("{base}:{suffix}");
    }
    Ok(identity)
}

/// Scope declarations precede direct ownership; a containing scope need not own text itself.
fn select(nodes: &mut [StructureNode], text: &str, document_id: &str, source: &Table,
    fragments: &[legal_structure_model::SourceFragment], profile: Option<legal_structure_model::DetectionProfile>,
    gold_scopes: Option<&[StructureScopeAnnotation]>) -> Result<Selection, String> {
    let training = gold_scopes.is_some();
    let (model, opening_model) = if training { (None, None) } else {
        match (crate::model::scope_model(), crate::model::scope_opening_model()) {
            (None, None) => return Ok(Selection::default()),
            (Some(scope), Some(opening)) => (Some(scope), Some(opening)),
            _ => return Err("Scope inference requires both compatible scope-opening and scope-identity models".into()),
        }
    };
    let mut units = source_units(nodes, text, source, fragments)?;
    if let Some(scopes) = gold_scopes { gold_ownership(&mut units, scopes, text, document_id)?; }
    let gold = gold_scopes.map(|scopes| gold_openings(scopes, &units, text, document_id)).transpose()?;
    let mut result = Selection::default();
    if let Some(gold) = &gold { result.unavailable_openings = gold.unavailable.clone(); }
    let mut histories = BTreeMap::<(u8, String), BTreeMap<String, History>>::new();
    let mut stream_documents = BTreeMap::<String, String>::new();
    let mut opened_at = BTreeSet::new();
    let mut occupied = BTreeSet::from([document_id.to_owned()]);
    for node in nodes.iter() {
        occupied.insert(node.id.clone());
        if let Some(owner) = &node.ownership {
            occupied.insert(owner.document_id.clone());
            occupied.extend(owner.quotation_id.iter().chain(owner.reading_stream_id.iter()).cloned());
        }
    }
    for (position, unit) in units.iter().enumerate() {
        if training && !unit.trainable { continue; }
        let original = unit.ownership.clone();
        if training && original.is_none() { return Err(format!("Gold unit {} has no scope identity", unit.id)); }
        if opened_at.insert(unit.opening.start) {
            for axis in [0u8, 1] {
                let supplied = gold.as_ref().and_then(|gold| gold.at.get(&(position, axis)));
                let count = if training { supplied.map_or(0, Vec::len) }
                    else { opening_count(opening_model.unwrap(), axis, unit, profile)? };
                result.declarations.try_reserve(count).map_err(|_| "Scope opening count exceeds available declaration storage")?;
                for slot in 0..count {
                    let annotation = supplied.map(|scopes| scopes[slot]);
                    let id = if let Some(scope) = annotation { scope.id.clone() }
                        else { fresh_identity(format!("{}:{}:opening:{}", AXES[axis as usize], unit.id, slot + 1), &mut occupied)? };
                    let query = format!("{}:opening:{}:{}", unit.id, AXES[axis as usize], slot + 1);
                    let document = if axis == 0 { id.clone() } else {
                        let history = histories.entry((0, document_id.to_owned())).or_default();
                        let mut candidates = vec![(document_id.to_owned(), "root")];
                        candidates.extend(history.keys().filter(|id| id.as_str() != document_id).cloned().map(|id| (id, "resume")));
                        let expected = annotation.map(|scope| scope.ownership.document_id.as_str());
                        let chosen = choose(5, position, slot + 1, unit, &query, candidates, history,
                            expected, model, profile, &mut result.rows)?;
                        let Some(chosen) = chosen else {
                            result.unavailable.push(UnavailableScopeParent { scope: id.clone(), axis: AXES[5],
                                parent: expected.map(str::to_owned), unit: Some(query), reason: "owning_document_has_no_available_opening" });
                            continue;
                        };
                        chosen
                    };
                    let context = if axis == 0 { document_id.to_owned() } else { document.clone() };
                    let history = histories.entry((axis, context)).or_default();
                    // An explicit null root and a named transport-document parent are distinct.
                    let named_root = if axis == 0 { document_id } else { &document };
                    let mut candidates = vec![(String::new(), "root"), (named_root.to_owned(), "resume")];
                    candidates.extend(history.keys().filter(|prior| !prior.is_empty() && prior.as_str() != named_root)
                        .cloned().map(|prior| (prior, "resume")));
                    let expected = annotation.map(|scope| scope.parent_scope_id.as_deref().unwrap_or(""));
                    let parent = choose(axis + 3, position, slot + 1, unit, &query, candidates, history,
                        expected, model, profile, &mut result.rows)?;
                    let Some(parent) = parent else {
                        result.unavailable.push(UnavailableScopeParent { scope: id.clone(), axis: AXES[(axis + 3) as usize],
                            parent: expected.filter(|id| !id.is_empty()).map(str::to_owned), unit: Some(query),
                            reason: "parent_has_no_available_opening" });
                        continue;
                    };
                    let depth = if parent.is_empty() { 0 } else { history.get(&parent).map_or(1, |prior| prior.depth + 1) };
                    history.insert(id.clone(), History { position, opened: position, slot: slot + 1, depth,
                        marker: crate::numbering::parse(unit.marker.as_deref(), None), values: unit.values.clone() });
                    result.declarations.push(StructureScopeAnnotation {
                        id: id.clone(), kind: if axis == 0 { StructureScopeKind::Document } else { StructureScopeKind::Quotation },
                        ownership: StructureOwnership { document_id: document,
                            quotation_id: (axis == 1).then_some(id), reading_stream_id: None },
                        parent_scope_id: (!parent.is_empty()).then_some(parent),
                        ranges: vec![unit.opening], owned_ranges: Vec::new(), title_ranges: Vec::new(),
                        layout: None, attribution_ranges: Vec::new(), relationship: None,
                    });
                }
            }
        }
        let mut selected_document = document_id.to_owned();
        let mut selected_quote = None;
        let mut selected_stream = None;
        for axis in [0u8, 1, 2] {
            let root = if axis == 0 { document_id.to_owned() } else { String::new() };
            let context = if axis == 0 { document_id.to_owned() } else { selected_document.clone() };
            let history = histories.entry((axis, context)).or_default();
            let expected = original.as_ref().filter(|_| training).map(|scope| {
                if axis == 0 { scope.document_id.clone() }
                else if axis == 1 { scope.quotation_id.clone().unwrap_or_default() }
                else { scope.reading_stream_id.clone().unwrap_or_default() }
            });
            let mut candidates = vec![(root.clone(), "root")];
            if axis == 2 {
                let new_identity = if let Some(identity) = expected.as_ref().filter(|id| *id != &root && !history.contains_key(*id)) { identity.clone() }
                    else { fresh_identity(format!("reading_stream:{}", unit.id), &mut occupied)? };
                candidates.push((new_identity, "new"));
            }
            candidates.extend(history.iter().filter(|(id, _)| *id != &root)
                .map(|(id, prior)| (id.clone(), if prior.opened == position { "new" } else { "resume" })));
            if training && axis == 0 {
                if let Some(known) = original.as_ref().and_then(|scope| scope.reading_stream_id.as_ref()).and_then(|id| stream_documents.get(id)) {
                    if expected.as_ref() != Some(known) { return Err("One reading stream crosses document identities".into()); }
                    candidates.retain(|(id, _)| id == known);
                }
            }
            let chosen = choose(axis, position, 0, unit, &unit.id, candidates, history,
                expected.as_deref(), model, profile, &mut result.rows)?;
            let chosen = if !training && axis == 2 {
                original.as_ref().and_then(|scope| scope.reading_stream_id.clone())
                    .filter(|id| stream_documents.get(id).is_none_or(|document| document == &selected_document)).or(chosen)
            } else { chosen };
            let chosen = match chosen {
                Some(id) => id,
                None => {
                    let expected = expected.ok_or("Scope model supplied no available ownership")?;
                    result.unavailable.push(UnavailableScopeParent { scope: expected.clone(), axis: AXES[axis as usize],
                        parent: None, unit: Some(unit.id.clone()), reason: "ownership_has_no_available_opening" });
                    // Keep teacher-forced ownership for later targets without inventing an opening.
                    if axis == 0 { selected_document = expected; }
                    else if axis == 1 && !expected.is_empty() { selected_quote = Some(expected); }
                    else if axis == 2 && !expected.is_empty() { selected_stream = Some(expected); }
                    continue;
                }
            };
            if let Some(prior) = history.get_mut(&chosen) {
                prior.position = position;
                prior.values.clone_from(&unit.values);
                if let Some(marker) = crate::numbering::parse(unit.marker.as_deref(), prior.marker.as_ref()) {
                    prior.marker = Some(marker);
                }
            } else {
                history.insert(chosen.clone(), History { position, opened: position, slot: 0, depth: 0,
                    marker: crate::numbering::parse(unit.marker.as_deref(), None), values: unit.values.clone() });
            }
            if axis == 0 { selected_document = chosen; }
            else if axis == 1 && !chosen.is_empty() { selected_quote = Some(chosen); }
            else if axis == 2 && !chosen.is_empty() { selected_stream = Some(chosen); }
        }
        if let Some(stream) = &selected_stream {
            if stream_documents.get(stream).is_some_and(|document| document != &selected_document) { return Err("One reading stream crosses document identities".into()); }
            stream_documents.insert(stream.clone(), selected_document.clone());
        }
        if !training { result.assignments.push((unit.opening, StructureOwnership {
            document_id: selected_document, quotation_id: selected_quote, reading_stream_id: selected_stream })); }
    }
    Ok(result)
}

pub fn scope_opening_training_rows(mut structure: DocumentStructure, source: &Table) -> Result<ScopeOpeningTrainingRows, String> {
    if structure.offset_unit.as_ref() != "utf16" { return Err("Gold must use UTF-16 coordinates".into()); }
    structure.nodes = crate::membership::gold_fragments(&structure)?.0;
    let mut units = source_units(&structure.nodes, &structure.text, source, &structure.source_fragments)?;
    gold_ownership(&mut units, &structure.scope_annotations, &structure.text, &structure.document_id)?;
    let gold = gold_openings(&structure.scope_annotations, &units, &structure.text, &structure.document_id)?;
    let mut rows = Vec::new();
    let mut anchors = BTreeSet::new();
    for (position, unit) in units.iter().enumerate() {
        if !unit.trainable || !anchors.insert(unit.opening.start) { continue; }
        for axis in [0u8, 1] {
            if gold.unavailable.iter().any(|target| target.axis == AXES[axis as usize]
                && target.start.is_none_or(|start| unit.opening.start <= start && start < unit.opening.end)) { continue; }
            rows.push(ScopeOpeningTrainingRow { unit: unit.id.clone(), start: unit.opening.start,
                axis: AXES[axis as usize], features: opening_features(axis, unit, structure.profile),
                count: gold.at.get(&(position, axis)).map_or(0, Vec::len) });
        }
    }
    Ok(ScopeOpeningTrainingRows { schema: SCOPE_OPENING_SCHEMA, feature_names: opening_feature_names(),
        rows, unavailable_targets: gold.unavailable })
}

/// A container has one ownership only when all its exact contained units agree.
/// Table/row/cell parent links are structural containment, so they move together.
fn propagate_container_ownership(nodes: &mut [StructureNode]) {
    use legal_structure_model::NodeKind;
    let units = nodes
        .iter()
        .filter(|node| crate::outline::outline_unit(node))
        .filter_map(|node| {
            node.ownership.as_ref().map(|scope| {
                (
                    if node.ranges.is_empty() {
                        vec![node.range]
                    } else {
                        node.ranges.clone()
                    },
                    scope.document_id.clone(),
                    scope.quotation_id.clone(),
                )
            })
        })
        .collect::<Vec<_>>();
    let owners = nodes
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                NodeKind::Table | NodeKind::Footnote | NodeKind::Endnote | NodeKind::List
            )
        })
        .map(|node| {
            let memberships = if node.ranges.is_empty() {
                std::slice::from_ref(&node.range)
            } else {
                &node.ranges
            };
            let mut observed = std::collections::BTreeSet::new();
            for (ranges, document, quotation) in &units {
                if !ranges.is_empty()
                    && ranges.iter().all(|range| {
                        memberships
                            .iter()
                            .any(|part| part.start <= range.start && range.end <= part.end)
                    })
                {
                    observed.insert((document.clone(), quotation.clone()));
                }
            }
            let owner = match observed.len() {
                1 => observed.into_iter().next(),
                0 => node
                    .ownership
                    .as_ref()
                    .map(|scope| (scope.document_id.clone(), scope.quotation_id.clone())),
                // Conflicting contained facts cannot authorize overwriting descendants.
                _ => None,
            };
            (node.id.clone(), node.kind, owner)
        })
        .collect::<Vec<_>>();
    let parents = nodes
        .iter()
        .map(|node| (node.id.clone(), node.parent_id.clone()))
        .collect::<BTreeMap<_, _>>();
    for (id, kind, owner) in owners {
        let Some((document, quotation)) = owner else {
            continue;
        };
        for node in nodes.iter_mut() {
            let belongs = node.id == id
                || kind == NodeKind::Table
                    && matches!(node.kind, NodeKind::Row | NodeKind::Cell)
                    && {
                        let mut ancestor = node.parent_id.as_deref();
                        let mut seen = std::collections::BTreeSet::new();
                        let mut found = false;
                        while let Some(parent) = ancestor {
                            if !seen.insert(parent) {
                                break;
                            }
                            if parent == id {
                                found = true;
                                break;
                            }
                            ancestor = parents.get(parent).and_then(|parent| parent.as_deref());
                        }
                        found
                    };
            if belongs {
                if let Some(scope) = &mut node.ownership {
                    scope.document_id.clone_from(&document);
                    scope.quotation_id.clone_from(&quotation);
                }
            }
        }
    }
}

/// Keep existing logical units intact until selected ownership actually differs.
fn apply_ownership(nodes: &mut Vec<StructureNode>, assignments: &[(ScalarRange, StructureOwnership)],
    text: &str, source: &Table, fragments: &[legal_structure_model::SourceFragment]) -> Result<(), String> {
    let coordinates = ScalarText::new(text);
    let mut table = source.clone();
    crate::witness::bind_source_fragments(&mut table, text, fragments)?;
    table.rows.sort_by_key(|row| row.source_range.map(|range| (range.start, range.end)));
    let mut occupied = nodes.iter().map(|node| node.id.clone()).collect::<BTreeSet<_>>();
    let mut output = Vec::new();
    for node in nodes.iter() {
        if node.role.is_none() && !crate::outline::outline_unit(node) { output.push(node.clone()); continue; }
        let parts = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
        let mut groups = Vec::<(Vec<ScalarRange>, StructureOwnership)>::new();
        for part in parts {
            let from = assignments.partition_point(|(range, _)| range.end <= part.start);
            for (range, owner) in assignments[from..].iter().take_while(|(range, _)| range.start < part.end) {
                let range = ScalarRange { start: part.start.max(range.start), end: part.end.min(range.end) };
                if range.start >= range.end { continue; }
                if let Some((ranges, _)) = groups.last_mut().filter(|(_, prior)| prior == owner) {
                    ranges.push(range);
                } else { groups.push((vec![range], owner.clone())); }
            }
        }
        if groups.is_empty() { output.push(node.clone()); continue; }
        if groups.len() == 1 {
            let mut retained = node.clone();
            retained.ownership = Some(groups.remove(0).1);
            output.push(retained);
            continue;
        }
        for (ordinal, (mut ranges, owner)) in groups.into_iter().enumerate() {
            ScalarRange::normalize_union(&mut ranges);
            let mut split = node.clone();
            split.range = ScalarRange { start: ranges[0].start, end: ranges.last().unwrap().end };
            if ordinal > 0 { split.id = fresh_identity(format!("{}:scope@{}", node.id, split.range.start), &mut occupied)?; }
            split.ranges = ranges;
            split.ownership = Some(owner);
            split.extent.clear();
            split.rendered_range = None;
            split.rendered_ranges.clear();
            split.rendered_membership.clear();
            let contains = |range: ScalarRange| split.ranges.iter().any(|part| part.start <= range.start && range.end <= part.end);
            split.marker_range = node.marker_range.filter(|range| contains(*range));
            split.content_start = node.content_start.filter(|start| split.ranges.iter().any(|range| range.start <= *start && *start < range.end));
            split.heading_range = node.heading_range.and_then(|heading| {
                let mut intersections = split.ranges.iter().map(|part| ScalarRange {
                    start: heading.start.max(part.start), end: heading.end.min(part.end) })
                    .filter(|range| range.start < range.end).collect::<Vec<_>>();
                ScalarRange::normalize_union(&mut intersections);
                (intersections.len() == 1).then(|| intersections[0])
            });
            if split.marker_range.is_none() && (ordinal > 0 || node.marker_range.is_some()) {
                split.locator_kind = None;
                split.aliases = None;
                split.anchor = None;
                split.proof = None;
            }
            split.line_ids.clear();
            split.page_indexes.clear();
            for part in &split.ranges {
                let from = table.rows.partition_point(|row| row.source_range.is_none_or(|range| range.end <= part.start));
                for row in table.rows[from..].iter().take_while(|row| row.source_range.is_some_and(|range| range.start < part.end)) {
                    if !split.line_ids.contains(&row.id) { split.line_ids.push(row.id.clone()); }
                    if !split.page_indexes.contains(&row.page) { split.page_indexes.push(row.page); }
                }
            }
            if split.kind == legal_structure_model::NodeKind::Heading {
                split.label = crate::witness::membership_text(&split, &coordinates);
            } else if split.marker_range.is_none() && (ordinal > 0 || node.marker_range.is_some()) { split.label = None; }
            output.push(split);
        }
    }
    *nodes = output;
    Ok(())
}

pub fn infer_scopes(
    nodes: &mut Vec<StructureNode>,
    text: &str,
    document_id: &str,
    source: &Table,
    fragments: &[legal_structure_model::SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Result<Vec<StructureScopeAnnotation>, String> {
    let enabled = crate::model::scope_model().is_some();
    let mut selection = select(nodes, text, document_id, source, fragments, profile, None)?;
    if enabled {
        for (range, owner) in &selection.assignments {
            for (kind, id) in [(StructureScopeKind::Document, Some(&owner.document_id)),
                (StructureScopeKind::Quotation, owner.quotation_id.as_ref()),
                (StructureScopeKind::ReadingStream, owner.reading_stream_id.as_ref())] {
                let Some(id) = id else { continue; };
                let index = if let Some(index) = selection.declarations.iter().position(|scope| scope.id == *id) { index }
                else {
                    selection.declarations.push(StructureScopeAnnotation { id: id.clone(), kind,
                        ownership: StructureOwnership { document_id: owner.document_id.clone(),
                            quotation_id: (kind == StructureScopeKind::Quotation).then(|| id.clone()),
                            reading_stream_id: (kind == StructureScopeKind::ReadingStream).then(|| id.clone()) },
                        parent_scope_id: if kind == StructureScopeKind::Document {
                            (id != document_id).then(|| document_id.to_owned())
                        } else { Some(owner.document_id.clone()) },
                        ranges: Vec::new(), owned_ranges: Vec::new(), title_ranges: Vec::new(),
                        layout: None, attribution_ranges: Vec::new(), relationship: None });
                    selection.declarations.len() - 1
                };
                selection.declarations[index].ranges.push(*range);
                selection.declarations[index].owned_ranges.push(*range);
            }
        }
        apply_ownership(nodes, &selection.assignments, text, source, fragments)?;
        propagate_container_ownership(nodes);
    }
    Ok(selection.declarations)
}

pub fn scope_training_rows(
    mut structure: DocumentStructure,
    source: &Table,
) -> Result<ScopeTrainingRows, String> {
    if structure.offset_unit.as_ref() != "utf16" {
        return Err("Gold must use UTF-16 coordinates".into());
    }
    structure.nodes = crate::membership::gold_fragments(&structure)?.0;
    let selection = select(
        &mut structure.nodes,
        &structure.text,
        &structure.document_id,
        source,
        &structure.source_fragments,
        structure.profile,
        Some(&structure.scope_annotations),
    )?;
    Ok(ScopeTrainingRows {
        schema: SCOPE_SCHEMA,
        feature_names: feature_names(),
        rows: selection.rows,
        unavailable_parent_targets: selection.unavailable,
        unavailable_opening_targets: selection.unavailable_openings,
    })
}

/// Exact owned membership records resumptions without manufacturing titles or relations.
pub fn scope_annotations(
    nodes: &[StructureNode],
    document_id: &str,
    declarations: &[StructureScopeAnnotation],
) -> Vec<legal_structure_model::StructureScopeAnnotation> {
    use legal_structure_model::{StructureScopeAnnotation, StructureScopeKind};
    let mut scopes = declarations.iter().map(|scope| ((scope.kind, scope.id.clone()), scope.clone()))
        .collect::<BTreeMap<_, _>>();
    for node in nodes
        .iter()
        .filter(|node| node.kind != legal_structure_model::NodeKind::Page)
    {
        let Some(owner) = &node.ownership else {
            continue;
        };
        let mut identities = Vec::new();
        identities.push((
            StructureScopeKind::Document,
            owner.document_id.clone(),
            (owner.document_id != document_id).then(|| document_id.to_owned()),
        ));
        if let Some(id) = &owner.quotation_id {
            identities.push((
                StructureScopeKind::Quotation,
                id.clone(),
                Some(owner.document_id.clone()),
            ));
        }
        if let Some(id) = &owner.reading_stream_id {
            identities.push((
                StructureScopeKind::ReadingStream,
                id.clone(),
                Some(owner.document_id.clone()),
            ));
        }
        for (kind, id, parent_scope_id) in identities {
            let scope =
                scopes
                    .entry((kind, id.clone()))
                    .or_insert_with(|| StructureScopeAnnotation {
                        id,
                        kind,
                        ownership: StructureOwnership {
                            document_id: owner.document_id.clone(),
                            quotation_id: if kind == StructureScopeKind::Quotation {
                                owner.quotation_id.clone()
                            } else {
                                None
                            },
                            reading_stream_id: if kind == StructureScopeKind::ReadingStream {
                                owner.reading_stream_id.clone()
                            } else {
                                None
                            },
                        },
                        parent_scope_id,
                        ranges: Vec::new(),
                        owned_ranges: Vec::new(),
                        title_ranges: Vec::new(),
                        layout: None,
                        attribution_ranges: Vec::new(),
                        relationship: None,
                    });
            let parts = if node.ranges.is_empty() {
                std::slice::from_ref(&node.range)
            } else {
                &node.ranges
            };
            scope.ranges.extend(parts);
            // Container envelopes describe containment, not assigned source ownership.
            // Use the same exact units whose ownership the scope selector establishes.
            if node.role.is_some() || crate::outline::outline_unit(node) {
                scope.owned_ranges.extend(parts);
            }
        }
    }
    let merge = |ranges: &mut Vec<legal_structure_model::ScalarRange>| {
        ranges.sort_by_key(|range| (range.start, range.end));
        let mut exact = Vec::<legal_structure_model::ScalarRange>::new();
        for range in ranges.iter() {
            if let Some(prior) = exact.last_mut().filter(|prior| range.start <= prior.end) {
                prior.end = prior.end.max(range.end);
            } else { exact.push(*range); }
        }
        *ranges = exact;
    };
    for scope in scopes.values_mut() {
        merge(&mut scope.ranges);
        merge(&mut scope.owned_ranges);
    }
    // The implicit root remains addressable even when every source unit belongs
    // to a constituent. Its inclusive extent never becomes its own membership.
    let document_ranges = scopes.values().filter(|scope| scope.kind == StructureScopeKind::Document)
        .flat_map(|scope| scope.ranges.iter().copied()).collect::<Vec<_>>();
    if !document_ranges.is_empty() {
        let root_scope = scopes.entry((StructureScopeKind::Document, document_id.to_owned()))
            .or_insert_with(|| StructureScopeAnnotation {
                id: document_id.to_owned(), kind: StructureScopeKind::Document,
                ownership: StructureOwnership { document_id: document_id.to_owned(), quotation_id: None, reading_stream_id: None },
                parent_scope_id: None, ranges: Vec::new(), owned_ranges: Vec::new(),
                title_ranges: Vec::new(), layout: None, attribution_ranges: Vec::new(), relationship: None,
            });
        root_scope.ranges = document_ranges;
        merge(&mut root_scope.ranges);
    }
    let by_id = scopes.keys().map(|key| (key.1.clone(), key.clone())).collect::<BTreeMap<_, _>>();
    let descendants = scopes.values().map(|scope|
        (scope.parent_scope_id.clone(), scope.ranges.clone())).collect::<Vec<_>>();
    for (mut parent, ranges) in descendants {
        let mut seen = std::collections::BTreeSet::new();
        while let Some(id) = parent {
            if !seen.insert(id.clone()) { break; }
            let Some(key) = by_id.get(&id) else { break; };
            let ancestor = scopes.get_mut(key).expect("indexed scope exists");
            ancestor.ranges.extend(ranges.iter().copied());
            parent = ancestor.parent_scope_id.clone();
        }
    }
    for scope in scopes.values_mut() { merge(&mut scope.ranges); }
    scopes.into_values().collect()
}
