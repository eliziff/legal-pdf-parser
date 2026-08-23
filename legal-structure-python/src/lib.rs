use legal_structure_core::{
    a2aj_document_structure, project_document_structure, A2ajInput, A2ajSourceKind, SourceDoc,
    SourceDocBlock, SourceDocKind, SourceDocOrigin,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::time::Instant;

const COMPILER: &str = "legal-structure";

fn required_string(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    payload
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{key} is required")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{key} must be a string")))
}

fn optional_string(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        Ok(None)
    } else {
        value
            .extract()
            .map(Some)
            .map_err(|_| PyValueError::new_err(format!("{key} must be a string or None")))
    }
}

fn section_map(payload: &Bound<'_, PyDict>) -> PyResult<Option<Vec<(String, String)>>> {
    let Some(value) = payload.get_item("sectionMap")? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let map = value
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("sectionMap must map labels to strings"))?;
    map.iter()
        .map(|(label, text)| {
            Ok((
                label
                    .extract()
                    .map_err(|_| PyValueError::new_err("sectionMap must map labels to strings"))?,
                text.extract()
                    .map_err(|_| PyValueError::new_err("sectionMap must map labels to strings"))?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn kind_name(kind: SourceDocKind) -> &'static str {
    match kind {
        SourceDocKind::Paragraph => "paragraph",
        SourceDocKind::Page => "page",
        SourceDocKind::Section => "section",
        SourceDocKind::Footnote => "footnote",
        SourceDocKind::Table => "table",
        SourceDocKind::Row => "row",
        SourceDocKind::Cell => "cell",
    }
}

fn origin_name(origin: SourceDocOrigin) -> &'static str {
    match origin {
        SourceDocOrigin::Native => "native",
        SourceDocOrigin::Heuristic => "heuristic",
    }
}

fn list<'py, T>(py: Python<'py>, values: T) -> PyResult<Bound<'py, PyList>>
where
    T: IntoIterator,
    T::Item: IntoPyObject<'py>,
    T::IntoIter: ExactSizeIterator,
{
    PyList::new(py, values)
}

fn block_dict<'py>(py: Python<'py>, block: &SourceDocBlock) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", kind_name(block.kind))?;
    value.set_item("label", &block.label)?;
    value.set_item(
        "aliases",
        list(py, block.aliases.iter().map(String::as_str))?,
    )?;
    value.set_item("start", block.start)?;
    value.set_item("end", block.end)?;
    value.set_item("origin", origin_name(block.origin))?;
    Ok(value)
}

fn text_slice(text: &[u16], start: usize, end: usize) -> PyResult<String> {
    let units = text
        .get(start..end)
        .ok_or_else(|| PyRuntimeError::new_err("SourceDoc block is outside its text"))?;
    String::from_utf16(units)
        .map_err(|_| PyRuntimeError::new_err("SourceDoc block splits a Unicode character"))
}

fn numbered<'py>(py: Python<'py>, label: &str) -> PyResult<Bound<'py, PyAny>> {
    let value = label
        .strip_prefix("par")
        .or_else(|| label.strip_prefix("page"))
        .unwrap_or(label);
    if let Ok(number) = value.parse::<u64>() {
        number.into_bound_py_any(py)
    } else if let Ok(number) = value.parse::<f64>() {
        if number.is_finite() {
            number.into_bound_py_any(py)
        } else {
            label.into_bound_py_any(py)
        }
    } else {
        label.into_bound_py_any(py)
    }
}

fn rendition<'py>(
    py: Python<'py>,
    document: &SourceDoc,
    selected: &[&SourceDocBlock],
    kind: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let text = document.text.encode_utf16().collect::<Vec<_>>();
    let mut ordered = selected.to_vec();
    ordered.sort_by_key(|block| block.start);
    let segments = PyList::empty(py);
    let mut cursor = 0;
    for block in ordered {
        if block.start > cursor {
            let segment = PyDict::new(py);
            segment.set_item("kind", "text")?;
            segment.set_item("text", text_slice(&text, cursor, block.start)?)?;
            segments.append(segment)?;
        }
        let segment = PyDict::new(py);
        segment.set_item("kind", kind_name(block.kind))?;
        segment.set_item("label", &block.label)?;
        segment.set_item(
            "aliases",
            list(py, block.aliases.iter().map(String::as_str))?,
        )?;
        segment.set_item("origin", origin_name(block.origin))?;
        segment.set_item("text", text_slice(&text, block.start, block.end)?)?;
        segments.append(segment)?;
        cursor = cursor.max(block.end);
    }
    if cursor < text.len() || segments.is_empty() {
        let segment = PyDict::new(py);
        segment.set_item("kind", "text")?;
        segment.set_item("text", text_slice(&text, cursor, text.len())?)?;
        segments.append(segment)?;
    }
    let value = PyDict::new(py);
    value.set_item("kind", kind)?;
    value.set_item("segments", segments)?;
    Ok(value)
}

fn receipt<'py>(py: Python<'py>, document: &SourceDoc) -> PyResult<Bound<'py, PyDict>> {
    let top = |kind| {
        document
            .blocks
            .iter()
            .filter(move |block| block.kind == kind && block.parent_label.is_none())
            .collect::<Vec<_>>()
    };
    let paragraphs = top(SourceDocKind::Paragraph);
    let pages = top(SourceDocKind::Page);
    let sections = top(SourceDocKind::Section);
    let (selected, kind) = if !paragraphs.is_empty() {
        (&paragraphs, "paragraphs")
    } else if !pages.is_empty() {
        (&pages, "pages")
    } else if !sections.is_empty() {
        (&sections, "sections")
    } else {
        (&sections, "none")
    };

    let blocks = PyDict::new(py);
    for (name, source) in [
        ("paragraph", &paragraphs),
        ("page", &pages),
        ("section", &sections),
    ] {
        let values = PyList::empty(py);
        for block in source {
            values.append(block_dict(py, block)?)?;
        }
        blocks.set_item(name, values)?;
    }

    let summary = PyDict::new(py);
    summary.set_item("kind", kind)?;
    summary.set_item("count", selected.len())?;
    let first = match selected.first() {
        Some(block) => numbered(py, &block.label)?,
        None => py.None().into_bound(py),
    };
    let last = match selected.last() {
        Some(block) => numbered(py, &block.label)?,
        None => py.None().into_bound(py),
    };
    summary.set_item("first", first)?;
    summary.set_item("last", last)?;
    let span = if selected.len() > 1 {
        let value = (selected.last().unwrap().start - selected[0].start) as f64
            / document.text.encode_utf16().count().max(1) as f64;
        (value * 10_000.0).round() / 10_000.0
    } else if selected.is_empty() {
        0.0
    } else {
        1.0
    };
    summary.set_item("span", span)?;

    let value = PyDict::new(py);
    value.set_item("compiler", COMPILER)?;
    value.set_item("summary", summary)?;
    value.set_item("rendition", rendition(py, document, selected, kind)?)?;
    value.set_item("blocks", blocks)?;
    Ok(value)
}

#[pyfunction]
fn compile_document<'py>(py: Python<'py>, payload: &Bound<'py, PyDict>) -> PyResult<Py<PyDict>> {
    let started = Instant::now();
    let doc_type = match required_string(payload, "docType")?.as_str() {
        "cases" => A2ajSourceKind::Cases,
        "laws" => A2ajSourceKind::Laws,
        _ => return Err(PyValueError::new_err("docType must be cases or laws")),
    };
    let citation = required_string(payload, "citation")?;
    let text = required_string(payload, "text")?;
    let sections = section_map(payload)?;
    let input = A2ajInput {
        citation,
        source_kind: doc_type,
        text: if sections.is_some() {
            String::new()
        } else {
            text
        },
        id: None,
        url: None,
        dataset: optional_string(payload, "dataset")?,
        name: optional_string(payload, "name")?,
        alternate_citation: optional_string(payload, "alternateCitation")?,
        section_map: sections,
        excerpt_of: None,
    };
    let document = py
        .detach(|| a2aj_document_structure(input).map(project_document_structure))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let value = receipt(py, &document)?;
    let elapsed_ms = (started.elapsed().as_secs_f64() * 1_000_000.0).round() / 1_000.0;
    value.set_item("elapsedMs", elapsed_ms)?;
    Ok(value.unbind())
}

#[pymodule]
fn legal_structure(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_document, module)?)
}
