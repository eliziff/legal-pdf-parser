use legal_structure_core::{
    a2aj_document_structure, project_document_structure, A2ajInput, A2ajSourceKind, ScalarText,
    SourceDoc, SourceDocBlock, SourceDocKind, SourceDocOrigin,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::IntoPyObjectExt;

fn kind_name(kind: SourceDocKind) -> &'static str {
    match kind {
        SourceDocKind::Paragraph => "paragraph",
        SourceDocKind::Page => "page",
        SourceDocKind::Section => "section",
        _ => unreachable!("primary blocks are paragraphs, pages, or sections"),
    }
}

fn origin_name(origin: SourceDocOrigin) -> &'static str {
    match origin {
        SourceDocOrigin::Native => "native",
        SourceDocOrigin::Heuristic => "heuristic",
    }
}

fn parse_kind(value: &str) -> PyResult<SourceDocKind> {
    match value {
        "paragraph" => Ok(SourceDocKind::Paragraph),
        "page" => Ok(SourceDocKind::Page),
        "section" => Ok(SourceDocKind::Section),
        _ => Err(PyValueError::new_err(
            "kind must be paragraph, page, or section",
        )),
    }
}

fn text_slice(text: &[u16], start: usize, end: usize) -> PyResult<String> {
    let units = text
        .get(start..end)
        .ok_or_else(|| PyRuntimeError::new_err("SourceDoc block is outside its text"))?;
    String::from_utf16(units)
        .map_err(|_| PyRuntimeError::new_err("SourceDoc block splits a Unicode character"))
}

fn numbered<'py>(py: Python<'py>, label: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let Some(label) = label else {
        return Ok(py.None().into_bound(py));
    };
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

#[pyclass(frozen)]
struct Document {
    source: SourceDoc,
}

impl Document {
    fn primary(&self) -> Option<(&'static str, SourceDocKind)> {
        [
            ("paragraphs", SourceDocKind::Paragraph),
            ("pages", SourceDocKind::Page),
            ("sections", SourceDocKind::Section),
        ]
        .into_iter()
        .find(|(_, kind)| {
            self.source
                .blocks
                .iter()
                .any(|block| block.kind == *kind && block.parent_label.is_none())
        })
    }

    fn top_blocks(&self, kind: SourceDocKind) -> impl Iterator<Item = &SourceDocBlock> {
        self.source
            .blocks
            .iter()
            .filter(move |block| block.kind == kind && block.parent_label.is_none())
    }
}

#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (
        doc_type,
        citation,
        text,
        *,
        alternate_citation=None,
        dataset=None,
        name=None,
        section_map=None
    ))]
    fn new(
        py: Python<'_>,
        doc_type: &str,
        citation: String,
        text: String,
        alternate_citation: Option<String>,
        dataset: Option<String>,
        name: Option<String>,
        section_map: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let source_kind = match doc_type {
            "cases" => A2ajSourceKind::Cases,
            "laws" => A2ajSourceKind::Laws,
            _ => return Err(PyValueError::new_err("doc_type must be cases or laws")),
        };
        let mut input = A2ajInput::new(
            citation,
            source_kind,
            if section_map.is_some() {
                String::new()
            } else {
                text
            },
        );
        input.dataset = dataset;
        input.name = name;
        input.alternate_citation = alternate_citation;
        input.section_map = section_map;
        let source = py
            .detach(|| a2aj_document_structure(input).map(project_document_structure))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { source })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.primary().map_or("none", |(name, _)| name)
    }

    #[getter]
    fn count(&self) -> usize {
        self.primary()
            .map(|(_, kind)| self.top_blocks(kind).count())
            .unwrap_or_default()
    }

    #[getter]
    fn first<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numbered(
            py,
            self.primary()
                .and_then(|(_, kind)| self.top_blocks(kind).next())
                .map(|block| block.label.as_str()),
        )
    }

    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numbered(
            py,
            self.primary()
                .and_then(|(_, kind)| self.top_blocks(kind).last())
                .map(|block| block.label.as_str()),
        )
    }

    #[getter]
    fn span(&self) -> f64 {
        let Some((_, kind)) = self.primary() else {
            return 0.0;
        };
        let mut blocks = self.top_blocks(kind);
        let Some(first) = blocks.next() else {
            return 0.0;
        };
        let Some(last) = blocks.last() else {
            return 1.0;
        };
        let value = (last.start - first.start) as f64
            / self.source.text.encode_utf16().count().max(1) as f64;
        (value * 10_000.0).round() / 10_000.0
    }

    fn blocks(&self, kind: &str) -> PyResult<Vec<(String, Vec<String>, usize, usize)>> {
        let kind = parse_kind(kind)?;
        let coordinates = ScalarText::new(&self.source.text);
        self.top_blocks(kind)
            .map(|block| {
                Ok((
                    block.label.clone(),
                    block.aliases.clone(),
                    coordinates.scalar_at_utf16(block.start).ok_or_else(|| {
                        PyRuntimeError::new_err("SourceDoc block starts inside a Unicode character")
                    })?,
                    coordinates.scalar_at_utf16(block.end).ok_or_else(|| {
                        PyRuntimeError::new_err("SourceDoc block ends inside a Unicode character")
                    })?,
                ))
            })
            .collect()
    }

    fn segments(&self) -> PyResult<Vec<(String, Option<String>, Option<String>, String)>> {
        let text = self.source.text.encode_utf16().collect::<Vec<_>>();
        let mut blocks = self
            .primary()
            .map(|(_, kind)| self.top_blocks(kind).collect::<Vec<_>>())
            .unwrap_or_default();
        blocks.sort_by_key(|block| block.start);
        let mut segments = Vec::with_capacity(blocks.len() * 2 + 1);
        let mut cursor = 0;
        for block in blocks {
            if block.start > cursor {
                segments.push((
                    "text".to_owned(),
                    None,
                    None,
                    text_slice(&text, cursor, block.start)?,
                ));
            }
            segments.push((
                kind_name(block.kind).to_owned(),
                Some(block.label.clone()),
                Some(origin_name(block.origin).to_owned()),
                text_slice(&text, block.start, block.end)?,
            ));
            cursor = cursor.max(block.end);
        }
        if cursor < text.len() || segments.is_empty() {
            segments.push((
                "text".to_owned(),
                None,
                None,
                text_slice(&text, cursor, text.len())?,
            ));
        }
        Ok(segments)
    }
}

#[pymodule]
fn legal_structure(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Document>()
}
