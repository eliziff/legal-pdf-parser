//! PyO3 Python bindings for pdf-inspector.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashSet;

use crate::detector::PdfType;
use crate::types::ItemType;

// ---------------------------------------------------------------------------
// Result wrapper
// ---------------------------------------------------------------------------

/// Result of processing a PDF file.
#[pyclass(name = "PdfResult")]
#[derive(Clone)]
pub struct PyPdfResult {
    /// The detected PDF type: "text_based", "scanned", "image_based", or "mixed".
    #[pyo3(get)]
    pub pdf_type: String,
    /// Markdown output (None if detect-only or scanned PDF).
    #[pyo3(get)]
    pub markdown: Option<String>,
    /// Total number of pages.
    #[pyo3(get)]
    pub page_count: u32,
    /// Processing time in milliseconds.
    #[pyo3(get)]
    pub processing_time_ms: u64,
    /// 1-indexed page numbers that need OCR.
    #[pyo3(get)]
    pub pages_needing_ocr: Vec<u32>,
    /// Title from PDF metadata.
    #[pyo3(get)]
    pub title: Option<String>,
    /// Detection confidence (0.0-1.0).
    #[pyo3(get)]
    pub confidence: f32,
    /// Whether the layout is complex (tables/columns detected).
    #[pyo3(get)]
    pub is_complex_layout: bool,
    /// Pages with tables detected.
    #[pyo3(get)]
    pub pages_with_tables: Vec<u32>,
    /// Pages with multi-column layout.
    #[pyo3(get)]
    pub pages_with_columns: Vec<u32>,
    /// Whether encoding issues were detected.
    #[pyo3(get)]
    pub has_encoding_issues: bool,
}

#[pymethods]
impl PyPdfResult {
    fn __repr__(&self) -> String {
        format!(
            "PdfResult(pdf_type='{}', pages={}, confidence={:.2})",
            self.pdf_type, self.page_count, self.confidence
        )
    }
}

fn pdf_type_str(t: PdfType) -> String {
    match t {
        PdfType::TextBased => "text_based".into(),
        PdfType::Scanned => "scanned".into(),
        PdfType::ImageBased => "image_based".into(),
        PdfType::Mixed => "mixed".into(),
    }
}

fn to_py_result(r: crate::PdfProcessResult) -> PyPdfResult {
    PyPdfResult {
        pdf_type: pdf_type_str(r.pdf_type),
        markdown: r.markdown,
        page_count: r.page_count,
        processing_time_ms: r.processing_time_ms,
        pages_needing_ocr: r.pages_needing_ocr,
        title: r.title,
        confidence: r.confidence,
        is_complex_layout: r.layout.is_complex,
        pages_with_tables: r.layout.pages_with_tables,
        pages_with_columns: r.layout.pages_with_columns,
        has_encoding_issues: r.has_encoding_issues,
    }
}

fn to_py_err(e: crate::PdfError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Text item wrapper
// ---------------------------------------------------------------------------

/// A positioned text item extracted from a PDF.
#[pyclass(name = "TextItem")]
#[derive(Clone)]
pub struct PyTextItem {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub width: f32,
    #[pyo3(get)]
    pub height: f32,
    #[pyo3(get)]
    pub font: String,
    #[pyo3(get)]
    pub font_size: f32,
    #[pyo3(get)]
    pub page: u32,
    #[pyo3(get)]
    pub is_bold: bool,
    #[pyo3(get)]
    pub is_italic: bool,
    #[pyo3(get)]
    pub item_type: String,
}

#[pymethods]
impl PyTextItem {
    fn __repr__(&self) -> String {
        format!(
            "TextItem(text='{}', page={}, x={:.1}, y={:.1})",
            self.text.chars().take(40).collect::<String>(),
            self.page,
            self.x,
            self.y,
        )
    }
}

fn item_type_str(t: &ItemType) -> String {
    match t {
        ItemType::Text => "text".into(),
        ItemType::Image => "image".into(),
        ItemType::Link(url) => format!("link:{url}"),
        ItemType::FormField => "form_field".into(),
    }
}

// ---------------------------------------------------------------------------
// Public Python API
// ---------------------------------------------------------------------------

/// Process a PDF file: detect type, extract text, and convert to Markdown.
///
/// Args:
///     path: Path to the PDF file.
///     pages: Optional list of 1-indexed page numbers to process.
///
/// Returns:
///     PdfResult with markdown, pdf_type, and metadata.
#[pyfunction]
#[pyo3(signature = (path, pages=None))]
fn process_pdf(path: &str, pages: Option<Vec<u32>>) -> PyResult<PyPdfResult> {
    let mut opts = crate::PdfOptions::new();
    if let Some(p) = pages {
        opts = opts.pages(p);
    }
    let result = crate::process_pdf_with_options(path, opts).map_err(to_py_err)?;
    Ok(to_py_result(result))
}

/// Process a PDF from bytes in memory.
///
/// Args:
///     data: PDF file contents as bytes.
///     pages: Optional list of 1-indexed page numbers to process.
///
/// Returns:
///     PdfResult with markdown, pdf_type, and metadata.
#[pyfunction]
#[pyo3(signature = (data, pages=None))]
fn process_pdf_bytes(data: &[u8], pages: Option<Vec<u32>>) -> PyResult<PyPdfResult> {
    let mut opts = crate::PdfOptions::new();
    if let Some(p) = pages {
        opts = opts.pages(p);
    }
    let result = crate::process_pdf_mem_with_options(data, opts).map_err(to_py_err)?;
    Ok(to_py_result(result))
}

/// Fast detection only — no text extraction or markdown.
///
/// Args:
///     path: Path to the PDF file.
///
/// Returns:
///     PdfResult with pdf_type and metadata (markdown will be None).
#[pyfunction]
fn detect_pdf(path: &str) -> PyResult<PyPdfResult> {
    let result = crate::detect_pdf(path).map_err(to_py_err)?;
    Ok(to_py_result(result))
}

/// Fast detection from bytes — no text extraction or markdown.
#[pyfunction]
fn detect_pdf_bytes(data: &[u8]) -> PyResult<PyPdfResult> {
    let result = crate::detect_pdf_mem(data).map_err(to_py_err)?;
    Ok(to_py_result(result))
}

/// Extract plain text from a PDF file.
///
/// Args:
///     path: Path to the PDF file.
///
/// Returns:
///     Extracted text as a string.
#[pyfunction]
fn extract_text(path: &str) -> PyResult<String> {
    crate::extract_text(path).map_err(to_py_err)
}

/// Extract text with position information.
///
/// Args:
///     path: Path to the PDF file.
///     pages: Optional list of 1-indexed page numbers.
///
/// Returns:
///     List of TextItem objects with text, position, font info.
#[pyfunction]
#[pyo3(signature = (path, pages=None))]
fn extract_text_with_positions(path: &str, pages: Option<Vec<u32>>) -> PyResult<Vec<PyTextItem>> {
    let items = match pages {
        Some(p) => {
            let page_set: HashSet<u32> = p.into_iter().collect();
            crate::extract_text_with_positions_pages(path, Some(&page_set)).map_err(to_py_err)?
        }
        None => crate::extract_text_with_positions(path).map_err(to_py_err)?,
    };

    Ok(items
        .into_iter()
        .map(|item| PyTextItem {
            text: item.text,
            x: item.x,
            y: item.y,
            width: item.width,
            height: item.height,
            font: item.font,
            font_size: item.font_size,
            page: item.page,
            is_bold: item.is_bold,
            is_italic: item.is_italic,
            item_type: item_type_str(&item.item_type),
        })
        .collect())
}

/// Python module definition.
#[pymodule]
fn pdf_inspector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPdfResult>()?;
    m.add_class::<PyTextItem>()?;
    m.add_function(wrap_pyfunction!(process_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(process_pdf_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(detect_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(detect_pdf_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text_with_positions, m)?)?;
    Ok(())
}
