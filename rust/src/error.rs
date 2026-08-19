use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("PDF extraction failed: {0}")]
    Pdf(#[from] pdf_inspector::PdfError),
    #[error("PDF parsing failed: {0}")]
    Lopdf(#[from] lopdf::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
