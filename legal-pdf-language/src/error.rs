use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] legal_pdf_core::Error),
    #[error("{0}")]
    Message(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Core(legal_pdf_core::Error::io(path, source))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
