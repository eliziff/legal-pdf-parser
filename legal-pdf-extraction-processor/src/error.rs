#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] legal_pdf_core::Error),
    #[error("{0}")]
    Message(String),
    #[error("PDF extraction failed: {0}")]
    Pdf(#[from] pdf_inspector_loader::PdfError),
    #[error("PDF parsing failed: {0}")]
    Lopdf(#[from] lopdf::Error),
}

impl From<Error> for legal_pdf_core::Error {
    fn from(error: Error) -> Self {
        Self::Message(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
