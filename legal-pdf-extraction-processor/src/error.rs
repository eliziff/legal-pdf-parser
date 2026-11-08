#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] legal_pdf_core::Error),
    #[error("{0}")]
    Message(String),
    #[error("PDF extraction failed: {0}")]
    Pdf(#[from] pdf_inspector::PdfError),
    #[error("PDF parsing failed: {0}")]
    Lopdf(#[from] lopdf::Error),
}

/// A PDF the reader cannot open is password-protected or corrupt.
impl From<Error> for legal_pdf_core::Error {
    fn from(error: Error) -> Self {
        use legal_pdf_core::FailureKind;
        match error {
            Error::Core(error) => error,
            Error::Message(message) => Self::Message(message),
            Error::Pdf(pdf_inspector::PdfError::Encrypted) => Self::Message(error.to_string()).failure(FailureKind::Encrypted),
            Error::Pdf(_) | Error::Lopdf(_) => Self::Message(error.to_string()).failure(FailureKind::Corrupt),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
