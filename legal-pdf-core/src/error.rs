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
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A failure of a kind the parser's caller acts on, with what went wrong.
    #[error("{}", kind.message())]
    Failure { kind: FailureKind, detail: String },
}

/// What kind of failure stopped a PDF's parse: the source (password-protected, unreadable, or changed while it was
/// read), the runtime of recognition or of the layout model, or the parser itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    Encrypted,
    Corrupt,
    SourceChanged,
    OcrRuntime,
    LayoutRuntime,
    Parser,
}

impl FailureKind {
    /// The failure as a reader is told it.
    pub fn message(self) -> &'static str {
        match self {
            Self::Encrypted => "PDF is password-protected. Remove its password, then upload it again.",
            Self::Corrupt => "PDF is invalid or corrupt",
            Self::SourceChanged => "PDF source changed after preparation began",
            Self::OcrRuntime => "Indra OCR could not start; check its local runtime assets",
            Self::LayoutRuntime => "PDF layout analysis could not start; check its local runtime assets",
            Self::Parser => "PDF structural parser failed",
        }
    }
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// This error as a failure of `kind`, keeping what went wrong.
    pub fn failure(self, kind: FailureKind) -> Self {
        match self {
            Self::Failure { .. } => self,
            error => Self::Failure { kind, detail: error.to_string() },
        }
    }

    /// The kind of failure this is: the parser's own where no stage named another.
    pub fn kind(&self) -> FailureKind {
        match self {
            Self::Failure { kind, .. } => *kind,
            _ => FailureKind::Parser,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
