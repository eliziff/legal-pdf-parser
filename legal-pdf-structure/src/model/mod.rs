//! The structure stage's contract: the data its stages read and write, and nothing that decides. Each line's slot
//! (`slots`), what a line is and the rule that decided it (`roles`), and the regions a line may be in (`regions`).
//! A change here rebuilds every stage: keep it small and stable.

mod decided;
pub mod evidence;
mod paragraphs;
pub mod regions;
mod runs;
mod roles;
mod slots;

pub use decided::*;
pub use evidence::*;
pub use paragraphs::*;
pub use roles::*;
pub use runs::*;
pub use slots::*;

/// The origin of the transition parser's nodes.
pub const TRANSITION_ORIGIN: &str = "legal-pdf-transition.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub profile: Option<legal_structure_model::DetectionProfile>,
    pub source_sha256: String,
    /// The page labels the PDF declares (`/PageLabels`), by page index; empty where it has none.
    pub page_labels: Vec<Option<String>>,
}

impl StructureIdentity {
    /// The document's id: its source's hash, named.
    pub fn document_id(&self) -> String {
        format!("doc-{}", &self.source_sha256[..20])
    }
}
