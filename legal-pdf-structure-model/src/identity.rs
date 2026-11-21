//! The source a parse reads: its hash, its profile, the page labels it declares.

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
