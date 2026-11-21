//! Where a constituent document of a packet starts.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Start {
    pub line_id: String,
    pub kind: String,
}

impl Start {
    pub fn new(line_id: String, kind: &str) -> Self {
        Self {
            line_id,
            kind: kind.to_owned(),
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }
}
