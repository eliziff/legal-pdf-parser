//! The parsed PDF as applications read it: the projected document (`document`), lookups by locator
//! (`lookup`).

use legal_pdf_core::model::{
    Footnote, Line, Page, PdfExtractionMetadata, ScalarRange, PARSER_VERSION,
};
use legal_structure_model::{
    document_fingerprint, serialization_sha256, utf16_len, DigestWriter, DocumentFingerprint,
    ScalarText,
};
use legal_structure_model::{
    Derivation, DocumentStructure, NodeKind, StructureAnalysis, StructureNode,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use unicode_normalization::UnicodeNormalization;

mod document;
pub use document::*;
mod lookup;
pub use lookup::*;

const MAX_UNITS: usize = 20;
const MAX_CONTEXT: usize = 2;
const MAX_RETURN_CHARS: usize = 60_000;
const LOCATOR_KINDS: [&str; 12] = [
    "page",
    "line",
    "paragraph",
    "footnote",
    "section",
    "subsection",
    "provision_paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "schedule",
    "article",
];
