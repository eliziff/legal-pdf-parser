//! The parsed PDF as applications read it: the projected document (`document`), lookups by locator
//! (`lookup`), and its rendered text (`rendered`).

pub use legal_structure_model::parse_ordinal;
use legal_pdf_core::model::{
    Footnote, Line, Page, Paragraph, PdfExtractionMetadata, ScalarRange, PARSER_VERSION,
};
use legal_structure_model::{Derivation, DocumentStructure, NodeKind, StructureNode};
use legal_structure_model::{
    document_fingerprint, normalize_compact_numbered_section_locator, serialization_sha256, utf16_len,
    DigestWriter, DocumentFingerprint, ScalarText,
};
use regex::Regex;
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
mod rendered;
pub(crate) use rendered::*;

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
