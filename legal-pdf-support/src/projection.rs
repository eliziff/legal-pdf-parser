//! The parsed PDF as applications read it: the projected document (`document`), lookups by locator
//! (`lookup`).

use legal_pdf_core::model::{
    Page, PdfExtractionMetadata, ScalarRange, PARSER_VERSION,
};
use legal_structure_model::{
    serialization_sha256, utf16_len,
    ScalarText,
};
use legal_structure_model::{
    DocumentStructure, NodeKind, StructureNode,
};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

mod document;
pub use document::*;
mod lookup;
pub use lookup::*;
mod passage;
pub use passage::*;

const MAX_UNITS: usize = 20;
const MAX_CONTEXT: usize = 2;
const MAX_RETURN_CHARS: usize = 60_000;
const LOCATOR_KINDS: [&str; 15] = [
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
    "rule",
    "part",
    "division",
];
