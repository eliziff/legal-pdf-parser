mod adapters;
mod codex;
mod contract;
mod deterministic_citations;
mod docx;
mod engine;
mod error;
mod grammar_tables;
mod grammar_word;
#[cfg(feature = "kraken")]
mod kraken;
#[cfg(feature = "kraken")]
mod kraken_process;
mod lookup;
mod model;
mod ocr;
#[cfg(any(feature = "kraken", feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ort_backend;
#[cfg(any(feature = "kraken", feature = "ppdoc"))]
mod ort_runtime;
mod pairing;
mod pairing_support;
mod pdf;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_openvino;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_postprocess;
mod profile;
mod projection;
mod separator;
mod storage;
pub mod structure;
#[cfg(feature = "kraken")]
mod tesseract_layout;

pub use adapters::{to_alr_payload, to_toa_text_units};
pub use contract::document_request;
pub use deterministic_citations::{
    extract_fields as extract_citation_fields, split_footnote as split_citations,
    split_footnote_recall_first as split_citations_recall_first, DeterministicFields,
    DeterministicPart, DeterministicSplit,
};
pub use docx::{
    apply_docx_links, assess_docx_route, deterministic_docx_intents, extract_docx_gold,
    plan_docx_links, plan_footnotes, validate_docx_response, DocxPlanOptions,
};
#[doc(hidden)]
pub use engine::{extract_common_input, replay_common_input};
pub use error::{Error, Result};
pub use grammar_tables::{compile_table_entry, load_tables, run_vectors as run_grammar_vectors};
#[cfg(feature = "kraken")]
pub use kraken::{
    KrakenBackend, KrakenBatchDiagnostics, KrakenBatchPerformance, KrakenImageDiagnostics,
    KrakenLayout, KrakenOcr, KrakenOptions, KrakenTier,
};
pub use lookup::lookup_footnote;
pub use model::*;
pub use ocr::{OcrLine, OcrOptions, OcrProvider, OcrWord, TesseractOcr, TesseractOptions};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
pub use ppdoc::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions};
pub use projection::{source_doc, structure_lookup};
