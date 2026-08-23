mod codex;
mod deterministic_citations;
mod docx;
mod error;
mod grammar_tables;

pub use deterministic_citations::{
    extract_fields as extract_citation_fields, split_footnote as split_citations,
    split_footnote_recall_first as split_citations_recall_first, DeterministicFields,
    DeterministicPart, DeterministicSplit,
};
pub use docx::{
    analyze_docx_bytes, apply_docx_links, assess_docx_route, deterministic_docx_intents,
    extract_docx_gold, fix_docx_supra_cross_references, has_docx_supra_references, plan_docx_links,
    plan_footnotes, validate_docx_response, DocxPlanOptions, DocxSupraCleanup,
    MAX_DOCX_SUPRA_BYTES,
};
pub use error::{Error, Result};
pub use grammar_tables::{compile_table_entry, load_tables, run_vectors as run_grammar_vectors};
