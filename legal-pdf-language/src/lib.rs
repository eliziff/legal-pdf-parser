mod codex;
mod deterministic_citations;
mod docx;
mod error;
mod grammar_tables;
mod grammar_word;

pub use deterministic_citations::{
    extract_fields as extract_citation_fields, split_footnote as split_citations,
    split_footnote_recall_first as split_citations_recall_first, DeterministicFields,
    DeterministicPart, DeterministicSplit,
};
pub use docx::{
    apply_docx_links, assess_docx_route, deterministic_docx_intents, extract_docx_gold,
    plan_docx_links, plan_footnotes, validate_docx_response, DocxPlanOptions,
};
pub use error::{Error, Result};
pub use grammar_tables::{compile_table_entry, load_tables, run_vectors as run_grammar_vectors};
