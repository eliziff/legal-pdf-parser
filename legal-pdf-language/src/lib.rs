mod error;
// Word's structure: its numbering derived and the document read and checked (analyze_docx).
mod docx_lint;
mod docx_numbering;
pub use docx_lint::*;
pub use docx_numbering::*;
pub use error::{Error, Result};
mod docx;
mod docx_spans;
mod docx_toa;
mod docx_compare;
mod docx_tracked;
mod text_ops;
mod process;

pub use docx::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, docx_to_toa_text_units,
    fix_docx_supra_cross_references, DocxSupraCleanup,
    MAX_DOCX_SUPRA_BYTES,
};
pub use docx_spans::{write_docx_spans, DocxEdit, DocxSpanWrite, DocxSpansWritten, DocxUnit};
pub use docx_toa::{apply_table_of_authorities, DocxAuthorityMark, DocxLinkedAuthority, DocxTableDelivery, PinpointLink};
pub use docx_tracked::{apply_tracked_edits, docx_body_text, docx_tracked_change_ids, insert_tracked_blocks, resolve_tracked_changes,
    apply_text_ops, AppliedChange, BlockInsert, BlockPosition, EditDiffSegment, EditError, EditInput, EditMode, TextOpReport, TextOpRequest,
    TextOpsApplied, TextScope, TextSpan, TrackedEdits};
pub use text_ops::{TextOpNote, TextOpParams};
pub use docx_compare::{compare_docx_versions, CompareAbstention, CompareChange, CompareKind, ComparedVersions};
