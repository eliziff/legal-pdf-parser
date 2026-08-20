//! Product orchestration for Page-backed structural derivation.

use crate::Result;
use legal_pdf_core::model::Page;

pub use legal_pdf_structure::{status, validate_document, StructureOutput, StructureReplay};

fn derive_prepared(
    pages: &mut [Page],
    diagnostics: Vec<legal_pdf_core::model::Diagnostic>,
) -> StructureOutput {
    let diagnostics = legal_pdf_structure::prepare_derivation(pages, diagnostics);
    let pairing = legal_pdf_pairing::pair_footnotes(pages);
    legal_pdf_structure::finish_derivation(pages, diagnostics, pairing)
}

pub fn derive(pages: &mut [Page], separators: &[Option<f64>]) -> Result<StructureOutput> {
    legal_pdf_structure::validate_input(pages, separators)?;
    let diagnostics = legal_pdf_structure::prepare_pages(pages, separators);
    Ok(derive_prepared(pages, diagnostics))
}

pub fn replay(pages: &mut [Page], separators: &[Option<f64>]) -> Result<StructureReplay> {
    legal_pdf_structure::validate_input(pages, separators)?;
    legal_pdf_support::profile::begin();
    let diagnostics = legal_pdf_structure::prepare_pages(pages, separators);
    let prepared_pages = pages.to_vec();
    let derived = derive_prepared(pages, diagnostics);
    legal_pdf_support::profile::end();
    Ok(StructureReplay {
        prepared_pages,
        derived,
    })
}
