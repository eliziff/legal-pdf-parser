//! Product orchestration for Page-backed structural derivation.

use crate::Result;
use legal_pdf_core::model::Page;

pub use legal_pdf_structure::{
    status, validate_document, StructureIdentity, StructureOutput, StructureReplay,
};

fn derive_prepared(
    pages: &mut [Page],
    prepared: legal_pdf_structure::PdfPreparation,
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    let prepared = legal_pdf_structure::prepare_derivation(pages, prepared);
    let pairing = legal_pdf_pairing::pair_footnotes(pages);
    legal_pdf_structure::finish_derivation(pages, prepared, pairing, identity)
}

pub fn derive(
    pages: &mut [Page],
    separators: &[Option<f64>],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    legal_pdf_structure::validate_input(pages, separators)?;
    let prepared = legal_pdf_structure::prepare_pages(pages, separators);
    derive_prepared(pages, prepared, identity)
}

pub fn replay(
    pages: &mut [Page],
    separators: &[Option<f64>],
    identity: StructureIdentity,
) -> Result<StructureReplay> {
    legal_pdf_structure::validate_input(pages, separators)?;
    let _profile = legal_pdf_support::profile::scope("structure_replay");
    let prepared = legal_pdf_structure::prepare_pages(pages, separators);
    let prepared_pages = pages.to_vec();
    let derived = derive_prepared(pages, prepared, identity)?;
    Ok(StructureReplay {
        prepared_pages,
        derived,
    })
}
