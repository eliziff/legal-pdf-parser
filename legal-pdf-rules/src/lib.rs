//! The PDF structure stage's line rules: what a line's text, markers, geometry and words show, which every stage reads.
//! On the stage's contract alone, so an edit to a stage recompiles no rule and an edit to a rule only what reads it.

use legal_pdf_structure_model as model;

pub mod evidence;
pub mod headings;
pub mod layout;
pub mod notes;
pub mod tables;
pub mod text;
