//! The transition machine's types every tier reads: each line as the state reads it and the moves it makes (`machine`),
//! and each line's readings, marks and features (`lines`).

pub mod lines;
pub mod machine;

use legal_pdf_core::model::{Line, Page};
use legal_pdf_evidence::{marker_readings, reference_spans};
use legal_pdf_decide::{Decision, Notes};
use legal_pdf_fences::Fences;
use legal_pdf_structure_model::evidence::*;
use legal_pdf_structure_model::{Held, Slot, Slots};
use std::collections::{HashMap, HashSet};
