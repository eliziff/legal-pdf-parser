//! The transition machine's types every tier reads: each line as the state reads it and the moves it makes (`machine`),
//! and each line's readings, marks and features (`lines`).

pub(crate) mod lines;
pub(crate) mod machine;

use legal_pdf_core::model::{Line, Page};
use crate::evidence::{marker_readings, reference_spans};
use crate::decide::{Decision, Notes};
use crate::fences::Fences;
use crate::model::evidence::*;
use crate::model::{Held, Slot, Slots};
use std::collections::{HashMap, HashSet};
