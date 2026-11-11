//! The structure stage's contract: the data its stages read and write, and nothing that decides. Each line's slot
//! (`slots`), what a line is and the rule that decided it (`roles`), and the regions a line may be in (`regions`).
//! A change here rebuilds every stage: keep it small and stable.

mod decided;
pub(crate) mod evidence;
mod paragraphs;
pub(crate) mod regions;
mod runs;
mod roles;
mod slots;

pub(crate) use decided::*;
pub(crate) use evidence::*;
pub(crate) use paragraphs::*;
pub(crate) use roles::*;
pub(crate) use runs::*;
pub(crate) use slots::*;

/// The origin of the transition parser's nodes.
pub(crate) const TRANSITION_ORIGIN: &str = "legal-pdf-transition.v1";

