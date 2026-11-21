//! The structure stage's contract (legal-pdf-structure reads it as `model`): the data its stages read and write, and nothing that decides. Each line's slot
//! (`slots`), what a line is and the rule that decided it (`roles`), and the regions a line may be in (`regions`).
//! A change here rebuilds every stage: keep it small and stable.

pub mod constituents;
pub mod contents_lists;
mod context;
mod decided;
pub mod decide;
pub mod fences;
pub mod machine;
pub mod notes;
pub mod tables;
mod geometry;
mod identity;
mod items;
pub mod evidence;
mod paragraphs;
pub mod regions;
mod roles;
mod runs;
mod slots;

pub use context::*;
pub use decided::*;
pub use geometry::*;
pub use identity::*;
pub use items::*;
pub use evidence::*;
pub use paragraphs::*;
pub use roles::*;
pub use runs::*;
pub use slots::*;

/// The origin of the parser's nodes.
pub const ORIGIN: &str = "legal-pdf-structure.v1";
