//! The gold's frame and the parser's, as the grader reads both: the gold's projection and the parser's
//! structure each as one `State` (`state`), the parser's blocks and annotations over its source lines
//! (`frame_parser`), a frame's blocks and annotations resolved (`frame`, `frame_annotations`), and the
//! two frames' lines aligned (`alignment`, `text_diff`).
pub mod alignment;
mod frame;
pub use frame::*;
mod frame_annotations;
mod frame_parser;
pub use frame_parser::*;
pub mod state;
pub mod text_diff;
