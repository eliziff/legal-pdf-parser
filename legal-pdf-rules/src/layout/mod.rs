//! A page's layout as every stage reads it: its languages, and the line a speaker's turn opens.

pub mod language;

use legal_pdf_core::model::Line;

/// Whether a transcript's line opens a unit of its own: a speaker's turn, or a line in capitals
/// (an examination's "BY MR. JONES:", "EXHIBIT NO. 3", "--- Upon resuming").
pub fn opens_turn(line: &Line) -> bool {
    let text = line.text.trim_start();
    legal_structure_lines::transcript::speaker_label(text).is_some()
        || text.starts_with("--")
        || (text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase))
}

/// All speaker label byte spans opening a transcript line (`legal_structure_lines::transcript::speaker_labels`).
pub fn speaker_labels(text: &str) -> Vec<(usize, usize)> {
    legal_structure_lines::transcript::speaker_labels(text)
}
