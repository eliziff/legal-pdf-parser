//! How a unit's lines join: the seam between two of its lines, and the hyphens inside a line that split a word.

use legal_pdf_core::model::Page;
use std::collections::HashSet;

pub use pdf_inspector::markdown::Join;

/// A document's own dictionary for rejoining words its lines break: its words and its hyphenated
/// compounds (pdf-inspector's `document_vocabulary`).
pub type Vocabulary = (HashSet<String>, HashSet<String>);

pub fn vocabulary(pages: &[Page]) -> Vocabulary {
    let text: Vec<&str> = pages.iter().flat_map(|page| &page.lines).map(|line| line.text.trim()).collect();
    pdf_inspector::markdown::document_vocabulary(&text.join(" "))
}

/// The seam between a unit's line (`before`, trimmed) and its next (`after`, trimmed). A word broken over the two is
/// rejoined as pdf-inspector rejoins it: plain where the document writes it whole ("de- fendant"), with its hyphen where
/// it writes the compound or the continuation is capitalized ("six- month", "Third- Party"), and otherwise plain before a
/// lowercase continuation ("judg- ment"), the unit's lines being one column's; a soft hyphen always joins. `Plain` drops
/// `before`'s last character, `Hyphen` keeps it with no space, `Keep` is a space.
pub fn seam(before: &str, after: &str, vocabulary: &Vocabulary) -> Join {
    use pdf_inspector::markdown::join_decision;
    let first: &str = after.split(|c: char| !c.is_alphabetic()).next().unwrap_or("");
    match before.chars().next_back() {
        Some('\u{00ad}') if !first.is_empty() => Join::Plain,
        Some(hyphen @ ('-' | '\u{00ac}')) => {
            let stem = &before[..before.len() - hyphen.len_utf8()];
            let last = stem.rsplit(|c: char| !c.is_alphabetic()).next().unwrap_or("");
            let decision = if last.chars().count() >= 2 && first.chars().count() >= 2 {
                join_decision(last, first, &vocabulary.0, &vocabulary.1)
            } else {
                Join::Keep
            };
            // Without the document's evidence, a lowercase continuation in one column's unit is the word going on.
            match decision {
                Join::Keep if !last.is_empty() && first.chars().next().is_some_and(char::is_lowercase) => Join::Plain,
                decision => decision,
            }
        }
        _ => Join::Keep,
    }
}

/// The hyphens in a line's text that split one word, each its UTF-16 offset in the line: a hyphen between letters where
/// no compound is written ("Companies' Cr-editors Arrangement Act"), a recognizer's read into the text, where
/// pdf-inspector's `join_decision` reads the two parts as one word, the document writing it whole elsewhere. A compound
/// whose second part opens with a capital or a digit ("Non-Renewable", "C-36"), or that the document never writes whole,
/// keeps it.
pub fn split_words(text: &str, vocabulary: &Vocabulary) -> Vec<usize> {
    use pdf_inspector::markdown::join_decision;
    let characters: Vec<char> = text.chars().collect();
    let mut hyphens = Vec::new();
    let mut unit = characters.first().map_or(0, |c| c.len_utf16());
    for at in 1..characters.len() {
        let here = unit;
        unit += characters[at].len_utf16();
        if characters[at] != '-' || !characters[at - 1].is_alphabetic() {
            continue;
        }
        let first = characters[..at].iter().rev().take_while(|c| c.is_alphabetic()).count();
        let second = characters[at + 1..].iter().take_while(|c| c.is_alphabetic()).count();
        if second < 2 || at > first && characters[at - first - 1] == '-' {
            continue;
        }
        let before: String = characters[at - first..at].iter().collect();
        let after: String = characters[at + 1..at + 1 + second].iter().collect();
        if join_decision(&before, &after, &vocabulary.0, &vocabulary.1) == Join::Plain {
            hyphens.push(here);
        }
    }
    hyphens
}
