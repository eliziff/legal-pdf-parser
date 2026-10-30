//! Parallel English and French text: which column, and which lines, read in each language.

use super::*;

/// English and French function words counted in some lines: (english, french).
pub(crate) fn language_votes<'a>(lines: impl Iterator<Item = &'a Line>) -> (usize, usize) {
    const ENGLISH: [&str; 24] = [
        "the", "of", "and", "to", "in", "that", "is", "be", "for", "by", "with", "or", "any",
        "which", "shall", "not", "was", "this", "as", "it", "are", "from", "under", "were",
    ];
    const FRENCH: [&str; 28] = [
        "le", "la", "les", "des", "du", "de", "et", "à", "au", "aux", "que", "qui", "une", "un",
        "est", "pour", "dans", "par", "sur", "ne", "pas", "il", "ou", "sont", "cette", "ces", "l",
        "d",
    ];
    let (mut english, mut french) = (0, 0);
    for line in lines {
        for word in line
            .text
            .split(|character: char| !character.is_alphabetic())
        {
            let word = word.to_lowercase();
            english += usize::from(ENGLISH.contains(&word.as_str()));
            french += usize::from(FRENCH.contains(&word.as_str()));
        }
    }
    (english, french)
}

/// A column's language from its votes: Some(true) for French, Some(false) for English.
pub(crate) fn column_language((english, french): (usize, usize), minimum: usize) -> Option<bool> {
    if french >= minimum && french >= english * 3 {
        Some(true)
    } else if english >= minimum && english >= french * 3 {
        Some(false)
    } else {
        None
    }
}

/// The lines printing the French of a parallel English and French text set column by
/// column, as the Supreme Court Reports and federal statutes are. Their headings,
/// numbers and contents repeat the English column's, which is the one read. A document
/// is parallel when some page holds an English column beside a French one; then the
/// French side of every page is the translation wherever its own words are French.
/// `models` are the pages' column models, by view; lines `set_aside` holds (a transcript's line
/// numbers, furniture) are no column's text.
pub(crate) fn translation_lines(
    views: &[PageView],
    models: &[ColumnModel],
    set_aside: impl Fn(Slot) -> bool,
) -> Vec<Slot> {
    let body = |&(slot, line): &(Slot, &Line)| has_valid_bbox(line) && !set_aside(slot);
    let splits = models
        .iter()
        .map(|model| (model.kind == "two_column").then_some(model.split_x))
        .collect::<Vec<_>>();
    let side = |line: &Line, split: f64| {
        (line.bbox[2] <= split || line.bbox[0] >= split).then(|| line_center_x(line) >= split)
    };
    // Pages whose two columns are confidently one language each, and which side is French.
    let parallel = views
        .iter()
        .zip(&splits)
        .filter_map(|(view, split)| {
            let split = (*split)?;
            let votes = |right: bool| {
                language_votes(
                    view.iter()
                        .filter(body)
                        .map(|(_, line)| line)
                        .filter(|line| side(line, split) == Some(right)),
                )
            };
            match (
                column_language(votes(false), 8),
                column_language(votes(true), 8),
            ) {
                (Some(false), Some(true)) => Some((view.page.index, split, true)),
                (Some(true), Some(false)) => Some((view.page.index, split, false)),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    if parallel.is_empty() {
        return Vec::new();
    }
    let french_right = parallel.iter().filter(|(_, _, right)| *right).count() * 2 >= parallel.len();
    let split = median(parallel.iter().map(|(_, split, _)| *split).collect());
    let confirmed = parallel
        .iter()
        .filter(|(_, _, right)| *right == french_right)
        .map(|(index, _, _)| *index)
        .collect::<HashSet<_>>();
    let mut translation = Vec::new();
    for (view, page_split) in views.iter().zip(&splits) {
        let split = page_split.unwrap_or(split);
        let mut french = view
            .iter()
            .filter(body)
            .filter(|(_, line)| side(line, split) == Some(french_right))
            .collect::<Vec<_>>();
        // What stands apart above the English column, a cover page's file number or a
        // memo's title, is no part of the translation beside it.
        let english_top = view
            .iter()
            .filter(body)
            .filter(|(_, line)| (line_center_x(line) >= split) != french_right)
            .map(|(_, line)| line.bbox[1])
            .min_by(f64::total_cmp);
        if let Some(top) = english_top {
            french.sort_by(|(_, a), (_, b)| a.bbox[1].total_cmp(&b.bbox[1]));
            let apart = french
                .windows(2)
                .filter(|pair| {
                    let (above, below) = (pair[0].1, pair[1].1);
                    let height = above.bbox[3] - above.bbox[1];
                    above.bbox[3] <= top && below.bbox[1] - above.bbox[3] >= height * 2.0
                })
                .map(|pair| pair[0].1.bbox[3])
                .max_by(f64::total_cmp);
            if let Some(cut) = apart {
                french.retain(|(_, line)| line.bbox[3] > cut);
            }
        }
        if confirmed.contains(&view.page.index)
            || column_language(language_votes(french.iter().map(|(_, line)| *line)), 3) == Some(true)
        {
            translation.extend(french.into_iter().map(|(slot, _)| slot));
        }
    }
    translation
}
