//! Recognized pages: recognition's lines made into the model's, a text layer's quality against
//! them, and a whole document recognized.

use super::*;

pub(crate) fn make_ocr_line(result: OcrLine, page: &Page, local_index: usize) -> Option<Line> {
    let mut text = result.text;
    let trimmed_end = text.trim_end().len();
    text.truncate(trimmed_end);
    let leading = text.len() - text.trim_start().len();
    // Word offsets count characters of the recognized text before it was trimmed.
    let shift = text[..leading].chars().count();
    text.replace_range(..leading, "");
    if text.is_empty() {
        return None;
    }
    let length = text.chars().count();
    let words = result
        .words
        .into_iter()
        .filter(|word| word.start >= shift && word.end - shift <= length)
        .enumerate()
        .map(|(index, word)| Word {
            id: format!("p{:04}-l{:04}-w{:03}", page.number, local_index, index + 1),
            text: word.text,
            bbox: word.bbox,
            start: word.start - shift,
            end: word.end - shift,
            underlined: false,
            struck: false,
            weight: 0.0,
        })
        .collect();
    Some(Line {
        id: format!("p{:04}-l{local_index:04}", page.number),
        page_index: page.index,
        page_number: page.number,
        source_index: 0,
        reading_order: 0,
        block_index: local_index,
        text,
        bbox: result.bbox,
        spans: vec![],
        words,
        detached_references: vec![],
        ocr: Some(legal_pdf_core::OcrEvidence {
            confidence: result.confidence,
            recovered: result.recovered,
            baseline: result.baseline,
            boundary: result.boundary,
            marker: None,
        }),
        exclude_from_body: false,
        ink: 0.0,
        weight: 0.0,
        cap_height: 0.0,
        suppress_footnote_label: false,
        paragraph_label_candidate: false,
        note_region_mode: String::new(),
        region_id: result.region_id,
        region_type: result.region_type,
        source: "ocr".to_owned(),
    })
}

pub(crate) fn reindex_source_lines(pages: &mut [Page]) {
    let mut source_index = 0;
    for page in pages {
        for line in &mut page.lines {
            source_index += 1;
            line.page_index = page.index;
            line.page_number = page.number;
            line.source_index = source_index;
            line.reading_order = source_index;
        }
    }
}

/// The share of a page's words that are recognition noise, and whether any word carries a
/// recognizer's reject mark ("~NSC", "~A~K~ROUP~~"). A word is noise when it holds a symbol no
/// prose writes inside a word, letters between digits or a number running into letters outside an
/// ordinal ("2Q02", "41ho"; "C-36", "T2P", "2nd" and "1601-06131" are words), three of one letter
/// in a row, no vowel in three letters or more, or a capital inside lower case.
pub(crate) fn recognition_noise(lines: &[Line]) -> (f64, bool) {
    let mut words = 0usize;
    let mut noise = 0usize;
    let mut rejected = false;
    for word in lines.iter().flat_map(|line| line.text.split_whitespace()) {
        let word = word.trim_matches(|character: char| !character.is_alphanumeric() && character != '~');
        if word.is_empty() {
            continue;
        }
        words += 1;
        let characters: Vec<char> = word.chars().collect();
        let reject = word.contains('~') && characters.iter().any(|character| character.is_alphanumeric());
        rejected |= reject;
        let symbol = reject || characters.iter().any(|character| "{}|^`<>\\".contains(*character));
        let letters = characters.iter().filter(|character| character.is_alphabetic()).count();
        let digits = characters.iter().filter(|character| character.is_ascii_digit()).count();
        let mixed = letters > 0 && digits > 0 && (characters.windows(3).any(|three|
            three[0].is_ascii_digit() && three[1].is_alphabetic() && three[2].is_ascii_digit())
            || characters.first().is_some_and(char::is_ascii_digit)
                && characters.last().is_some_and(|last| last.is_alphabetic())
                && !matches!(word.trim_start_matches(|character: char| character.is_ascii_digit()).to_lowercase().as_str(),
                    "st" | "nd" | "rd" | "th" | "d" | "e" | "er" | "re" | "s" | "a" | "b" | "c"));
        let alphabetic = letters == characters.len();
        let tripled = characters.windows(3).any(|three| three[0] == three[1] && three[1] == three[2] && three[0].is_alphabetic());
        let voiceless = alphabetic && letters >= 3
            && !characters.iter().any(|character| "aeiouyAEIOUYéèàÉ".contains(*character));
        let inner_capital = alphabetic && letters >= 3 && characters.windows(2).enumerate().any(|(at, two)|
            two[0].is_lowercase() && two[1].is_uppercase() && !(at == 1 && matches!(characters[0], 'M' | 'D' | 'O')));
        noise += usize::from(symbol || mixed || tripled || voiceless || inner_capital);
    }
    (if words == 0 { 1.0 } else { noise as f64 / words as f64 }, rejected)
}

/// How poorly a page reads, for choosing between its text layer and its recognition: the share
/// of its words that are recognition noise (a reject mark weighing five such words), and the share
/// of its words of four letters or more that the other reading of the document never writes.
pub(crate) fn reading_score(lines: &[Line], other: &HashSet<String>) -> f64 {
    let words = lines.iter().flat_map(|line| line.text.split(|c: char| !c.is_alphabetic()))
        .filter(|word| word.chars().count() >= 4).collect::<Vec<_>>();
    let foreign = words.iter().filter(|word| !other.contains(&word.to_lowercase())).count();
    // A recognizer's reject mark ("2014 ~NSC 4326") is a word it knows it misread.
    let rejected = lines.iter().flat_map(|line| line.text.split_whitespace())
        .filter(|word| word.contains('~') && word.chars().any(char::is_alphanumeric)).count();
    let total = lines.iter().map(|line| line.text.split_whitespace().count()).sum::<usize>().max(1);
    recognition_noise(lines).0 + 4.0 * rejected as f64 / total as f64
        + if words.is_empty() { 0.0 } else { foreign as f64 / words.len() as f64 }
}

/// Whether a page's text reads as recognized prose: no reject mark, and at most one word in ten
/// is noise.
pub(crate) fn layer_reads_well(lines: &[Line]) -> bool {
    let (share, rejected) = recognition_noise(lines);
    !rejected && share <= 0.1
}

/// How much printable text a page holds; decoding garbage is pdf-inspector's verdict, folded in
/// where the page is assembled.
pub(crate) fn text_quality(lines: &[Line]) -> f64 {
    // Python's str.isprintable accepts Unicode L/M/N/P/S plus ASCII space,
    // and rejects every other separator/control/format/private/unassigned
    // scalar. The already-required regex crate exposes those categories.
    static PRINTABLE: OnceLock<Regex> = OnceLock::new();
    let printable =
        PRINTABLE.get_or_init(|| Regex::new(r"[\p{L}\p{M}\p{N}\p{P}\p{S} ]").expect("valid regex"));
    let mut count = lines.len().saturating_sub(1);
    let mut printable_count = 0;
    let mut trimmed = 0;
    let mut trailing_whitespace = 0;
    let mut has_text = false;
    for (index, line) in lines.iter().enumerate() {
        printable_count += printable.find_iter(&line.text).count();
        if index > 0 && has_text {
            trailing_whitespace += 1;
        }
        for character in line.text.chars() {
            count += 1;
            if character.is_whitespace() {
                trailing_whitespace += usize::from(has_text);
            } else {
                trimmed += trailing_whitespace + 1;
                trailing_whitespace = 0;
                has_text = true;
            }
        }
    }
    let printable_share = printable_count as f64 / count.max(1) as f64;
    let quantity = (trimmed as f64 / 100.0).min(1.0);
    ((quantity * printable_share).max(0.0) * 10_000.0).round_ties_even() / 10_000.0
}

/// Apply recognition to extracted pages without loading or extracting the PDF again.
pub fn recognize_pdf(
    pdf: &[u8],
    extracted: &mut ExtractedPdf,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<()> {
    let ExtractedPdf {
        pages,
        separators,
        diagnostics,
        metadata,
        image_regions,
        ..
    } = extracted;
    let mut weak_pages: BTreeSet<usize> = metadata.pages_needing_ocr.iter().copied().collect();
    diagnostics.retain(|diagnostic| diagnostic.code != "OCR_REQUIRED");
    let routed_pages: Vec<_> = weak_pages
        .iter()
        .copied()
        .filter(|page| ocr_pages.is_none_or(|selected| selected.contains(page)))
        .collect();
    let mut reindex = false;
    if let Some(provider) = ocr {
        // Pages recognized whole, and the pictures on other pages, in page order.
        let picture_pages: BTreeMap<usize, &Vec<[f64; 4]>> = image_regions
            .iter()
            .filter(|(page, _)| {
                !weak_pages.contains(page) && ocr_pages.is_none_or(|selected| selected.contains(page))
            })
            .map(|(&page, regions)| (page, regions))
            .collect();
        let mut requests: Vec<_> = routed_pages
            .iter()
            .map(|&page_index| (page_index, Vec::new()))
            .chain(picture_pages.iter().map(|(&page_index, regions)| (page_index, regions.to_vec())))
            .filter_map(|(page_index, regions)| {
                pages.get(page_index).map(|page| OcrPageRequest {
                    page_index,
                    width: page.width,
                    height: page.height,
                    regions,
                    lines: Vec::new(),
                    words: Vec::new(),
                })
            })
            .collect();
        requests.sort_by_key(|request| request.page_index);
        let results = legal_pdf_core::profile::measure("extract.ocr", || {
            provider.extract_pages(pdf, &requests)
        })?;
        // Each page's recognition, read as lines.
        let mut recognized = Vec::new();
        for result in results {
            let Some(page) = pages.get_mut(result.page_index) else {
                return Err(Error::Message(format!(
                    "OCR returned an unknown page index: {}",
                    result.page_index
                )));
            };
            if let Some(regions) = picture_pages.get(&result.page_index) {
                let mut lines = result.lines;
                for region in regions.iter() {
                    let (inside, rest): (Vec<_>, Vec<_>) = lines.into_iter().partition(|line| {
                        let x = (line.bbox[0] + line.bbox[2]) / 2.0;
                        let y = (line.bbox[1] + line.bbox[3]) / 2.0;
                        region[0] <= x && x <= region[2] && region[1] <= y && y <= region[3]
                    });
                    lines = rest;
                    let first = page.lines.len() + 1;
                    // A picture's lines join its page when the recognizer is sure of them: at least
                    // 0.97 over their characters. A picture of type reads at 0.99 and more; a seal's
                    // lettering, a chart, a signature or a blurred letterhead at 0.95 and less.
                    let characters = |line: &OcrLine| line.text.trim().chars().count() as f64;
                    let total = inside.iter().map(characters).sum::<f64>();
                    let sure = inside.iter().map(|line| characters(line) * line.confidence).sum::<f64>();
                    if total == 0.0 || sure < 0.97 * total {
                        continue;
                    }
                    let mut read: Vec<_> = inside
                        .into_iter()
                        .enumerate()
                        .filter_map(|(offset, line)| make_ocr_line(line, page, first + offset))
                        .collect();
                    read.sort_by(|a, b| (a.bbox[1] + a.bbox[3]).total_cmp(&(b.bbox[1] + b.bbox[3])));
                    place_picture_lines(page, region, read);
                    reindex = true;
                }
                continue;
            }
            let lines: Vec<_> = result
                .lines
                .into_iter()
                .enumerate()
                .filter_map(|(index, line)| make_ocr_line(line, page, index + 1))
                .collect();
            recognized.push((result.page_index, lines, result.separator_y));
        }
        // A page that has a text layer of its own keeps it unless its recognition reads measurably
        // better by the same measure: noise words, and words the other reading never writes
        // anywhere in the document ("Arangement" where the layer writes "Arrangement" throughout;
        // "IPJTRODUCTION" where the recognition writes "INTRODUCTION").
        let vocabulary = |lines: &mut dyn Iterator<Item = &Line>| lines
            .flat_map(|line| line.text.split(|c: char| !c.is_alphabetic()).filter(|word| !word.is_empty())
                .map(str::to_lowercase).collect::<Vec<_>>())
            .collect::<HashSet<_>>();
        let layer_words = vocabulary(&mut pages.iter().filter(|page| page.source == "native")
            .flat_map(|page| page.lines.iter().filter(|line| line.source == "native")));
        let recognized_words = vocabulary(&mut recognized.iter().flat_map(|(_, lines, _)| lines.iter()));
        for (page_index, lines, separator_y) in recognized {
            let page = &mut pages[page_index];
            if lines.is_empty() {
                continue;
            }
            // A scan whose own words are only a stamp over it has no layer to keep: the stamp is
            // in its image, and the document's other pages give it no vocabulary to be read by.
            let stamped = metadata.scanned_pages.contains(&page_index)
                && page.lines.iter().map(|line| line.bbox[2] - line.bbox[0]).sum::<f64>() < 2.0 * page.width;
            let layered = page.source == "native" && !page.lines.is_empty() && !stamped;
            // A reading that holds fewer of the page's words than the other misses the rest of it.
            let count = |lines: &[Line]| lines.iter().map(|line| line.text.split_whitespace().count()).sum::<usize>() as f64;
            let (layer_count, recognized_count) = (count(&page.lines), count(&lines));
            let missing = |own: f64, other: f64| if other > 0.0 { (1.0 - own / other).max(0.0) } else { 0.0 };
            if layered && reading_score(&page.lines, &recognized_words) + missing(layer_count, recognized_count)
                <= reading_score(&lines, &layer_words) + missing(recognized_count, layer_count) {
                weak_pages.remove(&page_index);
                continue;
            }
            reindex = true;
            page.lines = lines;
            page.source = "ocr".to_owned();
            page.text_quality = 0.5;
            weak_pages.remove(&page_index);
            if let Some(separator) = separator_y {
                separators[page_index] = Some(separator);
            }
        }
        // A recognized page's lines and words are measured in its render too: the recognizer writes
        // no font flags or sizes, so their strokes and capitals are its only type evidence.
        let requests: Vec<_> = pages
            .iter()
            .enumerate()
            .filter(|(index, page)| page.source == "ocr" && !page.lines.is_empty() && ocr_pages.is_none_or(|selected| selected.contains(index)))
            .map(|(page_index, page)| OcrPageRequest {
                page_index,
                width: page.width,
                height: page.height,
                regions: Vec::new(),
                lines: page.lines.iter().map(|line| line.bbox).collect(),
                words: page.lines.iter().map(|line| line.words.iter().map(|word| word.bbox).collect()).collect(),
            })
            .collect();
        if !requests.is_empty() {
            provider.separators(pdf, &requests);
        }
        // A scan that keeps its own text layer still prints the rule above its notes in its
        // image: read the rule there where nothing else found one, and each line's ink.
        let requests: Vec<_> = metadata
            .scanned_pages
            .iter()
            .filter(|&&index| pages.get(index).is_some_and(|page| page.source == "native" && !page.lines.is_empty()))
            .filter_map(|&page_index| {
                pages.get(page_index).map(|page| OcrPageRequest {
                    page_index,
                    width: page.width,
                    height: page.height,
                    regions: Vec::new(),
                    lines: page.lines.iter().map(|line| line.bbox).collect(),
                    words: page.lines.iter().map(|line| line.words.iter().map(|word| word.bbox).collect()).collect(),
                })
            })
            .collect();
        if !requests.is_empty() {
            // The rule is the notes' only where a line below it opens as a note does, with its
            // number or mark: a signature line or a table's border has none under it.
            let opens_note = |line: &Line| {
                let text = line.text.trim_start();
                let label = text
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || "*†‡§".contains(*c))
                    .count();
                (1..=3).contains(&label)
                    && text.chars().nth(label).is_some_and(|c| c.is_whitespace() || ".)".contains(c))
            };
            for (request, separator) in requests.iter().zip(provider.separators(pdf, &requests)) {
                if separators[request.page_index].is_some() {
                    continue;
                }
                separators[request.page_index] = separator.filter(|y| {
                    pages[request.page_index]
                        .lines
                        .iter()
                        .any(|line| line.bbox[1] > *y && opens_note(line))
                });
            }
        }
        // A kept layer's lost note numbers, read again by our recognizer.
        let requests: Vec<_> = metadata
            .scanned_pages
            .iter()
            .filter_map(|&page_index| {
                pages
                    .get(page_index)
                    .filter(|page| page.source == "native" && !page.lines.is_empty())
                    .map(|page| OcrPageRequest {
                        page_index,
                        width: page.width,
                        height: page.height,
                        regions: Vec::new(),
                        lines: Vec::new(),
                        words: Vec::new(),
                    })
            })
            .collect();
        if !requests.is_empty() {
            for result in provider.extract_pages(pdf, &requests)? {
                let Some(page) = pages.get_mut(result.page_index) else {
                    continue;
                };
                let recognized: Vec<_> = result
                    .lines
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, line)| make_ocr_line(line, page, index + 1))
                    .collect();
                for line in page.lines.iter_mut().filter(|line| line.source == "native") {
                    let read = crate::reread::reading_of(line, &recognized);
                    crate::reread::splice_numbers(line, &read);
                }
            }
        }
        // Paragraph numbers the scan lost, from the run they stand in and the margin read again.
        if crate::paragraph_recovery::recover(pdf, pages, &metadata.scanned_pages, provider) > 0 {
            reindex = true;
        }
    }
    if reindex {
        reindex_source_lines(pages);
        legal_pdf_core::associate_ocr_markers(pages);
    }
    // A number standing on a line of its own, read or typeset, joins its paragraph's first line.
    // Off unless LEGALPDF_MARKER_JOIN=on: the join removes a source line, and the gold's lines are
    // the extraction's lines, so joined documents lose their gold alignment. The association the
    // structure stage reads is the lasting form of this.
    if std::env::var("LEGALPDF_MARKER_JOIN").is_ok_and(|v| v == "on") && legal_pdf_core::merge_marker_lines(pages) > 0 {
        reindex_source_lines(pages);
        legal_pdf_core::associate_ocr_markers(pages);
    }
    let unresolved_pages: Vec<_> = weak_pages.into_iter().collect();
    for &page_index in &unresolved_pages {
        diagnostics.push(Diagnostic::warning(
            "OCR_REQUIRED",
            "Page has no reliable embedded text and no usable OCR result.",
            Some(page_index),
        ));
    }

    metadata.pages_needing_ocr = unresolved_pages;
    metadata.ocr_routed_pages = routed_pages;
    Ok(())
}
