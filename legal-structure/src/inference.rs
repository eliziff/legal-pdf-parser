use super::*;

macro_rules! cached_regex {
    ($name:ident, $pattern:expr) => {{
        static $name: OnceLock<Regex> = OnceLock::new();
        $name.get_or_init(|| Regex::new($pattern).unwrap())
    }};
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MarkerStyle {
    Bracket,
    Dot,
    Bare,
}

#[derive(Clone)]
struct Marker {
    number: u32,
    start: usize,
    content_start: usize,
    style: MarkerStyle,
    score: f64,
    formal: bool,
    sentence: bool,
}

#[derive(Clone, Copy)]
struct Line<'a> {
    byte_start: usize,
    byte_end: usize,
    scalar_start: usize,
    text: &'a str,
}

fn lines<'a>(text: &'a ScalarText<'a>) -> impl Iterator<Item = Line<'a>> + 'a {
    text.lines().iter().map(move |line| Line {
        byte_start: line[0],
        byte_end: line[1],
        scalar_start: line[2],
        text: &text.value[line[0]..line[1]],
    })
}

fn javascript_lines<'a>(text: &'a ScalarText<'a>) -> Vec<Line<'a>> {
    // This is JavaScript regexp line segmentation, not coordinate conversion:
    // CRLF is one break while its two source scalars remain counted.
    let mut result = Vec::new();
    let mut chars = text.value.char_indices().peekable();
    let (mut byte_start, mut scalar_start, mut scalar) = (0, 0, 0);
    while let Some((byte, character)) = chars.next() {
        if matches!(character, '\r' | '\n' | '\u{2028}' | '\u{2029}') {
            result.push(Line {
                byte_start,
                byte_end: byte,
                scalar_start,
                text: &text.value[byte_start..byte],
            });
            scalar += 1;
            let mut next_byte = byte + character.len_utf8();
            if character == '\r' && chars.peek().is_some_and(|(_, next)| *next == '\n') {
                let (next, _) = chars.next().unwrap();
                next_byte = next + 1;
                scalar += 1;
            }
            byte_start = next_byte;
            scalar_start = scalar;
        } else {
            scalar += 1;
        }
    }
    result.push(Line {
        byte_start,
        byte_end: text.value.len(),
        scalar_start,
        text: &text.value[byte_start..],
    });
    result
}

fn leading_ascii_space(value: &str) -> usize {
    value
        .bytes()
        .take_while(|byte| matches!(byte, b' ' | b'\t'))
        .count()
}

fn decimal_prefix(value: &str, maximum: usize) -> Option<(&str, usize)> {
    let length = value.bytes().take_while(u8::is_ascii_digit).count();
    (length > 0 && length <= maximum).then(|| (&value[..length], length))
}

fn paragraph_markers(text: &ScalarText<'_>, contiguous: bool) -> Vec<Marker> {
    let mut result = Vec::new();
    for line in lines(text) {
        let lead = leading_ascii_space(line.text);
        let value = &line.text[lead..];
        let start = line.scalar_start;
        let basic = if let Some(rest) = value.strip_prefix('[') {
            decimal_prefix(rest, 4).and_then(|(number, length)| {
                (rest.as_bytes().get(length) == Some(&b']'))
                    .then(|| (number, MarkerStyle::Bracket, length + 2))
            })
        } else {
            decimal_prefix(value, 4).and_then(|(number, length)| {
                let rest = &value[length..];
                if rest.starts_with('.')
                    && (rest[1..].chars().next().is_some_and(char::is_whitespace)
                        || (rest.len() == 1 && line.byte_end < text.value.len()))
                {
                    Some((number, MarkerStyle::Dot, length + 1))
                } else if contiguous
                    && rest.starts_with('.')
                    && rest[1..].chars().next().is_some_and(char::is_uppercase)
                {
                    Some((number, MarkerStyle::Dot, length + 1))
                } else if rest.chars().next().is_some_and(char::is_whitespace)
                    || (rest.is_empty() && line.byte_end < text.value.len())
                {
                    Some((
                        number,
                        if contiguous {
                            MarkerStyle::Dot
                        } else {
                            MarkerStyle::Bare
                        },
                        length,
                    ))
                } else {
                    None
                }
            })
        };
        if let Some((number, style, marker_end)) = basic {
            let content = marker_end + leading_ascii_space(&value[marker_end..]);
            result.push(Marker {
                number: number.parse().unwrap(),
                start,
                content_start: line.scalar_start
                    + line.text[..line.text.len() - value.len()].chars().count()
                    + value[..content].chars().count(),
                style,
                score: 1.0,
                formal: false,
                sentence: false,
            });
        }
        {
            let glyph = value
                .chars()
                .next()
                .filter(|value| matches!(value, '¶' | '\u{95}' | '•'));
            if let Some(glyph) = glyph {
                let rest = value[glyph.len_utf8()..].trim_start_matches([' ', '\t']);
                if let Some((number, length)) = decimal_prefix(rest, 4) {
                    let after = rest[length..].chars().next();
                    if after.is_none_or(|value| value.is_whitespace() || ".,;:—-".contains(value))
                    {
                        result.push(Marker {
                            number: number.parse().unwrap(),
                            start,
                            content_start: start
                                + line.text[..line.text.len() - rest.len() + length]
                                    .chars()
                                    .count(),
                            style: MarkerStyle::Dot,
                            score: 1.0,
                            formal: false,
                            sentence: false,
                        });
                    }
                }
            }
        }
    }
    result.sort_by_key(|marker| marker.start);
    result.dedup_by_key(|marker| marker.start);
    result
}

fn marker_visible(marker: &Marker, excluded: &[ScalarRange]) -> bool {
    !excluded
        .iter()
        .any(|range| range.start <= marker.start && marker.start < range.end)
}

fn word_count(value: &str, letters_only: bool) -> usize {
    let mut count = 0;
    let mut inside = false;
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        let member = if letters_only {
            character.is_alphabetic()
        } else {
            character.is_alphanumeric()
        };
        if member {
            count += usize::from(!inside);
            inside = true;
        } else if matches!(character, '\'' | '’')
            && inside
            && characters.peek().is_some_and(|next| {
                if letters_only {
                    next.is_alphabetic()
                } else {
                    next.is_alphanumeric()
                }
            })
        {
            continue;
        } else {
            inside = false;
        }
    }
    count
}

fn median(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle] as f64
    } else {
        (values[middle - 1] + values[middle]) as f64 / 2.0
    }
}

fn heading_enumerator(value: &str) -> bool {
    cached_regex!(VALUE, r"^(?:\([\p{L}\p{N}]{1,5}\)|\p{L}[.)]|[IVXLCDM]{1,4}[.)]|[ivxlcdm]{1,4}[.)]|\d{1,3}(?:\.\d{1,3})*[.)])$").is_match(value)
}

fn level_opens(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| character.is_uppercase() || character.is_numeric())
}

fn heading_level(words: &[&str], enumerated: bool) -> bool {
    if words.is_empty() || words.len() > 12 {
        return false;
    }
    if words.len() == 1 && heading_enumerator(words[0]) {
        return true;
    }
    let text = words.join(" ");
    if !level_opens(&text) || text.ends_with(['.', ',', ';']) {
        return false;
    }
    if text.ends_with(['?', ':']) {
        return true;
    }
    let title = words.iter().all(|word| {
        let letters = word
            .chars()
            .filter(|value| value.is_alphabetic())
            .collect::<String>();
        utf16_len(&letters) < 4 || letters.chars().next().is_some_and(char::is_uppercase)
    });
    title || enumerated || words.len() <= 6
}

fn trim_leading_parenthetical(value: &str) -> &str {
    let value = value.trim();
    let Some(rest) = value.strip_prefix('(') else {
        return value;
    };
    let Some(close) = rest.find(')') else {
        return value;
    };
    if !rest[..close].is_empty()
        && rest[..close].chars().all(char::is_alphanumeric)
        && rest[close + 1..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        rest[close + 1..].trim_start()
    } else {
        value
    }
}

pub(super) fn formal_heading(value: &str) -> bool {
    let heading = trim_leading_parenthetical(value);
    if heading.is_empty()
        || utf16_len(heading) > 120
        || heading.chars().any(|value| ";![]{}".contains(value))
    {
        return false;
    }
    let words = heading.split_whitespace().collect::<Vec<_>>();
    let mut levels: Vec<(Vec<&str>, bool)> = vec![(Vec::new(), false)];
    for (index, word) in words.iter().enumerate() {
        let opener = words
            .get(index + 1)
            .is_some_and(|next| !heading_enumerator(next) && level_opens(next));
        if heading_enumerator(word) && opener {
            if levels.last().unwrap().0.is_empty() {
                levels.last_mut().unwrap().1 = true;
            } else {
                levels.push((Vec::new(), true));
            }
        } else {
            levels.last_mut().unwrap().0.push(word);
        }
    }
    levels
        .iter()
        .all(|(words, enumerated)| heading_level(words, *enumerated))
}

fn sentence_heading(value: &str, following: &str) -> bool {
    let heading = trim_leading_parenthetical(value);
    let words = heading.split_whitespace().collect::<Vec<_>>();
    utf16_len(heading) <= 120
        && (4..=18).contains(&words.len())
        && heading.chars().next().is_some_and(char::is_uppercase)
        && words
            .iter()
            .any(|word| word.chars().next().is_some_and(char::is_lowercase))
        && !heading.chars().any(|value| "[].,;:!?".contains(value))
        && following
            .trim_start()
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
}

fn heading_joined(
    text: &ScalarText<'_>,
    known: &HashSet<usize>,
    style: MarkerStyle,
) -> Vec<Marker> {
    if style == MarkerStyle::Bare {
        return Vec::new();
    }
    let mut result = Vec::new();
    for line in lines(text) {
        let bytes = line.text.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            let digits = if style == MarkerStyle::Bracket && bytes[at] == b'[' {
                at + 1
            } else if style == MarkerStyle::Dot && bytes[at].is_ascii_digit() {
                at
            } else {
                at += 1;
                continue;
            };
            let length = bytes[digits..]
                .iter()
                .take(4)
                .take_while(|byte| byte.is_ascii_digit())
                .count();
            if length == 0 {
                at += 1;
                continue;
            }
            let tail = digits + length;
            let end = if style == MarkerStyle::Bracket {
                (bytes.get(tail) == Some(&b']')).then_some(tail + 1)
            } else if bytes.get(tail) == Some(&b'.') {
                let after = tail + 1;
                if after == bytes.len() {
                    Some(after)
                } else {
                    line.text[after..]
                        .chars()
                        .next()
                        .filter(|character| character.is_whitespace())
                        .map(|character| after + character.len_utf8())
                }
            } else {
                None
            };
            let Some(end) = end else {
                at += 1;
                continue;
            };
            let start = line.scalar_start + line.text[..at].chars().count();
            if known.contains(&start) {
                at = end;
                continue;
            }
            let heading = &line.text[..at];
            let formal = formal_heading(heading)
                && (style == MarkerStyle::Bracket || !heading.contains('.'));
            let sentence = style == MarkerStyle::Bracket
                && sentence_heading(heading, &text.value[line.byte_start + end..]);
            if formal || sentence {
                result.push(Marker {
                    number: line.text[digits..tail].parse().unwrap(),
                    start,
                    content_start: line.scalar_start + line.text[..end].chars().count(),
                    style,
                    score: if formal { 0.6 } else { 0.35 },
                    formal,
                    sentence,
                });
            }
            at = end;
        }
    }
    result
}

fn rooted_chain(candidates: Vec<Marker>) -> (Vec<Marker>, f64) {
    let selected = select_numeric_sequence(
        candidates
            .iter()
            .enumerate()
            .map(|(index, marker)| NumericSequenceCandidate {
                index,
                value: marker.number,
                position: (marker.start, 0),
                page: 0,
                score: marker.score,
                start_supported: false,
            })
            .collect(),
        NumericSequencePolicy::RootedConsecutive,
    );
    (
        selected
            .indices
            .into_iter()
            .map(|index| candidates[index].clone())
            .collect(),
        selected.score,
    )
}

fn sole_chain(chain: &[Marker], candidates: &[Marker]) -> bool {
    let claimed = chain
        .iter()
        .map(|value| value.start)
        .collect::<HashSet<_>>();
    let last = chain.last().map_or(0, |value| value.number);
    let mut rest = candidates
        .iter()
        .filter(|value| !claimed.contains(&value.start))
        .collect::<Vec<_>>();
    rest.sort_by_key(|value| value.start);
    !rest
        .iter()
        .any(|value| (1..=last + 1).contains(&value.number))
        && rest.windows(2).all(|pair| pair[1].number <= pair[0].number)
}

fn endnote_shaped(text: &ScalarText<'_>, chain: &[Marker]) -> bool {
    chain.len() >= 8
        && text.utf16_len() > 0
        && chain
            .iter()
            .filter(|value| text.utf16(value.start) as f64 > text.utf16_len() as f64 * 0.75)
            .count() as f64
            / chain.len() as f64
            >= 0.7
}

fn monotone_scopes(markers: &[Marker], max_gap: u32) -> Vec<Vec<Marker>> {
    let mut scopes = Vec::<Vec<Marker>>::new();
    let mut by_last = HashMap::<u32, Vec<usize>>::new();
    for marker in markers {
        let candidates = (marker.number.saturating_sub(max_gap)..marker.number)
            .flat_map(|value| by_last.get(&value).into_iter().flatten().copied())
            .collect::<Vec<_>>();
        let index = candidates
            .into_iter()
            .reduce(|best, current| {
                let left = scopes[current][0].number;
                let right = scopes[best][0].number;
                if left < right || (left == right && current < best) {
                    current
                } else {
                    best
                }
            })
            .unwrap_or(scopes.len());
        if index == scopes.len() {
            scopes.push(vec![marker.clone()]);
        } else {
            let previous = scopes[index].last().unwrap().number;
            if let Some(values) = by_last.get_mut(&previous) {
                values.retain(|value| *value != index);
            }
            scopes[index].push(marker.clone());
        }
        by_last.entry(marker.number).or_default().push(index);
    }
    scopes
}

fn contiguous_scopes(markers: &[Marker]) -> Vec<Vec<Marker>> {
    let mut scopes: Vec<Vec<Marker>> = Vec::new();
    for marker in markers {
        if scopes
            .last()
            .and_then(|scope| scope.last())
            .is_some_and(|prior| marker.number == prior.number + 1)
        {
            scopes.last_mut().unwrap().push(marker.clone());
        } else {
            scopes.push(vec![marker.clone()]);
        }
    }
    scopes
}

fn next_boundary(boundaries: &[usize], start: usize, end: usize) -> usize {
    boundaries
        .get(boundaries.partition_point(|boundary| *boundary <= start))
        .copied()
        .unwrap_or(end)
}

pub(super) fn raw_numeric_runs(text: &ScalarText<'_>) -> Vec<StructureCandidateRun> {
    let all = paragraph_markers(text, false);
    let mut boundaries = all.iter().map(|marker| marker.start).collect::<Vec<_>>();
    boundaries.push(text.len());
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut runs = Vec::new();
    for style in [MarkerStyle::Bracket, MarkerStyle::Dot, MarkerStyle::Bare] {
        let markers = all
            .iter()
            .filter(|marker| marker.style == style)
            .cloned()
            .collect::<Vec<_>>();
        for scope in monotone_scopes(&markers, 8) {
            if scope.len() < 2 {
                continue;
            }
            let mut candidates = scope
                .iter()
                .map(|marker| {
                    let end = next_boundary(&boundaries, marker.start, text.len());
                    let surface_label = text
                        .slice(marker.start..marker.content_start)
                        .expect("numeric marker range is bounded")
                        .trim()
                        .to_owned();
                    StructureMarkerCandidate {
                        id: String::new(),
                        range: ScalarRange {
                            start: marker.start,
                            end,
                        },
                        marker_range: ScalarRange {
                            start: marker.start,
                            end: marker.content_start,
                        },
                        label: surface_label,
                        grammar_value: marker.number.to_string(),
                        parent_candidate_id: None,
                        level: 0,
                        content_start: marker.content_start,
                    }
                })
                .collect::<Vec<_>>();
            let range = ScalarRange {
                start: candidates[0].range.start,
                end: candidates.last().unwrap().range.end,
            };
            let ordinal = runs.len() + 1;
            for (index, candidate) in candidates.iter_mut().enumerate() {
                candidate.id = format!("numeric-{ordinal:06}-{:04}", index + 1);
            }
            runs.push(StructureCandidateRun {
                id: format!("numeric-{ordinal:06}"),
                grammar: CandidateGrammar::Numeric,
                range,
                rooted: scope[0].number == 1,
                consecutive: scope
                    .windows(2)
                    .all(|pair| pair[1].number == pair[0].number + 1),
                markers: candidates,
            });
        }
    }
    runs.sort_by_key(|run| (run.range.start, run.range.end));
    for (run_index, run) in runs.iter_mut().enumerate() {
        run.id = format!("numeric-{:06}", run_index + 1);
        for (marker_index, marker) in run.markers.iter_mut().enumerate() {
            marker.id = format!("{}-{:04}", run.id, marker_index + 1);
        }
    }
    runs
}

pub(super) fn raw_enumerator_runs(text: &ScalarText<'_>) -> Vec<StructureCandidateRun> {
    #[derive(Clone)]
    struct RawEnumerator {
        family: u8,
        value: u32,
        start: usize,
        content_start: usize,
    }

    let mut by_family = BTreeMap::<u8, Vec<RawEnumerator>>::new();
    for line in lines(text) {
        let trimmed = line.text.trim_start_matches(instrument_space);
        let start =
            line.scalar_start + line.text[..line.text.len() - trimmed.len()].chars().count();
        let Some((token, at)) = instrument_marker(trimmed, true, true) else {
            continue;
        };
        let content_start = start + trimmed[..at].chars().count();
        for (family, value) in enum_readings(token).into_iter().flatten() {
            let Ok(value) = value.parse::<u32>() else {
                continue;
            };
            by_family.entry(family).or_default().push(RawEnumerator {
                family,
                value,
                start,
                content_start,
            });
        }
    }
    let mut boundaries = by_family
        .values()
        .flatten()
        .map(|marker| marker.start)
        .chain([text.len()])
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut runs = Vec::new();
    for markers in by_family.into_values() {
        let mut scopes = Vec::<Vec<RawEnumerator>>::new();
        for marker in markers {
            let target = scopes
                .iter()
                .enumerate()
                .filter(|(_, scope)| {
                    scope.last().is_some_and(|prior| {
                        prior.value < marker.value && marker.value - prior.value <= 8
                    })
                })
                .max_by_key(|(_, scope)| scope.last().unwrap().value)
                .map(|(index, _)| index);
            if marker.value == 1 || target.is_none() {
                scopes.push(vec![marker]);
            } else {
                scopes[target.unwrap()].push(marker);
            }
        }
        for scope in scopes.into_iter().filter(|scope| scope.len() >= 2) {
            let mut candidates = scope
                .iter()
                .map(|marker| {
                    let end = next_boundary(&boundaries, marker.start, text.len());
                    let surface_label = text
                        .slice(marker.start..marker.content_start)
                        .expect("enumerator marker range is bounded")
                        .trim()
                        .to_owned();
                    StructureMarkerCandidate {
                        id: String::new(),
                        range: ScalarRange {
                            start: marker.start,
                            end,
                        },
                        marker_range: ScalarRange {
                            start: marker.start,
                            end: marker.content_start,
                        },
                        label: surface_label,
                        grammar_value: format!("{}:{}", marker.family, marker.value),
                        parent_candidate_id: None,
                        level: 0,
                        content_start: marker.content_start,
                    }
                })
                .collect::<Vec<_>>();
            let range = ScalarRange {
                start: candidates[0].range.start,
                end: candidates.last().unwrap().range.end,
            };
            let ordinal = runs.len() + 1;
            for (index, candidate) in candidates.iter_mut().enumerate() {
                candidate.id = format!("enumerator-{ordinal:06}-{:04}", index + 1);
            }
            runs.push(StructureCandidateRun {
                id: format!("enumerator-{ordinal:06}"),
                grammar: CandidateGrammar::Enumerator,
                range,
                rooted: scope[0].value == 1,
                consecutive: scope
                    .windows(2)
                    .all(|pair| pair[1].value == pair[0].value + 1),
                markers: candidates,
            });
        }
    }
    runs.sort_by_key(|run| (run.range.start, run.range.end));
    runs
}

fn recover_contiguous(
    text: &ScalarText<'_>,
    markers: &[Marker],
    style: MarkerStyle,
) -> Vec<Marker> {
    let line = markers
        .iter()
        .filter(|value| value.style == style)
        .cloned()
        .collect::<Vec<_>>();
    if line.is_empty() {
        return line;
    }
    let candidates = heading_joined(text, &line.iter().map(|value| value.start).collect(), style);
    let within = |number: u32, from: usize, to: usize, formal: bool, sentence: bool| {
        candidates
            .iter()
            .filter(|value| {
                value.number == number
                    && value.start > from
                    && value.start < to
                    && ((!formal || value.formal) && (!sentence || value.sentence))
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    let mut recovered = HashMap::<usize, Marker>::new();
    for pair in line.windows(2) {
        if pair[0].number >= pair[1].number {
            continue;
        }
        let mut found = Vec::new();
        for number in pair[0].number + 1..pair[1].number {
            let candidates = within(number, pair[0].start, pair[1].start, true, false);
            if candidates.len() != 1 {
                found.clear();
                break;
            }
            found.push(candidates[0].clone());
        }
        for marker in found {
            recovered.insert(marker.start, marker);
        }
        if pair[1].number == pair[0].number + 2 {
            let candidates = within(
                pair[0].number + 1,
                pair[0].start,
                pair[1].start,
                false,
                true,
            );
            if candidates.len() == 1 {
                recovered.insert(candidates[0].start, candidates[0].clone());
            }
        }
    }
    if let Some(first) = line.first().filter(|value| value.number > 1) {
        let candidates = candidates
            .iter()
            .filter(|value| {
                value.number == first.number - 1
                    && value.start < first.start
                    && value.formal
                    && text.utf16(first.start) - text.utf16(value.start) <= 2_000
            })
            .cloned()
            .collect::<Vec<_>>();
        if candidates.len() == 1 {
            recovered.insert(candidates[0].start, candidates[0].clone());
        }
    }
    let mut result = line;
    result.extend(recovered.into_values());
    result.sort_by_key(|value| value.start);
    result
}

fn fill_lossy_marker_gaps(
    text: &ScalarText<'_>,
    spine: &[Marker],
    style: MarkerStyle,
) -> Vec<Marker> {
    let candidates = heading_joined(
        text,
        &spine.iter().map(|value| value.start).collect(),
        style,
    );
    let mut recovered = HashMap::<u32, Vec<Marker>>::new();
    for candidate in candidates {
        let before = spine
            .iter()
            .rev()
            .find(|value| value.start < candidate.start);
        let after = spine.iter().find(|value| value.start > candidate.start);
        let between = before.zip(after).is_some_and(|(left, right)| {
            left.number < candidate.number && candidate.number < right.number
        });
        let leading = before.is_none()
            && after.is_some_and(|right| {
                candidate.number > 0
                    && candidate.number < right.number
                    && right.number - candidate.number <= 2
                    && text.utf16(right.start) - text.utf16(candidate.start) <= 2_000
            });
        let sentence = before.zip(after).is_some_and(|(left, right)| {
            left.number + 1 == candidate.number && candidate.number + 1 == right.number
        }) && candidate.sentence;
        if (between || leading) && (candidate.formal || sentence) {
            recovered
                .entry(candidate.number)
                .or_default()
                .push(candidate);
        }
    }
    let mut result = spine.to_vec();
    result.extend(
        recovered
            .into_values()
            .filter(|values| values.len() == 1)
            .flatten(),
    );
    result.sort_by_key(|value| value.start);
    result
}

fn quoted_dot(text: &ScalarText<'_>, marker: &Marker) -> bool {
    if marker.style != MarkerStyle::Dot {
        return false;
    }
    let start = text.byte(marker.start);
    let end = text.value[start..]
        .find('\n')
        .map_or(text.value.len(), |at| start + at);
    let line = &text.value[start..end];
    cached_regex!(OPEN, r"^\d{1,4}\.\s+\(\d{1,4}\)\s+").is_match(line)
        && cached_regex!(WORD, r"(?i)\b(?:Act|Code|Regulations?|Rules?|shall|must)\b")
            .is_match(line)
}

struct Hypothesis {
    style: MarkerStyle,
    markers: Vec<Marker>,
    all: Vec<Marker>,
    short: bool,
    score: f64,
}

fn paragraph_ranges(
    text: &ScalarText<'_>,
    selected: &[Marker],
    all: &[Marker],
    style: MarkerStyle,
    fill_gaps: bool,
    extra: &[usize],
) -> Vec<Block> {
    let selected = if fill_gaps && style != MarkerStyle::Bare {
        fill_lossy_marker_gaps(text, selected, style)
    } else {
        selected.to_vec()
    };
    let mut boundaries = all
        .iter()
        .filter(|value| value.style == style)
        .map(|value| value.start)
        .chain(selected.iter().map(|value| value.start))
        .chain(extra.iter().copied())
        .chain([text.len()])
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    selected
        .into_iter()
        .map(|marker| {
            let end = next_boundary(&boundaries, marker.start, text.len());
            Block::labelled(
                NodeKind::Paragraph,
                format!("par{}", marker.number),
                marker.start,
                end,
            )
        })
        .collect()
}

fn detect_paragraphs(
    text: &ScalarText<'_>,
    profile: DetectionProfile,
    excluded: &[ScalarRange],
) -> Vec<Block> {
    let strict = profile != DetectionProfile::CaseLossy;
    let contiguous = profile == DetectionProfile::CaseContiguousComplete;
    let markers = paragraph_markers(text, contiguous);
    let visible = markers
        .iter()
        .filter(|value| marker_visible(value, excluded) && (!strict || !quoted_dot(text, value)))
        .cloned()
        .collect::<Vec<_>>();
    let mut hypotheses = Vec::<Hypothesis>::new();
    for style in [MarkerStyle::Bracket, MarkerStyle::Dot, MarkerStyle::Bare] {
        if profile == DetectionProfile::CaseRootedComplete {
            let mut candidates = visible
                .iter()
                .filter(|value| value.style == style)
                .cloned()
                .collect::<Vec<_>>();
            if style != MarkerStyle::Bare {
                let known = candidates.iter().map(|value| value.start).collect();
                candidates.extend(heading_joined(text, &known, style));
            }
            let (chain, score) = rooted_chain(candidates.clone());
            if chain.len() < 2 || endnote_shaped(text, &chain) {
                continue;
            }
            if chain.len() >= 5 {
                hypotheses.push(Hypothesis {
                    style,
                    markers: chain,
                    all: candidates,
                    short: false,
                    score,
                });
            } else if style == MarkerStyle::Bracket && sole_chain(&chain, &candidates) {
                hypotheses.push(Hypothesis {
                    style,
                    markers: chain,
                    all: candidates,
                    short: true,
                    score,
                });
            }
            continue;
        }
        let style_markers: Vec<Marker> = if contiguous && style != MarkerStyle::Bare {
            recover_contiguous(text, &visible, style)
                .into_iter()
                .filter(|value| marker_visible(value, excluded))
                .collect()
        } else {
            markers
                .iter()
                .filter(|value| value.style == style)
                .cloned()
                .collect()
        };
        let scopes = if contiguous {
            contiguous_scopes(&style_markers)
        } else {
            monotone_scopes(&style_markers, 8)
        };
        for scope in scopes.iter() {
            if scope.len() >= 5 {
                hypotheses.push(Hypothesis {
                    style,
                    markers: scope.clone(),
                    all: style_markers.clone(),
                    short: false,
                    score: 0.0,
                });
            } else if style == MarkerStyle::Bracket
                && scope.len() >= 2
                && scope
                    .iter()
                    .enumerate()
                    .all(|(index, value)| value.number == index as u32 + 1)
                && (!strict
                    || (scopes
                        .iter()
                        .all(|other| std::ptr::eq(other, scope) || other.len() == 1)
                        && style_markers.iter().all(|value| {
                            scope.iter().any(|mark| mark.start == value.start)
                                || value.number > scope.last().unwrap().number + 1
                        })))
            {
                hypotheses.push(Hypothesis {
                    style,
                    markers: scope.clone(),
                    all: style_markers.clone(),
                    short: true,
                    score: 0.0,
                });
            }
        }
    }
    if profile != DetectionProfile::CaseRootedComplete
        && hypotheses
            .iter()
            .any(|value| !value.short && value.markers[0].number <= 5)
    {
        hypotheses.retain(|value| value.short || value.markers[0].number <= 5);
    }
    let rank = |style| match style {
        MarkerStyle::Bracket => 2,
        MarkerStyle::Dot => 1,
        MarkerStyle::Bare => 0,
    };
    hypotheses.sort_by(|left, right| {
        left.short.cmp(&right.short).then_with(|| {
            if profile == DetectionProfile::CaseRootedComplete {
                right
                    .score
                    .total_cmp(&left.score)
                    .then(rank(right.style).cmp(&rank(left.style)))
            } else {
                right
                    .markers
                    .len()
                    .cmp(&left.markers.len())
                    .then(rank(right.style).cmp(&rank(left.style)))
                    .then(left.markers[0].number.cmp(&right.markers[0].number))
            }
        })
    });
    for hypothesis in hypotheses {
        let offsets = hypothesis
            .all
            .iter()
            .filter(|value| value.style == hypothesis.style)
            .map(|value| value.start)
            .collect::<Vec<_>>();
        let mut next = HashMap::new();
        for (index, start) in offsets.iter().enumerate() {
            next.insert(
                *start,
                offsets.get(index + 1).copied().unwrap_or(text.len()),
            );
        }
        let preliminary = hypothesis
            .markers
            .iter()
            .map(|marker| ScalarRange {
                start: marker.start,
                end: next.get(&marker.start).copied().unwrap_or(text.len()),
            })
            .collect::<Vec<_>>();
        let counts = preliminary
            .iter()
            .map(|range| {
                if range.end >= range.start {
                    word_count(
                        text.slice(range.start..range.end)
                            .expect("section range is bounded"),
                        contiguous,
                    )
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let bounded = if counts.len() > 1 {
            &counts[..counts.len() - 1]
        } else {
            &counts[..]
        };
        let median = median(bounded);
        let mean = bounded.iter().sum::<usize>() as f64 / bounded.len().max(1) as f64;
        let maximum = bounded.iter().copied().max().unwrap_or(0);
        // SourceDocs scores the unmodified hypothesis. Lossy heading inference
        // changes only the returned ranges after that hypothesis is accepted.
        let start = text.utf16(preliminary[0].start) as f64 / text.utf16_len().max(1) as f64;
        let span = (text.utf16(preliminary.last().unwrap().start)
            - text.utf16(preliminary[0].start)) as f64
            / text.utf16_len().max(1) as f64;
        if hypothesis.short {
            if text.utf16_len() <= 6_000
                && (text.utf16(preliminary[0].start) <= 1_200 || start <= 0.5)
                && counts.iter().copied().max().unwrap_or(0) >= 30
            {
                return paragraph_ranges(
                    text,
                    &hypothesis.markers,
                    &hypothesis.all,
                    hypothesis.style,
                    !strict,
                    &excluded.iter().map(|value| value.start).collect::<Vec<_>>(),
                );
            }
            continue;
        }
        let substantive =
            counts.iter().filter(|value| **value >= 12).count() as f64 / preliminary.len() as f64;
        if !(median >= 12.0 || mean >= 20.0 || maximum >= 30)
            || span < 0.05
            || (hypothesis.style == MarkerStyle::Bracket
                && text.utf16_len() > 6_000
                && start > 0.7
                && substantive < 0.5)
            || (hypothesis.style != MarkerStyle::Bracket && substantive < 0.7)
            || (hypothesis.style == MarkerStyle::Bare
                && (median < 20.0 || span < 0.15 || start > 0.7))
        {
            continue;
        }
        return paragraph_ranges(
            text,
            &hypothesis.markers,
            &hypothesis.all,
            hypothesis.style,
            !strict,
            &excluded.iter().map(|value| value.start).collect::<Vec<_>>(),
        );
    }
    Vec::new()
}

fn gapped_paragraphs(blocks: &[Block]) -> bool {
    let values = blocks
        .iter()
        .filter(|value| value.kind == NodeKind::Paragraph)
        .filter_map(|value| {
            value
                .label
                .as_deref()?
                .strip_prefix("par")?
                .parse::<u32>()
                .ok()
        })
        .collect::<Vec<_>>();
    values.windows(2).any(|pair| pair[1] != pair[0] + 1)
}

fn clipped_case_paragraphs(
    mut blocks: Vec<Block>,
    excluded: &[ScalarRange],
    text: &ScalarText<'_>,
) -> Vec<Block> {
    blocks.retain_mut(|block| {
        for range in excluded {
            if range.start >= block.range.end {
                break;
            }
            if range.end <= block.range.start {
                continue;
            }
            if range.start <= block.range.start {
                return false;
            }
            block.range.end = range.start;
        }
        block.range.end > block.range.start
            && text
                .slice(block.range.start..block.range.end)
                .expect("block range is bounded")
                .chars()
                .any(char::is_alphabetic)
    });
    blocks
}

#[derive(Clone)]
struct PageMarker {
    number: u32,
    start: usize,
    content_start: usize,
}

fn page_markers(text: &ScalarText<'_>, report_start: Option<u32>) -> Vec<PageMarker> {
    let page_word = cached_regex!(PAGE_WORD, r"(?iu)\bpage\b");
    if !page_word.is_match(text.value) {
        return Vec::new();
    }
    let regex = cached_regex!(
        VALUE,
        r"(?imu)\[[ \t]*pages?[ \t]*[.:,;]?[ \t]*(\d{1,4})[ \t]*[.:,;]?[ \t]*[\]\[)}]?[ \t]*[.,;:]?|^[ \t]*\[?[ \t]*page[ \t]*[.:,;]?[ \t]*(\d{1,4})[ \t]*[\])}]?[ \t]*[.,;:]?[ \t]*$"
    );
    let mut result = Vec::new();
    for line in lines(text).filter(|line| {
        line.text
            .as_bytes()
            .windows(4)
            .any(|word| word.eq_ignore_ascii_case(b"page"))
    }) {
        for capture in regex.captures_iter(line.text) {
            let whole = capture.get(0).unwrap();
            let number = capture
                .get(1)
                .or_else(|| capture.get(2))
                .unwrap()
                .as_str()
                .parse::<u32>()
                .unwrap();
            if report_start.is_some_and(|start| number < start) {
                continue;
            }
            let start = line.scalar_start + line.text[..whole.start()].chars().count();
            let content_start = line.scalar_start + line.text[..whole.end()].chars().count();
            result.push(PageMarker {
                number,
                start,
                content_start,
            });
        }
    }
    result
}

fn detect_pages(
    text: &ScalarText<'_>,
    report_start: Option<u32>,
    require_report_start: bool,
) -> Vec<Block> {
    if require_report_start && report_start.is_none() {
        return Vec::new();
    }
    let mut scopes = Vec::<Vec<PageMarker>>::new();
    let mut by_last = HashMap::<u32, Vec<usize>>::new();
    for marker in page_markers(text, report_start) {
        let candidates = marker
            .number
            .checked_sub(1)
            .and_then(|number| by_last.get(&number))
            .cloned()
            .unwrap_or_default();
        let index = candidates
            .into_iter()
            .reduce(|best, current| {
                if scopes[current].last().unwrap().start > scopes[best].last().unwrap().start {
                    current
                } else {
                    best
                }
            })
            .unwrap_or(scopes.len());
        if index == scopes.len() {
            scopes.push(vec![marker.clone()]);
        } else {
            let previous = scopes[index].last().unwrap().number;
            if let Some(values) = by_last.get_mut(&previous) {
                values.retain(|value| *value != index);
            }
            scopes[index].push(marker.clone());
        }
        by_last.entry(marker.number).or_default().push(index);
    }
    scopes.retain(|scope| scope.len() >= 3);
    scopes.sort_by_key(|scope| std::cmp::Reverse(scope.len()));
    if scopes.is_empty()
        || scopes
            .get(1)
            .is_some_and(|other| other.len() == scopes[0].len())
    {
        return Vec::new();
    }
    let best = &scopes[0];
    let mut blocks = best
        .windows(2)
        .map(|pair| {
            Block::labelled(
                NodeKind::Page,
                format!("page{}", pair[0].number),
                pair[0].content_start,
                pair[1].start,
            )
        })
        .collect::<Vec<_>>();
    if report_start.is_some_and(|start| best[0].number == start + 1) {
        blocks.insert(
            0,
            Block::labelled(
                NodeKind::Page,
                format!("page{}", report_start.unwrap()),
                0,
                best[0].start,
            ),
        );
    }
    blocks
}

fn detect_case(
    text: &ScalarText<'_>,
    profile: DetectionProfile,
    report_start_page: Option<u32>,
    require_report_start: bool,
    excluded: &[ScalarRange],
) -> Vec<Block> {
    let mut paragraphs = match profile {
        DetectionProfile::CaseContiguousComplete => {
            let complete = clipped_case_paragraphs(
                detect_paragraphs(text, DetectionProfile::CaseLossy, excluded),
                excluded,
                text,
            );
            if !complete.is_empty() && !gapped_paragraphs(&complete) {
                complete
            } else {
                detect_paragraphs(text, DetectionProfile::CaseContiguousComplete, excluded)
            }
        }
        profile => detect_paragraphs(text, profile, excluded),
    };
    paragraphs.extend(detect_pages(text, report_start_page, require_report_start));
    paragraphs
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum SectionStyle {
    Integer,
    Dot,
    DotTerm,
    Hyphen,
    Mixed,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum SectionFamily {
    Bare,
    DotTerm,
    Markdown,
    Emphasis,
    Range,
}

#[derive(Clone)]
pub(super) struct SectionMark {
    pub(super) label: String,
    pub(super) start: usize,
    pub(super) content_start: usize,
    pub(super) style: SectionStyle,
    pub(super) family: SectionFamily,
    pub(super) aliases: Vec<String>,
}

#[derive(Clone)]
struct LabelPart {
    separator: char,
    digits: Option<String>,
    text: String,
    suffix: u32,
}

fn suffix_value(value: &str) -> u32 {
    value
        .to_ascii_uppercase()
        .bytes()
        .fold(0, |total, value| total * 26 + u32::from(value - b'A' + 1))
}

fn label_parts(label: &str) -> Vec<LabelPart> {
    let mut separator = '\0';
    label
        .split_inclusive(['.', '-'])
        .filter_map(|piece| {
            let (body, next) = piece
                .strip_suffix('.')
                .map(|value| (value, '.'))
                .or_else(|| piece.strip_suffix('-').map(|value| (value, '-')))
                .unwrap_or((piece, '\0'));
            if body.is_empty() {
                separator = next;
                return None;
            }
            let digits = body.bytes().take_while(u8::is_ascii_digit).count();
            let numeric = (digits > 0
                && body[digits..]
                    .chars()
                    .all(|value| value.is_ascii_alphabetic()))
            .then(|| body[..digits].to_owned());
            let value = LabelPart {
                separator,
                digits: numeric,
                text: body.to_owned(),
                suffix: suffix_value(&body[digits..]),
            };
            separator = next;
            Some(value)
        })
        .collect()
}

fn compare_parts(left: &[LabelPart], right: &[LabelPart], fraction: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    for index in 0..left.len().max(right.len()) {
        let (Some(a), Some(b)) = (left.get(index), right.get(index)) else {
            return left.len().cmp(&right.len());
        };
        if a.separator != b.separator {
            return a.separator.cmp(&b.separator);
        }
        match (&a.digits, &b.digits) {
            (Some(a_digits), Some(b_digits)) => {
                let width = a_digits.len().max(b_digits.len());
                let ordered = if fraction && a.separator == '.' {
                    format!("{a_digits:0<width$}").cmp(&format!("{b_digits:0<width$}"))
                } else {
                    format!("{:0>width$}", a_digits.trim_start_matches('0'))
                        .cmp(&format!("{:0>width$}", b_digits.trim_start_matches('0')))
                };
                if ordered != Equal {
                    return ordered;
                }
                if a_digits.len() != b_digits.len() {
                    return a_digits.len().cmp(&b_digits.len());
                }
                if a.suffix != b.suffix {
                    return a.suffix.cmp(&b.suffix);
                }
            }
            (Some(_), None) => return Less,
            (None, Some(_)) => return Greater,
            (None, None) => {
                let ordered = a
                    .text
                    .to_ascii_uppercase()
                    .cmp(&b.text.to_ascii_uppercase());
                if ordered != Equal {
                    return ordered;
                }
            }
        }
    }
    Equal
}

pub(super) fn compare_labels(left: &str, right: &str, fraction: bool) -> std::cmp::Ordering {
    compare_parts(&label_parts(left), &label_parts(right), fraction)
}

fn numeric_label(value: &str, markdown: bool) -> Option<(&str, usize)> {
    let matched = cached_regex!(VALUE, r"^\d{1,8}(?:[.-]\d{1,8}){0,3}[A-Z]{0,2}").find(value)?;
    let label = matched.as_str();
    (!markdown || label.contains(['.', '-'])).then_some((label, matched.end()))
}

fn provision_label(value: &str) -> Option<(&str, usize)> {
    let matched = cached_regex!(VALUE,
    r"^(?:\d{1,8}[A-Za-z]{0,3}(?:[.-]\d{1,8}[A-Za-z]{0,3}){0,3}|[A-Za-z]{1,3}(?:[.-][0-9A-Za-z]{1,8}){1,3})"
).find(value)?;
    Some((matched.as_str(), matched.end()))
}

fn section_style(label: &str, trailing: bool) -> SectionStyle {
    if trailing {
        SectionStyle::DotTerm
    } else if label.contains('.') && label.contains('-') {
        SectionStyle::Mixed
    } else if label.contains('-') {
        SectionStyle::Hyphen
    } else if label.contains('.') {
        SectionStyle::Dot
    } else {
        SectionStyle::Integer
    }
}

fn bare_content_starts(value: &str) -> bool {
    value.chars().next().is_some_and(|character| {
        character.is_alphabetic() || character.is_ascii_digit() || "([*“\"«".contains(character)
    })
}

fn dotterm_content_starts(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| character.is_alphanumeric() || "\"'“«(".contains(character))
}

fn markdown_range_continuation(value: &str) -> bool {
    cached_regex!(
        VALUE,
        r"(?iu)^[ \t]*#{1,6}[ \t]+.*(?:[ \t](?:to|à)|[-–—])[ \t]*$"
    )
    .is_match(value)
}

fn previous_nonblank<'a>(source: &'a [Line<'a>], index: usize) -> Option<&'a str> {
    source[..index]
        .iter()
        .rev()
        .map(|line| line.text)
        .find(|value| !value.trim().is_empty())
}

fn section_mark(
    text: &ScalarText<'_>,
    source: &[Line<'_>],
    index: usize,
    family: SectionFamily,
) -> Option<SectionMark> {
    let line = &source[index];
    let lead = leading_ascii_space(line.text);
    let mut value = &line.text[lead..];
    if family == SectionFamily::Markdown {
        let hashes = value.bytes().take_while(|byte| *byte == b'#').count();
        if !(1..=6).contains(&hashes) || !value[hashes..].starts_with([' ', '\t']) {
            return None;
        }
        value = value[hashes..].trim_start_matches([' ', '\t']);
    }
    let bold = value.starts_with("**");
    if bold {
        value = &value[2..];
    }
    let (label, length) = numeric_label(value, family == SectionFamily::Markdown)?;
    let mut after = length;
    if bold {
        if !value[after..].starts_with("**") {
            return None;
        }
        after += 2;
    }
    let mut trailing = false;
    if family == SectionFamily::DotTerm {
        let punctuation = value[after..]
            .chars()
            .next()
            .filter(|value| matches!(value, '.' | ')'))?;
        after += punctuation.len_utf8();
        trailing = true;
    } else if family == SectionFamily::Markdown && value[after..].starts_with('.') {
        after += 1;
        trailing = true;
    }
    let rest = &value[after..];
    let spaces = leading_ascii_space(rest);
    let content = &rest[spaces..];
    let accepted = match family {
        SectionFamily::Bare => {
            content.is_empty()
                || (spaces > 0 && bare_content_starts(content))
                || (spaces == 0 && content.starts_with('('))
        }
        SectionFamily::DotTerm => {
            !content.is_empty()
                && dotterm_content_starts(content)
                && (spaces > 0 || content.starts_with('('))
        }
        SectionFamily::Markdown => content.is_empty() || (spaces > 0 && !content.is_empty()),
        _ => false,
    };
    if !accepted
        || family == SectionFamily::Bare
            && content.is_empty()
            && previous_nonblank(source, index).is_some_and(markdown_range_continuation)
    {
        return None;
    }
    Some(SectionMark {
        label: label.to_owned(),
        start: text.scalar(line.byte_start + lead),
        content_start: text.scalar(line.byte_end - content.len()),
        style: section_style(label, trailing),
        family,
        aliases: Vec::new(),
    })
}

fn collect_section_families(text: &ScalarText<'_>) -> [Vec<SectionMark>; 3] {
    const FAMILIES: [SectionFamily; 3] = [
        SectionFamily::Bare,
        SectionFamily::DotTerm,
        SectionFamily::Markdown,
    ];
    let source = lines(text).collect::<Vec<_>>();
    let mut result = std::array::from_fn(|_| Vec::new());
    for index in 0..source.len() {
        for (family, marks) in FAMILIES.into_iter().zip(&mut result) {
            if let Some(mark) = section_mark(text, &source, index, family) {
                marks.push(mark);
            }
        }
    }
    result
}

fn section_key(label: &str) -> impl Iterator<Item = u64> + '_ {
    label.split(['.', '-']).filter_map(|value| {
        value
            .bytes()
            .take_while(u8::is_ascii_digit)
            .fold(None, |total, digit| {
                Some(total.unwrap_or(0) * 10 + u64::from(digit - b'0'))
            })
    })
}

fn section_scopes(
    marks: &[SectionMark],
    styles: &[SectionStyle],
    root: bool,
    fraction: bool,
) -> Vec<Vec<SectionMark>> {
    let mut scopes = Vec::<Vec<SectionMark>>::new();
    for mark in marks.iter().filter(|value| styles.contains(&value.style)) {
        let parts = label_parts(&mark.label);
        let best = scopes
            .iter()
            .enumerate()
            .filter(|(_, scope)| {
                let last = scope.last().unwrap();
                let prior = label_parts(&last.label);
                parts.len() == prior.len() && compare_parts(&parts, &prior, fraction).is_gt()
            })
            .reduce(|best, candidate| {
                if compare_labels(
                    &candidate.1.last().unwrap().label,
                    &best.1.last().unwrap().label,
                    fraction,
                )
                .is_gt()
                {
                    candidate
                } else {
                    best
                }
            })
            .map(|value| value.0);
        if let Some(best) = best {
            scopes[best].push(mark.clone());
        } else {
            scopes.push(vec![mark.clone()]);
        }
        if scopes.len() > 8 {
            let smallest = (0..scopes.len())
                .min_by_key(|index| scopes[*index].len())
                .unwrap();
            scopes.remove(smallest);
        }
    }
    scopes
        .into_iter()
        .filter(|scope| {
            scope.len() >= 3 && (!root || section_key(&scope[0].label).all(|value| value == 1))
        })
        .collect()
}

fn expand_descendants(
    scope: Vec<SectionMark>,
    marks: &[SectionMark],
    length: usize,
) -> Vec<SectionMark> {
    if scope
        .first()
        .is_none_or(|value| section_key(&value.label).count() != 1)
    {
        return scope;
    }
    let mut result = Vec::with_capacity(scope.len());
    let mut cursor = 0;
    for (index, parent) in scope.iter().enumerate() {
        let end = scope.get(index + 1).map_or(length, |value| value.start);
        while marks
            .get(cursor)
            .is_some_and(|mark| mark.start <= parent.start)
        {
            cursor += 1;
        }
        let begin = cursor;
        while marks.get(cursor).is_some_and(|mark| mark.start < end) {
            cursor += 1;
        }
        let root = section_key(&parent.label).next();
        let mut descendants = Vec::new();
        let mut counts = HashMap::<&str, usize>::new();
        for mark in &marks[begin..cursor] {
            if matches!(mark.style, SectionStyle::Dot | SectionStyle::DotTerm)
                && mark.label.contains('.')
                && section_key(&mark.label).next() == root
            {
                descendants.push(mark);
                *counts.entry(mark.label.as_str()).or_default() += 1;
            }
        }
        result.push(parent.clone());
        result.extend(
            descendants
                .into_iter()
                .filter(|value| counts.get(value.label.as_str()) == Some(&1))
                .cloned(),
        );
    }
    result
}

fn same_labels(left: &[SectionMark], right: &[SectionMark]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.label == right.label)
}

fn choose_sections(
    left: Option<Vec<SectionMark>>,
    right: Option<Vec<SectionMark>>,
) -> Option<Vec<SectionMark>> {
    match (left, right) {
        (None, value) | (value, None) => value,
        (Some(left), Some(right)) if same_labels(&left, &right) => Some(left),
        (Some(left), Some(right)) if left[0].start != right[0].start => {
            Some(if left[0].start < right[0].start {
                left
            } else {
                right
            })
        }
        (Some(left), Some(right)) if left.len() != right.len() => {
            Some(if left.len() > right.len() {
                left
            } else {
                right
            })
        }
        _ => None,
    }
}

fn section_guard(value: &[SectionMark], text: &ScalarText<'_>) -> bool {
    !value.is_empty()
        && text.utf16_len() > 0
        && text.utf16(value[0].start) as f64 / text.utf16_len() as f64 <= 0.7
}

fn scope_winner(
    scopes: Vec<Vec<SectionMark>>,
    marks: &[SectionMark],
    text: &ScalarText<'_>,
) -> Option<Vec<SectionMark>> {
    let mut values = scopes
        .into_iter()
        .map(|scope| {
            if scope.first().is_some_and(|value| {
                value.style != SectionStyle::DotTerm && section_key(&value.label).count() == 1
            }) {
                expand_descendants(scope, marks, text.len())
            } else {
                scope
            }
        })
        .filter(|scope| section_guard(scope, text))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then(left[0].start.cmp(&right[0].start))
    });
    let best = values.first()?.clone();
    (!values.iter().skip(1).any(|value| {
        value.len() == best.len() && value[0].start == best[0].start && !same_labels(value, &best)
    }))
    .then_some(best)
}

fn statute_winner(
    marks: &[SectionMark],
    text: &ScalarText<'_>,
    allow_hyphen: bool,
) -> Option<Vec<SectionMark>> {
    if marks.len() < 3 {
        return None;
    }
    let mut component = section_scopes(
        marks,
        &[
            SectionStyle::Integer,
            SectionStyle::Dot,
            SectionStyle::DotTerm,
        ],
        false,
        false,
    );
    if allow_hyphen {
        component.extend(section_scopes(marks, &[SectionStyle::Hyphen], true, false));
        component.extend(section_scopes(marks, &[SectionStyle::Mixed], true, false));
    }
    choose_sections(
        scope_winner(component, marks, text),
        scope_winner(
            section_scopes(marks, &[SectionStyle::Dot], false, true),
            marks,
            text,
        ),
    )
}

fn inline_section(text: &ScalarText<'_>, mark: &SectionMark) -> bool {
    let start = text.byte(mark.content_start);
    let end = text.value[start..]
        .find('\n')
        .map_or(text.value.len(), |value| start + value);
    !text.value[start..end].trim().is_empty()
}

fn next_nonblank<'a>(source: &'a [Line<'a>], start: usize) -> Option<&'a Line<'a>> {
    source
        .iter()
        .find(|line| line.scalar_start > start && !line.text.trim().is_empty())
}

fn short_root(text: &ScalarText<'_>, families: &[Vec<SectionMark>; 3]) -> Vec<SectionMark> {
    let status = cached_regex!(
        STATUS,
        r"(?iu)^(?:\[\s*)?(?:repealed|revoked|abrog(?:ated|é|ée|és|ées)|renumbered|spent|not (?:yet )?in force|omitted)\b"
    );
    let heading = cached_regex!(HEADING, r#"^(?:(?:["'“«]\s*)?\p{Lu}|\(\d+\))"#);
    let source = lines(text).collect::<Vec<_>>();
    let mut candidates = families.iter().flatten().cloned().collect::<Vec<_>>();
    for line in &source {
        let value = line.text.trim_matches([' ', '\t']);
        if matches!(value, "1" | "2") {
            candidates.push(SectionMark {
                label: value.to_owned(),
                start: line.scalar_start + leading_ascii_space(line.text),
                content_start: line.scalar_start + line.text.chars().count(),
                style: SectionStyle::Integer,
                family: SectionFamily::Bare,
                aliases: Vec::new(),
            });
        }
    }
    candidates.retain(|value| matches!(value.label.as_str(), "1" | "2"));
    let mut invalid = false;
    for marker in &mut candidates {
        let start = text.byte(marker.content_start);
        let end = text.value[start..]
            .find('\n')
            .map_or(text.value.len(), |value| start + value);
        // Preserve the source grammar's raw offset check: on CRLF input a
        // label ending immediately before `\r` is not "at" the `\n` line
        // end, even though the intervening scalar is whitespace.
        if start >= end {
            let Some(next) = next_nonblank(&source, marker.start) else {
                invalid = true;
                continue;
            };
            let parenthetical = next.text.trim_start().starts_with('(')
                && next.text.trim_start()[1..]
                    .chars()
                    .next()
                    .is_some_and(char::is_numeric);
            if !heading.is_match(next.text.trim_start())
                && !parenthetical
                && !status.is_match(next.text.trim_start())
            {
                invalid = true;
            } else {
                marker.content_start = next.scalar_start + leading_ascii_space(next.text);
            }
        }
    }
    if invalid {
        return Vec::new();
    }
    candidates.sort_by_key(|value| value.start);
    candidates.dedup_by(|left, right| left.label == right.label && left.start == right.start);
    let ones = candidates
        .iter()
        .filter(|value| value.label == "1")
        .cloned()
        .collect::<Vec<_>>();
    let twos = candidates
        .iter()
        .filter(|value| value.label == "2")
        .cloned()
        .collect::<Vec<_>>();
    if ones.len() != 1
        || twos.len() > 1
        || twos.first().is_some_and(|two| two.start <= ones[0].start)
    {
        return Vec::new();
    }
    let result = if twos.is_empty() {
        vec![ones[0].clone()]
    } else {
        vec![ones[0].clone(), twos[0].clone()]
    };
    if text.utf16(result[0].start) as f64 / text.utf16_len().max(1) as f64 <= 0.7 {
        result
    } else {
        Vec::new()
    }
}

fn statute_spine_over(
    text: &ScalarText<'_>,
    allow_hyphen: bool,
    inline_only: bool,
    all_families: &[Vec<SectionMark>; 3],
) -> Vec<SectionMark> {
    let families = all_families.clone().map(|marks| {
        marks
            .into_iter()
            .filter(|mark| !inline_only || inline_section(text, mark))
            .collect::<Vec<_>>()
    });
    let mut candidates = families
        .iter()
        .filter_map(|marks| statute_winner(marks, text, allow_hyphen))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|value| value[0].start);
    if candidates.is_empty() {
        return short_root(text, all_families);
    }
    let mut best = candidates[0].clone();
    let first_start = best[0].start;
    for candidate in candidates
        .into_iter()
        .skip(1)
        .take_while(|value| value[0].start == first_start)
    {
        let Some(chosen) = choose_sections(Some(best), Some(candidate)) else {
            return Vec::new();
        };
        best = chosen;
    }
    if best[0].family == SectionFamily::DotTerm {
        let mut all = [families[0].clone(), families[1].clone()].concat();
        all.sort_by_key(|value| value.start);
        expand_descendants(best, &all, text.len())
    } else {
        best
    }
}

pub(super) fn statute_spine(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
    let families = collect_section_families(text);
    let result = statute_spine_over(text, allow_hyphen, false, &families);
    if result.is_empty() || result.iter().any(|value| inline_section(text, value)) {
        result
    } else {
        statute_spine_over(text, allow_hyphen, true, &families)
    }
}

fn dotted_order(marks: &[SectionMark]) -> Option<bool> {
    let dotted = marks
        .iter()
        .map(|value| &value.label)
        .filter(|value| value.contains('.') && !value.contains('-'))
        .collect::<Vec<_>>();
    let inversions = |fraction| {
        dotted
            .windows(2)
            .filter(|pair| compare_labels(pair[0], pair[1], fraction).is_gt())
            .count()
    };
    let component = inversions(false);
    let fraction = inversions(true);
    if component != fraction {
        return Some(fraction < component);
    }
    if dotted.windows(2).any(|pair| {
        compare_labels(pair[0], pair[1], false) != compare_labels(pair[0], pair[1], true)
    }) {
        None
    } else {
        Some(false)
    }
}

fn emphasis_sections(text: &ScalarText<'_>) -> Vec<SectionMark> {
    let mut candidates = Vec::new();
    for line in lines(text) {
        let lead = leading_ascii_space(line.text);
        let value = &line.text[lead..];
        let Some(value) = value.strip_prefix("**") else {
            continue;
        };
        let Some((label, length)) = provision_label(value) else {
            continue;
        };
        if !value[length..].starts_with("**") {
            continue;
        }
        let rest = &value[length + 2..];
        if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
            continue;
        }
        candidates.push(SectionMark {
            label: label.to_owned(),
            start: line.scalar_start + lead,
            content_start: line.scalar_start
                + lead
                + 2
                + label.chars().count()
                + 2
                + leading_ascii_space(rest),
            style: section_style(label, false),
            family: SectionFamily::Emphasis,
            aliases: Vec::new(),
        });
    }
    let Some(first) = candidates.first() else {
        return candidates;
    };
    let numeric = first
        .label
        .starts_with(|value: char| value.is_ascii_digit());
    candidates.retain(|value| {
        value
            .label
            .starts_with(|character: char| character.is_ascii_digit())
            == numeric
    });
    let Some(fraction) = dotted_order(&candidates) else {
        return Vec::new();
    };
    let mut result = Vec::<SectionMark>::new();
    for marker in candidates {
        if result
            .last()
            .is_none_or(|prior| compare_labels(&marker.label, &prior.label, fraction).is_gt())
        {
            result.push(marker);
        }
    }
    if result.is_empty()
        || text.utf16(result[0].start) as f64 / text.utf16_len().max(1) as f64 > 0.7
        || (text.utf16_len() - text.utf16(result[0].start)) as f64
            / (text.utf16_len().max(1) as f64)
            < 0.1
    {
        Vec::new()
    } else {
        result
    }
}

fn status_sections(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
    let regex = cached_regex!(
        VALUE,
        r"(?imu)^[ \t]*(?:\*\*)?(\d{1,4})(?:[ \t]+(?:to|through|and|à|a|et)[ \t]+|[ \t]*([-–—])[ \t]*)(\d{1,4})(?:\*\*)?[ \t]*[,;:]?[ \t]*(?:\[[ \t]*)?(?:repealed|revoked|abrog(?:ated|é|ée|és|ées)|renumbered|spent|not (?:yet )?in force|omitted)\b"
    );
    regex
        .captures_iter(text.value)
        .filter_map(|capture| {
            if allow_hyphen && capture.get(2).is_some() {
                return None;
            }
            let from = capture[1].parse::<u32>().ok()?;
            let to = capture[3].parse::<u32>().ok()?;
            if from >= to || to > from + 400 {
                return None;
            }
            let whole = capture.get(0).unwrap();
            Some(SectionMark {
                label: from.to_string(),
                start: text.scalar(whole.start() + leading_ascii_space(whole.as_str())),
                content_start: text.scalar(whole.end()),
                style: SectionStyle::Integer,
                family: SectionFamily::Range,
                aliases: (from + 1..=to).map(|value| value.to_string()).collect(),
            })
        })
        .collect()
}

fn coherent_sections(marks: &[SectionMark]) -> bool {
    let Some(fraction) = dotted_order(marks) else {
        return false;
    };
    marks
        .windows(2)
        .all(|pair| compare_labels(&pair[0].label, &pair[1].label, fraction).is_lt())
}

fn selected_sections(text: &ScalarText<'_>, allow_hyphen: bool) -> Vec<SectionMark> {
    let emphasis = emphasis_sections(text);
    let flat = statute_spine(text, allow_hyphen);
    let mut selected = if emphasis.is_empty() {
        flat
    } else if flat.is_empty() {
        emphasis
    } else {
        let occurrences = emphasis
            .iter()
            .map(|value| (value.label.to_ascii_lowercase(), value.content_start))
            .collect::<HashSet<_>>();
        if !flat.iter().any(|value| {
            occurrences.contains(&(value.label.to_ascii_lowercase(), value.content_start))
        }) {
            emphasis
        } else {
            let mut labels = flat
                .into_iter()
                .map(|value| (value.label.to_ascii_lowercase(), value))
                .collect::<HashMap<_, _>>();
            for marker in emphasis.iter().cloned() {
                let key = marker.label.to_ascii_lowercase();
                if labels
                    .get(&key)
                    .is_none_or(|value| value.content_start == marker.content_start)
                {
                    labels.insert(key, marker);
                }
            }
            let mut combined = labels.into_values().collect::<Vec<_>>();
            combined.sort_by_key(|value| value.start);
            if coherent_sections(&combined) {
                combined
            } else {
                emphasis
            }
        }
    };
    let ranges = status_sections(text, allow_hyphen);
    if !ranges.is_empty() {
        let mut labels = selected
            .iter()
            .cloned()
            .map(|value| (value.label.to_ascii_lowercase(), value))
            .collect::<HashMap<_, _>>();
        for marker in ranges {
            for alias in &marker.aliases {
                labels.remove(&alias.to_ascii_lowercase());
            }
            labels.insert(marker.label.to_ascii_lowercase(), marker);
        }
        let mut combined = labels.into_values().collect::<Vec<_>>();
        combined.sort_by_key(|value| value.start);
        if coherent_sections(&combined) {
            selected = combined;
        }
    }
    selected
}

fn roman_value(value: &str) -> Option<u32> {
    let mut total = 0i32;
    let mut prior = 0;
    for character in value.to_ascii_lowercase().bytes().rev() {
        let value = match character {
            b'i' => 1,
            b'v' => 5,
            b'x' => 10,
            b'l' => 50,
            b'c' => 100,
            b'd' => 500,
            b'm' => 1000,
            _ => return None,
        };
        total += if value < prior { -value } else { value };
        prior = prior.max(value);
    }
    (total > 0).then_some(total as u32)
}

struct EnumFrame {
    family: u8,
    value: String,
    label: String,
}

struct ChildMark<'a> {
    token: &'a str,
    start: usize,
    content_start: usize,
}

#[derive(Clone)]
pub(super) struct GrammarPoint {
    pub(super) range: ScalarRange,
    pub(super) label: String,
    pub(super) parent_label: Option<String>,
    pub(super) content_start: usize,
    pub(super) diagnostic: Option<&'static str>,
}

impl GrammarPoint {
    fn into_section(self) -> Block {
        Block {
            kind: NodeKind::Section,
            range: self.range,
            label: Some(self.label),
            aliases: Vec::new(),
            parent_label: self.parent_label,
            content_start: Some(self.content_start),
            diagnostic: self.diagnostic,
            source: Derivation::Heuristic,
            origin_id: ENGINE_ORIGIN,
        }
    }
}

#[derive(Default)]
struct StructureState {
    nodes: Vec<(GrammarPoint, usize)>,
    container: Option<String>,
    section: Option<(String, usize)>,
    stack: Vec<EnumFrame>,
    used: HashMap<String, usize>,
}

fn enum_readings(token: &str) -> [Option<(u8, String)>; 2] {
    if token
        .split('.')
        .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return [Some((0, token.to_owned())), None];
    }
    let lower = token.to_ascii_lowercase();
    let alpha = match lower.as_bytes() {
        [value @ b'a'..=b'z'] => Some(u32::from(*value - b'a' + 1)),
        [left @ b'a'..=b'z', right] if left == right => Some(26 + u32::from(*left - b'a' + 1)),
        _ => None,
    };
    let upper = token != lower;
    let alpha = alpha.map(|value| (if upper { 3 } else { 1 }, value.to_string()));
    let roman = roman_value(&lower).map(|value| (if upper { 4 } else { 2 }, value.to_string()));
    if lower.len() > 1 {
        [
            roman
                .clone()
                .filter(|(_, value)| value.parse::<u32>().unwrap() <= 50)
                .or(alpha)
                .or(roman),
            None,
        ]
    } else {
        [alpha, roman]
    }
}

fn instrument_marker(value: &str, tail: bool, dot: bool) -> Option<(&str, usize)> {
    let found = cached_regex!(MARKER,
        r"^(?:\((\d{1,3}(?:\.\d{1,3})?|[a-z]{1,2}|[ivxlcdm]{1,6}|[A-Z]{1,2}|[IVXLCDM]{1,6})\)[ \t]*(.*)|([a-z]{1,2}|[ivxlcdm]{1,6})\)[ \t]+(\S.*)|([a-z]{1,2}|[ivxlcdm]{1,6})\.[ \t]+(\S.*))$"
    ).captures(value)?;
    [(1, 2, true), (3, 4, tail), (5, 6, dot)]
        .into_iter()
        .find_map(|(token, rest, enabled)| {
            enabled.then_some((found.get(token)?.as_str(), found.get(rest)?.start()))
        })
}

fn instrument_space(character: char) -> bool {
    // Instrument grammar historically admits Rust whitespace, including U+0085;
    // it is therefore not the shared ECMAScript whitespace contract.
    character.is_whitespace() || character == '\u{feff}'
}

fn legislation_marker(value: &str, followed_by_newline: bool) -> Option<(&str, usize, usize)> {
    let found = cached_regex!(
        MARKER,
        r"^\((\d+(?:\.\d+)?|[A-Za-z](?:\.\d+)?|[ivxlcdmIVXLCDM]+)\)"
    )
    .captures(value)?;
    let whole = found.get(0).unwrap();
    let rest = &value[whole.end()..];
    (rest.chars().next().is_some_and(char::is_whitespace)
        || (rest.is_empty() && followed_by_newline))
        .then(|| {
            (
                found.get(1).unwrap().as_str(),
                whole.end() + leading_ascii_space(rest),
                whole.end() - 1,
            )
        })
}

fn compare_child_values(left: &str, right: &str) -> std::cmp::Ordering {
    let left = left.split('.').collect::<Vec<_>>();
    let right = right.split('.').collect::<Vec<_>>();
    for index in 0..left.len().max(right.len()) {
        let (left, right) = (
            left.get(index).copied().unwrap_or_default(),
            right.get(index).copied().unwrap_or_default(),
        );
        if left == right {
            continue;
        }
        let ordered = if index == 0 {
            left.parse::<f64>()
                .unwrap_or_default()
                .partial_cmp(&right.parse::<f64>().unwrap_or_default())
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            let width = left.len().max(right.len());
            format!("{left:0<width$}").cmp(&format!("{right:0<width$}"))
        };
        if !ordered.is_eq() {
            return ordered;
        }
    }
    std::cmp::Ordering::Equal
}

fn admitted_dialects(text: &ScalarText<'_>) -> (bool, bool) {
    let mut live = [HashMap::<u8, (String, usize)>::new(), HashMap::new()];
    let mut best = [0, 0];
    for line in lines(text) {
        let value = line.text.trim_matches(instrument_space);
        if instrument_marker(value, false, false).is_some() {
            continue;
        }
        for (index, dialect) in [(true, false), (false, true)].into_iter().enumerate() {
            if let Some((token, _)) = instrument_marker(value, dialect.0, dialect.1) {
                for (family, value) in enum_readings(token).into_iter().flatten() {
                    let state = live[index].entry(family).or_default();
                    if value == "1" {
                        *state = (value, 1);
                    } else if state.1 > 0 && compare_labels(&value, &state.0, true).is_gt() {
                        *state = (value, state.1 + 1);
                    }
                    best[index] = best[index].max(state.1);
                }
            }
        }
    }
    (best[0] >= 3, best[1] >= 3)
}

impl StructureState {
    fn emit_child(
        &mut self,
        token: &str,
        start: usize,
        content_start: usize,
        parent: String,
        depth: usize,
        code: &'static str,
    ) -> String {
        let base = format!("{parent}({token})");
        let occurrence = self.used.entry(base.clone()).or_insert(1);
        let label = (*occurrence > 1)
            .then(|| format!("{base}@{occurrence}"))
            .unwrap_or(base);
        *occurrence += 1;
        self.nodes.push((
            GrammarPoint {
                range: ScalarRange {
                    start,
                    end: usize::MAX,
                },
                label: label.clone(),
                parent_label: Some(parent),
                content_start,
                diagnostic: Some(code),
            },
            depth,
        ));
        label
    }

    fn child(&mut self, token: &str, start: usize, content_start: usize) {
        let (root, root_depth) = self.section.clone().unwrap();
        let readings = enum_readings(token);
        let selected = (0..4).find_map(|pass| {
            readings.iter().flatten().find_map(|(family, value)| {
                let at = self.stack.iter().rposition(|frame| frame.family == *family);
                match (pass, at) {
                    (0, Some(index)) => {
                        let prior = &self.stack[index].value;
                        match (prior.parse::<u32>(), value.parse::<u32>()) {
                            (Ok(prior), Ok(value)) => prior + 1 == value,
                            _ => compare_labels(value, prior, true).is_gt(),
                        }
                    }
                    (1, None) => value == "1" && self.stack.len() < 6,
                    (2, Some(index)) => {
                        value == "1"
                            || compare_labels(value, &self.stack[index].value, true).is_gt()
                    }
                    (3, None) => self.stack.len() < 6,
                    _ => false,
                }
                .then(|| (pass, *family, value.clone(), at))
            })
        });
        let (pass, family, value, at) = selected.unwrap_or_else(|| (4, 0, String::new(), None));
        let (parent, depth) = if pass == 4 {
            (root.clone(), root_depth + 1)
        } else if let Some(index) = at {
            self.stack.truncate(index + 1);
            self.stack[index].value = value.clone();
            (
                index
                    .checked_sub(1)
                    .map_or_else(|| root.clone(), |parent| self.stack[parent].label.clone()),
                root_depth + index + 1,
            )
        } else {
            let parent = self
                .stack
                .last()
                .map_or_else(|| root.clone(), |frame| frame.label.clone());
            let depth = root_depth + self.stack.len() + 1;
            self.stack.push(EnumFrame {
                family,
                value: value.clone(),
                label: String::new(),
            });
            (parent, depth)
        };
        let code = match (pass, value.as_str()) {
            (0, _) => "instrument_ladder_increment",
            (1, _) => "instrument_ladder_level_open",
            (2, "1") => "instrument_ladder_restart",
            (2, _) => "instrument_ladder_forward_jump",
            (3, _) => "instrument_ladder_midcounter_open",
            _ => "instrument_ladder_violation",
        };
        let label = self.emit_child(token, start, content_start, parent, depth, code);
        if pass < 4 {
            self.stack.last_mut().unwrap().label = label.clone();
        }
    }

    fn legislation_child(
        &mut self,
        token: &str,
        next: Option<&str>,
        start: usize,
        content_start: usize,
    ) {
        let (head, suffix) = token.split_once('.').unwrap_or((token, ""));
        let numeric = token
            .as_bytes()
            .first()
            .is_some_and(|value| value.is_ascii_digit());
        let roman = roman_value(head);
        let upper = head != head.to_ascii_lowercase();
        let alpha_level = if upper { 4 } else { 2 };
        let alpha_value = head
            .as_bytes()
            .first()
            .filter(|value| value.is_ascii_alphabetic())
            .map(|value| u32::from(value.to_ascii_lowercase() - b'a' + 1));
        let prior = |family| {
            self.stack
                .iter()
                .find(|frame| frame.family == family)
                .map(|frame| frame.value.as_str())
        };
        let roman_preferred = if head.len() > 1 {
            true
        } else if let Some(value) = roman {
            if prior(3)
                .and_then(|prior| prior.parse::<u32>().ok())
                .is_some_and(|prior| prior + 1 == value)
            {
                true
            } else if alpha_value.is_some_and(|value| {
                prior(alpha_level)
                    .and_then(|prior| prior.parse::<u32>().ok())
                    .is_some_and(|prior| prior + 1 == value)
            }) {
                head.eq_ignore_ascii_case("i")
                    && next.is_some_and(|next| next.eq_ignore_ascii_case("ii"))
            } else {
                !upper && head == "i" && self.stack.iter().any(|frame| frame.family == 2)
            }
        } else {
            false
        };
        let (family, value) = if numeric {
            (1, token.to_owned())
        } else if roman_preferred {
            (3, roman.unwrap().to_string())
        } else {
            let Some(alpha) = alpha_value else { return };
            (
                alpha_level,
                if suffix.is_empty() {
                    alpha.to_string()
                } else {
                    format!("{alpha}.{suffix}")
                },
            )
        };
        if prior(family).is_some_and(|prior| !compare_child_values(&value, prior).is_gt()) {
            return;
        }

        self.stack.retain(|frame| frame.family <= family);
        let at = self.stack.iter().position(|frame| frame.family == family);
        let (root, root_depth) = self.section.clone().unwrap();
        let parent = at
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| self.stack.get(index))
            .or_else(|| self.stack.iter().rev().find(|frame| frame.family < family))
            .map_or_else(|| root.clone(), |frame| frame.label.clone());
        if let Some(index) = at {
            self.stack[index].value = value.clone();
        } else {
            self.stack.push(EnumFrame {
                family,
                value,
                label: String::new(),
            });
        }
        let label = self.emit_child(
            token,
            start,
            content_start,
            parent,
            root_depth + usize::from(family),
            "legislation_child",
        );
        self.stack
            .iter_mut()
            .find(|frame| frame.family == family)
            .unwrap()
            .label = label;
    }
}

fn enumerated_children(
    value: &str,
    offset: usize,
    root: String,
    root_depth: usize,
    content_start: usize,
    inline_at_root: bool,
    leading_label: Option<&str>,
) -> Vec<Block> {
    let text = ScalarText::new(value);
    let public_parent = root.clone();
    let mut state = StructureState {
        section: Some((root, root_depth)),
        ..Default::default()
    };
    let mut markers = Vec::new();
    let leading = leading_label.and_then(|label| {
        let lead = leading_ascii_space(value);
        let rest = value[lead..].strip_prefix(label)?;
        let gap = leading_ascii_space(rest);
        let prefix = lead + label.len() + gap;
        let line = value[prefix..].lines().next().unwrap_or_default();
        let newline = line.len() < value[prefix..].len();
        let (token, content, close) = legislation_marker(line, newline)?;
        Some(ChildMark {
            token,
            start: value[..prefix + close].chars().count(),
            content_start: value[..prefix + content].chars().count(),
        })
    });
    if let Some(marker) = leading {
        markers.push(marker);
    } else {
        let inline = &text.value[text.byte(content_start)..];
        let inline_line = inline.lines().next().unwrap_or_default();
        if let Some((token, at, _)) = legislation_marker(inline_line, inline.contains('\n')) {
            markers.push(ChildMark {
                token,
                start: if inline_at_root { 0 } else { content_start },
                content_start: content_start + inline_line[..at].chars().count(),
            });
        }
    }
    for line in lines(&text) {
        let value = line.text.trim_start_matches(instrument_space);
        if let Some((token, at, _)) = legislation_marker(value, line.byte_end < text.value.len()) {
            if !markers
                .iter()
                .any(|marker| marker.start == line.scalar_start)
            {
                markers.push(ChildMark {
                    token,
                    start: line.scalar_start,
                    content_start: line.scalar_start
                        + line.text[..line.text.len() - value.len()].chars().count()
                        + value[..at].chars().count(),
                });
            }
        }
    }
    markers.sort_by_key(|marker| marker.start);
    for index in 0..markers.len() {
        let marker = &markers[index];
        state.legislation_child(
            marker.token,
            markers.get(index + 1).map(|next| next.token),
            marker.start,
            marker.content_start,
        );
    }
    for index in 0..state.nodes.len() {
        let start = state.nodes[index].0.range.start;
        let end = markers
            .iter()
            .find(|marker| marker.start > start)
            .map_or(text.len(), |marker| marker.start);
        state.nodes[index].0.range.end = end;
        state.nodes[index].0.range.start += offset;
        state.nodes[index].0.range.end += offset;
        state.nodes[index].0.content_start += offset;
        state.nodes[index].0.parent_label = Some(public_parent.clone());
    }
    state
        .nodes
        .into_iter()
        .map(|(point, _)| point.into_section())
        .collect()
}

fn direct_section(value: &str) -> Option<(String, usize)> {
    let found = cached_regex!(SECTION,
        r#"^(?:(?:Section|SECTION)\s+(\d{1,3}(?:\.\d{1,3})*[A-Za-z]?)[.)]?\s*[—–\-:]?\s*(["'“(A-Z].*|)|(\d{1,3}\.\d{1,3}(?:\.\d{1,3})*)\s+(["'“(A-Z].*)|((?:[0-4]?\d{1,2}|500))[.)]\s+(["'“(A-Z].*)|(\d{1,3}(?:\.\d{1,3}){0,3})[ \t]+(\(\d.*))$"#
    ).captures(value)?;
    let label = [1, 3, 5, 7]
        .into_iter()
        .find(|index| found.get(*index).is_some())?;
    Some((found[label].to_owned(), found.get(label + 1)?.start()))
}

fn instrument_top(value: &str, direct: bool) -> Option<(String, usize, bool)> {
    if let Some(found) = cached_regex!(TOP,
        r"^(?:(ARTICLE|Article|PART|Part|DIVISION|Division)\s+([IVXLCDM]+|\d{1,3})\b\s*[—–\-.:]?\s*(.*)|(SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)\s+([A-Z0-9][\w.\-]*)\s*[—–\-.:]?\s*(.*))$"
    ).captures(value) {
        let container = found.get(1).is_some();
        let (word, token, rest) = if container { (1, 2, 3) } else { (4, 5, 6) };
        let heading = &found[rest];
        if !container
            && !(heading.is_empty()
                || heading.starts_with(['"', '\'', '“', '('])
                || heading
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_uppercase()))
        {
            return None;
        }
        let word = found[word].to_ascii_lowercase();
        let prefix = match word.as_str() {
            "part" | "annex" => word.as_str(),
            "schedule" => &word[..5],
            _ => &word[..3],
        };
        let suffix = if container {
            found[token].parse().ok()
                .or_else(|| roman_value(&found[token]))?
                .to_string()
        } else {
            found[token].to_ascii_lowercase()
        };
        return Some((format!("{prefix}{suffix}"), found.get(rest)?.start(), true));
    }
    direct
        .then_some(value)
        .and_then(direct_section)
        .map(|(label, at)| (format!("sec{label}"), at, false))
}

pub(super) fn detect_instrument_grammar(text: &ScalarText<'_>) -> Vec<GrammarPoint> {
    let mut spine = statute_spine(text, false).into_iter().peekable();
    let direct = spine.peek().is_none();
    let dialects = admitted_dialects(text);
    let mut state = StructureState::default();
    for line in lines(text) {
        let trimmed_start = line.text.trim_start_matches(instrument_space);
        let value = trimmed_start.trim_end_matches(instrument_space);
        if value.is_empty() {
            continue;
        }
        let start = line.scalar_start
            + line.text[..line.text.len() - trimmed_start.len()]
                .chars()
                .count();
        let selected = spine
            .next_if(|mark| mark.start == start)
            .map(|mark| (format!("sec{}", mark.label), mark.content_start, false))
            .or_else(|| {
                instrument_top(value, direct).map(|(label, at, container)| {
                    (label, start + value[..at].chars().count(), container)
                })
            });
        if let Some((label, content_start, container)) = selected {
            let depth = usize::from(!container && state.container.is_some());
            state.nodes.push((
                GrammarPoint {
                    range: ScalarRange {
                        start,
                        end: usize::MAX,
                    },
                    label: label.clone(),
                    parent_label: (!container).then(|| state.container.clone()).flatten(),
                    content_start,
                    diagnostic: None,
                },
                depth,
            ));
            state.stack.clear();
            if container {
                state.container = Some(label);
                state.section = None;
            } else {
                state.section = Some((label, depth));
                let content_byte = text.byte(content_start);
                let inline = if content_byte <= line.byte_end {
                    &text.value[content_byte..line.byte_end]
                } else {
                    ""
                };
                if let Some((token, at)) = instrument_marker(inline, false, false) {
                    state.child(
                        token,
                        content_start,
                        content_start + inline[..at].chars().count(),
                    );
                }
            }
            continue;
        }
        if let (Some((token, at)), Some(_)) = (
            instrument_marker(value, dialects.0, dialects.1),
            state.section.as_ref(),
        ) {
            state.child(token, start, start + value[..at].chars().count());
        }
    }
    let mut open = Vec::<usize>::new();
    for index in 0..state.nodes.len() {
        let (start, depth) = (state.nodes[index].0.range.start, state.nodes[index].1);
        while open
            .last()
            .is_some_and(|prior| state.nodes[*prior].1 >= depth)
        {
            state.nodes[open.pop().expect("open node")].0.range.end = start;
        }
        open.push(index);
    }
    for index in open {
        state.nodes[index].0.range.end = text.len();
    }
    state.nodes.into_iter().map(|(point, _)| point).collect()
}

pub(crate) fn detect_instrument(text: &ScalarText<'_>) -> Vec<Block> {
    detect_instrument_grammar(text)
        .into_iter()
        .map(GrammarPoint::into_section)
        .collect()
}

fn detect_legislation(
    text: &ScalarText<'_>,
    allow_hyphenated_sections: bool,
    native_claims: &[NativeClaim],
) -> Vec<Block> {
    let sections = selected_sections(text, allow_hyphenated_sections);
    let mut result = Vec::new();
    for (index, section) in sections.iter().enumerate() {
        let end = sections
            .get(index + 1)
            .map_or(text.len(), |value| value.start);
        let label = format!("sec{}", section.label);
        let mut top = Block::labelled(NodeKind::Section, label, section.start, end);
        top.aliases = section
            .aliases
            .iter()
            .map(|value| format!("sec{value}"))
            .collect();
        top.content_start = Some(section.content_start);
        result.push(top);
        let value = text
            .slice(section.start..end)
            .expect("instrument section range is bounded");
        result.extend(enumerated_children(
            value,
            section.start,
            format!("sec{}", section.label),
            0,
            section.content_start - section.start,
            matches!(section.family, SectionFamily::Bare | SectionFamily::DotTerm),
            None,
        ));
    }
    for claim in native_claims
        .iter()
        .filter(|claim| claim.kind == EvidenceKind::Section && claim.parent_label.is_none())
    {
        let Some(label) = claim
            .label
            .as_deref()
            .and_then(|value| value.strip_prefix("sec"))
        else {
            continue;
        };
        if !provision_label(label).is_some_and(|(value, end)| value == label && end == label.len())
        {
            continue;
        }
        let value = text
            .slice(claim.range.start..claim.range.end)
            .expect("native claim range is bounded");
        let lead = leading_ascii_space(value);
        let content_start = value[lead..]
            .strip_prefix(label)
            .map_or(claim.range.start, |_| {
                claim.range.start + value[..lead].chars().count() + label.chars().count()
            });
        let children = enumerated_children(
            value,
            claim.range.start,
            format!("sec{label}"),
            0,
            content_start - claim.range.start,
            false,
            Some(label),
        );
        result.retain(|block| {
            !block
                .parent_label
                .as_deref()
                .is_some_and(|parent| parent.eq_ignore_ascii_case(&format!("sec{label}")))
                || block.range.start < claim.range.start
                || block.range.end > claim.range.end
        });
        result.extend(children);
    }
    result
}

fn add_ranges(mut blocks: Vec<Block>, length: usize) -> Vec<Block> {
    for index in 0..blocks.len() {
        blocks[index].range.end = blocks
            .get(index + 1)
            .map_or(length, |value| value.range.start);
    }
    blocks
}

fn detect_journal(text: &ScalarText<'_>) -> Vec<Block> {
    let source = javascript_lines(text);
    let mut result = add_ranges(
        source
            .iter()
            .filter_map(|line| {
                cached_regex!(
                    SECTION,
                    r"(?u)^[ \t]*([IVXLCDM]+|[A-Z])\.[ \t]+([^\x08]{3,180})$"
                )
                .captures(line.text)
                .map(|capture| (line, capture))
            })
            .map(|(line, capture)| {
                let whole = capture.get(0).unwrap();
                let title = capture[2].split_whitespace().collect::<Vec<_>>().join(" ");
                let alias = title
                    .to_lowercase()
                    .chars()
                    .map(|value| if value.is_alphanumeric() { value } else { ' ' })
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut block = Block::labelled(
                    NodeKind::Section,
                    format!("sec{}", &capture[1]),
                    line.scalar_start + line.text[..whole.start()].chars().count(),
                    0,
                );
                block.aliases = std::iter::once(capture[1].to_owned())
                    .chain((!alias.is_empty()).then(|| format!("sectitle:{alias}")))
                    .collect();
                block
            })
            .collect(),
        text.len(),
    );
    result.extend(add_ranges(
        source
            .iter()
            .filter_map(|line| {
                cached_regex!(NOTE, r"(?u)^[ \t]*(\d{1,5})\t[ \t]*$")
                    .captures(line.text)
                    .map(|capture| (line, capture))
            })
            .map(|(line, capture)| {
                let whole = capture.get(0).unwrap();
                let mut block = Block::labelled(
                    NodeKind::Footnote,
                    format!("fn{}", capture[1].parse::<u32>().unwrap()),
                    line.scalar_start + line.text[..whole.start()].chars().count(),
                    0,
                );
                block.aliases.push(capture[1].to_owned());
                block
            })
            .collect(),
        text.len(),
    ));
    let mut start = None;
    for (index, line) in source.iter().enumerate() {
        let blank = line.text.trim_matches([' ', '\t', '\r']).is_empty();
        if start.is_none() && !blank {
            start = line
                .text
                .char_indices()
                .find(|(_, value)| !javascript_whitespace(*value))
                .map(|(at, _)| line.scalar_start + line.text[..at].chars().count());
        }
        let next_blank = source
            .get(index + 1)
            .is_some_and(|value| value.text.trim_matches([' ', '\t', '\r']).is_empty());
        if let Some(block_start) = start.filter(|_| next_blank || index + 1 == source.len()) {
            let end = if index + 1 == source.len() {
                text.len()
            } else {
                line.scalar_start + line.text.chars().count()
            };
            let value = text
                .slice(block_start..end)
                .expect("journal block range is bounded");
            let block_start =
                if cached_regex!(PAGE, r"(?iu)^\[page [^\]\n]{1,40}\]").is_match(value) {
                    value
                        .find('\n')
                        .map_or(end, |at| block_start + value[..=at].chars().count())
                } else {
                    block_start
                };
            if block_start < end {
                result.push(Block {
                    kind: NodeKind::Prose,
                    range: ScalarRange {
                        start: block_start,
                        end,
                    },
                    label: None,
                    aliases: Vec::new(),
                    parent_label: None,
                    content_start: None,
                    diagnostic: None,
                    source: Derivation::Heuristic,
                    origin_id: ENGINE_ORIGIN,
                });
            }
            start = None;
        }
    }
    result
}

pub(super) fn inferred_blocks(evidence: &DocumentInput, text: &ScalarText<'_>) -> Vec<Block> {
    let mut paragraph_exclusions = evidence
        .exclusions
        .iter()
        .filter(|value| value.applies_to.iter().any(|name| name == "paragraph"))
        .map(|value| value.range)
        .collect::<Vec<_>>();
    paragraph_exclusions.sort_by_key(|range| (range.start, range.end));
    match evidence.profile {
        DetectionProfile::Legislation => detect_legislation(
            text,
            evidence.allow_hyphenated_sections,
            &evidence.native_claims,
        ),
        DetectionProfile::Instrument => detect_instrument(text),
        DetectionProfile::Journal => detect_journal(text),
        _ => detect_case(
            text,
            evidence.profile,
            evidence.report_start_page,
            evidence.require_report_start,
            &paragraph_exclusions,
        ),
    }
}
