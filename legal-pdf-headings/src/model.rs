//! PageIndex Flash's span, line and block aggregates over native lines and regions.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/model/{char_stats,rects,span_line,block}.py`
//! and `tokens/{tokenizer,token_types}.py` at 6d23caf416858f2ca136840305d1f479a86f6ef7.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced in `mod.rs`. Coordinates follow the
//! upstream convention: y grows upward, so a block's top edge is above its bottom edge.

use legal_pdf_core::model::{Line, Span};
use regex::Regex;
use std::sync::OnceLock;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Rect {
    pub(super) left: f64,
    pub(super) right: f64,
    pub(super) top: f64,
    pub(super) bottom: f64,
}

impl Rect {
    pub(super) const EMPTY: Rect = Rect {
        left: f64::INFINITY,
        right: f64::NEG_INFINITY,
        top: f64::NEG_INFINITY,
        bottom: f64::INFINITY,
    };

    /// A native top-down bbox on a page of `height`.
    pub(super) fn from_native(bbox: [f64; 4], height: f64) -> Rect {
        Rect {
            left: bbox[0],
            right: bbox[2],
            top: height - bbox[1],
            bottom: height - bbox[3],
        }
    }

    pub(super) fn width(&self) -> f64 {
        nan_max(0.0, self.right - self.left)
    }

    pub(super) fn height(&self) -> f64 {
        nan_max(0.0, self.top - self.bottom)
    }

    pub(super) fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub(super) fn center_x(&self) -> f64 {
        (self.left + self.right) / 2.0
    }

    pub(super) fn union(&self, other: &Rect) -> Rect {
        Rect {
            left: nan_min(self.left, other.left),
            right: nan_max(self.right, other.right),
            top: nan_max(self.top, other.top),
            bottom: nan_min(self.bottom, other.bottom),
        }
    }
}

pub(super) fn nan_max(value: f64, other: f64) -> f64 {
    if value.is_nan() || other.is_nan() {
        f64::NAN
    } else if value >= other {
        value
    } else {
        other
    }
}

pub(super) fn nan_min(value: f64, other: f64) -> f64 {
    if value.is_nan() || other.is_nan() {
        f64::NAN
    } else if value <= other {
        value
    } else {
        other
    }
}

pub(super) fn left_aligned(a: &Rect, b: &Rect, tolerance: f64) -> bool {
    (a.left - b.left).abs() <= tolerance
}

pub(super) fn right_aligned(a: &Rect, b: &Rect, tolerance: f64) -> bool {
    (a.right - b.right).abs() <= tolerance
}

pub(super) fn center_aligned(a: &Rect, b: &Rect, tolerance: f64) -> bool {
    let left = a.left - b.left;
    let right = a.right - b.right;
    let sign = |value: f64| {
        if value > 0.0 {
            1
        } else if value < 0.0 {
            -1
        } else {
            0
        }
    };
    if sign(left) != -sign(right) {
        return false;
    }
    (a.center_x() - b.center_x()).abs() <= tolerance.max(left.abs().min(right.abs()) / 2.0)
}

pub(super) fn x_aligned(a: &Rect, b: &Rect, tolerance: f64) -> bool {
    left_aligned(a, b, tolerance)
        || right_aligned(a, b, tolerance)
        || center_aligned(a, b, tolerance)
}

pub(super) fn intervals_overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> bool {
    (a0 <= b0 && b0 <= a1) || (b0 <= a0 && a0 <= b1)
}

pub(super) fn y_overlaps(a: &Rect, b: &Rect) -> bool {
    intervals_overlap(a.bottom, a.top, b.bottom, b.top)
}

/// Tokenizer character categories: 1 number, 2 uppercase, 3 lowercase, 4 other letter, 5 mark,
/// 6 sentence end, 7 dash/connector, 8 punctuation, 9 math symbol, 10 whitespace, 11 other.
pub(super) fn char_category(character: char) -> u8 {
    // ASCII, nearly every character a page carries, from a table read off the general rule once.
    static ASCII: OnceLock<[u8; 128]> = OnceLock::new();
    if character.is_ascii() {
        return ASCII.get_or_init(|| std::array::from_fn(|code| unicode_char_category(char::from(code as u8))))
            [character as usize];
    }
    unicode_char_category(character)
}

fn unicode_char_category(character: char) -> u8 {
    use GeneralCategory::*;
    let category = character.general_category();
    match category {
        LowercaseLetter => return 3,
        UppercaseLetter | TitlecaseLetter => return 2,
        OtherLetter => return 4,
        _ => {}
    }
    if matches!(
        character,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | '\u{feff}'
    ) || matches!(
        category,
        SpaceSeparator | LineSeparator | ParagraphSeparator
    ) {
        return 10;
    }
    if ".?!\u{ff61}\u{3002}\u{ff1f}\u{ff01}\u{ff0e}".contains(character) {
        return 6;
    }
    if matches!(category, ConnectorPunctuation | DashPunctuation)
        || matches!(character, '\u{2212}' | '\u{207b}' | '\u{208b}')
    {
        return 7;
    }
    match category {
        OpenPunctuation | ClosePunctuation | InitialPunctuation | FinalPunctuation
        | OtherPunctuation => 8,
        DecimalNumber | LetterNumber | OtherNumber => 1,
        NonspacingMark | SpacingMark | EnclosingMark => 5,
        MathSymbol => 9,
        _ => 11,
    }
}

pub(super) fn is_word_category(category: u8) -> bool {
    matches!(category, 1 | 2 | 3 | 5)
}

pub(super) fn is_punct_category(category: u8) -> bool {
    matches!(category, 6 | 7 | 8)
}

/// The package whitespace set that upstream trims spans with.
fn trim_unicode_ws(text: &str) -> &str {
    text.trim_matches(|character: char| char_category(character) == 10)
}

#[derive(Debug, Clone, Default)]
pub(super) struct CharStats {
    pub(super) first: u8,
    pub(super) last: u8,
    pub(super) counts: [u32; 12],
    pub(super) total: u32,
}

impl CharStats {
    pub(super) fn of(text: &str) -> CharStats {
        let mut stats = CharStats::default();
        for character in text.chars() {
            let category = char_category(character);
            if stats.first == 0 {
                stats.first = category;
            }
            stats.last = category;
            stats.counts[category as usize] += 1;
            stats.total += 1;
        }
        stats
    }

    pub(super) fn merge(&mut self, other: &CharStats) {
        if self.first == 0 {
            self.first = other.first;
        }
        if other.last != 0 {
            self.last = other.last;
        }
        for (count, added) in self.counts.iter_mut().zip(other.counts) {
            *count += added;
        }
        self.total += other.total;
    }

    pub(super) fn letters(&self) -> u32 {
        self.counts[3] + self.counts[2] + self.counts[4]
    }

    pub(super) fn punctuation(&self) -> u32 {
        self.counts[6] + self.counts[7] + self.counts[8]
    }

    /// Letters count once, other letters twice and everything else half.
    pub(super) fn info_weight(&self) -> f64 {
        f64::from(self.counts[3])
            + f64::from(self.counts[2])
            + 2.0 * f64::from(self.counts[4])
            + 0.5 * f64::from(self.total - self.letters())
    }

    pub(super) fn upper_dominant(&self) -> bool {
        let upper = f64::from(self.counts[2]);
        let letters = f64::from(self.letters());
        upper > (letters * 3.0 / 4.0).max(letters - 4.0)
            && upper > 3f64.max(f64::from(self.total) / 3.0)
    }

    pub(super) fn caps_heavy(&self) -> bool {
        self.upper_dominant() || self.counts[2] >= 2.max(self.total)
    }
}

fn bold_font() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i-u)(bold|timesb)").expect("bold font"))
}

fn italic_font() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i-u)(ital|it\z|i[1-9][0-9]*\z|obliq)").expect("italic font"))
}

fn subset_prefix() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z]{6}\+").expect("subset prefix"))
}

#[derive(Debug, Clone)]
pub(super) struct FlashSpan {
    pub(super) rect: Rect,
    pub(super) stats: CharStats,
    pub(super) font_name: String,
    pub(super) size: f64,
    pub(super) bold: bool,
    pub(super) italic: bool,
}

impl FlashSpan {
    pub(super) fn of(span: &Span, height: f64) -> FlashSpan {
        let mut font_name = span.font.as_str();
        if subset_prefix().is_match(font_name) {
            font_name = &font_name[7..];
        }
        let font_name = match font_name.to_lowercase().as_str() {
            "timesnewroman" | "times-new-roman" | "timesroman" | "times-roman" | "timesnew"
            | "times-new" => "Times".to_owned(),
            _ => font_name.to_owned(),
        };
        FlashSpan {
            // Flash measures glyph boxes; a native span box rises to the font's ascent, about a
            // quarter of the size above the capitals, so its top is lowered to match.
            rect: Rect::from_native(
                [
                    span.bbox[0],
                    (span.bbox[1] + 0.25 * span.size).min(span.bbox[3]),
                    span.bbox[2],
                    span.bbox[3],
                ],
                height,
            ),
            stats: CharStats::of(trim_unicode_ws(&span.text)),
            bold: span.flags & 16 != 0 || bold_font().is_match(&font_name),
            italic: italic_font().is_match(&font_name),
            font_name,
            size: span.size,
        }
    }

    pub(super) fn char_count(&self) -> u32 {
        self.stats.total
    }

    pub(super) fn font_style(&self) -> String {
        format!("{} {}", self.font_name, if self.bold { "B" } else { "R" })
    }

    /// `"<fontName> <B|R> <size rounded half-up to 0.1>"`.
    pub(super) fn style_key(&self) -> String {
        format!("{} {}", self.font_style(), half_up_one_decimal(self.size))
    }
}

pub(super) fn half_up_one_decimal(value: f64) -> String {
    let scaled = value * 10.0;
    let floor = scaled.floor();
    let rounded = if scaled - floor < 0.5 {
        floor
    } else {
        floor + 1.0
    };
    format!("{:.1}", rounded / 10.0)
}

#[derive(Debug, Clone)]
pub(super) struct FlashLine {
    pub(super) spans: Vec<FlashSpan>,
    pub(super) stats: CharStats,
    pub(super) bold: f64,
    pub(super) italic: f64,
    pub(super) skew: f64,
    pub(super) size: f64,
    pub(super) rect: Rect,
    /// Tallest span.
    pub(super) span_height: f64,
    /// Share of the line's box its spans cover.
    pub(super) fill: f64,
    pub(super) text: String,
}

impl FlashLine {
    pub(super) fn of(line: &Line, height: f64) -> FlashLine {
        let mut flash = FlashLine {
            spans: Vec::new(),
            stats: CharStats::default(),
            bold: 0.0,
            italic: 0.0,
            skew: 0.0,
            size: 0.0,
            rect: Rect::EMPTY,
            span_height: 0.0,
            fill: 0.0,
            text: line.text.clone(),
        };
        for span in &line.spans {
            flash.append(FlashSpan::of(span, height));
        }
        flash
    }

    fn append(&mut self, span: FlashSpan) {
        let weight = self.stats.info_weight();
        let added = span.stats.info_weight();
        let total = weight + added;
        if total > 0.0 {
            self.bold = (self.bold * weight + if span.bold { added } else { 0.0 }) / total;
            self.italic = (self.italic * weight + if span.italic { added } else { 0.0 }) / total;
            self.skew = self.skew * weight / total;
            self.size = (self.size * weight + span.size * added) / total;
        }
        self.stats.merge(&span.stats);
        if span.char_count() > 0 {
            let before = self.rect.area();
            self.span_height = self.span_height.max(span.rect.height());
            self.rect = self.rect.union(&span.rect);
            self.fill =
                1f64.min((self.fill * before + span.rect.area()) / 1f64.max(self.rect.area()));
        }
        self.spans.push(span);
    }

    pub(super) fn char_count(&self) -> u32 {
        self.stats.total
    }
}

#[derive(Debug, Clone)]
pub(super) struct FlashBlock {
    pub(super) lines: Vec<FlashLine>,
    pub(super) stats: CharStats,
    pub(super) centered_lines: bool,
    pub(super) bold: f64,
    pub(super) italic: f64,
    pub(super) skew: f64,
    pub(super) size: f64,
    /// Information-weighted span coverage.
    pub(super) fill: f64,
    /// Area-weighted span coverage.
    pub(super) area_fill: f64,
    /// Tallest line.
    pub(super) line_height: f64,
    pub(super) rect: Rect,
    style_counts: Vec<(String, u32)>,
    size_counts: Vec<(f64, u32)>,
}

impl FlashBlock {
    pub(super) fn new() -> FlashBlock {
        FlashBlock {
            lines: Vec::new(),
            stats: CharStats::default(),
            centered_lines: true,
            bold: 0.0,
            italic: 0.0,
            skew: 0.0,
            size: 0.0,
            fill: 0.0,
            area_fill: 0.0,
            line_height: 0.0,
            rect: Rect::EMPTY,
            style_counts: Vec::new(),
            size_counts: Vec::new(),
        }
    }

    pub(super) fn add_line(&mut self, line: FlashLine) {
        self.centered_lines = self.centered_lines
            && (self.lines.is_empty() || center_aligned(&self.rect, &line.rect, 1.0));
        let weight = self.stats.info_weight();
        let added = line.stats.info_weight();
        let total = weight + added;
        if total > 0.0 {
            self.bold = (self.bold * weight + line.bold * added) / total;
            self.italic = (self.italic * weight + line.italic * added) / total;
            self.skew = (self.skew * weight + line.skew * added) / total;
            self.size = (self.size * weight + line.size * added) / total;
            self.fill = (self.fill * weight + line.fill * added) / total;
        }
        self.stats.merge(&line.stats);
        if line.char_count() > 0 {
            let before = self.rect.area();
            self.line_height = self.line_height.max(line.span_height);
            self.rect = self.rect.union(&line.rect);
            let after = self.rect.area();
            if after > 0.0 {
                self.area_fill = (self.area_fill * before + line.fill * line.rect.area()) / after;
            }
            for span in &line.spans {
                let key = span.style_key();
                match self.style_counts.iter_mut().find(|(seen, _)| *seen == key) {
                    Some((_, count)) => *count += span.char_count(),
                    None => self.style_counts.push((key, span.char_count())),
                }
                let size = (span.size * 10.0 + 0.5).floor() / 10.0;
                match self.size_counts.iter_mut().find(|(seen, _)| *seen == size) {
                    Some((_, count)) => *count += span.char_count(),
                    None => self.size_counts.push((size, span.char_count())),
                }
            }
        }
        self.lines.push(line);
    }

    pub(super) fn char_count(&self) -> u32 {
        self.stats.total
    }

    pub(super) fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub(super) fn first_span(&self) -> Option<&FlashSpan> {
        self.lines.first().and_then(|line| line.spans.first())
    }

    pub(super) fn last_span(&self) -> Option<&FlashSpan> {
        self.lines.last().and_then(|line| line.spans.last())
    }

    /// The style with the most characters; the first seen of equals.
    pub(super) fn dominant_style(&self) -> &str {
        let mut best: Option<&(String, u32)> = None;
        for entry in &self.style_counts {
            if best.is_none_or(|best| entry.1 > best.1) {
                best = Some(entry);
            }
        }
        best.map_or("", |entry| entry.0.as_str())
    }

    pub(super) fn dominant_size(&self) -> f64 {
        let mut best: Option<&(f64, u32)> = None;
        for entry in &self.size_counts {
            if best.is_none_or(|best| entry.1 > best.1) {
                best = Some(entry);
            }
        }
        best.map_or(0.0, |entry| entry.0)
    }

    pub(super) fn caps_heavy(&self) -> bool {
        self.stats.caps_heavy()
    }

    /// Dominant font size, plus two for capitals and one for bold.
    pub(super) fn heading_score(&self) -> f64 {
        self.dominant_size()
            + if self.caps_heavy() { 2.0 } else { 0.0 }
            + if self.bold > 0.5 { 1.0 } else { 0.0 }
    }

    /// 1 justified, 2 left-aligned, 3 centered, 4 right-aligned, 5 mixed.
    pub(super) fn alignment_code(&self) -> u8 {
        if self.lines.is_empty() {
            return 0;
        }
        let (mut left, mut right, mut any) = (true, true, true);
        for line in &self.lines {
            let tolerance = 1f64.max(line.rect.width() / 20.0);
            let line_left = left_aligned(&self.rect, &line.rect, tolerance);
            let line_right = right_aligned(&self.rect, &line.rect, tolerance);
            left &= line_left;
            right &= line_right;
            any &= line_left || line_right;
        }
        if left && !right {
            2
        } else if right && !left {
            4
        } else if any {
            1
        } else if self.centered_lines {
            3
        } else {
            5
        }
    }

    pub(super) fn tokens(&self) -> Vec<Token> {
        tokenize(self.lines.iter().map(|line| line.text.as_str()))
    }

    /// Mixed-case body text rather than a title: enough capitalized words, no long lowercase one.
    pub(super) fn sentence_like(&self) -> bool {
        let tokens = self.tokens();
        if tokens.len() < 3 || self.caps_heavy() {
            return false;
        }
        let upper = self.stats.counts[2] as usize;
        if upper <= 2 || (upper as f64) < tokens.len() as f64 / 10.0 {
            return false;
        }
        let mut matched = 0;
        for token in &tokens {
            if token.kind != 2 || token.text.chars().count() <= 2 || token.first == 4 {
                continue;
            }
            if token.first == 2 {
                matched += 1;
            } else if token.text.chars().count() >= 5 {
                return false;
            }
        }
        matched >= 3
    }
}

/// Token kinds: 1 number, 2 word, 3 sentence end, 4 dash, 5 punctuation, 6 math, 7 other.
#[derive(Debug, Clone)]
pub(super) struct Token {
    pub(super) kind: u8,
    pub(super) text: String,
    pub(super) first: u8,
    pub(super) last: u8,
    boundary: bool,
    line: usize,
}

const SCRIPT_FAMILY: [u8; 12] = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7];

fn can_extend(last: u8, next: u8) -> bool {
    (next == last && !is_punct_category(next)) || (is_word_category(last) && is_word_category(next))
}

/// The upstream line tokenizer over line texts: whitespace bounds tokens, letters and digits run
/// together, each punctuation mark stands alone, and a word hyphenated across lines rejoins.
pub(super) fn tokenize<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut current: Option<Token> = None;
    let mut pending_space = false;
    for (line_index, text) in lines.enumerate() {
        if let Some(token) = current.take() {
            let boundary = token.text != "-" || tokens.last().is_none_or(|prior| prior.boundary);
            tokens.push(Token { boundary, ..token });
        }
        for character in text.chars() {
            let category = char_category(character);
            if category == 10 {
                pending_space = true;
                continue;
            }
            if pending_space {
                if let Some(token) = current.take() {
                    if category == 5 && is_word_category(token.last) {
                        current = Some(token);
                    } else {
                        tokens.push(Token {
                            boundary: true,
                            ..token
                        });
                    }
                }
            }
            if current.is_none() && tokens.len() >= 2 && category == 3 {
                let hyphen = &tokens[tokens.len() - 1];
                let word = &tokens[tokens.len() - 2];
                if word.last == 3
                    && !word.boundary
                    && hyphen.text == "-"
                    && hyphen.line != line_index
                {
                    tokens.pop();
                    let mut word = tokens.pop().expect("word");
                    word.text.push(character);
                    word.last = category;
                    word.line = line_index;
                    current = Some(word);
                    pending_space = false;
                    continue;
                }
            }
            match current.as_mut() {
                Some(token) if can_extend(token.last, category) => {
                    token.text.push(character);
                    token.last = category;
                    let family = SCRIPT_FAMILY[category as usize];
                    if token.kind == 1 && family != 1 {
                        token.kind = 2;
                    }
                }
                _ => {
                    if let Some(token) = current.take() {
                        tokens.push(Token {
                            boundary: false,
                            ..token
                        });
                    }
                    current = Some(Token {
                        kind: SCRIPT_FAMILY[category as usize],
                        text: character.to_string(),
                        first: category,
                        last: category,
                        boundary: false,
                        line: line_index,
                    });
                }
            }
            pending_space = false;
        }
        pending_space = true;
    }
    if let Some(token) = current.take() {
        tokens.push(Token {
            boundary: true,
            ..token
        });
    }
    tokens
}

/// Upstream `assign_reading_order`'s isolated-centered flag: a block whose lines share a centre
/// at the page's centre, apart from the next block or not edge-aligned with it; or one centred
/// on a page-centred next block it is not edge-aligned with.
pub(super) fn x_aligned_center_close(
    block: &FlashBlock,
    page_width: f64,
    next: Option<&Rect>,
) -> bool {
    let page_center = page_width / 2.0;
    let centers_close =
        |rect: &Rect| (rect.center_x() - page_center).abs() <= 1f64.max(rect.width() / 10.0);
    let rect = &block.rect;
    if block.centered_lines && centers_close(rect) {
        return match next {
            None => true,
            Some(next) => {
                next.top > rect.bottom
                    || (!left_aligned(rect, next, 1.0) && !right_aligned(rect, next, 1.0))
            }
        };
    }
    block.centered_lines
        && next.is_some_and(|next| {
            !left_aligned(rect, next, 1.0)
                && !right_aligned(rect, next, 1.0)
                && center_aligned(rect, next, rect.width() / 10.0)
                && centers_close(next)
        })
}
