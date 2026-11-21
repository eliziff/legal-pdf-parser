//! The assistant's deterministic text operations: each a named transform of a scope's accepted text that never adds,
//! removes or moves a paragraph ("\n"), reporting the sites it left unchanged. docx_tracked.rs writes the difference.

use fancy_regex::{Captures, Regex as Lookaround};
use crate::{Error, Result};
use legal_structure::javascript_whitespace;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// An operation's own parameters.
#[derive(Deserialize, Default)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
pub struct TextOpParams {
    pub find: Option<String>,
    pub replace: Option<String>,
    pub match_case: Option<bool>,
    pub whole_word: Option<bool>,
    pub occurrence: Option<usize>,
    pub style: Option<String>,
}

/// A site left unchanged, and for a flagged spelling its context and the dictionary's suggestions.
#[derive(Serialize, Clone)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
pub struct TextOpNote {
    pub site: String,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

type Notes = Vec<TextOpNote>;

fn note(notes: &mut Notes, site: &str, reason: impl Into<String>, context: Option<String>, suggestions: Option<Vec<String>>) {
    if notes.len() < 40 {
        let site = if site.chars().count() > 60 { format!("{}…", site.chars().take(60).collect::<String>()) } else { site.to_owned() };
        notes.push(TextOpNote { site, reason: reason.into(), suggestions, context });
    }
}

fn pattern(source: &str) -> Regex {
    Regex::new(source).expect("text op pattern")
}

fn lookaround(source: &str) -> Lookaround {
    Lookaround::new(source).expect("text op pattern")
}

fn replaced(pattern: &Lookaround, text: &str, with: impl FnMut(&Captures<'_, str>) -> String) -> String {
    pattern.replace_all(text, with).into_owned()
}

static LETTER: LazyLock<Regex> = LazyLock::new(|| pattern(r"^\p{L}$"));
static UPPER: LazyLock<Regex> = LazyLock::new(|| pattern(r"^\p{Lu}"));
static NUMBER: LazyLock<Regex> = LazyLock::new(|| pattern(r"\p{N}"));

fn letter(character: char) -> bool {
    LETTER.is_match(character.encode_utf8(&mut [0; 4]))
}

fn number(character: char) -> bool {
    NUMBER.is_match(character.encode_utf8(&mut [0; 4]))
}

/// Letters cased one by one, the first upper: lowercasing a whole word would change Greek final sigma.
fn capitalize_letters(text: &str) -> String {
    let mut first = true;
    text.chars().map(|character| if letter(character) {
        let cased = if first { character.to_uppercase().collect::<String>() } else { character.to_lowercase().collect() };
        first = false;
        cased
    } else { character.to_string() }).collect()
}

/// Everything lower, the first letter of each sentence (after `. ! ? …` and any closers) upper, and "I" kept.
fn sentence_case(text: &str) -> String {
    static PRONOUN: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(?m)(^|[^\p{L}'’])i(?=$|[^\p{L}])"));
    static CONTRACTION: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(?m)(^|[^\p{L}'’])i(?=['’](?:m|ve|ll|d)(?:$|[^\p{L}]))"));
    let mut next = true;
    let cased = text.to_lowercase().chars().map(|character| {
        if letter(character) {
            let cased = if next { character.to_uppercase().collect::<String>() } else { character.to_string() };
            next = false;
            return cased;
        }
        if matches!(character, '.' | '!' | '?' | '…' | '\n') { next = true; }
        else if !(javascript_whitespace(character) || matches!(character, '"' | '\'' | '”' | '’' | ')' | ']')) { next = false; }
        character.to_string()
    }).collect::<String>();
    let pronoun = replaced(&PRONOUN, &cased, |caps| format!("{}I", &caps[1]));
    replaced(&CONTRACTION, &pronoun, |caps| format!("{}I", &caps[1]))
}

/// Each word capitalized, a small word lower unless first or last on its line, an all-capitals word kept.
fn title_case(text: &str) -> String {
    const SMALL: [&str; 17] = ["a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "at", "by", "in", "of", "on", "to", "up"];
    static TOKEN: LazyLock<Regex> = LazyLock::new(|| pattern("[^ ]+"));
    text.split('\n').map(|line| {
        let last = TOKEN.find_iter(line).count().saturating_sub(1);
        let mut index = 0;
        TOKEN.replace_all(line, |caps: &regex::Captures| {
            let token = &caps[0];
            let edge = index == 0 || index == last;
            index += 1;
            let letters = token.chars().filter(|character| letter(*character)).collect::<String>();
            if letters.chars().count() >= 2 && letters == letters.to_uppercase() { token.to_owned() }
            else if !edge && SMALL.contains(&letters.to_lowercase().as_str()) { token.to_lowercase() }
            else { capitalize_letters(token) }
        }).into_owned()
    }).collect::<Vec<_>>().join("\n")
}

/// Word's find and replace over the scope (UTF-16 offsets, as Word's own text reads).
fn replace_text(text: &str, params: &TextOpParams, notes: &mut Notes) -> Result<String> {
    let (find, replace) = (params.find.as_deref().unwrap_or_default(), params.replace.as_deref().unwrap_or_default());
    if find.is_empty() { return Err(Error::Message("replace_text requires a non-empty find string".to_owned())); }
    if find.contains(['\n', '\r']) || replace.contains('\r') {
        return Err(Error::Message("replace_text find/replace must not span paragraphs".to_owned()));
    }
    let cased = |text: &str| if params.match_case == Some(true) { text.to_owned() } else { text.to_lowercase() };
    let (units, hay, needle) = (text.encode_utf16().collect::<Vec<_>>(), cased(text).encode_utf16().collect::<Vec<_>>(), cased(find).encode_utf16().collect::<Vec<_>>());
    let length = find.encode_utf16().count();
    let word = |unit: Option<&u16>| unit.and_then(|unit| char::from_u32(u32::from(*unit))).is_some_and(|character| letter(character) || number(character));
    let starts = (0..=hay.len().saturating_sub(needle.len())).filter(|&at| hay[at..].starts_with(&needle))
        .filter(|&at| params.whole_word != Some(true) || !(at > 0 && word(units.get(at - 1)) || word(units.get(at + length)))).collect::<Vec<_>>();
    let wanted = match params.occurrence {
        Some(occurrence) => starts.get(occurrence.wrapping_sub(1)).copied().into_iter().collect::<Vec<_>>(),
        None => starts.clone(),
    };
    if let Some(occurrence) = params.occurrence.filter(|_| wanted.is_empty() && !starts.is_empty()) {
        note(notes, find, format!("occurrence {occurrence} not found ({} matches)", starts.len()), None, None);
    }
    let (mut out, mut cursor) = (Vec::new(), 0);
    for at in wanted {
        if at < cursor {
            note(notes, &String::from_utf16_lossy(&units[at..(at + length).min(units.len())]), "overlaps a previous replacement", None, None);
            continue;
        }
        out.extend(&units[cursor..at.min(units.len())]);
        out.extend(replace.encode_utf16());
        cursor = (at + length).min(units.len());
    }
    out.extend(&units[cursor..]);
    Ok(String::from_utf16_lossy(&out))
}

/// One or two spaces after a sentence's end; an abbreviation's or a citation's period left as it is.
fn sentence_spacing(text: &str, params: &TextOpParams, notes: &mut Notes) -> Result<String> {
    const ABBREVIATIONS: &[&str] = &["v", "vs", "s", "ss", "no", "nos", "para", "paras", "art", "arts", "sec", "secs", "cl", "ch", "pt",
        "fn", "n", "p", "pp", "cf", "id", "ibid", "etc", "e.g", "i.e", "seq", "supra", "infra", "ed", "rev", "op", "loc", "st", "dr", "mr",
        "mrs", "ms", "prof", "hon", "jr", "sr", "inc", "ltd", "corp", "co", "llc", "llp", "dept", "div", "reg", "stat", "misc", "approx"];
    static END: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r#"([\p{L}\p{N}])([.!?…])(["'”’)\]]*)( {1,4})(?=["“'‘(\[]?\p{Lu})"#));
    static WORD: LazyLock<Regex> = LazyLock::new(|| pattern(r"([\p{L}.]+)$"));
    static INITIALS: LazyLock<Regex> = LazyLock::new(|| pattern(r"^(?:[a-z]\.)+[a-z]?$"));
    let style = match params.style.as_deref() {
        Some("one") => " ",
        Some("two") => "  ",
        _ => return Err(Error::Message("sentence_spacing requires style \"one\" or \"two\"".to_owned())),
    };
    Ok(replaced(&END, text, |caps| {
        let (whole, last, term) = (caps.get(0).unwrap(), &caps[1], &caps[2]);
        let end = whole.start() + last.len();
        let before = text[..end].char_indices().rev().nth(12).map_or(&text[..end], |(at, _)| &text[at..end]);
        let word = WORD.captures(before).map_or(String::new(), |found| found[1].trim_end_matches('.').to_lowercase());
        let last_char = last.chars().next().unwrap_or_default();
        let skip = term == "." && (number(last_char) || UPPER.is_match(last) && (word.chars().count() == 1 || INITIALS.is_match(&word))
            || ABBREVIATIONS.contains(&word.as_str()));
        if skip {
            if &caps[4] != style { note(notes, before.trim_start(), "possible abbreviation or citation — spacing left unchanged", None, None); }
            return whole.as_str().to_owned();
        }
        format!("{last}{term}{}{style}", &caps[3])
    }))
}

/// Straight quotes made typographic; a quote after a digit, or an ambiguous apostrophe, left as it is.
fn curl_quotes(text: &str, notes: &mut Notes) -> String {
    const ELISIONS: [&str; 8] = ["tis", "twas", "twere", "em", "cause", "til", "till", "n"];
    let characters = text.chars().collect::<Vec<_>>();
    let mut out = characters.clone();
    for (at, &character) in characters.iter().enumerate() {
        if character != '"' && character != '\'' { continue; }
        let (previous, next) = (at.checked_sub(1).map(|before| characters[before]), characters.get(at + 1).copied());
        let context = characters[at.saturating_sub(12)..(at + 12).min(characters.len())].iter().collect::<String>();
        if previous.is_some_and(number) {
            note(notes, &context, "quote after a digit may be a measurement — left unchanged", None, None);
            continue;
        }
        let opens = previous.is_none_or(|previous| javascript_whitespace(previous) || "([{—–-".contains(previous));
        out[at] = if character == '"' { if opens { '“' } else { '”' } }
        else if previous.is_some_and(letter) || next.is_some_and(number) { '’' }
        else if opens && next.is_some_and(letter) {
            let word = characters[at + 1..].iter().take_while(|character| letter(**character)).collect::<String>();
            let mut rest = characters[at + 1..].iter().take_while(|character| **character != '\n');
            if ELISIONS.contains(&word.to_lowercase().as_str()) { '’' }
            else if rest.any(|character| *character == '\'') { '‘' }
            else { note(notes, &context, "apostrophe or opening quote is ambiguous — left unchanged", None, None); continue; }
        } else if opens { '‘' } else { '’' };
    }
    out.into_iter().collect()
}

/// An ascending number range given an en dash, a spaced or doubled hyphen an em dash.
fn normalize_dashes(text: &str, notes: &mut Notes) -> String {
    static RANGE: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(?<![\p{L}\p{N}.-])([0-9]{1,4})-([0-9]{1,4})(?![\p{L}\p{N}.-])"));
    static SPACED: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(\S) -{1,2} (?=\S)"));
    static DOUBLED: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(\S)--(?=\S)"));
    let ranges = replaced(&RANGE, text, |caps| {
        if caps[1].parse::<u32>().ok() < caps[2].parse::<u32>().ok() { return format!("{}–{}", &caps[1], &caps[2]); }
        note(notes, &caps[0], "hyphenated numbers are not an ascending range — left unchanged", None, None);
        caps[0].to_owned()
    });
    let spaced = replaced(&SPACED, &ranges, |caps| format!("{}—", &caps[1]));
    replaced(&DOUBLED, &spaced, |caps| format!("{}—", &caps[1]))
}

/// "..." as one character ("character", the default), or "…" as three periods ("periods").
fn normalize_ellipses(text: &str, params: &TextOpParams) -> Result<String> {
    static PERIODS: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(?<!\.)\.{3}(?!\.)"));
    match params.style.as_deref().unwrap_or("character") {
        "periods" => Ok(text.replace('…', "...")),
        "character" => Ok(replaced(&PERIODS, text, |_| "…".to_owned())),
        _ => Err(Error::Message("normalize_ellipses style must be \"character\" or \"periods\"".to_owned())),
    }
}

/// Canadian English, with US English beside it where BEAVER_SPELLING_DICTIONARY names it ("en-CA,en-US"), each with
/// the drafting terms neither holds.
static SPELLERS: LazyLock<Vec<spellbook::Dictionary>> = LazyLock::new(|| {
    const LEGAL: &str = "arguendo assignee assignor bailee bailor chattel choate covenantor covenantee delegee disseisin emolument \
        estop estoppel garnishee garnishor gravamen hereinabove hereinbelow indemnitee indemnitor indemnitors indemnitees laches \
        mortgagee mortgagor novation obligee obligor offeree offeror pled promisee promisor recoupment remittitur replevin subrogee \
        subrogor sublessee sublessor testatrix tortfeasor tortfeasors usufruct curtilage interpleader remand remanded affiant \
        declarant tortious tortiously justiciable justiciability Blackacre Whiteacre Greenacre judgement judgements analyse \
        analysed analysing analyses fulfil fulfils fulfilling fulfilment instalment instalments enrol enrolment wilful wilfully \
        skilful bona fide fides mala prima facie alia alios judicata sui generis vires mens rea actus reus decidendi obiter dicta \
        dictum certiorari mandamus habeas proferentem meruit ejusdem delicto pendens curiae novo banc forma pauperis subpoena \
        duces tecum voir dire";
    let configured = std::env::var("BEAVER_SPELLING_DICTIONARY").unwrap_or_default().to_lowercase();
    let mut codes = configured.split(',').map(str::trim).filter(|code| matches!(*code, "en-ca" | "en-us")).collect::<Vec<_>>();
    codes.dedup();
    if codes.is_empty() { codes.push("en-ca"); }
    codes.into_iter().map(|code| {
        let (aff, dic) = if code == "en-us" { (include_str!("../dictionaries/en-us.aff"), include_str!("../dictionaries/en-us.dic")) }
            else { (include_str!("../dictionaries/en-ca.aff"), include_str!("../dictionaries/en-ca.dic")) };
        let mut speller = spellbook::Dictionary::new(aff, dic).expect("dictionary");
        LEGAL.split_whitespace().for_each(|word| { let _ = speller.add(word); });
        speller
    }).collect()
});

/// Possible misspellings flagged, never changed: quoted text, short words, acronyms, mixed case and a citation's
/// neighbourhood are out of scope, and a proper-noun-shaped word is flagged without lookalikes.
fn check_spelling(text: &str, notes: &mut Notes) {
    static QUOTED: LazyLock<Regex> = LazyLock::new(|| pattern(r#""[^"\n]*"|“[^“”\n]*”"#));
    static WORD: LazyLock<Regex> = LazyLock::new(|| pattern(r"\p{L}[\p{L}'’]*"));
    static POSSESSIVE: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)'s$|'$"));
    static PREVIOUS: LazyLock<Regex> = LazyLock::new(|| pattern(r"([\p{L}§¶.]+)\s*$"));
    static NEXT: LazyLock<Regex> = LazyLock::new(|| pattern(r"^\s*([\p{L}§¶]+)"));
    static NUMBERED: LazyLock<Regex> = LazyLock::new(|| pattern(r"^\s*\p{N}"));
    static CITATION: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)^(?:v|vs|s|ss|no|nos|para|paras|art|arts|sec|secs|cl|cf|id|ibid|supra|infra|et|al|seq|c)$"));
    let quoted = QUOTED.find_iter(text).map(|found| found.range()).collect::<Vec<_>>();
    let around = |from: usize, back: bool| if back { text[..from].char_indices().rev().nth(23).map_or(&text[..from], |(at, _)| &text[at..from]) }
        else { text[from..].char_indices().nth(24).map_or(&text[from..], |(at, _)| &text[from..from + at]) };
    for found in WORD.find_iter(text) {
        let word = found.as_str();
        let (preceding, following) = (around(found.start(), true), around(found.end(), false));
        let core = POSSESSIVE.replace(&word.replace('’', "'"), "").into_owned();
        if core.encode_utf16().count() < 3 || preceding.chars().rev().take(2).any(number) || NUMBERED.is_match(following)
            || quoted.iter().any(|range| range.contains(&found.start())) || core == core.to_uppercase()
            || core.chars().skip(1).collect::<String>() != core.chars().skip(1).collect::<String>().to_lowercase() {
            continue;
        }
        let (previous, next) = (PREVIOUS.captures(preceding).map_or("", |caps| caps.get(1).unwrap().as_str()), NEXT.captures(following).map_or("", |caps| caps.get(1).unwrap().as_str()));
        if CITATION.is_match(previous.trim_end_matches('.')) || CITATION.is_match(next) || previous.contains(['§', '¶']) || next.contains(['§', '¶']) { continue; }
        if SPELLERS.iter().any(|speller| speller.check(&core) || speller.check(&core.to_lowercase())) { continue; }
        let context = Some(format!("{preceding}{word}{following}").trim().to_owned());
        if UPPER.is_match(&core) {
            // A party name is never shown beside a lookalike (Hansman is not Hangman).
            note(notes, word, "possible proper noun — verify manually", context, None);
            continue;
        }
        let suggestions = SPELLERS.iter().find_map(|speller| {
            let mut found = Vec::new();
            speller.suggest(&core, &mut found);
            (!found.is_empty()).then_some(found)
        }).unwrap_or_default().into_iter().take(3).collect();
        note(notes, word, "possible misspelling", context, Some(suggestions));
    }
}

/// `op` run on a scope's text: its result and the sites it left unchanged; an unknown op, or one that moved a paragraph,
/// refuses.
pub(crate) fn run_text_op(op: &str, text: &str, params: &TextOpParams) -> Result<(String, Notes)> {
    static WORDS: LazyLock<Regex> = LazyLock::new(|| pattern(r"[\p{L}\p{N}'’]+"));
    static INDENT: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?m)(^ *)|( {2,})"));
    static SECTION: LazyLock<Lookaround> = LazyLock::new(|| lookaround(r"(§§?|¶¶?|\bss?\.) (?=\d)"));
    static TRAILING: LazyLock<Regex> = LazyLock::new(|| pattern(r"[ \t]+(\n|\z)"));
    let mut notes = Vec::new();
    let result = match op {
        "uppercase" => text.to_uppercase(),
        "lowercase" => text.to_lowercase(),
        "sentence_case" => sentence_case(text),
        "capitalize_each_word" => WORDS.replace_all(text, |caps: &regex::Captures| capitalize_letters(&caps[0])).into_owned(),
        "toggle_case" => text.chars().map(|character| {
            let (upper, lower) = (character.to_uppercase().collect::<String>(), character.to_lowercase().collect::<String>());
            if upper == lower { character.to_string() } else if upper == character.to_string() { lower } else { upper }
        }).collect(),
        "title_case" => title_case(text),
        "replace_text" => replace_text(text, params, &mut notes)?,
        "sentence_spacing" => sentence_spacing(text, params, &mut notes)?,
        "check_spelling" => { check_spelling(text, &mut notes); text.to_owned() }
        "straighten_quotes" => text.chars().map(|character| match character { '“' | '”' | '„' | '‟' => '"', '‘' | '’' | '‚' | '‛' => '\'', other => other }).collect(),
        "curl_quotes" => curl_quotes(text, &mut notes),
        "collapse_double_spaces" => INDENT.replace_all(text, |caps: &regex::Captures| caps.get(1).map_or(" ", |indent| indent.as_str()).to_owned()).into_owned(),
        "normalize_dashes" => normalize_dashes(text, &mut notes),
        "normalize_ellipses" => normalize_ellipses(text, params)?,
        "nonbreaking_section_refs" => replaced(&SECTION, text, |caps| format!("{}\u{a0}", &caps[1])),
        "remove_trailing_whitespace" => TRAILING.replace_all(text, "$1").into_owned(),
        _ => return Err(Error::Message(format!("Unknown text op: {op}"))),
    };
    if result.split('\n').count() != text.split('\n').count() {
        return Err(Error::Message(format!("Text op {op} altered paragraph boundaries")));
    }
    Ok((result, notes))
}
