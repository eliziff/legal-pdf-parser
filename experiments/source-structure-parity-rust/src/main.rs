use flate2::read::GzDecoder;
use legal_structure::{
    a2aj_source_doc, journal_source_doc, journal_text_source_doc, native_markup_source_doc,
    A2ajInput, A2ajSourceKind, JournalPageLabel, NativeMarkupInput, SourceDoc, SourceDocOrigin,
};
use rayon::prelude::*;
use regex::Regex;
use rusqlite::{Connection, OpenFlags, Row};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Config {
    baseline: PathBuf,
    a2aj_db: PathBuf,
    courtlistener_db: PathBuf,
    journal_db: PathBuf,
    journal_final_db: PathBuf,
    providers: HashSet<String>,
    limit: usize,
    start_id: i64,
    batch: usize,
    max_seconds: u64,
}

#[derive(Clone, Deserialize)]
struct Expected {
    provider: String,
    source_id: String,
    status: String,
    mode: Option<String>,
    canonical_bytes: Option<usize>,
    canonical_sha256: Option<String>,
    blocks: Option<usize>,
    failure: Option<String>,
}

#[derive(Default)]
struct Totals {
    checked: usize,
    matched: usize,
    intentional_changes: usize,
    mismatches: Vec<String>,
}

enum Comparison {
    Match,
    IntentionalChange,
    Mismatch(String),
}

struct Actual {
    status: &'static str,
    mode: Option<&'static str>,
    bytes: Option<usize>,
    sha256: Option<String>,
    blocks: Option<usize>,
    text_chars: Option<usize>,
    failure: Option<&'static str>,
    diagnostic: Option<String>,
}

#[derive(Clone)]
struct A2ajRow {
    id: i64,
    doc_type: String,
    dataset: Option<String>,
    citation_en: Option<String>,
    citation_fr: Option<String>,
    citation2_en: Option<String>,
    citation2_fr: Option<String>,
    name_en: Option<String>,
    name_fr: Option<String>,
    url_en: Option<String>,
    url_fr: Option<String>,
    text_en: Option<String>,
    text_fr: Option<String>,
}

#[derive(Clone)]
struct CourtRow {
    id: i64,
    markup: Option<String>,
    plain: Option<String>,
}

#[derive(Clone)]
struct JournalRow {
    id: usize,
    text: Option<String>,
    url: Option<String>,
    pages: Vec<JournalPageLabel>,
    final_pages: Option<PathBuf>,
}

fn arg_path(args: &HashMap<String, String>, name: &str, fallback: PathBuf) -> PathBuf {
    args.get(name).map(PathBuf::from).unwrap_or(fallback)
}

fn config() -> Result<Config, String> {
    let mut values = HashMap::new();
    let mut args = env::args().skip(1);
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("unexpected argument: {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{} requires a value", key))?;
        values.insert(key[2..].to_owned(), value);
    }
    let local = env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or("LOCALAPPDATA is not set")?
        .join("OpenLegalProducts/LegalData/providers");
    let baseline = arg_path(
        &values,
        "baseline",
        PathBuf::from(
            "../backend/experiments/source-structure-parity/results/installed-provider-freeze-full",
        ),
    );
    let a2aj_db = arg_path(&values, "a2aj-db", local.join("a2aj/a2aj.sqlite"));
    let courtlistener_db = arg_path(
        &values,
        "courtlistener-db",
        local.join("courtlistener/courtlistener.sqlite"),
    );
    let journal_final_db = arg_path(
        &values,
        "journal-final-db",
        env::var_os("MIKE_JOURNAL_FINAL_CONTRACT_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|| local.join("journals/journals.db")),
    );
    let journal_db = if let Some(value) = values.get("journal-db") {
        PathBuf::from(value)
    } else if let Some(value) = env::var_os("MIKE_PUBLIC_ENDPOINT_DB") {
        PathBuf::from(value)
    } else {
        journal_source_path(&local.join("journals"))?
    };
    let providers = values
        .get("provider")
        .map_or("a2aj,courtlistener,journal", String::as_str)
        .split(',')
        .map(str::to_owned)
        .collect();
    Ok(Config {
        baseline,
        a2aj_db,
        courtlistener_db,
        journal_db,
        journal_final_db,
        providers,
        limit: number(&values, "limit", 0)?,
        start_id: number(&values, "start-id", 0)? as i64,
        batch: number(&values, "batch", 512)?.clamp(1, 5_000),
        max_seconds: number(&values, "max-seconds", 180)? as u64,
    })
}

fn number(values: &HashMap<String, String>, name: &str, fallback: usize) -> Result<usize, String> {
    values.get(name).map_or(Ok(fallback), |value| {
        value.parse().map_err(|_| format!("invalid --{name}"))
    })
}

fn readonly(path: &Path) -> Result<Connection, String> {
    if !path.is_file() {
        return Err(format!("required input is absent: {}", path.display()));
    }
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn journal_source_path(directory: &Path) -> Result<PathBuf, String> {
    let direct = directory.join("public_endpoint.db");
    if direct.is_file() {
        return Ok(direct);
    }
    let search = readonly(&directory.join("public_endpoint-search.sqlite"))?;
    search
        .query_row(
            "SELECT value FROM meta WHERE key='source_path'",
            [],
            |row| row.get::<_, String>(0),
        )
        .map(PathBuf::from)
        .map_err(|error| format!("journal source path: {error}"))
}

fn load_expected(root: &Path, provider: &str) -> Result<HashMap<String, Expected>, String> {
    let mut files = Vec::new();
    let provider_root = root.join(provider);
    if !provider_root.is_dir() && root.join("parts").is_dir() {
        for file in fs::read_dir(root.join("parts")).map_err(|error| error.to_string())? {
            let path = file.map_err(|error| error.to_string())?.path();
            if path.extension().is_some_and(|value| value == "gz") {
                files.push(path);
            }
        }
    }
    let shards = if provider_root.is_dir() {
        Some(fs::read_dir(provider_root).map_err(|error| error.to_string())?)
    } else {
        None
    };
    for shard in shards.into_iter().flatten() {
        let parts = shard
            .map_err(|error| error.to_string())?
            .path()
            .join("parts");
        if !parts.is_dir() {
            continue;
        }
        for file in fs::read_dir(parts).map_err(|error| error.to_string())? {
            let path = file.map_err(|error| error.to_string())?.path();
            if path.extension().is_some_and(|value| value == "gz") {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut expected = HashMap::new();
    for path in files {
        let reader = BufReader::new(GzDecoder::new(
            File::open(&path).map_err(|e| e.to_string())?,
        ));
        for line in reader.lines() {
            let row: Expected = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            expected.insert(format!("{}\0{}", row.provider, row.source_id), row);
        }
    }
    Ok(expected)
}

fn document_actual(result: Result<SourceDoc, impl std::fmt::Display>) -> Actual {
    match result {
        Ok(doc) if !doc.text.is_empty() => {
            let native = doc
                .blocks
                .iter()
                .any(|block| block.origin == SourceDocOrigin::Native);
            let heuristic = doc
                .blocks
                .iter()
                .any(|block| block.origin == SourceDocOrigin::Heuristic);
            let mode = if native && heuristic {
                "hybrid"
            } else if native {
                "native"
            } else {
                "flat"
            };
            match doc.json_bytes(true) {
                Ok(bytes) => Actual {
                    status: "pass",
                    mode: Some(mode),
                    bytes: Some(bytes.len()),
                    sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
                    blocks: Some(doc.blocks.len()),
                    text_chars: Some(doc.text.chars().count()),
                    failure: None,
                    diagnostic: None,
                },
                Err(_) => failed("compile_exception"),
            }
        }
        Ok(_) => failed("provider_unavailable"),
        Err(error) => {
            let mut actual = failed("compile_exception");
            actual.diagnostic = Some(error.to_string());
            actual
        }
    }
}

fn failed(reason: &'static str) -> Actual {
    Actual {
        status: "failure",
        mode: None,
        bytes: None,
        sha256: None,
        blocks: None,
        text_chars: None,
        failure: Some(reason),
        diagnostic: None,
    }
}

fn compare(expected: &Expected, actual: Actual, allow_changed_output: bool) -> Comparison {
    let same = expected.status == actual.status
        && (actual.status != "pass"
            || (expected.mode.as_deref() == actual.mode
                && expected.canonical_bytes == actual.bytes
                && expected.canonical_sha256 == actual.sha256
                && expected.blocks == actual.blocks))
        && (actual.status != "failure" || expected.failure.as_deref() == actual.failure);
    if same {
        Comparison::Match
    } else if allow_changed_output && expected.status == actual.status && actual.status == "pass" {
        Comparison::IntentionalChange
    } else {
        Comparison::Mismatch(format!(
            "{}:{} expected {} {:?}/{:?}/{:?}, got {} {:?}/{:?}/{:?} text_chars={:?} diagnostic={:?}",
            expected.provider,
            expected.source_id,
            expected.status,
            expected.mode,
            expected.canonical_bytes,
            expected.blocks,
            actual.status,
            actual.mode,
            actual.bytes,
            actual.blocks,
            actual.text_chars,
            actual.diagnostic,
        ))
    }
}

fn record(totals: &mut Totals, comparison: Comparison) {
    totals.checked += 1;
    match comparison {
        Comparison::Match => totals.matched += 1,
        Comparison::IntentionalChange => totals.intentional_changes += 1,
        Comparison::Mismatch(value) => {
            if totals.mismatches.len() < 50 {
                totals.mismatches.push(value);
            }
        }
    }
}

fn trimmed(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let result = js_trim(value.split('\0').next().unwrap_or_default());
        (!result.is_empty()).then(|| result.to_owned())
    })
}

fn js_trim(value: &str) -> &str {
    value.trim_matches(|character: char| character.is_whitespace() || character == '\u{feff}')
}

fn a2aj_value(row: &A2ajRow, name: &str, language: &str) -> Option<String> {
    let (primary, fallback) = match (name, language) {
        ("citation", "en") => (&row.citation_en, &row.citation_fr),
        ("citation", _) => (&row.citation_fr, &row.citation_en),
        ("citation2", "en") => (&row.citation2_en, &row.citation2_fr),
        ("citation2", _) => (&row.citation2_fr, &row.citation2_en),
        ("name", "en") => (&row.name_en, &row.name_fr),
        ("name", _) => (&row.name_fr, &row.name_en),
        ("url", "en") => (&row.url_en, &row.url_fr),
        ("url", _) => (&row.url_fr, &row.url_en),
        ("text", "en") => (&row.text_en, &row.text_fr),
        ("text", _) => (&row.text_fr, &row.text_en),
        _ => return None,
    };
    trimmed(primary.clone()).or_else(|| trimmed(fallback.clone()))
}

fn compile_a2aj(row: A2ajRow) -> Actual {
    let language = if trimmed(row.text_en.clone()).is_some() {
        "en"
    } else {
        "fr"
    };
    let Some(text) = a2aj_value(&row, "text", language) else {
        return failed("provider_unavailable");
    };
    let citation =
        a2aj_value(&row, "citation", language).or_else(|| a2aj_value(&row, "citation2", language));
    let Some(citation) = citation else {
        return failed("provider_unavailable");
    };
    let mut input = A2ajInput::new(
        citation,
        if row.doc_type == "laws" {
            A2ajSourceKind::Laws
        } else {
            A2ajSourceKind::Cases
        },
        text,
    );
    input.name = a2aj_value(&row, "name", language);
    input.alternate_citation = a2aj_value(&row, "citation2", language);
    input.url = a2aj_value(&row, "url", language);
    input.dataset = trimmed(row.dataset);
    document_actual(a2aj_source_doc(input))
}

fn a2aj_row(row: &Row<'_>) -> rusqlite::Result<A2ajRow> {
    Ok(A2ajRow {
        id: row.get(0)?,
        doc_type: row.get(1)?,
        dataset: row.get(2)?,
        citation_en: row.get(3)?,
        citation_fr: row.get(4)?,
        citation2_en: row.get(5)?,
        citation2_fr: row.get(6)?,
        name_en: row.get(7)?,
        name_fr: row.get(8)?,
        url_en: row.get(9)?,
        url_fr: row.get(10)?,
        text_en: row.get(11)?,
        text_fr: row.get(12)?,
    })
}

fn run_a2aj(
    config: &Config,
    expected: &HashMap<String, Expected>,
    started: Instant,
) -> Result<Totals, String> {
    let db = readonly(&config.a2aj_db)?;
    let mut statement = db.prepare(
        "SELECT id, doc_type, dataset, citation_en, citation_fr, citation2_en, citation2_fr, name_en, name_fr, url_en, url_fr, unofficial_text_en, unofficial_text_fr FROM document WHERE id >= ? ORDER BY id",
    ).map_err(|e| e.to_string())?;
    let mut rows = statement
        .query([config.start_id])
        .map_err(|e| e.to_string())?;
    let mut totals = Totals::default();
    loop {
        let mut batch = Vec::with_capacity(config.batch);
        while batch.len() < config.batch
            && (config.limit == 0 || totals.checked + batch.len() < config.limit)
        {
            match rows.next().map_err(|e| e.to_string())? {
                Some(row) => batch.push(a2aj_row(row).map_err(|e| e.to_string())?),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let batch_len = batch.len();
        let results = batch
            .into_par_iter()
            .map(|row| {
                let key = format!("a2aj\0{}", row.id);
                let actual = compile_a2aj(row);
                expected
                    .get(&key)
                    .map(|wanted| compare(wanted, actual, false))
                    .unwrap_or_else(|| Comparison::Mismatch(format!("missing oracle row {key}")))
            })
            .collect::<Vec<_>>();
        for result in results {
            record(&mut totals, result);
        }
        progress("a2aj", &totals, batch_len, started);
        deadline(config, started)?;
        if config.limit > 0 && totals.checked >= config.limit {
            break;
        }
    }
    Ok(totals)
}

fn court_row(row: &Row<'_>) -> rusqlite::Result<CourtRow> {
    let mut markup = None;
    for index in 1..=6 {
        let value: Option<String> = row.get(index)?;
        if value
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            && markup.is_none()
        {
            markup = value;
        }
    }
    let plain: Option<String> = row.get(7)?;
    Ok(CourtRow {
        id: row.get(0)?,
        markup,
        plain: plain.filter(|v| !v.trim().is_empty()),
    })
}

fn native_input(
    id: i64,
    text: String,
    markup: Option<String>,
) -> Result<NativeMarkupInput, serde_json::Error> {
    serde_json::from_value(
        json!({ "provider": "courtlistener", "id": id.to_string(), "text": text, "markup": markup }),
    )
}

fn compile_court(row: CourtRow) -> Actual {
    if row.markup.is_none() && row.plain.is_none() {
        return failed("provider_unavailable");
    }
    let text = if row.markup.is_some() {
        String::new()
    } else {
        opinion_text(row.plain.as_deref().unwrap_or(""))
    };
    let input = match native_input(row.id, text, row.markup.clone()) {
        Ok(value) => value,
        Err(_) => return failed("compile_exception"),
    };
    match native_markup_source_doc(input) {
        Ok(doc) if doc.text.is_empty() && row.markup.is_some() => {
            let fallback = opinion_text(row.markup.as_deref().unwrap());
            if fallback.is_empty() {
                return failed("provider_unavailable");
            }
            match native_input(row.id, fallback, row.markup) {
                Ok(input) => document_actual(native_markup_source_doc(input)),
                Err(_) => failed("compile_exception"),
            }
        }
        result => document_actual(result),
    }
}

fn opinion_text(value: &str) -> String {
    static PAGE: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH: OnceLock<Regex> = OnceLock::new();
    static BREAK: OnceLock<Regex> = OnceLock::new();
    static BLOCK: OnceLock<Regex> = OnceLock::new();
    static TAG: OnceLock<Regex> = OnceLock::new();
    static SPACE: OnceLock<Regex> = OnceLock::new();
    static LINES: OnceLock<Regex> = OnceLock::new();
    let mut text = PAGE
        .get_or_init(|| Regex::new(r"(?is)<page-number[^>]*>(.*?)</page-number>").unwrap())
        .replace_all(value, "$1")
        .into_owned();
    text = PARAGRAPH
        .get_or_init(|| Regex::new(r"(?i)</p>").unwrap())
        .replace_all(&text, "\n\n")
        .into_owned();
    text = BREAK
        .get_or_init(|| Regex::new(r"(?i)<br\s*/?>").unwrap())
        .replace_all(&text, "\n")
        .into_owned();
    text = BLOCK
        .get_or_init(|| Regex::new(r"(?i)</(?:div|section|opinion|blockquote|li|h[1-6])>").unwrap())
        .replace_all(&text, "\n")
        .into_owned();
    text = TAG
        .get_or_init(|| Regex::new(r"(?s)<[^>]+>").unwrap())
        .replace_all(&text, "")
        .into_owned();
    text = decode_html(&text);
    text = SPACE
        .get_or_init(|| Regex::new(r"[ \t]+\n").unwrap())
        .replace_all(&text, "\n")
        .into_owned();
    LINES
        .get_or_init(|| Regex::new(r"\n{3,}").unwrap())
        .replace_all(&text, "\n\n")
        .trim()
        .to_owned()
}

fn decode_html(value: &str) -> String {
    static DECIMAL: OnceLock<Regex> = OnceLock::new();
    static HEX: OnceLock<Regex> = OnceLock::new();
    let value = value
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    let value = DECIMAL
        .get_or_init(|| Regex::new(r"&#(\d+);").unwrap())
        .replace_all(&value, |caps: &regex::Captures<'_>| entity(&caps[1], 10));
    HEX.get_or_init(|| Regex::new(r"(?i)&#x([0-9a-f]+);").unwrap())
        .replace_all(&value, |caps: &regex::Captures<'_>| entity(&caps[1], 16))
        .into_owned()
}

fn entity(value: &str, radix: u32) -> String {
    u32::from_str_radix(value, radix)
        .ok()
        .and_then(char::from_u32)
        .map(|value| value.to_string())
        .unwrap_or_default()
}

fn run_court(
    config: &Config,
    expected: &HashMap<String, Expected>,
    started: Instant,
) -> Result<Totals, String> {
    let db = readonly(&config.courtlistener_db)?;
    let mut statement = db.prepare("SELECT id, html_with_citations, xml_harvard, html_columbia, html_lawbox, html_anon_2020, html, plain_text FROM opinion WHERE id >= ? ORDER BY id").map_err(|e| e.to_string())?;
    let mut rows = statement
        .query([config.start_id])
        .map_err(|e| e.to_string())?;
    let mut totals = Totals::default();
    loop {
        let mut batch = Vec::with_capacity(config.batch);
        while batch.len() < config.batch
            && (config.limit == 0 || totals.checked + batch.len() < config.limit)
        {
            match rows.next().map_err(|e| e.to_string())? {
                Some(row) => batch.push(court_row(row).map_err(|e| e.to_string())?),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let batch_len = batch.len();
        let results = batch
            .into_par_iter()
            .map(|row| {
                let key = format!("courtlistener\0{}", row.id);
                let actual = compile_court(row);
                expected
                    .get(&key)
                    .map(|wanted| compare(wanted, actual, false))
                    .unwrap_or_else(|| Comparison::Mismatch(format!("missing oracle row {key}")))
            })
            .collect::<Vec<_>>();
        for result in results {
            record(&mut totals, result);
        }
        progress("courtlistener", &totals, batch_len, started);
        deadline(config, started)?;
        if config.limit > 0 && totals.checked >= config.limit {
            break;
        }
    }
    Ok(totals)
}

fn registrations(config: &Config) -> Result<HashMap<usize, PathBuf>, String> {
    let db = readonly(&config.journal_final_db)?;
    let mut statement = db
        .prepare("SELECT article_id, source_dir FROM article_final_contracts")
        .map_err(|e| e.to_string())?;
    let mut rows = statement.query([]).map_err(|e| e.to_string())?;
    let mut result = HashMap::new();
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let id = usize::try_from(row.get::<_, i64>(0).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let source: Option<String> = row.get(1).map_err(|e| e.to_string())?;
        if let Some(path) =
            source.and_then(|value| registered_pages(&config.journal_final_db, &value))
        {
            result.insert(id, path);
        }
    }
    Ok(result)
}

fn registered_pages(database: &Path, source: &str) -> Option<PathBuf> {
    let relative = Path::new(source);
    if source.is_empty() || relative.is_absolute() {
        return None;
    }
    let directory = database.parent()?;
    for base in [directory, directory.parent()?] {
        let candidate = base.join(relative).join("pages.jsonl");
        if candidate.is_file() {
            let real_base = base.canonicalize().ok()?;
            let real_candidate = candidate.canonicalize().ok()?;
            if real_candidate.starts_with(real_base) {
                return Some(real_candidate);
            }
        }
    }
    None
}

fn trusted_url(value: Option<String>) -> Option<String> {
    trimmed(value).and_then(|mut value| {
        if !(value.starts_with("http://") || value.starts_with("https://")) {
            return None;
        }
        if let Some(at) = value.find('#') {
            value.truncate(at);
        }
        Some(value)
    })
}

fn compile_journal(row: JournalRow) -> Actual {
    let Some(text) = trimmed(row.text) else {
        return failed("provider_unavailable");
    };
    let Some(url) = trusted_url(row.url) else {
        return failed("provider_unavailable");
    };
    if let Some(path) = row.final_pages {
        match File::open(path) {
            Ok(file) => document_actual(journal_source_doc(
                row.id,
                Some(url),
                BufReader::new(file),
                &row.pages,
            )),
            Err(_) => failed("compile_exception"),
        }
    } else {
        document_actual(journal_text_source_doc(row.id, Some(url), text, &row.pages))
    }
}

fn run_journal(
    config: &Config,
    expected: &HashMap<String, Expected>,
    started: Instant,
) -> Result<Totals, String> {
    let registrations = registrations(config)?;
    let db = readonly(&config.journal_db)?;
    let mut statement = db
        .prepare("SELECT article_id, text, galley_url, url_en FROM articles WHERE article_id >= ? ORDER BY article_id")
        .map_err(|e| e.to_string())?;
    let mut rows = statement
        .query([config.start_id])
        .map_err(|e| e.to_string())?;
    let mut page_statement = db.prepare("SELECT CAST(page_label AS TEXT), pdf_page FROM article_pages WHERE article_id = ? ORDER BY page_order").map_err(|e| e.to_string())?;
    let mut totals = Totals::default();
    loop {
        let mut batch = Vec::with_capacity(config.batch);
        while batch.len() < config.batch
            && (config.limit == 0 || totals.checked + batch.len() < config.limit)
        {
            let Some(row) = rows.next().map_err(|e| e.to_string())? else {
                break;
            };
            let raw_id: i64 = row.get(0).map_err(|e| e.to_string())?;
            let id = usize::try_from(raw_id).map_err(|e| e.to_string())?;
            let mut page_rows = page_statement.query([raw_id]).map_err(|e| e.to_string())?;
            let mut pages = Vec::new();
            while let Some(page) = page_rows.next().map_err(|e| e.to_string())? {
                let label: Option<String> = page.get(0).map_err(|e| e.to_string())?;
                let pdf_page: Option<i64> = page.get(1).map_err(|e| e.to_string())?;
                if let (Some(label), Some(pdf_page)) = (label, pdf_page.filter(|value| *value > 0))
                {
                    pages.push(JournalPageLabel {
                        label,
                        pdf_page: pdf_page as usize,
                    });
                }
            }
            let galley: Option<String> = row.get(2).map_err(|e| e.to_string())?;
            let url: Option<String> = row.get(3).map_err(|e| e.to_string())?;
            batch.push(JournalRow {
                id,
                text: row.get(1).map_err(|e| e.to_string())?,
                url: trimmed(galley).or(url),
                pages,
                final_pages: registrations.get(&id).cloned(),
            });
        }
        if batch.is_empty() {
            break;
        }
        let batch_len = batch.len();
        let results = batch
            .into_par_iter()
            .map(|row| {
                let key = format!("journal\0{}", row.id);
                let actual = compile_journal(row);
                expected
                    .get(&key)
                    .map(|wanted| compare(wanted, actual, true))
                    .unwrap_or_else(|| Comparison::Mismatch(format!("missing oracle row {key}")))
            })
            .collect::<Vec<_>>();
        for result in results {
            record(&mut totals, result);
        }
        progress("journal", &totals, batch_len, started);
        deadline(config, started)?;
        if config.limit > 0 && totals.checked >= config.limit {
            break;
        }
    }
    if config.limit == 0 {
        for expected in expected
            .values()
            .filter(|row| row.provider == "journal-final-contract")
        {
            record(
                &mut totals,
                compare(expected, failed("not_applicable_missing_source_row"), false),
            );
        }
    }
    Ok(totals)
}

fn deadline(config: &Config, started: Instant) -> Result<(), String> {
    (started.elapsed() <= Duration::from_secs(config.max_seconds))
        .then_some(())
        .ok_or_else(|| format!("time limit exceeded: {} seconds", config.max_seconds))
}

fn progress(provider: &str, totals: &Totals, batch: usize, started: Instant) {
    if totals.checked / 5_000 > totals.checked.saturating_sub(batch) / 5_000 {
        eprintln!(
            "{provider} checked={} matched={} intentional_changes={} mismatched={} elapsed={:.1}s",
            totals.checked,
            totals.matched,
            totals.intentional_changes,
            totals.checked - totals.matched - totals.intentional_changes,
            started.elapsed().as_secs_f64()
        );
    }
}

fn main() -> Result<(), String> {
    let config = config()?;
    rayon::ThreadPoolBuilder::new()
        .build_global()
        .map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut all = Totals::default();
    for provider in ["a2aj", "courtlistener", "journal"] {
        if !config.providers.contains(provider) {
            continue;
        }
        let expected = load_expected(&config.baseline, provider)?;
        let totals = match provider {
            "a2aj" => run_a2aj(&config, &expected, started)?,
            "courtlistener" => run_court(&config, &expected, started)?,
            _ => run_journal(&config, &expected, started)?,
        };
        all.checked += totals.checked;
        all.matched += totals.matched;
        all.intentional_changes += totals.intentional_changes;
        all.mismatches.extend(
            totals
                .mismatches
                .into_iter()
                .take(50 - all.mismatches.len()),
        );
    }
    println!(
        "checked={} matched={} intentional_changes={} mismatched={} elapsed_ms={}",
        all.checked,
        all.matched,
        all.intentional_changes,
        all.checked - all.matched - all.intentional_changes,
        started.elapsed().as_millis()
    );
    for mismatch in &all.mismatches {
        println!("MISMATCH {mismatch}");
    }
    if all.checked != all.matched + all.intentional_changes {
        return Err("source-structure parity failed".to_owned());
    }
    Ok(())
}
