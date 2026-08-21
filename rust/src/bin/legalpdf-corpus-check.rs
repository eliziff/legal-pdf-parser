use legalpdf::corpus_check_cached_extraction;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

const BASELINE_SCHEMA: &str = "legalpdf.corpus-check-baseline.v1";
const SUMMARY_SCHEMA: &str = "legalpdf.corpus-check-summary.v2";
const RECEIPT_SCHEMA: &str = "legalpdf.corpus-check-document.v2";
const MAX_SECONDS: f64 = 180.0;
const DEFAULT_JOBS: usize = 3;
const MAX_JOBS: usize = 16;

type AppResult<T> = Result<T, String>;

#[derive(Clone, Debug, Deserialize)]
struct DocumentRow {
    id: String,
    cache_path: PathBuf,
    relative_path: String,
    source_sha256: String,
    pages: usize,
    jurisdiction: String,
    source_family: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct KnownFailure {
    id: String,
    relative_path: String,
    source_sha256: String,
    pages: usize,
    error_code: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct Denominator {
    attempts: usize,
    successful_documents: usize,
    successful_pages: usize,
    failures: usize,
    failure_pages: usize,
}

#[derive(Debug, Deserialize)]
struct Baseline {
    schema_version: String,
    corpus_id: String,
    denominator: Denominator,
    documents: Vec<DocumentRow>,
    #[serde(default)]
    known_failures: Vec<KnownFailure>,
}

impl Baseline {
    fn validate(&self) -> AppResult<()> {
        if self.schema_version != BASELINE_SCHEMA || self.corpus_id.trim().is_empty() {
            return Err("invalid corpus baseline identity".to_owned());
        }
        let mut ids = HashSet::new();
        for row in &self.documents {
            if row.id.trim().is_empty()
                || !ids.insert(row.id.as_str())
                || row.cache_path.as_os_str().is_empty()
                || row.cache_path.is_absolute()
                || row.relative_path.trim().is_empty()
                || !is_sha256(&row.source_sha256)
                || row.pages == 0
                || row.jurisdiction.trim().is_empty()
                || row.source_family.trim().is_empty()
            {
                return Err(format!("invalid baseline document {}", row.id));
            }
        }
        for row in &self.known_failures {
            if row.id.trim().is_empty()
                || !ids.insert(row.id.as_str())
                || row.relative_path.trim().is_empty()
                || !is_sha256(&row.source_sha256)
                || row.pages == 0
                || row.error_code.trim().is_empty()
            {
                return Err(format!("invalid known failure {}", row.id));
            }
        }
        let expected = Denominator {
            attempts: self.documents.len() + self.known_failures.len(),
            successful_documents: self.documents.len(),
            successful_pages: self.documents.iter().map(|row| row.pages).sum(),
            failures: self.known_failures.len(),
            failure_pages: self.known_failures.iter().map(|row| row.pages).sum(),
        };
        if self.denominator.attempts != expected.attempts
            || self.denominator.successful_documents != expected.successful_documents
            || self.denominator.successful_pages != expected.successful_pages
            || self.denominator.failures != expected.failures
            || self.denominator.failure_pages != expected.failure_pages
        {
            return Err("baseline denominator does not match its rows".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    baseline: PathBuf,
    corpus_root: PathBuf,
    out: PathBuf,
    jobs: usize,
}

fn usage() -> &'static str {
    "usage: legalpdf-corpus-check --baseline <accepted-catalog> --corpus-root <corpus> --out <new-empty-dir> [--jobs 3]"
}

fn parse_args(values: impl IntoIterator<Item = String>) -> AppResult<Args> {
    let mut values = values.into_iter();
    let (mut baseline, mut corpus_root, mut out) = (None, None, None);
    let mut jobs = DEFAULT_JOBS;
    while let Some(value) = values.next() {
        match value.as_str() {
            "--baseline" => baseline = values.next().map(PathBuf::from),
            "--corpus-root" => corpus_root = values.next().map(PathBuf::from),
            "--out" => out = values.next().map(PathBuf::from),
            "--jobs" => {
                jobs = values
                    .next()
                    .ok_or_else(|| "--jobs requires a value".to_owned())?
                    .parse()
                    .map_err(|_| "--jobs must be an integer".to_owned())?;
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unknown argument {value}\n{}", usage())),
        }
    }
    if !(1..=MAX_JOBS).contains(&jobs) {
        return Err(format!("--jobs must be between 1 and {MAX_JOBS}"));
    }
    Ok(Args {
        baseline: baseline.ok_or_else(|| format!("--baseline is required\n{}", usage()))?,
        corpus_root: corpus_root
            .ok_or_else(|| format!("--corpus-root is required\n{}", usage()))?,
        out: out.ok_or_else(|| format!("--out is required\n{}", usage()))?,
        jobs,
    })
}

#[derive(Debug)]
struct Completion {
    row: DocumentRow,
    elapsed_seconds: f64,
    outcome: AppResult<Value>,
}

fn read_json(path: &Path) -> AppResult<Value> {
    serde_json::from_reader(BufReader::new(
        File::open(path).map_err(|error| format!("{}: {error}", path.display()))?,
    ))
    .map_err(|error| format!("{}: {error}", path.display()))
}

fn process(row: DocumentRow, root: &Path) -> Completion {
    let started = Instant::now();
    let outcome = (|| {
        let result =
            corpus_check_cached_extraction(root.join(&row.cache_path), row.relative_path.clone())
                .map_err(|error| error.to_string())?;
        if result.get("source_sha256").and_then(Value::as_str) != Some(&row.source_sha256)
            || result.get("page_count").and_then(Value::as_u64) != Some(row.pages as u64)
        {
            return Err("production result disagrees with accepted source identity".to_owned());
        }
        Ok(result)
    })();
    Completion {
        row,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        outcome,
    }
}

fn write_atomic(path: &Path, value: &Value) -> AppResult<()> {
    let temporary = path.with_extension(format!(
        "tmp-{}",
        thread::current().name().unwrap_or("worker")
    ));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("{}: {error}", temporary.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())?;
    drop(writer);
    fs::rename(&temporary, path).map_err(|error| format!("{}: {error}", path.display()))
}

fn add_counts(target: &mut BTreeMap<String, usize>, value: Option<&Value>) {
    if let Some(values) = value.and_then(Value::as_object) {
        for (key, count) in values {
            *target.entry(key.clone()).or_default() += count.as_u64().unwrap_or(0) as usize;
        }
    }
}

fn ranking_entry(row: &DocumentRow, result: &Value) -> Value {
    json!({
        "id": row.id,
        "relative_path": row.relative_path,
        "pages": row.pages,
        "nodes": result.pointer("/structure/node_count").and_then(Value::as_u64).unwrap_or(0),
        "diagnostics": result.pointer("/structure/diagnostic_count").and_then(Value::as_u64).unwrap_or(0),
    })
}

fn run() -> AppResult<i32> {
    let args = parse_args(std::env::args().skip(1))?;
    let baseline: Baseline = serde_json::from_value(read_json(&args.baseline)?)
        .map_err(|error| format!("{}: {error}", args.baseline.display()))?;
    baseline.validate()?;
    if !args.corpus_root.is_dir() {
        return Err("--corpus-root must be a directory".to_owned());
    }
    if args.out.exists() {
        if !args.out.is_dir()
            || fs::read_dir(&args.out)
                .map_err(|error| error.to_string())?
                .next()
                .is_some()
        {
            return Err("--out must name a new or empty directory".to_owned());
        }
    } else {
        fs::create_dir_all(&args.out).map_err(|error| error.to_string())?;
    }

    let started = Instant::now();
    let queue = Arc::new(Mutex::new(VecDeque::from(baseline.documents.clone())));
    let (sender, receiver) = mpsc::channel();
    let mut workers = Vec::new();
    for worker in 0..args.jobs.min(baseline.documents.len().max(1)) {
        let queue = Arc::clone(&queue);
        let sender = sender.clone();
        let root = args.corpus_root.clone();
        workers.push(
            thread::Builder::new()
                .name(format!("{worker}"))
                .spawn(move || loop {
                    let row = queue.lock().expect("corpus queue poisoned").pop_front();
                    let Some(row) = row else { break };
                    if sender.send(process(row, &root)).is_err() {
                        break;
                    }
                })
                .map_err(|error| error.to_string())?,
        );
    }
    drop(sender);

    let mut passed = 0_usize;
    let mut passed_pages = 0_usize;
    let mut failures = Vec::new();
    let mut by_kind = BTreeMap::new();
    let mut by_rule = BTreeMap::new();
    let mut sections_by_locator_kind = BTreeMap::new();
    let mut by_diagnostic = BTreeMap::new();
    let mut by_group = BTreeMap::<String, Map<String, Value>>::new();
    let mut ranked = Vec::new();
    let mut heading_derived_sections = 0_usize;
    let mut abstentions = 0_usize;
    let mut partial_resolutions = 0_usize;
    for completion in receiver {
        let elapsed = completion.elapsed_seconds;
        let row = completion.row;
        let receipt = match completion.outcome {
            Ok(result) => {
                passed += 1;
                passed_pages += row.pages;
                add_counts(&mut by_kind, result.pointer("/structure/by_kind"));
                add_counts(&mut by_rule, result.pointer("/structure/by_rule"));
                add_counts(
                    &mut sections_by_locator_kind,
                    result.pointer("/structure/sections_by_locator_kind"),
                );
                add_counts(
                    &mut by_diagnostic,
                    result.pointer("/structure/diagnostics_by_code"),
                );
                heading_derived_sections += result
                    .pointer("/structure/heading_derived_section_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                abstentions += result
                    .pointer("/structure/abstention_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                partial_resolutions += result
                    .pointer("/structure/partial_resolution_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                ranked.push(ranking_entry(&row, &result));
                json!({"schema_version":RECEIPT_SCHEMA,"id":row.id,"ok":true,"pages":row.pages,
                    "elapsed_seconds":elapsed,"structure":result["structure"]})
            }
            Err(error) => {
                failures.push(json!({"id":row.id,"relative_path":row.relative_path,"error":error}));
                json!({"schema_version":RECEIPT_SCHEMA,"id":row.id,"ok":false,"pages":row.pages,
                    "elapsed_seconds":elapsed,"error":failures.last().unwrap()["error"]})
            }
        };
        for group in [
            format!("jurisdiction:{}", row.jurisdiction),
            format!("source_family:{}", row.source_family),
        ] {
            let entry = by_group.entry(group).or_insert_with(|| {
                Map::from_iter([
                    ("documents".to_owned(), json!(0)),
                    ("pages".to_owned(), json!(0)),
                    ("failures".to_owned(), json!(0)),
                ])
            });
            entry["documents"] = json!(entry["documents"].as_u64().unwrap_or(0) + 1);
            entry["pages"] = json!(entry["pages"].as_u64().unwrap_or(0) + row.pages as u64);
            if !receipt["ok"].as_bool().unwrap_or(false) {
                entry["failures"] = json!(entry["failures"].as_u64().unwrap_or(0) + 1);
            }
        }
        write_atomic(&args.out.join(format!("{}.json", row.id)), &receipt)?;
        eprintln!(
            "documents {}/{} | pages {}/{} | failures {} | {:.1}s | {:.1} pages/s",
            passed + failures.len(),
            baseline.documents.len(),
            passed_pages,
            baseline.denominator.successful_pages,
            failures.len(),
            started.elapsed().as_secs_f64(),
            passed_pages as f64 / started.elapsed().as_secs_f64().max(0.001)
        );
    }
    let worker_panics = workers
        .into_iter()
        .filter_map(|worker| worker.join().err())
        .count();
    if worker_panics > 0 {
        failures.push(json!({"error":format!("{worker_panics} corpus workers panicked")}));
    }
    let mut highest_concentration = ranked.clone();
    highest_concentration.sort_by_key(|item| {
        std::cmp::Reverse((
            item["nodes"].as_u64().unwrap_or(0),
            item["pages"].as_u64().unwrap_or(0),
        ))
    });
    highest_concentration.truncate(10);
    ranked.sort_by_key(|item| std::cmp::Reverse(item["diagnostics"].as_u64().unwrap_or(0)));
    ranked.truncate(10);
    let elapsed = started.elapsed().as_secs_f64();
    let pass = failures.is_empty()
        && passed == baseline.denominator.successful_documents
        && passed_pages == baseline.denominator.successful_pages
        && heading_derived_sections == 0
        && elapsed <= MAX_SECONDS;
    let summary = json!({
        "schema_version": SUMMARY_SCHEMA, "corpus_id": baseline.corpus_id, "pass": pass,
        "documents": {"successful":passed,"expected":baseline.denominator.successful_documents},
        "pages": {"successful":passed_pages,"expected":baseline.denominator.successful_pages},
        "known_failures": baseline.known_failures, "known_failure_pages": baseline.denominator.failure_pages,
        "processing_errors": failures.len(), "elapsed_seconds": elapsed,
        "pages_per_second": passed_pages as f64 / elapsed.max(0.001), "max_seconds": MAX_SECONDS,
        "structure": {"by_kind":by_kind,"by_rule":by_rule,"diagnostics_by_code":by_diagnostic,
            "sections_by_locator_kind":sections_by_locator_kind,
            "heading_derived_section_count":heading_derived_sections,
            "abstention_count":abstentions,"partial_resolution_count":partial_resolutions},
        "results_by_group": by_group, "highest_concentration_documents": highest_concentration,
        "worst_documents": ranked, "failures": failures,
    });
    write_atomic(&args.out.join("summary.json"), &summary)?;
    println!("{summary}");
    Ok(if pass { 0 } else { 1 })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("legalpdf-corpus-check: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_has_one_fresh_output_mode() {
        let args = parse_args(
            [
                "--baseline",
                "accepted.json",
                "--corpus-root",
                "corpus",
                "--out",
                "receipts",
                "--jobs",
                "3",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.jobs, 3);
        assert!(parse_args(
            ["--baseline", "accepted.json", "--out", "receipts"]
                .into_iter()
                .map(str::to_owned)
        )
        .is_err());
        assert!(parse_args(["--resume", "old"].into_iter().map(str::to_owned)).is_err());
    }
}
