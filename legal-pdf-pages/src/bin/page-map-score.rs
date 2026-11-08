//! page-map-score: the page map scored against known page maps, from saved evidence; no PDF is read.
//!
//!   page-map-score KEY EVIDENCE [--wrong FILE] [--starts] [--explain ID]
//!
//! With --explain, the shifts article ID's pages support are printed with their anchors. With --starts, each article's own first page (the key's `first_page`, what its citation gives a caller) is the page
//! map's start, as a caller holding the citation passes it.
//!
//! KEY: JSON lines {id, map: [{pdf_page, labels: [printed, ...]}, ...]}, each article's confirmed page map, its article
//! pages only (journals.db `articles.confirmed_page_map`). EVIDENCE: JSON lines {id, pages: [PageEvidence, ...]}
//! (`page-evidence` writes them). Only the key's pages are scored: a PDF's leading and trailing pages outside the article
//! (a cover sheet, front matter, the next article's opening) have no number to get right, and a number put on one is
//! counted apart. A page is right when its number is one of its labels.
use legal_pdf_pages::{page_map, PageEvidence};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};

#[derive(Deserialize)]
struct KeyPage {
    pdf_page: u32,
    labels: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct Key {
    id: serde_json::Value,
    map: Vec<KeyPage>,
    #[serde(default)]
    first_page: Option<serde_json::Value>,
    /// The working system's own raw page map (journals.db `raw_detected_page_map`), [[pdf_page, printed], ...].
    #[serde(default)]
    detected: Option<Vec<Vec<serde_json::Value>>>,
}

#[derive(Deserialize)]
struct Evidence {
    id: serde_json::Value,
    pages: Vec<PageEvidence>,
}

fn lines<T: for<'a> Deserialize<'a>>(path: &str) -> Vec<T> {
    BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"))).lines().map_while(Result::ok)
        .filter_map(|line| serde_json::from_str(&line).ok()).collect()
}

fn labels_right(labels: Option<&Vec<String>>, got: &str) -> bool {
    labels.is_some_and(|labels| labels.iter().any(|label| label.eq_ignore_ascii_case(got)))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(key), Some(evidence)) = (args.first(), args.get(1)) else {
        eprintln!("page-map-score KEY EVIDENCE [--wrong FILE]");
        std::process::exit(2);
    };
    let wrong_path = args.iter().position(|arg| arg == "--wrong").and_then(|at| args.get(at + 1));
    let with_starts = args.iter().any(|arg| arg == "--starts");
    let mut starts_of: HashMap<String, Vec<u32>> = HashMap::new();
    let mut detected_of: HashMap<String, HashMap<u32, String>> = HashMap::new();
    let keys: HashMap<String, HashMap<u32, Vec<String>>> = lines::<Key>(key).into_iter().map(|key| {
        let first = key.first_page.as_ref().and_then(|v| v.as_u64().map(|n| n as u32).or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())));
        if let Some(first) = first { starts_of.insert(key.id, vec![first]); }
        let label = |v: &serde_json::Value| v.as_str().map(|s| s.trim().to_owned()).or_else(|| v.as_i64().map(|n| n.to_string()));
        detected_of.insert(key.id, key.detected.iter().flatten()
            .filter_map(|row| Some((row.first()?.as_u64()? as u32, label(row.get(1)?)?))).collect());
        (key.id, key.map.into_iter()
        .map(|page| (page.pdf_page, page.labels.iter().filter_map(|l| l.as_str().map(|s| s.trim().to_owned()).or_else(|| l.as_i64().map(|n| n.to_string()))).collect()))
        .collect())
    }).collect();
    let mut tally: BTreeMap<&str, u64> = BTreeMap::new();
    let mut by_source: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut wrong = wrong_path.map(|path| std::fs::File::create(path).expect("--wrong"));
    let explain: Option<String> = args.iter().position(|arg| arg == "--explain").and_then(|at| args.get(at + 1)).map(|id| id.trim_matches('"').to_owned());
    let name = |id: &serde_json::Value| id.as_str().map_or_else(|| id.to_string(), str::to_owned);
    for article in lines::<Evidence>(evidence) {
        let Some(expected) = keys.get(&article.id) else { continue };
        if explain == Some(article.id) {
            for (shift, anchors) in legal_pdf_pages::weighed(&article.pages).into_iter().take(6) {
                println!("shift {shift:+}: {} anchors", anchors.len());
                for (page, number, text) in anchors.iter().take(8) {
                    println!("    pdf {page:>4} -> {number:<6} {text}");
                }
            }
        }
        let starts = if with_starts { starts_of.get(&article.id).cloned().unwrap_or_default() } else { Vec::new() };
        let map = page_map(&article.pages, &starts, &|_, _| false);
        *tally.entry("articles").or_default() += 1;
        let mut all_right = true;
        for (&page, printed) in detected_of.get(&article.id).into_iter().flatten() {
            if !expected.contains_key(&page) {
                continue;
            }
            *tally.entry("pages the working system numbered").or_default() += 1;
            let got = map.get(page as usize - 1).and_then(|b| b.label.as_deref());
            *tally.entry(match got {
                Some(got) if got.eq_ignore_ascii_case(printed) => "  of those, numbered the same",
                Some(_) => "  of those, numbered otherwise",
                None => "  of those, not numbered",
            }).or_default() += 1;
            if labels_right(expected.get(&page), printed) {
                *tally.entry("  of those, the working system right").or_default() += 1;
            }
        }
        for (&page, labels) in expected {
            *tally.entry("pages").or_default() += 1;
            let binding = map.get(page as usize - 1);
            let got = binding.and_then(|b| b.label.as_deref());
            let source = binding.and_then(|b| b.source.clone()).unwrap_or_else(|| "none".to_owned());
            let right = got.is_some_and(|got| labels.iter().any(|label| label.eq_ignore_ascii_case(got)));
            let entry = by_source.entry(source).or_default();
            entry.0 += 1;
            if right {
                entry.1 += 1;
                *tally.entry("right").or_default() += 1;
            } else {
                all_right = false;
                *tally.entry(if got.is_some() { "wrong number" } else { "no number" }).or_default() += 1;
                if let Some(file) = wrong.as_mut() {
                    let _ = writeln!(file, "{}", serde_json::json!({"id": article.id, "pdf_page": page, "labels": labels, "got": got,
                        "observed": binding.and_then(|b| b.observed.clone()), "source": binding.and_then(|b| b.source.clone())}));
                }
            }
        }
        if all_right {
            *tally.entry("articles with every page right").or_default() += 1;
        }
        *tally.entry("pages outside the article given a number").or_default() +=
            map.iter().filter(|b| b.label.is_some() && !expected.contains_key(&b.pdf_page)).count() as u64;
    }
    for (what, n) in &tally {
        println!("  {n:>8}  {what}");
    }
    let pages = tally.get("pages").copied().unwrap_or(0).max(1) as f64;
    println!("  page numbers right: {:.2}%", 100.0 * tally.get("right").copied().unwrap_or(0) as f64 / pages);
    for (source, (n, right)) in &by_source {
        println!("    by source {source:<9} {n:>7} pages, {:.2}% right", 100.0 * *right as f64 / (*n).max(1) as f64);
    }
}
