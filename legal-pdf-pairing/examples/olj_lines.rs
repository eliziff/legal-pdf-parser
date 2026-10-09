//! Pair the Open Access Journals Database goldset's own input lines, so the pairer can be
//! scored against the original on identical text: `olj_lines <lines.jsonl> <markers.jsonl>`.
//! Each line carries its text, region and reading order as the original pairer read them.

use legal_pdf_core::model::{Line, Page};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// A line box, as an object or as the text of one ("{'x0': 192.0, ...}").
fn bbox(value: &Value) -> [f64; 4] {
    let parsed;
    let value = match value.as_str() {
        Some(text) => {
            parsed = serde_json::from_str::<Value>(&text.replace('\'', "\"")).unwrap_or(Value::Null);
            &parsed
        }
        None => value,
    };
    let get = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| value.get(*key).and_then(Value::as_f64))
    };
    match (
        get(&["x0", "left", "x"]),
        get(&["y0", "top", "y"]),
        get(&["x1", "right"]),
        get(&["y1", "bottom"]),
    ) {
        (Some(x0), Some(y0), Some(x1), Some(y1)) => [x0, y0, x1, y1],
        _ => [0.0; 4],
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let input = std::fs::File::open(&args[1]).expect("goldset lines");
    let mut out = BufWriter::new(std::fs::File::create(&args[2]).expect("markers output"));
    let mut articles = BTreeMap::<(String, String), Vec<Value>>::new();
    for row in BufReader::new(input).lines() {
        let row = row.expect("goldset line");
        let Ok(row) = serde_json::from_str::<Value>(row.trim_start_matches('\u{feff}')) else {
            continue;
        };
        let key = |field: &str| match &row[field] {
            Value::String(text) => text.clone(),
            Value::Number(number) => number.to_string(),
            _ => String::new(),
        };
        let article = (key("dataset"), key("article_id"));
        if !article.0.is_empty() && !article.1.is_empty() {
            articles.entry(article).or_default().push(row);
        }
    }
    let citations = BTreeMap::new();
    for ((dataset, article_id), rows) in articles {
        let mut pages = BTreeMap::<u64, Vec<Line>>::new();
        for (index, row) in rows.iter().enumerate() {
            let number = row["pdf_page"].as_u64().unwrap_or(1).max(1);
            let text = ["raw_transcription", "normalized_transcription", "line_text"]
                .iter()
                .find_map(|field| row[*field].as_str().filter(|text| !text.is_empty()))
                .unwrap_or_default();
            let line = json!({
                "id": row["line_id"],
                "page_index": number - 1,
                "page_number": number,
                "source_index": index,
                "reading_order": row["reading_order_index"].as_u64().unwrap_or(index as u64),
                "block_index": 0,
                "text": text,
                "bbox": bbox(&row["line_bbox_px"]),
                // OLJ_NO_REGIONS=1 reads every line as running text, to weigh the regions.
                "region_type": if std::env::var_os("OLJ_NO_REGIONS").is_some() {
                    "text"
                } else {
                    row["region_type"].as_str().unwrap_or("unknown")
                },
                "source": "ocr",
            });
            pages
                .entry(number)
                .or_default()
                .push(serde_json::from_value(line).expect("goldset line as a parser line"));
        }
        let pages = pages
            .into_iter()
            .map(|(number, mut lines)| {
                lines.sort_by_key(|line| line.reading_order);
                let width = lines.iter().map(|line| line.bbox[2]).fold(1000.0, f64::max);
                let height = lines.iter().map(|line| line.bbox[3]).fold(1400.0, f64::max);
                serde_json::from_value::<Page>(json!({
                    "id": format!("p{number:04}"),
                    "index": number - 1,
                    "number": number,
                    "width": width,
                    "height": height,
                    "lines": lines,
                    "regions": [],
                    "source": "ocr",
                }))
                .expect("goldset page")
            })
            .collect::<Vec<_>>();
        let output = legal_pdf_pairing::pair_footnotes(&pages, &citations, true);
        for mut marker in output
            .pairing_audit
            .map(|audit| audit.markers)
            .unwrap_or_default()
        {
            marker["dataset"] = json!(dataset);
            marker["article_id"] = json!(article_id);
            writeln!(out, "{marker}").expect("markers output");
        }
    }
}
