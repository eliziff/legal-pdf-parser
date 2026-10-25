//! Whether each PDF's text is its own (digital-born) or a recognition of its image (an OCR layer), by
//! pdf-inspector's detection, the one extraction reads: a PDF is digital-born where it is text-based and no page
//! needs recognition, OCR otherwise (a scan, an invisible text layer, no text).
//!
//!   cargo run --release --example text_layer_kinds < ID_TAB_PATH.tsv > KINDS.tsv
//!
//! Each input line is `id <TAB> path`; each output line `id <TAB> digital|ocr|unread <TAB> pdf type <TAB> reasons`.
use std::io::{BufRead, Write};

fn main() {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Some((id, path)) = line.split_once('\t') else { continue };
        let row = match pdf_inspector::detect_pdf(path) {
            Ok(result) => {
                let mut reasons: Vec<&str> = result.ocr_reasons_by_page.iter()
                    .flat_map(|page| page.reasons.iter().map(String::as_str)).collect();
                reasons.sort_unstable();
                reasons.dedup();
                let text_based = matches!(result.pdf_type, pdf_inspector::PdfType::TextBased);
                let kind = if text_based && result.pages_needing_ocr.is_empty() { "digital" } else { "ocr" };
                format!("{id}\t{kind}\t{:?}\t{}", result.pdf_type, reasons.join(","))
            }
            Err(error) => format!("{id}\tunread\t\t{}", error.to_string().replace(['\t', '\n'], " ")),
        };
        let _ = writeln!(out, "{row}");
    }
}
