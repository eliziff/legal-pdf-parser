use legal_pdf_extraction_processor::{assemble_pdf, load_extraction_document, page_geometries};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let page: usize = args[2].parse().unwrap();
    let document = load_extraction_document(&bytes).unwrap();
    let geometry = page_geometries(&document);
    let ((items, _, _, rules), detection) = pdf_inspector::extract_fidelity_from_doc(&document).unwrap();
    for rule in &rules { if rule.page as usize == page { eprintln!("RULE {rule:?}"); } }
    let extracted = assemble_pdf(&bytes, &geometry, items, rules, detection, None, None).unwrap();
    eprintln!("SEPARATOR {:?}", extracted.separators[page - 1]);
    println!("{}", serde_json::to_string(&extracted.pages[page - 1]).unwrap());
}
