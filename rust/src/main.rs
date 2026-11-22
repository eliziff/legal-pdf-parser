// The citation engine's data (legal-citations-data): this program runs the citation engine, whose data the program supplies.
use legal_citations_data as _;
use legal_citations as _;
#[cfg(feature = "fast-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legalpdf::{
    read_pdf, Error, PdfLookupRequest, PdfRequest,
    PdfStructureLookup, Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};

const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_PDF_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PdfInspectorManifest {
    schema_version: String,
    documents: Vec<PdfInspectorDocument>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PdfInspectorDocument {
    path: String,
    sha256: String,
    pages: usize,
}

#[derive(Serialize)]
struct PdfInspectorProduct<'a> {
    structure: &'a legal_structure_model::DocumentStructure,
    pages: Vec<PdfStructureLookup>,
}

#[derive(Serialize)]
struct PdfInspectorRow<'a> {
    path: &'a str,
    product_sha256: String,
    #[serde(flatten)]
    product: &'a PdfInspectorProduct<'a>,
}

struct TempCacheRoot(PathBuf);

impl TempCacheRoot {
    fn create(prefix: &str) -> Result<Self> {
        let temporary = std::env::temp_dir();
        for attempt in 0..100 {
            let path = temporary.join(format!(
                "{prefix}-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(Error::io(&path, source)),
            }
        }
        Err(Error::Message(format!(
            "could not create temporary cache root for {prefix}"
        )))
    }

    fn document(&self, index: usize) -> Result<PathBuf> {
        let path = self.0.join(index.to_string());
        std::fs::create_dir(&path).map_err(|source| Error::io(&path, source))?;
        Ok(path)
    }
}

impl Drop for TempCacheRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn usage() -> &'static str {
    "usage:\n  legalpdf extract-text <input.pdf> [--out <path>] [--cache-dir <dir>] [--profile <name>] [--tier t1|t2] [--ocr <provider>] [--ocr-pack <dir>] [--dpi <n>] [--doc <dir>] [--json-units | --json-structure]\n  legalpdf --version"
}

fn pdf_inspector_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative.len() > 1_024
        || relative.chars().any(char::is_control)
        || relative_path.is_absolute()
        || !relative_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        || !relative_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
    {
        return Err(Error::Message(format!(
            "invalid PDF Inspector manifest path: {relative}"
        )));
    }
    let path = root.join(relative_path);
    let canonical = std::fs::canonicalize(&path).map_err(|source| Error::io(&path, source))?;
    if !canonical.starts_with(root) {
        return Err(Error::Message(format!(
            "PDF Inspector manifest path escapes the repository root: {relative}"
        )));
    }
    Ok(canonical)
}

fn pdf_inspector_gate_command(arguments: &[String]) -> Result<i32> {
    let [manifest, repository_root] = arguments else {
        return Err(Error::Message(
            "_pdf-inspector-gate requires <manifest.json> <repository-root>".to_owned(),
        ));
    };
    let manifest_path = PathBuf::from(manifest);
    let metadata =
        std::fs::metadata(&manifest_path).map_err(|source| Error::io(&manifest_path, source))?;
    if metadata.len() > MAX_MANIFEST_BYTES as u64 {
        return Err(Error::Message("gate manifest exceeds 1 MiB".to_owned()));
    }
    let bytes =
        std::fs::read(&manifest_path).map_err(|source| Error::io(&manifest_path, source))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(Error::Message(
            "gate manifest changed while it was being read".to_owned(),
        ));
    }
    let manifest: PdfInspectorManifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != "legalpdf.cache-contract-corpus.v1"
        || manifest.documents.is_empty()
    {
        return Err(Error::Message(
            "invalid PDF Inspector gate manifest".to_owned(),
        ));
    }
    let root_path = PathBuf::from(repository_root);
    let root = std::fs::canonicalize(&root_path).map_err(|source| Error::io(&root_path, source))?;
    if !root.is_dir() {
        return Err(Error::Message(format!(
            "PDF Inspector repository root is not a directory: {}",
            root.display()
        )));
    }

    let caches = TempCacheRoot::create("legalpdf-pdf-inspector-gate")?;
    let mut seen = HashSet::new();
    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let total = manifest.documents.len();
    for (index, entry) in manifest.documents.iter().enumerate() {
        if entry.pages == 0
            || entry.sha256.len() != 64
            || !entry
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(Error::Message(format!(
                "invalid PDF Inspector manifest entry: {}",
                entry.path
            )));
        }
        let path = pdf_inspector_path(&root, &entry.path)?;
        if !seen.insert(path.clone()) {
            return Err(Error::Message(format!(
                "duplicate PDF Inspector manifest path: {}",
                entry.path
            )));
        }
        let metadata = std::fs::metadata(&path).map_err(|source| Error::io(&path, source))?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PDF_BYTES {
            return Err(Error::Message(format!(
                "PDF Inspector source size is invalid: {}",
                entry.path
            )));
        }
        let bytes = std::fs::read(&path).map_err(|source| Error::io(&path, source))?;
        if bytes.len() as u64 != metadata.len() {
            return Err(Error::Message(format!(
                "PDF Inspector source changed while it was being read: {}",
                entry.path
            )));
        }
        let source_sha256 = Sha256::digest(&bytes).iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        if source_sha256 != entry.sha256 {
            return Err(Error::Message(format!(
                "PDF Inspector source SHA-256 mismatch: {}",
                entry.path
            )));
        }

        eprintln!(
            "gate {}/{}: {} ({} bytes, {} pages)",
            index + 1,
            total,
            entry.path,
            bytes.len(),
            entry.pages
        );
        let cache_dir = caches.document(index)?;
        let request: PdfRequest = serde_json::from_value(serde_json::json!({
            "cacheDir": cache_dir,
            "expectedSourceSha256": entry.sha256,
        }))?;
        let document = read_pdf(Some(&bytes), &request, None)?.ok_or_else(|| Error::Message("PDF cache miss after parsing source bytes".to_owned()))?;
        if document.page_count() != entry.pages {
            return Err(Error::Message(format!(
                "PDF Inspector page count mismatch: {} expected {}, got {}",
                entry.path,
                entry.pages,
                document.page_count()
            )));
        }
        let query = legal_document_query::DocumentQuery::new();
        let mut pages = Vec::with_capacity(entry.pages);
        for page in 1..=entry.pages {
            pages.push(document.lookup(&query, &PdfLookupRequest::new("page", page.to_string())));
        }
        let product = PdfInspectorProduct {
            structure: document.structure(),
            pages,
        };
        let product_sha256 = Sha256::digest(serde_json::to_vec(&product)?).iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        serde_json::to_writer(
            &mut output,
            &PdfInspectorRow {
                path: &entry.path,
                product_sha256: product_sha256.clone(),
                product: &product,
            },
        )?;
        output
            .write_all(b"\n")
            .map_err(|source| Error::io("<stdout>", source))?;
        output
            .flush()
            .map_err(|source| Error::io("<stdout>", source))?;
        eprintln!("  product {product_sha256}");
    }
    Ok(0)
}

fn next_flag_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    let Some(value) = args.get(*index) else {
        return Err(Error::Message(format!("{flag} requires a value")));
    };
    Ok(value.clone())
}

/// Extract deterministic UTF-8 text (or structured JSON) from a single PDF via `read_pdf`.
fn extract_text_command(arguments: &[String]) -> Result<i32> {
    let mut input: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut cache_dir_arg: Option<PathBuf> = None;
    let mut profile: Option<String> = None;
    let mut tier_t2 = false;
    let mut ocr_enabled = false;
    let mut ocr_pack: Option<PathBuf> = None;
    let mut ocr_dpi: Option<u16> = None;
    let mut doc_pack: Option<PathBuf> = None;
    let mut json_units = false;
    let mut json_structure = false;

    let mut index = 0;
    while index < arguments.len() {
        let arg = &arguments[index];
        match arg.as_str() {
            "--out" | "-o" => out_path = Some(PathBuf::from(next_flag_value(arguments, &mut index, "--out")?)),
            "--cache-dir" => cache_dir_arg = Some(PathBuf::from(next_flag_value(arguments, &mut index, "--cache-dir")?)),
            "--profile" => profile = Some(next_flag_value(arguments, &mut index, "--profile")?),
            "--tier" => {
                let tier = next_flag_value(arguments, &mut index, "--tier")?;
                match tier.as_str() {
                    "t1" => tier_t2 = false,
                    "t2" => tier_t2 = true,
                    _ => {
                        return Err(Error::Message(format!(
                            "unsupported --tier value: {tier} (expected t1 or t2)"
                        )))
                    }
                }
            }
            "--ocr" => {
                let provider = next_flag_value(arguments, &mut index, "--ocr")?;
                if provider != "indra" {
                    return Err(Error::Message(format!("unsupported --ocr provider: {provider} (expected indra)")));
                }
                ocr_enabled = true;
            }
            "--ocr-pack" => {
                ocr_pack = Some(PathBuf::from(next_flag_value(arguments, &mut index, "--ocr-pack")?));
                ocr_enabled = true;
            }
            "--dpi" => {
                let raw = next_flag_value(arguments, &mut index, "--dpi")?;
                let dpi: u16 = raw
                    .parse()
                    .map_err(|_| Error::Message(format!("invalid --dpi value: {raw}")))?;
                if dpi == 0 {
                    return Err(Error::Message("--dpi must be a positive integer".to_owned()));
                }
                ocr_dpi = Some(dpi);
                ocr_enabled = true;
            }
            "--doc" => doc_pack = Some(PathBuf::from(next_flag_value(arguments, &mut index, "--doc")?)),
            "--json-units" => json_units = true,
            "--json-structure" => json_structure = true,
            _ if arg.starts_with('-') => {
                return Err(Error::Message(format!("unknown flag for extract-text: {arg}")));
            }
            _ => {
                if input.is_some() {
                    return Err(Error::Message(
                        "extract-text accepts only one <input.pdf> path".to_owned(),
                    ));
                }
                input = Some(PathBuf::from(arg));
            }
        }
        index += 1;
    }

    if tier_t2 {
        ocr_enabled = true;
        ocr_dpi.get_or_insert(300);
    }
    if json_units && json_structure {
        return Err(Error::Message(
            "--json-units and --json-structure are mutually exclusive".to_owned(),
        ));
    }

    let input_path = input.ok_or_else(|| Error::Message(usage().to_owned()))?;
    let metadata =
        std::fs::metadata(&input_path).map_err(|source| Error::io(&input_path, source))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PDF_BYTES {
        return Err(Error::Message(format!(
            "PDF input size is invalid: {}",
            input_path.display()
        )));
    }
    let bytes = std::fs::read(&input_path).map_err(|source| Error::io(&input_path, source))?;
    let source_sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let _temp_cache = if cache_dir_arg.is_none() {
        Some(TempCacheRoot::create("legalpdf-extract-text")?)
    } else {
        None
    };
    let cache_dir = cache_dir_arg
        .or_else(|| _temp_cache.as_ref().map(|root| root.0.clone()))
        .expect("cache dir resolved");
    std::fs::create_dir_all(&cache_dir).map_err(|source| Error::io(&cache_dir, source))?;

    let mut request_body = serde_json::Map::from_iter([
        ("cacheDir".to_owned(), serde_json::json!(cache_dir)),
        ("expectedSourceSha256".to_owned(), serde_json::json!(source_sha256)),
    ]);
    if let Some(profile) = profile {
        request_body.insert("profile".to_owned(), serde_json::json!(profile));
    }
    if ocr_enabled {
        let mut settings = serde_json::Map::new();
        if let Some(pack) = ocr_pack {
            settings.insert("model_pack".to_owned(), serde_json::json!(pack));
        }
        if let Some(dpi) = ocr_dpi {
            settings.insert("dpi".to_owned(), serde_json::json!(dpi));
        }
        request_body.insert(
            "ocr".to_owned(),
            serde_json::json!({ "provider": "indra", "settings": settings }),
        );
    }
    if let Some(pack) = doc_pack {
        request_body.insert(
            "structure".to_owned(),
            serde_json::json!({ "provider": "doc", "settings": { "model_pack": pack } }),
        );
    }
    let request: PdfRequest = serde_json::from_value(serde_json::Value::Object(request_body))?;
    let document = read_pdf(Some(&bytes), &request, None)?
        .ok_or_else(|| Error::Message("PDF extraction returned no document".to_owned()))?;

    let payload: Vec<u8> = if json_structure {
        let mut structure_json = serde_json::to_vec_pretty(&serde_json::json!({
            "sourceSha256": source_sha256,
            "summary": document.summary(),
            "structure": document.structure(),
        }))?;
        structure_json.push(b'\n');
        structure_json
    } else if json_units {
        let mut units_json = serde_json::to_vec_pretty(&serde_json::json!({
            "sourceSha256": source_sha256,
            "pageCount": document.page_count(),
            "units": document.authority_text_units(),
            "pages": document.page_texts(),
        }))?;
        units_json.push(b'\n');
        units_json
    } else {
        let unit_texts: Vec<&str> = document
            .authority_text_units()
            .iter()
            .map(|unit| unit.text.trim())
            .filter(|text| !text.is_empty())
            .collect();
        let mut combined = if !unit_texts.is_empty() {
            unit_texts.join("\n\n")
        } else {
            document
                .page_texts()
                .into_iter()
                .map(|page| page.trim().to_owned())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.into_bytes()
    };

    if let Some(path) = out_path {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|source| Error::io(parent, source))?;
        }
        std::fs::write(&path, &payload).map_err(|source| Error::io(&path, source))?;
    } else {
        let stdout = std::io::stdout();
        let mut handle = BufWriter::new(stdout.lock());
        handle
            .write_all(&payload)
            .map_err(|source| Error::io("<stdout>", source))?;
        handle.flush().map_err(|source| Error::io("<stdout>", source))?;
    }
    Ok(0)
}

fn run() -> Result<i32> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    match command.as_str() {
        "extract-text" => extract_text_command(rest),
        "_pdf-inspector-gate" => pdf_inspector_gate_command(rest),
        "--version" | "-V" => {
            println!("legalpdf {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "--help" | "-h" => {
            println!("{}", usage());
            Ok(0)
        }
        _ => Err(Error::Message(usage().to_owned())),
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("legalpdf: {error}");
            std::process::exit(1);
        }
    }
}
