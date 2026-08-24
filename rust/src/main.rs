#[cfg(feature = "fast-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legalpdf::{
    derive_pdf_document, digest_cached_extraction, query_pdf_document, Error, PdfLookupRequest,
    PdfRequest, PdfStructureLookup, Result,
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
    structure: &'a legal_structure::DocumentStructure,
    pages: Vec<PdfStructureLookup>,
}

#[derive(Serialize)]
struct PdfInspectorRow<'a> {
    path: &'a str,
    product_sha256: String,
    #[serde(flatten)]
    product: &'a PdfInspectorProduct<'a>,
}

struct GateCacheRoot(PathBuf);

impl GateCacheRoot {
    fn create() -> Result<Self> {
        let temporary = std::env::temp_dir();
        for attempt in 0..100 {
            let path = temporary.join(format!(
                "legalpdf-pdf-inspector-gate-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(Error::io(&path, source)),
            }
        }
        Err(Error::Message(
            "could not create an empty PDF Inspector gate cache root".to_owned(),
        ))
    }

    fn document(&self, index: usize) -> Result<PathBuf> {
        let path = self.0.join(index.to_string());
        std::fs::create_dir(&path).map_err(|source| Error::io(&path, source))?;
        Ok(path)
    }
}

impl Drop for GateCacheRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn usage() -> &'static str {
    "usage:\n  legalpdf --version"
}

fn parity_replay_batch_command(arguments: &[String]) -> Result<i32> {
    let [manifest] = arguments else {
        return Err(Error::Message(
            "_parity-replay-batch requires <manifest.json>".to_owned(),
        ));
    };
    let path = PathBuf::from(manifest);
    let bytes = std::fs::read(&path).map_err(|source| Error::io(&path, source))?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(Error::Message("replay manifest exceeds 1 MiB".to_owned()));
    }
    let jobs: Vec<(PathBuf, String)> = serde_json::from_slice(&bytes)?;
    for (index, batch) in jobs.chunks(25).enumerate() {
        for (input, source_name) in batch {
            let value = digest_cached_extraction(input, source_name.clone())?;
            println!("{}", serde_json::to_string(&value)?);
        }
        let completed = ((index + 1) * 25).min(jobs.len());
        eprintln!("replayed {completed}/{}", jobs.len());
    }
    Ok(0)
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

    let caches = GateCacheRoot::create()?;
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
        let source_sha256 = format!("{:x}", Sha256::digest(&bytes));
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
            "cache_dir": cache_dir,
            "expected_source_sha256": entry.sha256,
        }))?;
        let document = derive_pdf_document(&bytes, &request)?;
        if document.page_count() != entry.pages {
            return Err(Error::Message(format!(
                "PDF Inspector page count mismatch: {} expected {}, got {}",
                entry.path,
                entry.pages,
                document.page_count()
            )));
        }
        let mut pages = Vec::with_capacity(entry.pages);
        for page in 1..=entry.pages {
            pages.push(query_pdf_document(
                &document,
                &PdfLookupRequest::new("page", page.to_string()),
            ));
        }
        let product = PdfInspectorProduct {
            structure: document.structure(),
            pages,
        };
        let product_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&product)?));
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

fn run() -> Result<i32> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    match command.as_str() {
        "_parity-replay-batch" => parity_replay_batch_command(rest),
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
