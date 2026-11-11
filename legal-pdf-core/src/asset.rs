use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Hashes an immutable provider asset once per process.
pub fn provider_asset_sha256(path: &Path) -> Result<String> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, String>>> = OnceLock::new();

    let path = fs::canonicalize(path).map_err(|source| Error::io(path, source))?;
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(digest) = cache.get(&path) {
        return Ok(digest.clone());
    }

    let mut digest = Sha256::new();
    let mut reader = BufReader::new(File::open(&path).map_err(|source| Error::io(&path, source))?);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|source| Error::io(&path, source))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let digest = digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    cache.insert(path, digest.clone());
    Ok(digest)
}

/// A provider's path, given explicitly or else by the environment variable `variable`; an error naming `label` where
/// neither gives one.
pub fn required_path(value: &Option<PathBuf>, variable: &str, label: &str) -> Result<PathBuf> {
    value
        .clone()
        .or_else(|| std::env::var_os(variable).map(PathBuf::from))
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| Error::Message(format!("{label} is required; pass it explicitly or set {variable}")))
}

/// An existing file's canonical path; an error naming `label` where there is none.
pub fn canonical_file(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::Message(format!("{label} does not exist: {}", path.display())));
    }
    fs::canonicalize(path).map_err(|source| Error::io(path, source))
}

/// A model pack's file, named by its path relative to the pack's canonical `root`: a plain relative path (no drive,
/// root, `..` or backslash) to an existing file inside the pack.
pub fn pack_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf> {
    let plain = !relative.contains(['\\', ':'])
        && Path::new(relative).components().all(|part| matches!(part, Component::Normal(_) | Component::CurDir));
    let path = if plain { canonical_file(&root.join(relative), label)? } else { PathBuf::new() };
    if !plain || !path.starts_with(root) {
        return Err(Error::Message(format!("{label} path must stay inside its model pack: {relative}")));
    }
    Ok(path)
}
