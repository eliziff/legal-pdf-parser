use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
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
    let digest = format!("{:x}", digest.finalize());
    cache.insert(path, digest.clone());
    Ok(digest)
}
