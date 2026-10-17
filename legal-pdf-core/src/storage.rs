use crate::{Error, Result};
use flate2::bufread::GzDecoder;
use flate2::{Compression, GzBuilder};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T> {
    result.map_err(|source| Error::io(path, source))
}

fn temporary_path(path: &Path, attempt: u64) -> Result<PathBuf> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| Error::Message(format!("unsafe output path: {}", path.display())))?;
    #[cfg(not(target_arch = "wasm32"))]
    let process_id = std::process::id();
    // Browser WASI has no process IDs; create_new and the sequence distinguish files.
    #[cfg(target_arch = "wasm32")]
    let process_id = 0;
    Ok(path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        process_id,
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed) + attempt
    )))
}

fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        io(parent, fs::create_dir_all(parent))?;
    }
    let mut chosen = None;
    for attempt in 0..32 {
        let candidate = temporary_path(path, attempt)?;
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                chosen = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => return Err(Error::io(candidate, source)),
        }
    }
    let (temporary, file) = chosen.ok_or_else(|| {
        Error::Message(format!(
            "could not allocate a temporary for {}",
            path.display()
        ))
    })?;
    let result = (|| {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        io(&temporary, writer.flush())?;
        drop(writer);
        if path.exists() {
            io(path, fs::remove_file(path))?;
        }
        io(path, fs::rename(&temporary, path))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn write_gzip_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    atomic_write_with(path, |writer| {
        let mut gzip = GzBuilder::new().mtime(0).write(writer, Compression::fast());
        // Batch JSON's tiny token writes before feeding the compressor.
        let mut buffered = BufWriter::new(&mut gzip);
        serde_json::to_writer(&mut buffered, value)?;
        buffered.flush().map_err(|source| Error::io(path, source))?;
        drop(buffered);
        gzip.finish().map_err(|source| Error::io(path, source))?;
        Ok(())
    })
}

pub fn read_gzip_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = io(path, File::open(path))?;
    serde_json::from_reader(BufReader::new(GzDecoder::new(BufReader::new(file))))
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn compressed_cache_is_one_round_trippable_file() {
        let path = std::env::temp_dir().join(format!(
            "legalpdf-cache-test-{}-{}.json.gz",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let value = json!({"hello": ["world"]});
        write_gzip_json(&path, &value).unwrap();
        assert_eq!(read_gzip_json::<Value>(&path).unwrap(), value);
        fs::remove_file(path).unwrap();
    }
}
