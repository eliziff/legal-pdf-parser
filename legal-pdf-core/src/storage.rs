use crate::{Error, Result};
use flate2::bufread::GzDecoder;
use flate2::{Compression, GzBuilder};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::ser::Formatter;
use serde_json::Value;
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
    Ok(path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed) + attempt
    )))
}

#[doc(hidden)]
pub fn atomic_write_with(
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
        io(&temporary, writer.get_ref().sync_all())?;
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

#[doc(hidden)]
pub fn write_json(path: &Path, value: &Value) -> Result<()> {
    atomic_write_with(path, |writer| {
        serde_json::to_writer_pretty(&mut *writer, value)?;
        writer
            .write_all(b"\n")
            .map_err(|source| Error::io(path, source))
    })
}

struct PythonFormatter;

#[doc(hidden)]
pub fn python_json(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonFormatter);
    value.serialize(&mut serializer)?;
    String::from_utf8(bytes).map_err(|error| Error::Message(format!("JSON is not UTF-8: {error}")))
}

impl Formatter for PythonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        writer.write_all(b": ")
    }
}

pub fn write_gzip_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    atomic_write_with(path, |writer| {
        let mut gzip = GzBuilder::new().mtime(0).write(writer, Compression::fast());
        serde_json::to_writer(&mut gzip, value)?;
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
    use serde_json::json;

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
