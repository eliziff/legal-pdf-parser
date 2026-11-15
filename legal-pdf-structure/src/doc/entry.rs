//! A document's dump entry: one file, `<sha>.dump` (safetensors), holding all the dump keeps of the document: the arrays
//! DOC reads (`inputs.<name>`, `inputs::Inputs`), the fit's targets (`targets.<name>`), the parse as tier 0 left it
//! (`parse`: its pages, outline and derivations, MessagePack), the gold's side as the grader reads it (`gold`, the
//! grader's), and its record in the file's own metadata (`record`, JSON: its lines and their places, reference marks and
//! note numbers, what it was read and labelled from). Every reader and writer of the dump goes through it.

use super::tensors::{self, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct Entry {
    tensors: HashMap<String, Tensor>,
    /// Its record: null where it has none.
    pub record: serde_json::Value,
}

impl Entry {
    pub fn path(folder: &Path, sha256: &str) -> PathBuf {
        folder.join(format!("{sha256}.dump"))
    }

    /// The entry kept for `sha256` in the dump at `folder`, where there is one.
    pub fn read(folder: &Path, sha256: &str) -> Option<Self> {
        let (tensors, metadata) = tensors::read_with_metadata(&Self::path(folder, sha256)).ok()?;
        let record = metadata.get("record").and_then(|record| serde_json::from_str(record).ok()).unwrap_or_default();
        Some(Self { tensors, record })
    }

    pub fn write(&self, folder: &Path, sha256: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(folder)?;
        let mut all: Vec<Tensor> = self.tensors.values().cloned().collect();
        all.sort_by(|a, b| a.name.cmp(&b.name));
        tensors::write(&Self::path(folder, sha256), &all, &[("record", self.record.to_string())])
    }

    /// A part's arrays (`inputs`, `targets`), by their names within it.
    pub fn part(&self, part: &str) -> HashMap<String, Tensor> {
        let prefix = format!("{part}.");
        self.tensors.iter().filter_map(|(name, tensor)| {
            let name = name.strip_prefix(&prefix)?.to_owned();
            Some((name.clone(), Tensor { name, ..tensor.clone() }))
        }).collect()
    }

    /// A part's arrays put in place of the ones it held.
    pub fn set_part(&mut self, part: &str, arrays: Vec<Tensor>) {
        let prefix = format!("{part}.");
        self.tensors.retain(|name, _| !name.starts_with(&prefix));
        for tensor in arrays {
            let name = format!("{prefix}{}", tensor.name);
            self.tensors.insert(name.clone(), Tensor { name, ..tensor });
        }
    }

    pub fn has_part(&self, part: &str) -> bool {
        let prefix = format!("{part}.");
        self.tensors.keys().any(|name| name.starts_with(&prefix))
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|tensor| &tensor.bytes[..])
    }

    pub fn set_bytes(&mut self, name: &str, bytes: &[u8]) {
        self.tensors.insert(name.to_owned(), Tensor::bytes(name, vec![bytes.len()], bytes));
    }
}
