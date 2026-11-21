//! Arrays written as safetensors (the format's header: an 8-byte little-endian length, then a JSON object naming each
//! tensor's dtype, shape and byte range, then the bytes), so the fit maps a document's arrays and parses nothing.

use std::path::Path;

/// One array: its name, dtype ("F32", "I64", "I32", "I8", "U8", "BOOL"), shape and little-endian bytes.
#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Tensor {
    pub fn f32(name: &str, shape: Vec<usize>, values: &[f32]) -> Self {
        Self {
            name: name.to_owned(),
            dtype: "F32",
            shape,
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub fn i64(name: &str, shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Self {
        Self {
            name: name.to_owned(),
            dtype: "I64",
            shape,
            bytes: values.into_iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// A vector of integers, one dimension, each in 32 bits: every index and count the fit reads fits them.
    pub fn ints<T: Copy + Into<i64>>(name: &str, values: &[T]) -> Self {
        Self::i32(name, vec![values.len()], values.iter().map(|&v| v.into()))
    }

    /// Integers in 32 bits where every one fits them, else in 64: the narrowest that holds them all.
    pub fn i32(name: &str, shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Self {
        let values: Vec<i64> = values.into_iter().collect();
        if values.iter().all(|&v| i32::try_from(v).is_ok()) {
            Self {
                name: name.to_owned(),
                dtype: "I32",
                shape,
                bytes: values
                    .iter()
                    .flat_map(|&v| (v as i32).to_le_bytes())
                    .collect(),
            }
        } else {
            Self::i64(name, shape, values)
        }
    }

    pub fn i8(name: &str, shape: Vec<usize>, values: &[i8]) -> Self {
        Self {
            name: name.to_owned(),
            dtype: "I8",
            shape,
            bytes: values.iter().map(|&v| v as u8).collect(),
        }
    }

    pub fn flags(name: &str, values: &[bool]) -> Self {
        Self {
            name: name.to_owned(),
            dtype: "BOOL",
            shape: vec![values.len()],
            bytes: values.iter().map(|&v| u8::from(v)).collect(),
        }
    }

    pub fn bytes(name: &str, shape: Vec<usize>, values: &[u8]) -> Self {
        Self {
            name: name.to_owned(),
            dtype: "U8",
            shape,
            bytes: values.to_vec(),
        }
    }

    /// An array of any dtype from its bytes (little-endian, row-major).
    pub fn raw(name: &str, dtype: &'static str, shape: Vec<usize>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.to_owned(),
            dtype,
            shape,
            bytes,
        }
    }

    /// Its values as 32-bit floats (an F32 array's own).
    pub fn f32s(&self) -> Vec<f32> {
        match self.dtype {
            "F32" => self
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                .collect(),
            _ => self.values().into_iter().map(|v| v as f32).collect(),
        }
    }

    /// Its values as 64-bit integers (an I64 array's own).
    pub fn i64s(&self) -> Vec<i64> {
        match self.dtype {
            "I64" => self
                .bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes")))
                .collect(),
            "I32" => self
                .bytes
                .chunks_exact(4)
                .map(|b| i64::from(i32::from_le_bytes(b.try_into().expect("four bytes"))))
                .collect(),
            _ => self.values().into_iter().map(|v| v as i64).collect(),
        }
    }

    /// Its values as flags (non-zero true).
    pub fn bools(&self) -> Vec<bool> {
        match self.dtype {
            "U8" | "BOOL" => self.bytes.iter().map(|&b| b != 0).collect(),
            _ => self.values().into_iter().map(|v| v != 0.0).collect(),
        }
    }

    /// Its values as numbers, whatever its dtype.
    pub fn values(&self) -> Vec<f64> {
        match self.dtype {
            "F32" => self
                .bytes
                .chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes(b.try_into().expect("four bytes"))))
                .collect(),
            "I64" => self
                .bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes")) as f64)
                .collect(),
            "I32" => self
                .bytes
                .chunks_exact(4)
                .map(|b| f64::from(i32::from_le_bytes(b.try_into().expect("four bytes"))))
                .collect(),
            "I8" => self.bytes.iter().map(|&b| f64::from(b as i8)).collect(),
            _ => self.bytes.iter().map(|&b| f64::from(b)).collect(),
        }
    }
}

/// The arrays of a safetensors file `write` wrote, by name.
pub fn read(path: &Path) -> std::io::Result<std::collections::HashMap<String, Tensor>> {
    read_with_metadata(path).map(|(arrays, _)| arrays)
}

/// The arrays of a safetensors file `write` wrote, by name, and the metadata (string pairs) of its header.
#[allow(clippy::type_complexity)]
pub fn read_with_metadata(
    path: &Path,
) -> std::io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    let bytes = std::fs::read(path)?;
    let bad = |what: &str| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {what}", path.display()),
        )
    };
    let length = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")) as usize)
        .ok_or_else(|| bad("no header"))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(
        bytes
            .get(8..8 + length)
            .ok_or_else(|| bad("short header"))?,
    )?;
    let data = &bytes[8 + length..];
    let metadata = header
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_owned())))
                .collect()
        })
        .unwrap_or_default();
    let dtype = |d: &str| {
        ["F32", "I64", "I32", "I8", "U8", "BOOL"]
            .into_iter()
            .find(|&k| k == d)
    };
    header
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, entry)| {
            let range = entry["data_offsets"]
                .as_array()
                .map(|r| {
                    r.iter()
                        .filter_map(serde_json::Value::as_u64)
                        .map(|v| v as usize)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let dtype = entry["dtype"]
                .as_str()
                .and_then(dtype)
                .ok_or_else(|| bad(&format!("{name}'s dtype")))?;
            let shape = entry["shape"]
                .as_array()
                .map(|s| {
                    s.iter()
                        .filter_map(serde_json::Value::as_u64)
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default();
            let bytes = data
                .get(*range.first().unwrap_or(&0)..*range.get(1).unwrap_or(&0))
                .ok_or_else(|| bad(&format!("{name}'s bytes")))?
                .to_vec();
            Ok((
                name.clone(),
                Tensor {
                    name,
                    dtype,
                    shape,
                    bytes,
                },
            ))
        })
        .collect::<std::io::Result<_>>()
        .map(|arrays| (arrays, metadata))
}

/// `tensors` written to `path` as one safetensors file, with `metadata` (string pairs) in its header.
pub fn write(path: &Path, tensors: &[Tensor], metadata: &[(&str, String)]) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut at = 0usize;
    for t in tensors {
        header.insert(t.name.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [at, at + t.bytes.len()]}));
        at += t.bytes.len();
    }
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".to_owned(),
            serde_json::Value::Object(
                metadata
                    .iter()
                    .map(|(k, v)| (k.to_string(), serde_json::Value::from(v.as_str())))
                    .collect(),
            ),
        );
    }
    let mut text = serde_json::to_vec(&serde_json::Value::Object(header))?;
    // The header padded with spaces to a multiple of 8, so the data starts aligned.
    while (8 + text.len()) % 8 != 0 {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + at);
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text);
    tensors.iter().for_each(|t| out.extend(&t.bytes));
    // Written whole beside it, then renamed into place: a reader following the dump never sees half a file.
    let partial = path.with_extension("partial");
    std::fs::write(&partial, out)?;
    std::fs::rename(&partial, path)
}
