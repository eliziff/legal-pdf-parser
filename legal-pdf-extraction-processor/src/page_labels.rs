use lopdf::{decode_text_string, Document, Object};
use std::collections::{BTreeMap, HashSet};

/// Read optional labels from the document already opened for extraction.
pub fn embedded_page_labels(document: &Document, count: usize) -> Vec<Option<String>> {
    fn read(document: &Document, count: usize) -> Option<Vec<Option<String>>> {
        let mut pending = vec![document.catalog().ok()?.get(b"PageLabels").ok()?];
        let mut seen = HashSet::new();
        let mut rules = BTreeMap::new();
        let mut visited = 0;
        while let Some(object) = pending.pop() {
            visited += 1;
            if visited > 10_000 {
                return None;
            }
            if let Object::Reference(id) = object {
                if !seen.insert(*id) {
                    return None;
                }
            }
            let node = document.dereference(object).ok()?.1.as_dict().ok()?;
            if let Ok(nums) = node.get(b"Nums") {
                let nums = document.dereference(nums).ok()?.1.as_array().ok()?;
                if nums.len() % 2 != 0 {
                    return None;
                }
                for pair in nums.chunks_exact(2) {
                    let index = usize::try_from(pair[0].as_i64().ok()?).ok()?;
                    let spec = document.dereference(&pair[1]).ok()?.1.as_dict().ok()?;
                    if rules.insert(index, spec).is_some() {
                        return None;
                    }
                }
            }
            if let Ok(kids) = node.get(b"Kids") {
                pending.extend(document.dereference(kids).ok()?.1.as_array().ok()?);
            }
        }
        let mut labels = vec![None; count];
        for (index, slot) in labels.iter_mut().enumerate() {
            let Some((&first, spec)) = rules.range(..=index).next_back() else {
                continue;
            };
            let prefix = spec
                .get(b"P")
                .ok()
                .map(decode_text_string)
                .transpose()
                .ok()?
                .unwrap_or_default();
            let start = spec
                .get(b"St")
                .ok()
                .map(Object::as_i64)
                .transpose()
                .ok()?
                .unwrap_or(1);
            let number = start.checked_add((index - first) as i64)?;
            if !(1..=100_000).contains(&number) {
                return None;
            }
            let style = spec
                .get(b"S")
                .ok()
                .map(Object::as_name)
                .transpose()
                .ok()?
                .unwrap_or(b"");
            let mut suffix = match style {
                b"D" => number.to_string(),
                b"R" | b"r" => {
                    let mut remainder = number;
                    let mut text = String::new();
                    for (value, numeral) in [
                        (1000, "M"),
                        (900, "CM"),
                        (500, "D"),
                        (400, "CD"),
                        (100, "C"),
                        (90, "XC"),
                        (50, "L"),
                        (40, "XL"),
                        (10, "X"),
                        (9, "IX"),
                        (5, "V"),
                        (4, "IV"),
                        (1, "I"),
                    ] {
                        while remainder >= value {
                            text.push_str(numeral);
                            remainder -= value;
                        }
                    }
                    text
                }
                b"A" | b"a" => char::from(b'A' + ((number - 1) % 26) as u8)
                    .to_string()
                    .repeat(((number - 1) / 26 + 1) as usize),
                b"" => String::new(),
                _ => return None,
            };
            if style == b"r" || style == b"a" {
                suffix.make_ascii_lowercase();
            }
            let value = prefix + &suffix;
            if !value.is_empty() {
                *slot = Some(value);
            }
        }
        Some(labels)
    }
    read(document, count).unwrap_or_else(|| vec![None; count])
}
