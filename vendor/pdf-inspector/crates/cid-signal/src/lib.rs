use lopdf::Object;
use std::collections::HashSet;

const MAX_CID_W_EXPANSION: usize = 65_536;

/// Check if a CIDFont's width keys look like Unicode code points rather than
/// low-value glyph IDs.
pub fn cid_values_look_like_unicode(cid_font_dict: &lopdf::Dictionary) -> bool {
    let w_arr = match cid_font_dict.get(b"W").ok() {
        Some(Object::Array(arr)) => arr,
        _ => return false,
    };
    // Repeated full-width ranges must not grow temporary work per copy.
    let mut seen = HashSet::new();
    let mut i = 0;
    while i < w_arr.len() && seen.len() < MAX_CID_W_EXPANSION {
        let Ok(cid) = w_arr[i].as_i64() else {
            i += 1;
            continue;
        };
        let start = cid as u16;
        if i + 1 >= w_arr.len() {
            seen.insert(start);
            i += 1;
        } else if let Object::Array(widths) = &w_arr[i + 1] {
            for j in 0..widths.len() {
                if seen.len() >= MAX_CID_W_EXPANSION {
                    break;
                }
                seen.insert(start.wrapping_add(j as u16));
            }
            i += 2;
        } else if i + 2 < w_arr.len() {
            if let Ok(end) = w_arr[i + 1].as_i64() {
                let end = end as u16;
                if start <= end {
                    for cid in start..=end {
                        if seen.len() >= MAX_CID_W_EXPANSION {
                            break;
                        }
                        seen.insert(cid);
                    }
                }
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    if seen.is_empty() {
        return false;
    }
    let mut cids: Vec<_> = seen.into_iter().collect();
    cids.sort_unstable();
    cids[cids.len() / 2] >= 0x41
}
