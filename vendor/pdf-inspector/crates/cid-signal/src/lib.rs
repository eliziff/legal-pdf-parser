use lopdf::Object;

/// Check if a CIDFont's width keys look like Unicode code points rather than
/// low-value glyph IDs.
pub fn cid_values_look_like_unicode(cid_font_dict: &lopdf::Dictionary) -> bool {
    let w_arr = match cid_font_dict.get(b"W").ok() {
        Some(Object::Array(arr)) => arr,
        _ => return false,
    };
    let mut cids = Vec::<u16>::new();
    let mut i = 0;
    while i < w_arr.len() {
        let Ok(cid) = w_arr[i].as_i64() else {
            i += 1;
            continue;
        };
        cids.push(cid as u16);
        if i + 1 >= w_arr.len() {
            i += 1;
        } else if let Object::Array(widths) = &w_arr[i + 1] {
            cids.extend((1..widths.len()).map(|j| (cid as u16).wrapping_add(j as u16)));
            i += 2;
        } else if i + 2 < w_arr.len() {
            if let Ok(end) = w_arr[i + 1].as_i64() {
                cids.extend((cid as u16)..=(end as u16));
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    if cids.is_empty() {
        return false;
    }
    cids.sort_unstable();
    cids[cids.len() / 2] >= 0x41
}
