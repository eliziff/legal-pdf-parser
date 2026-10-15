//! Source-preserving PDF reading normalization. Returns only witnessed deletions.
use crate::model::ScalarRange;
use std::collections::HashSet;

/// Identify split-word hyphens using whole-word witnesses elsewhere in the document.
/// Each deletion is in the corresponding input text's UTF-16 coordinates.
/// Consumers compose these edits into their own source runs; inserted/replaced text is never inferred.
pub fn split_word_deletions(texts: &[&str]) -> Vec<Vec<ScalarRange>> {
    let whole: HashSet<String> = texts.iter().flat_map(|text|
        text.split(|character: char| !character.is_alphabetic()))
        .filter(|word| !word.is_empty()).map(str::to_lowercase).collect();
    texts.iter().map(|text| {
        let characters = text.char_indices().collect::<Vec<_>>();
        let mut removed = Vec::new();
        let (mut at, mut utf16, mut counted_byte) = (0,0,0);
        while at < characters.len() {
            let (byte, character) = characters[at];
            if character != '-' || at == 0 || !characters[at-1].1.is_alphabetic() { at+=1;continue; }
            let first=characters[..at].iter().rev().take_while(|(_,c)|c.is_alphabetic()).count();
            let mut next=at+1;
            while next<characters.len() && matches!(characters[next].1,' '|'\n'|'\r') { next+=1; }
            let gap=&characters[at+1..next];
            let broken=gap.is_empty() || gap.iter().any(|(_,c)|*c=='\n');
            let second=characters[next..].iter().take_while(|(_,c)|c.is_alphabetic()).count();
            if !broken || second<2 || !characters[next].1.is_lowercase()
                || at>=first+1 && characters[at-first-1].1=='-' { at+=1;continue; }
            let start_byte=characters[at-first].0;
            let end_byte=characters.get(next+second).map_or(text.len(),|(byte,_)|*byte);
            let joined=text[start_byte..end_byte].chars().filter(|c|c.is_alphabetic()).collect::<String>().to_lowercase();
            if !whole.contains(&joined) { at+=1;continue; }
            utf16+=text[counted_byte..byte].encode_utf16().count();
            let end=characters[next].0;
            let length=text[byte..end].encode_utf16().count();
            removed.push(ScalarRange {start:utf16,end:utf16+length});
            utf16+=length;counted_byte=end;at=next;
        }
        removed
    }).collect()
}
