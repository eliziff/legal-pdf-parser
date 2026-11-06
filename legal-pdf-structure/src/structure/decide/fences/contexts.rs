//! Each line's reading context.
use super::*;

impl Fences {
    /// Each line's reading context: its document, its language and the quotation it stands in.
    pub(super) fn read_contexts(&mut self) {
        let mut contexts: Vec<Context> = Vec::new();
        self.ctx = (0..self.document.len())
            .map(|slot| {
                let context = Context { document: self.document[slot], translation: self.translation[slot], quotation: self.quotation[slot] };
                contexts.iter().position(|known| *known == context).unwrap_or_else(|| {
                    contexts.push(context);
                    contexts.len() - 1
                }) as u16
            })
            .collect();
        self.contexts = contexts;
    }
}
