pub mod text_utils {
    pub use pdf_inspector_core::text_utils::*;
}

pub mod types {
    pub use pdf_inspector_core::types::*;
}

pub mod extractor {
    use crate::types::TextItem;

    pub(crate) fn trace_text_preview(text: &str, max_chars: usize) -> &str {
        match text.char_indices().nth(max_chars) {
            Some((index, _)) => &text[..index],
            None => text,
        }
    }

    pub(crate) fn is_text_layout_item(item: &TextItem) -> bool {
        !matches!(item.item_type, crate::types::ItemType::Image)
    }

    mod layout;
    mod reading_order;

    pub use layout::*;
}

pub use extractor::*;
