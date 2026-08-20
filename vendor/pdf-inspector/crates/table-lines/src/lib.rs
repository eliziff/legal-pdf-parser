pub mod types {
    pub use pdf_inspector_core::types::*;
}

pub mod tables {
    pub use pdf_inspector_table_kernel::{Table, TableKind};

    use crate::types::TextItem;

    pub(crate) fn is_text_layout_item(item: &TextItem) -> bool {
        !matches!(item.item_type, crate::types::ItemType::Image)
    }

    mod detect_lines;

    pub use detect_lines::detect_tables_from_lines;
}

pub use tables::*;
