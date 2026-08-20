pub mod types {
    pub use pdf_inspector_core::types::*;
}

pub mod tables {
    pub use pdf_inspector_table_kernel::{Table, TableKind};

    use crate::types::TextItem;

    pub(crate) fn is_text_layout_item(item: &TextItem) -> bool {
        !matches!(item.item_type, crate::types::ItemType::Image)
    }

    mod detect_rects;

    pub use detect_rects::{detect_chart_regions, detect_tables_from_rects, RectHintRegion};
}

pub use tables::*;
