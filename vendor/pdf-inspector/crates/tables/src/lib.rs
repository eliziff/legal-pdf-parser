pub use pdf_inspector_core::{text_utils, types};

pub mod tables {
    pub use pdf_inspector_table_heuristic::{detect_tables, is_table_of_contents};
    pub use pdf_inspector_table_kernel::{Table, TableKind};
    pub use pdf_inspector_table_lines::detect_tables_from_lines;
    pub use pdf_inspector_table_rects::{
        detect_chart_regions, detect_tables_from_rects, RectHintRegion,
    };
}

pub use tables::*;
