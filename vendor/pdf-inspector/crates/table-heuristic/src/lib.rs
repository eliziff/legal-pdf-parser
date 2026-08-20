pub mod text_utils {
    pub use pdf_inspector_core::text_utils::*;
}

pub mod types {
    pub use pdf_inspector_core::types::*;
}

pub mod tables {
    pub use pdf_inspector_table_kernel::{Table, TableDetectionMode, TableKind};

    mod detect_heuristic;
    mod financial;
    mod grid;

    pub use detect_heuristic::{detect_tables, is_table_of_contents};
}

pub use tables::*;
