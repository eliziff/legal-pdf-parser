mod adapters;
mod lookup;
pub mod pairing_support;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_openvino;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_postprocess;
pub mod profile;
mod projection;

pub use adapters::{to_alr_payload, to_toa_text_units};
pub use lookup::lookup_footnote;
pub use pairing_support::{
    enumerator_interpretations, has_citation_signal, has_legal_citation_cue,
    heading_text_plausible, is_legal_citation_continuation, parse_heading_ladder,
    protected_citation_spans,
};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
pub use ppdoc::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions};
pub use projection::{numeric_range, parse_ordinal, source_doc, structure_lookup};
