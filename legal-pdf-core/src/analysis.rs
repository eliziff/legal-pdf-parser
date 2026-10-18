//! The structure engine's services on laid-out text, which these parsers run. The program that
//! links them supplies the engine, so the parsers build against the layout crate alone and an
//! engine or document-model edit recompiles none of them.

use legal_layout::LayoutAnalysis;
use std::sync::OnceLock;

/// This checkout's structure engine, for the command-line tools and tests built with `structure-analysis`.
#[cfg(feature = "structure-analysis")]
pub static CHECKOUT_ENGINE: &legal_structure::EngineAnalysis = &legal_structure::STRUCTURE_ANALYSIS;

static ANALYSIS: OnceLock<&'static dyn LayoutAnalysis> = OnceLock::new();

/// Supplies the structure engine's services on laid-out text. The first one supplied stays.
pub fn install_layout_analysis(analysis: &'static dyn LayoutAnalysis) {
    let _ = ANALYSIS.set(analysis);
}

/// The supplied engine, or this checkout's when built with `structure-analysis`.
pub fn layout_analysis() -> &'static dyn LayoutAnalysis {
    #[cfg(feature = "structure-analysis")]
    return *ANALYSIS.get_or_init(|| &legal_structure::STRUCTURE_ANALYSIS);
    #[cfg(not(feature = "structure-analysis"))]
    *ANALYSIS
        .get()
        .expect("the program must supply the structure engine with install_layout_analysis")
}
