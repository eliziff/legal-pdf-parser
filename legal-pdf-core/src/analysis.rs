//! The structure engine these parsers run. The program that links them supplies it, so
//! the parsers build against the structure model alone and an engine edit recompiles only
//! the engine and that program.

use legal_structure_model::StructureAnalysis;
use std::sync::OnceLock;

static ANALYSIS: OnceLock<&'static dyn StructureAnalysis> = OnceLock::new();

/// Supplies the structure engine. The first one supplied stays.
pub fn install_structure_analysis(analysis: &'static dyn StructureAnalysis) {
    let _ = ANALYSIS.set(analysis);
}

/// The supplied structure engine, or this checkout's when built with `structure-analysis`.
pub fn structure_analysis() -> &'static dyn StructureAnalysis {
    #[cfg(feature = "structure-analysis")]
    return *ANALYSIS.get_or_init(|| &legal_structure::STRUCTURE_ANALYSIS);
    #[cfg(not(feature = "structure-analysis"))]
    *ANALYSIS
        .get()
        .expect("the program must supply the structure engine with install_structure_analysis")
}
