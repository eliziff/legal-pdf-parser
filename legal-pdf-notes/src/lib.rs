//! A PDF's notes as the structure stage reads them: the note band and the lanes that take lines for notes (`notes`),
//! each note's propositions and cross-references (`note_context`), and the engine's pairer over the written pages
//! (`note_pairing`).

pub mod note_context;
pub mod note_pairing;
pub mod notes;
