//! Every heading of a document, read once after the containers have claimed their lines (line
//! labels, furniture, tables, contents, captions, notes) and quotations are read. The readers run
//! in this order, each correcting the last:
//!
//! 1. `native`: each page's own readers, per reading context outside quotations (a source's heading
//!    tags, the numbering ladder's headings and their wrapped titles, title-case lines on
//!    recognized pages).
//! 2. `bookmarks` and `contents`: the document's bookmarks and printed contents list, merged into
//!    one outline and reconciled with the lines they name.
//! 3. `rules` (a composed decision, after the readers' state is cached): the whole-document rules
//!    that demote what is no heading, promote a transcript's parts, lift lines set apart, and level
//!    every heading.
//!
//! The structure tagger (the engine's fast ML tier) reads these as its tier 0 and re-labels the
//! regions; the typography reader that stood here (a PageIndex Flash port) read no better under it.
//!
//! Each line's heading trail records every pass and rule that made it a heading or took that away
//! (`mark`, `stamp`): a parse's rows carry it, so each one's catches can be counted on the gold.

use super::*;

pub(crate) mod bookmarks;
pub(crate) mod contents;
mod native;
pub(crate) use native::*;
pub(crate) mod rules;

/// Whether a trail leaves its line a heading: its last mark is a "+".
pub(crate) fn heads(trail: &str) -> bool {
    trail.rfind(['+', '-']).is_some_and(|at| trail[at..].starts_with('+'))
}

/// A line made a heading (`heading`) or taken from the headings by `why`.
pub(crate) fn mark(trail: &mut HashMap<String, String>, id: &str, heading: bool, why: &str) {
    let marks = trail.entry(id.to_owned()).or_default();
    marks.push(if heading { '+' } else { '-' });
    marks.push_str(why);
}

/// After a pass: every line it made a heading, or took from them, that no rule marked.
pub(crate) fn stamp(pages: &[Page], trail: &mut HashMap<String, String>, pass: &str) {
    for line in pages.iter().flat_map(|page| &page.lines) {
        let now = line.region_type == "heading";
        if now != trail.get(&line.id).is_some_and(|marks| heads(marks)) {
            mark(trail, &line.id, now, pass);
        }
    }
}

/// Every heading of the document, read once, after every container has claimed its lines (line
/// labels, furniture, tables and their notes, contents, captions, notes) and quotations are read:
/// in each reading context, the native readers (a source's own tags, numbering ladders, capitals,
/// type size) except inside a quotation, then the typography of the whole document (PageIndex
/// Flash) with every heading's depth, then the document's bookmarks or printed contents list.
pub(crate) fn read_headings(pages: &mut [Page], outline: &[PdfOutlineEntry], primitives: &mut PdfPrimitiveEvidence) {
    stamp(pages, &mut primitives.headings.trail, "layout");
    legal_pdf_support::profile::measure("prepare.native_headings", || {
        let reading = primitives.contexts.reading_lines().clone();
        let translation = primitives.contexts.translation_line_ids.clone();
        let mut levels = HashMap::new();
        constituents::within_document_contexts(pages, &reading, &translation, |document| {
            // A quotation's titles and labels structure only it: the typography reader below
            // reads them.
            if document
                .iter()
                .flat_map(|page| &page.lines)
                .any(|line| primitives.quotations.scopes.contains_key(&line.id))
            {
                return;
            }
            // LEGALPDF_ABLATE=native leaves the native readers out, for their measure under the tagger.
            if legal_structure_model::ablated("native") {
                return;
            }
            demote_prose_heading_tags(document, primitives);
            levels.extend(apply_text_fidelity_headings(document, primitives));
        });
        primitives.headings.heading_levels = levels;
    });
    stamp(pages, &mut primitives.headings.trail, "native");
    legal_pdf_support::profile::measure("prepare.bookmarks", || {
        // The document's own bookmarks and its printed contents list, read together.
        let authored = bookmarks::usable(outline, pages.len());
        let listed = contents::contents_outline(pages, &primitives.claims);
        let merged = bookmarks::merge_outlines(if authored { outline } else { &[] }, &listed);
        bookmarks::reconcile_bookmarks(pages, &merged, primitives, bookmarks::complete(outline, pages.len()));
    });
    stamp(pages, &mut primitives.headings.trail, "outline");
}
