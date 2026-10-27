//! Each page's lines classified: furniture, tables and their cells, contents lists, captions and
//! the note band, per document and language.

use super::*;

/// Source labels survive ordering and classification. Unknown labels carry no evidence;
/// they must not erase the known labels of other lines. Ambiguous identities invalidate the map.
pub(super) fn source_region_contract(pages: &[Page]) -> Option<HashMap<String, String>> {
    let mut regions = HashMap::new();
    let mut ids = HashSet::new();
    for line in pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
    {
        if line.id.is_empty() || !ids.insert(line.id.as_str()) {
            return None;
        }
        let source = line.region_type.trim();
        let tag = matches!(source, "text" | "body" | "" | "unknown" | "unknown_region")
            .then(|| {
                legal_pdf_core::tagged_heading(line, |span| span.structure_role.as_deref()).or_else(
                    || {
                        legal_pdf_core::tagged_line_role(line, |span| {
                            span.structure_role.as_deref().filter(|role| *role == "LI")
                        })
                    },
                )
            })
            .flatten();
        let region = match tag {
            Some(tag) => tag.to_owned(),
            None if legal_pdf_core::is_heading_role(source) || source == "LI" => source.to_owned(),
            None => source.to_ascii_lowercase(),
        };
        if !matches!(region.as_str(), "" | "unknown" | "unknown_region") {
            regions.insert(line.id.clone(), region);
        }
    }
    (!regions.is_empty()).then_some(regions)
}

pub(super) fn source_regions_cover_pages(pages: &[Page], regions: Option<&HashMap<String, String>>) -> bool {
    regions.is_some_and(|regions| {
        pages
            .iter()
            .flat_map(|page| &page.lines)
            .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
            .all(|line| regions.contains_key(&line.id))
    })
}

// A contents entry runs its title into a dot leader ("4. Land Use ........ 12"), as pdf-inspector reads one.

pub(super) fn classify_pages_with_source(
    pages: &mut [Page],
    separators: &[Option<f64>],
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    // Discover openings with the existing furniture/reference filters before calibrating each document.
    let source = pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| (line.region_type.clone(), line.exclude_from_body))
        .collect::<Vec<_>>();
    legal_pdf_support::profile::measure("prepare.furniture", || mark_repeated_furniture(pages));
    constituents::reconcile_starts(pages, &mut evidence.contexts.constituent_starts);
    let documents = constituents::document_lines(pages, &evidence.contexts.constituent_starts);
    let diagnostics = if documents.is_empty() {
        drop(source);
        classify_document_pages(pages, separators, &HashSet::new(), evidence)
    } else {
        for (line, (region, excluded)) in pages.iter_mut().flat_map(|page| &mut page.lines).zip(source) {
            line.region_type = region;
            line.exclude_from_body = excluded;
        }
        let separators_by_page = pages
            .iter()
            .zip(separators)
            .map(|(page, separator)| (page.index, *separator))
            .collect::<HashMap<_, _>>();
        let shared_pages = pages
            .iter()
            .filter(|page| {
                let mut contexts = page
                    .lines
                    .iter()
                    .map(|line| documents.get(&line.id).copied().unwrap_or(0));
                contexts
                    .next()
                    .is_some_and(|first| contexts.any(|context| context != first))
            })
            .map(|page| page.index)
            .collect::<HashSet<_>>();
        let mut diagnostics = Vec::new();
        constituents::within_document_contexts(pages, &documents, &HashSet::new(), |document| {
            let separators = document
                .iter()
                .map(|page| separators_by_page[&page.index])
                .collect::<Vec<_>>();
            legal_pdf_support::profile::measure("prepare.furniture", || {
                mark_repeated_furniture(document)
            });
            diagnostics.extend(classify_document_pages(
                document,
                &separators,
                &shared_pages,
                evidence,
            ));
        });
        evidence
            .claims.contents_pages
            .retain(|page| !shared_pages.contains(page));
        diagnostics
    };
    evidence.contexts.document_line_ids = documents;
    legal_pdf_support::profile::measure("classify.paragraphs", || {
        constituents::within_document_contexts(
            pages,
            &evidence.contexts.document_line_ids,
            &evidence.contexts.translation_line_ids,
            |document| {
                crate::paragraphs::segment_paragraphs(document);
                contents::group_rows(document, &evidence.claims);
            },
        );
        legal_pdf_tables::part_cell_paragraphs(pages, &evidence.claims.tables);
    });
    diagnostics
}

pub(super) fn classify_document_pages(
    pages: &mut [Page],
    separators: &[Option<f64>],
    shared_pages: &HashSet<usize>,
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut contents_line_ids = HashSet::new();
    let translation =
        legal_pdf_support::profile::measure("classify.translation", || translation_lines(pages));
    // A table's rulings underline none of its words.
    for page in pages.iter_mut() {
        legal_pdf_tables::clear_table_underlines(page);
    }
    let (tables, listed) = legal_pdf_support::profile::measure("classify.tables", || {
        legal_pdf_tables::document_tables(pages, &translation, &TableReadings)
    });
    // A contents list pdf-inspector read as a table is contents, beside the rows read below.
    contents_line_ids.extend(listed);
    // The document's own lists, read by their typography before any table keeps their lines: a
    // table that is a contents list or a word index is that list.
    let lists = legal_pdf_support::profile::measure("classify.contents_lists", || {
        contents_lists::read(pages, &legal_pdf_tables::table_line_ids(&tables), &evidence.embedded_page_labels)
    });
    let tables: Vec<_> = tables
        .into_iter()
        .filter(|table| {
            let ids = table.line_ids();
            let listed = ids.iter().filter(|id| lists.contents.contains(id.as_str())).count();
            if std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some() {
                eprintln!("contents\ttable of {} lines, {listed} listed: {}", ids.len(), if listed * 2 <= ids.len() { "kept" } else { "dropped" });
            }
            listed * 2 <= ids.len()
        })
        .collect();
    let table_lines = legal_pdf_tables::table_line_ids(&tables);
    contents_line_ids.extend(lists.contents);
    let mut contents_pages = lists.pages;
    evidence.claims.contents_entries.extend(lists.entries);
    // A row the extractor read across both columns parts into its two languages.
    for page in pages.iter_mut() {
        let base = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0)
            + 1;
        for line in page
            .lines
            .iter_mut()
            .filter(|line| translation.contains(&line.id))
        {
            line.block_index += base;
        }
    }
    // A page holding a table's cells.
    let table_pages: Vec<bool> = pages
        .iter()
        .map(|page| page.lines.iter().any(|line| table_lines.contains(&line.id)))
        .collect();
    evidence.claims.table_cell_line_ids.extend(
        pages.iter().flat_map(|page| &page.lines).filter(|line| table_lines.contains(&line.id)).map(|line| line.id.clone()),
    );

    // Read note history within one language; tables and physical order retain the whole page.
    if translation.is_empty() {
        diagnostics.extend(classify_line_roles(
            pages,
            separators,
            &contents_pages,
            evidence,
        ));
    } else {
        let separators_by_page = pages
            .iter()
            .zip(separators)
            .map(|(page, separator)| (page.index, *separator))
            .collect::<HashMap<_, _>>();
        constituents::within_document_contexts(pages, &HashMap::new(), &translation, |language| {
            let separators = language
                .iter()
                .map(|page| separators_by_page[&page.index])
                .collect::<Vec<_>>();
            diagnostics.extend(classify_line_roles(
                language,
                &separators,
                &contents_pages,
                evidence,
            ));
        });
    }
    legal_pdf_support::profile::measure("classify.order_pages", || {
        for (page, table_page) in pages.iter_mut().zip(table_pages) {
            diagnostics.extend(order_page(page, table_page, contents_pages.contains(&page.index)));
            legal_pdf_tables::order_table_lines(page, &tables);
        }
    });
    evidence.claims.tables.extend(tables);
    // Keep page-wide contents evidence precise when two documents share a physical page.
    contents_line_ids.extend(
        pages
            .iter()
            .filter(|page| {
                shared_pages.contains(&page.index) && contents_pages.contains(&page.index)
            })
            .flat_map(|page| page.lines.iter().map(|line| line.id.clone())),
    );
    // A table of authorities is no contents list, whatever its leaders and locators.
    let authority_pages: HashSet<usize> = pages
        .iter()
        .filter(|page| page.lines.iter().any(|line| lists.authorities.contains(&line.id)))
        .map(|page| page.index)
        .collect();
    contents_line_ids.retain(|id| !lists.authorities.contains(id));
    contents_pages.retain(|page| !authority_pages.contains(page));
    evidence.claims.contents_line_ids.extend(contents_line_ids);
    evidence.claims.contents_pages.extend(contents_pages);
    legal_pdf_support::profile::measure("classify.captions", || {
        constituents::within_document_contexts(pages, &HashMap::new(), &translation, |document| {
            captions::classify(document, &evidence.claims, &mut evidence.headings);
        });
    });
    evidence.contexts.translation_line_ids.extend(translation);
    diagnostics
}
