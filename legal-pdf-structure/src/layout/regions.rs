//! Regions, a view of the lines' final roles, built once every reader has run.

use super::*;

/// Each page's regions: its lines in order, a region where their kind and extraction block run on.
/// On a transcript's page (one with line numbers) a line number stands apart and parts no region,
/// and its body reads by turns: a turn runs from the line that opens it to the next that opens one.
pub(crate) fn build_regions(pages: &mut [Page]) {
    for page in pages {
        let transcript = page.lines.iter().any(|line| line.region_type == "line_label");
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut labels: Vec<Vec<usize>> = Vec::new();
        for index in 0..page.lines.len() {
            let line = &page.lines[index];
            if transcript && line.region_type == "line_label" {
                labels.push(vec![index]);
                continue;
            }
            if groups.last().is_some_and(|group| {
                let prior = &page.lines[*group.last().expect("non-empty group")];
                prior.region_type == line.region_type
                    && if transcript && line.region_type == "body" { !opens_turn(line) } else { prior.block_index == line.block_index }
            }) {
                groups.last_mut().expect("group exists").push(index);
            } else {
                groups.push(vec![index]);
            }
        }
        groups.extend(labels);
        groups.sort_by_key(|group| group[0]);
        page.regions.clear();
        for (region_index, indexes) in groups.into_iter().enumerate() {
            let id = format!("p{:04}-r{:04}", page.number, region_index + 1);
            let kind = page.lines[indexes[0]].region_type.clone();
            let line_ids = indexes
                .iter()
                .map(|&index| page.lines[index].id.clone())
                .collect();
            let bbox = union_bbox(indexes.iter().map(|&index| page.lines[index].bbox));
            let reading_order = indexes
                .iter()
                .map(|&index| page.lines[index].reading_order)
                .min()
                .unwrap_or(0);
            for &index in &indexes {
                page.lines[index].region_id.clone_from(&id);
            }
            page.regions.push(Region {
                id,
                page_index: page.index,
                kind,
                line_ids,
                bbox,
                reading_order,
            });
        }
    }
}
