//! Scanned pages and the pictures of text on typeset pages, and the recognized lines placed in them.

use super::*;

/// The pages that are a picture of their text: an image covers at least half of the page and the
/// page's visible text, laid end to end, would not fill one line across it (a filing stamp, a record
/// page number, "Page: 17", a heading over a pasted image), or two where the image covers the whole
/// page (a repository's stamp over each page of a scan: its title, its address, its folio). A text
/// layer behind the picture (invisible, or painted beneath it) does not count. A page of ordinary rendered text, or a
/// tab divider's one line with no image under it, is never one.
/// Each page's last picture covering nine tenths of it, as its index in `items`, which keep the order the page
/// is painted in: text painted before it lies beneath it, unseen, as an invisible layer is (a scanner's reading of
/// the scan, laid under it).
pub(crate) fn covering_pictures(geometries: &PageGeometryMap, items: &[TextItem]) -> HashMap<u32, usize> {
    let mut covering = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let Some((_, geometry)) = geometries.get(&item.page) else { continue };
        let span = |start: f32, extent: f32, limit: f64| (f64::from(start + extent).min(limit) - f64::from(start).max(0.0)).max(0.0);
        let area = geometry.raw_width * geometry.raw_height;
        if matches!(item.item_type, ItemType::Image) && area > 0.0
            && span(item.x, item.width, geometry.raw_width) * span(item.y, item.height, geometry.raw_height) >= 0.9 * area {
            covering.insert(item.page, index);
        }
    }
    covering
}

pub(crate) fn scanned_pages(geometries: &PageGeometryMap, items: &[TextItem]) -> Vec<usize> {
    let mut covered = HashMap::<u32, f64>::new();
    let mut written = HashMap::<u32, f64>::new();
    let beneath = covering_pictures(geometries, items);
    for (index, item) in items.iter().enumerate() {
        if beneath.get(&item.page).is_some_and(|&picture| index < picture) && matches!(item.item_type, ItemType::Text) {
            continue;
        }
        match &item.item_type {
            // Image rects are in the visible box's own frame, before any page rotation.
            ItemType::Image => if let Some((_, geometry)) = geometries.get(&item.page) {
                let area = geometry.raw_width * geometry.raw_height;
                let span = |start: f32, extent: f32, limit: f64|
                    (f64::from(start + extent).min(limit) - f64::from(start).max(0.0)).max(0.0);
                if area > 0.0 {
                    *covered.entry(item.page).or_default() +=
                        span(item.x, item.width, geometry.raw_width) * span(item.y, item.height, geometry.raw_height) / area;
                }
            },
            // A run's length is its longer side, whichever way it is turned (a margin stamp).
            ItemType::Text if item.render_mode != Some(3) && !item.text.trim().is_empty() =>
                *written.entry(item.page).or_default() += f64::from(item.width.max(item.height)),
            _ => {}
        }
    }
    geometries.iter()
        .filter(|(number, (_, geometry))| {
            let share = covered.get(number).copied().unwrap_or(0.0);
            let written = written.get(number).copied().unwrap_or(0.0);
            share >= 0.5 && written < geometry.width || share >= 0.9 && written < 2.0 * geometry.width
        })
        .map(|(number, _)| usize::try_from(number.saturating_sub(1)).unwrap_or(usize::MAX))
        .collect()
}

/// Each page's pictures, `[x0, y0, x1, y1]` in points from the top-left of the page as it is shown.
pub(crate) fn pictures(geometries: &PageGeometryMap, items: &[TextItem]) -> HashMap<u32, Vec<[f64; 4]>> {
    let mut pictures = HashMap::<u32, Vec<[f64; 4]>>::new();
    for item in items.iter().filter(|item| matches!(&item.item_type, ItemType::Image)) {
        let Some((_, geometry)) = geometries.get(&item.page) else {
            continue;
        };
        // Image rects are in the visible box's own frame, before any page rotation.
        let frame = PageGeometry { x0: 0.0, y0: 0.0, ..*geometry };
        let (bbox, visible) =
            transform_bbox(frame, [item.x, item.y, item.x + item.width, item.y + item.height]);
        if visible && bbox[2] > bbox[0] && bbox[3] > bbox[1] {
            let height = geometry.height;
            pictures.entry(item.page).or_default().push([
                f64::from(bbox[0]),
                height - f64::from(bbox[3]),
                f64::from(bbox[2]),
                height - f64::from(bbox[1]),
            ]);
        }
    }
    pictures
}

/// The pictures on pages with text of their own that may be pictures of text: a quotation, an
/// extract or a letterhead pasted in as an image. Pictures one under another by less than a line
/// of the smaller, or side by side by less than a list marker's half-inch indent, are one picture,
/// as text pasted in line by line is. One is at least an inch wide and a line of small print high,
/// holds none of the page's own words (text over a background, or a scan's text layer), and is
/// neither all in the page's top or bottom band, where letterheads, logos and running marks are
/// drawn, nor drawn in the same place on two other pages (a running mark wherever it is). Pages
/// recognized whole have none.
pub(crate) fn image_regions(
    pages: &[Page],
    pictures: HashMap<u32, Vec<[f64; 4]>>,
    recognized: &BTreeSet<usize>,
) -> BTreeMap<usize, Vec<[f64; 4]>> {
    const MIN_WIDTH: f64 = 72.0;
    const MIN_HEIGHT: f64 = 6.0;
    const INDENT: f64 = 36.0;
    let inside = |rect: &[f64; 4], bbox: &[f64; 4]| {
        let (x, y) = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0);
        rect[0] <= x && x <= rect[2] && rect[1] <= y && y <= rect[3]
    };
    let mut joined_by_page = BTreeMap::<usize, Vec<[f64; 4]>>::new();
    for (number, mut rects) in pictures {
        let index = usize::try_from(number.saturating_sub(1)).unwrap_or(usize::MAX);
        let Some(page) = pages.get(index).filter(|page| !page.lines.is_empty()) else {
            continue;
        };
        if recognized.contains(&index) {
            continue;
        }
        let mut joined = Vec::<[f64; 4]>::new();
        while let Some(mut rect) = rects.pop() {
            // Grow the picture by every other one beside or under it until none is.
            loop {
                let before = rects.len();
                rects.retain(|other| {
                    let line = (rect[3] - rect[1]).min(other[3] - other[1]);
                    let across = (rect[0].max(other[0]) - rect[2].min(other[2])).max(0.0);
                    let down = (rect[1].max(other[1]) - rect[3].min(other[3])).max(0.0);
                    let touches = across <= 1.0 && down <= line || down == 0.0 && across <= INDENT;
                    if touches {
                        rect = [
                            rect[0].min(other[0]),
                            rect[1].min(other[1]),
                            rect[2].max(other[2]),
                            rect[3].max(other[3]),
                        ];
                    }
                    !touches
                });
                if rects.len() == before {
                    break;
                }
            }
            joined.push(rect);
        }
        joined.retain(|rect| {
            rect[2] - rect[0] >= MIN_WIDTH
                && rect[3] - rect[1] >= MIN_HEIGHT
                && rect[3] > page.height * 0.12
                && rect[1] < page.height * 0.90
                && !page.lines.iter().any(|line| {
                    if line.words.is_empty() {
                        inside(rect, &line.bbox)
                    } else {
                        line.words.iter().any(|word| inside(rect, &word.bbox))
                    }
                })
        });
        if !joined.is_empty() {
            joined.sort_by(|a, b| a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0])));
            joined_by_page.insert(index, joined);
        }
    }
    let same = |a: &[f64; 4], b: &[f64; 4]| a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 2.0);
    let elsewhere = |page: usize, rect: &[f64; 4]| {
        joined_by_page
            .iter()
            .filter(|(&other, rects)| other != page && rects.iter().any(|other| same(rect, other)))
            .count()
    };
    joined_by_page
        .iter()
        .filter_map(|(&page, rects)| {
            let rects: Vec<_> = rects.iter().filter(|rect| elsewhere(page, rect) < 2).copied().collect();
            (!rects.is_empty()).then_some((page, rects))
        })
        .collect()
}

/// Place a picture's recognized lines on its page in reading order: after the nearest of the
/// page's lines above it in its column (an indented extract is in the column of the heading or
/// paragraph introducing it; a column gutter is wider), else before the nearest one below it.
pub(crate) fn place_picture_lines(page: &mut Page, region: &[f64; 4], lines: Vec<Line>) {
    const GUTTER: f64 = 12.0;
    let in_column = |line: &Line| line.bbox[0] < region[2] + GUTTER && region[0] - GUTTER < line.bbox[2];
    let middle = (region[1] + region[3]) / 2.0;
    let center = |line: &Line| (line.bbox[1] + line.bbox[3]) / 2.0;
    let nearest = |above: bool| {
        page.lines
            .iter()
            .enumerate()
            .filter(|(_, line)| in_column(line) && (center(line) < middle) == above)
            .min_by(|(_, a), (_, b)| {
                let distance = |line: &Line| if above { middle - line.bbox[3] } else { line.bbox[1] - middle };
                distance(a).total_cmp(&distance(b))
            })
            .map(|(at, _)| at)
    };
    let at = nearest(true)
        .map(|at| at + 1)
        .or_else(|| nearest(false))
        .or_else(|| page.lines.iter().position(|line| center(line) > middle))
        .unwrap_or(page.lines.len());
    page.lines.splice(at..at, lines);
}
