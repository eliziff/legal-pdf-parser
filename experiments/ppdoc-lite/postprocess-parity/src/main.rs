use serde::{Deserialize, Serialize};
use std::io::{self, Read};

mod model {
    pub struct Line {
        pub exclude_from_body: bool,
        pub text: String,
        pub bbox: [f64; 4],
    }

    pub struct Page {
        pub index: usize,
        pub width: f64,
        pub height: f64,
        pub lines: Vec<Line>,
    }
}

mod ppdoc {
    pub struct PPDocDetection {
        pub label: String,
        pub score: f32,
        pub bbox: [f32; 4],
    }
}

#[path = "../../../../rust/src/ppdoc_postprocess.rs"]
mod ppdoc_postprocess;

use model::{Line, Page};
use ppdoc::PPDocDetection;
use ppdoc_postprocess::{best_region_index, postprocess_document, scale_detections};

#[derive(Deserialize)]
struct ContractInput {
    cases: Vec<CaseInput>,
}

#[derive(Deserialize)]
struct CaseInput {
    name: String,
    pages: Vec<PageInput>,
}

#[derive(Deserialize)]
struct PageInput {
    page_number: usize,
    width: f64,
    height: f64,
    lines: Vec<LineInput>,
    regions: Vec<RegionInput>,
}

#[derive(Deserialize)]
struct LineInput {
    line_id: String,
    text: String,
    bbox: [f64; 4],
}

#[derive(Deserialize)]
struct RegionInput {
    label: String,
    score: f32,
    bbox: [f64; 4],
}

#[derive(Serialize)]
struct ContractOutput {
    cases: Vec<CaseOutput>,
}

#[derive(Serialize)]
struct CaseOutput {
    name: String,
    pages: Vec<PageOutput>,
}

#[derive(Serialize)]
struct PageOutput {
    page_number: usize,
    regions: Vec<RegionOutput>,
    assignments: Vec<AssignmentOutput>,
}

#[derive(Serialize)]
struct RegionOutput {
    label: String,
    score: f32,
    bbox: [f64; 4],
    order: usize,
    raw_index: usize,
}

#[derive(Serialize)]
struct AssignmentOutput {
    line_id: String,
    label: Option<String>,
    raw_index: Option<usize>,
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let contract: ContractInput = serde_json::from_str(&input).unwrap();
    let mut output = Vec::with_capacity(contract.cases.len());

    for case in contract.cases {
        let pages: Vec<Page> = case
            .pages
            .iter()
            .map(|page| Page {
                index: page.page_number - 1,
                width: page.width,
                height: page.height,
                lines: page
                    .lines
                    .iter()
                    .map(|line| Line {
                        exclude_from_body: false,
                        text: line.text.clone(),
                        bbox: line.bbox,
                    })
                    .collect(),
            })
            .collect();
        let mut regions_by_page = case
            .pages
            .iter()
            .map(|page| {
                let detections: Vec<PPDocDetection> = page
                    .regions
                    .iter()
                    .map(|region| PPDocDetection {
                        label: region.label.clone(),
                        score: region.score,
                        bbox: region.bbox.map(|value| value as f32),
                    })
                    .collect();
                scale_detections(
                    page.width,
                    page.height,
                    page.width as u32,
                    page.height as u32,
                    &detections,
                )
            })
            .collect::<Vec<_>>();
        postprocess_document(&pages, &mut regions_by_page);

        let page_outputs = case
            .pages
            .into_iter()
            .zip(regions_by_page.into_iter())
            .map(|(page, regions)| {
                let assignments = page
                    .lines
                    .into_iter()
                    .map(|line| {
                        let region =
                            best_region_index(line.bbox, &regions).map(|index| &regions[index]);
                        AssignmentOutput {
                            line_id: line.line_id,
                            label: region.map(|value| value.label.clone()),
                            raw_index: region.map(|value| value.raw_index),
                        }
                    })
                    .collect();
                PageOutput {
                    page_number: page.page_number,
                    regions: regions
                        .into_iter()
                        .map(|region| RegionOutput {
                            label: region.label,
                            score: region.score,
                            bbox: region.bbox,
                            order: region.order,
                            raw_index: region.raw_index,
                        })
                        .collect(),
                    assignments,
                }
            })
            .collect();
        output.push(CaseOutput {
            name: case.name,
            pages: page_outputs,
        });
    }

    print!(
        "{}",
        serde_json::to_string(&ContractOutput { cases: output }).unwrap()
    );
}
