//! Experiment-only observations of the ordinary native OCR path and controlled crop readings.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod crop_ab;

#[derive(Debug, Deserialize)]
pub struct ProbeBox {
    pub id: String,
    pub bbox: [f64; 4],
}

fn rect(b: LineBox) -> [usize; 4] {
    [b.left, b.top, b.right, b.bottom]
}

fn overlap(a: [usize; 4], b: [usize; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

fn logadd(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// Bounded CTC prefix search: eight prefixes and eight classes per timestep, plus blank and
/// each prefix's last class. It is an ablation, not the production decoder or an exact search.
fn beam(logits: &[f32], classes: usize, steps: usize) -> Vec<(Vec<usize>, f64)> {
    let mut active = BTreeMap::from([(Vec::<usize>::new(), (0.0, f64::NEG_INFINITY))]);
    for values in logits.chunks_exact(classes).take(steps) {
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z = f64::from(maximum)
            + values
                .iter()
                .map(|v| f64::from(*v - maximum).exp())
                .sum::<f64>()
                .ln();
        let mut top = (0..classes).collect::<Vec<_>>();
        top.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        top.truncate(8);
        top.push(0);
        let mut next = BTreeMap::<Vec<usize>, (f64, f64)>::new();
        for (prefix, (blank, nonblank)) in &active {
            let total = logadd(*blank, *nonblank);
            let mut candidates = top.clone();
            candidates.extend(prefix.last());
            candidates.sort_unstable();
            candidates.dedup();
            for token in candidates {
                let p = f64::from(values[token]) - z;
                let entry = next
                    .entry(prefix.clone())
                    .or_insert((f64::NEG_INFINITY, f64::NEG_INFINITY));
                if token == 0 {
                    entry.0 = logadd(entry.0, total + p);
                    continue;
                }
                let repeated = prefix.last() == Some(&token);
                if repeated {
                    entry.1 = logadd(entry.1, *nonblank + p);
                }
                let mut extended = prefix.clone();
                extended.push(token);
                let entry = next
                    .entry(extended)
                    .or_insert((f64::NEG_INFINITY, f64::NEG_INFINITY));
                entry.1 = logadd(entry.1, if repeated { *blank + p } else { total + p });
            }
        }
        let mut ranked = next.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            logadd(b.1 .0, b.1 .1)
                .total_cmp(&logadd(a.1 .0, a.1 .1))
                .then(a.0.cmp(&b.0))
        });
        active = ranked.into_iter().take(8).collect();
    }
    let mut result = active
        .into_iter()
        .map(|(p, (a, b))| (p, logadd(a, b)))
        .collect::<Vec<_>>();
    result.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    result
}

fn reading(
    session: &mut Session,
    codec: &Codec,
    line: &PreparedLine,
    height: usize,
    image_path: Option<&Path>,
) -> Result<Value> {
    let width = line.width.div_ceil(8) * 8;
    let mut pixels = vec![0.0_f32; height * width];
    for row in 0..height {
        for x in 0..line.width {
            pixels[row * width + x] = 1.0 - f32::from(line.pixels[row * line.width + x]) / 255.0;
        }
    }
    let tensor = TensorRef::from_array_view(([1, 1, height, width], pixels.as_slice()))
        .map_err(ort_error)?;
    let lengths = [line.width as i64];
    let widths = TensorRef::from_array_view(([1], lengths.as_slice())).map_err(ort_error)?;
    let options = RunOptions::new()
        .map_err(ort_error)?
        .with_outputs(OutputSelector::no_default().with("logits").with("lengths"));
    let output = session
        .run_with_options(
            ort::inputs!["image" => tensor, "widths" => widths],
            &options,
        )
        .map_err(ort_error)?;
    let (shape, logits) = output["logits"]
        .try_extract_tensor::<f32>()
        .map_err(ort_error)?;
    let length = if let Ok((_, n)) = output["lengths"].try_extract_tensor::<i64>() {
        n[0] as usize
    } else {
        output["lengths"]
            .try_extract_tensor::<i32>()
            .map_err(ort_error)?
            .1[0] as usize
    };
    let classes = shape[2] as usize;
    let (text, confidence) =
        codec.decode(&greedy_ctc(logits, 0, classes, shape[1] as usize, length));
    let alternatives = beam(logits, classes, length)
        .into_iter()
        .take(5)
        .map(|(ids, score)| {
            let spans = ids.into_iter().map(|id| (id, 1.0)).collect::<Vec<_>>();
            json!({"text": codec.decode(&spans).0, "log_probability": score})
        })
        .collect::<Vec<_>>();
    if let Some(path) = image_path {
        GrayImage::from_raw(line.width as u32, height as u32, line.pixels.clone())
            .unwrap()
            .save(path)
            .map_err(|e| Error::Message(e.to_string()))?;
    }
    Ok(
        json!({"text": text, "confidence": confidence, "beam": alternatives,
        "normalized_width": line.width, "normalized_height": height,
        "normalized_sha256": format!("{:x}", Sha256::digest(&line.pixels))}),
    )
}

impl IndraOcr {
    /// Read one complete PDF page with the production renderer, orientation, detector, crop
    /// preparation and greedy decoder. Authoring rectangles are diagnostic interventions only.
    pub fn attribution_page(
        &mut self,
        pdf: &[u8],
        request: &OcrPageRequest,
        targets: &[ProbeBox],
        images: &Path,
    ) -> Result<Value> {
        debug_assert_eq!(
            beam(&[0., 8., 0., 8., 0., 0., 0., 8., 0.], 3, 3)[0].0,
            vec![1, 1]
        );
        debug_assert_eq!(beam(&[0.1, 0.0, 0.1, 0.0], 2, 2)[0].0, vec![1]);
        fs::create_dir_all(images).map_err(|e| Error::io(images, e))?;
        let document = Pdf::new(pdf.to_vec()).map_err(|e| Error::Message(format!("PDF: {e:?}")))?;
        let orientation = self.orientation.as_mut().map(std::sync::Mutex::new);
        let mut rendered = render_pdf_pages(
            &document,
            &RenderCache::new(),
            &InterpreterSettings::default(),
            &pdf_render_settings(self.dpi),
            std::slice::from_ref(request),
            orientation.as_ref(),
        )?;
        if rendered.len() != 1 {
            return Err(Error::Message(
                "Probe expects one complete rendered page".into(),
            ));
        }
        let (metadata, image) = rendered.remove(0);
        image
            .save(images.join("page.png"))
            .map_err(|e| Error::Message(e.to_string()))?;
        let page = PagePixels::Rgba {
            pixels: image.as_raw(),
            width: image.width(),
            height: image.height(),
        };
        let boxes = page.layout(&mut self.layouts[0])?;
        let core = self.layouts[0]
            .core_boxes()
            .iter()
            .copied()
            .map(rect)
            .collect::<Vec<_>>();
        let supplied = targets
            .iter()
            .filter_map(|t| {
                if metadata.5 .0 != 0 {
                    return None;
                }
                let b = t.bbox;
                let clamp = |v: f64, limit: u32| v.round().clamp(0.0, f64::from(limit)) as usize;
                let bbox = LineBox {
                    left: clamp(b[0] / metadata.2, image.width()),
                    top: clamp(b[1] / metadata.3, image.height()),
                    right: clamp(b[2] / metadata.2, image.width()),
                    bottom: clamp(b[3] / metadata.3, image.height()),
                    recovered: false,
                };
                (bbox.width() > 0 && bbox.height() > 0).then_some((t.id.as_str(), bbox))
            })
            .collect::<Vec<_>>();
        let prepared = boxes
            .iter()
            .filter_map(|&b| prepare_page_line(page, b, self.width_scale, self.input_height))
            .collect::<Vec<_>>();
        let codec = self.codec.as_ref().unwrap();
        let (mut recognized, _) = Self::recognize_prepared_pages(
            &mut self.sessions,
            codec,
            self.batch_size,
            self.width_bucket,
            self.input_height,
            std::slice::from_ref(&prepared),
        )?;
        let mut observations = Vec::new();
        for (n, (line, result)) in prepared.iter().zip(recognized.remove(0)).enumerate() {
            let kept = !result.text.trim().is_empty()
                && (!line.bbox.recovered || result.confidence >= RECOVERED_CONFIDENCE);
            let mut value = json!({"bbox": rect(line.bbox), "recovered": line.bbox.recovered,
                "text": result.text, "confidence": result.confidence, "kept": kept});
            if supplied
                .iter()
                .any(|(_, b)| overlap(rect(*b), rect(line.bbox)))
            {
                let detail = reading(
                    &mut self.sessions[0],
                    codec,
                    line,
                    self.input_height,
                    Some(&images.join(format!("line-{n}.png"))),
                )?;
                value["greedy_parity"] = json!(detail["text"] == value["text"]);
                value["reading"] = detail;
            }
            observations.push(value);
        }
        let mut crop_readings = Vec::new();
        for (id, bbox) in supplied {
            for (variant, padding) in [("authoring_box", 0.0), ("expanded_box", 0.35)] {
                let pad = (bbox.height() as f64 * padding).ceil() as usize;
                let expanded = LineBox {
                    left: bbox.left.saturating_sub(pad),
                    top: bbox.top.saturating_sub(pad),
                    right: (bbox.right + pad).min(image.width() as usize),
                    bottom: (bbox.bottom + pad).min(image.height() as usize),
                    ..bbox
                };
                if let Some(line) =
                    prepare_page_line(page, expanded, self.width_scale, self.input_height)
                {
                    let mut value =
                        reading(&mut self.sessions[0], codec, &line, self.input_height, None)?;
                    value["id"] = json!(id);
                    value["variant"] = json!(variant);
                    value["bbox"] = json!(rect(expanded));
                    crop_readings.push(value);
                }
            }
        }
        Ok(
            json!({"schema": "ocr-attribution.page.v1", "page": request.page_index + 1, "dpi": self.dpi,
            "width": image.width(), "height": image.height(), "points_per_pixel": [metadata.2, metadata.3],
            "turns": metadata.5.0, "raster_sha256": format!("{:x}", Sha256::digest(image.as_raw())),
            "core_boxes": core, "lines": observations, "crops": crop_readings,
            "decoder": {"beam_width": 8, "classes_per_step": 8}}),
        )
    }
}
