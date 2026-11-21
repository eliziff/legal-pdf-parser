//! What DOC reads of a document, each line's in reading order, as arrays: made once from the parse (`Inputs::of`), its
//! dump entry's `inputs` as they are (`tensors`, `Entry`), and read back from there where a parse is resumed
//! (`Inputs::read`). The reader reads these and nothing else of the lines' measures, so the fit, the dump and a
//! resumed parse read one thing.

use crate::model::machine::lines::T0;
use crate::model::machine::lines::SHAPE;
use crate::model::machine::lines::WIDTH;
use crate::model::machine::lines::LineTable;
use super::tensors::Tensor;
use super::*;

pub struct Inputs {
    /// Each line's own features as bytes on their scales (`scale`), WIDTH a line; the document's shape, SHAPE of them.
    pub own: Vec<i8>,
    pub shape: Vec<i8>,
    /// Each line's box (x0, y0, x1, y1) and type size, as measured.
    pub boxes: Vec<f32>,
    pub size: Vec<f32>,
    pub pages: Vec<usize>,
    pub looks: Vec<i64>,
    pub templates: Vec<i64>,
    pub anchors: Vec<u8>,
    /// Each template's measured row, ROW wide.
    pub rows: Vec<f32>,
    pub t0: Vec<u8>,
    /// Each line's first marker reading: its family, value and prefix's length (`reader::line_marker`).
    pub markers: Vec<(i64, i64, i64)>,
    /// Each line's tokens end to end, and each line's count of them, where a text model tokenized them.
    pub tokens: Option<(Vec<u32>, Vec<usize>)>,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        slots: &Slots,
        fences: &Fences,
        decision: &Decision,
        table: &LineTable,
        texts: &[&str],
        tokenize: Option<&dyn Fn(&[&str]) -> Result<(Vec<u32>, Vec<usize>), String>>,
        measure_lines: &dyn Fn() -> crate::model::machine::lines::Measures,
    ) -> Result<Self, String> {
        let measures = measure_lines();
        let sheet = Stylesheet::read(slots, fences, table, &measures, decision);
        let n = fences.order.len();
        let rows = sheet.rows(table, &measures);
        Ok(Self {
            own: scale::quantized(&measures.own),
            shape: scale::shape(&measures.shape).to_vec(),
            boxes: reader::boxes(&measures),
            size: (0..n).map(|k| measures.row(k)[feature("size")]).collect(),
            pages: fences.order.iter().map(|&slot| slots.page(slot)).collect(),
            looks: table.looks.iter().map(|&l| i64::from(l)).collect(),
            templates: sheet.template.iter().map(|&t| i64::from(t)).collect(),
            anchors: sheet.anchor.iter().map(|&a| a as u8).collect(),
            rows: if rows.is_empty() {
                vec![0.0; stylesheet::ROW]
            } else {
                rows.concat()
            },
            t0: measures.t0.clone(),
            markers: reader::line_markers(table),
            tokens: tokenize.map(|tokenize| tokenize(texts)).transpose()?,
        })
    }

    /// As the dump writes them.
    pub fn tensors(&self) -> Vec<Tensor> {
        let n = self.pages.len();
        let int = |name: &str, values: Vec<i64>| Tensor::i32(name, vec![n], values);
        let mut out = vec![
            Tensor::i8("own", vec![n, WIDTH], &self.own),
            Tensor::i8("shape", vec![SHAPE], &self.shape),
            Tensor::f32("boxes", vec![n, 4], &self.boxes),
            Tensor::f32("size", vec![n], &self.size),
            int("pages", self.pages.iter().map(|&p| p as i64).collect()),
            Tensor::i64("looks", vec![n], self.looks.iter().copied()),
            int("templates", self.templates.clone()),
            Tensor::bytes("anchors", vec![n], &self.anchors),
            Tensor::f32(
                "template_rows",
                vec![self.rows.len() / stylesheet::ROW, stylesheet::ROW],
                &self.rows,
            ),
            Tensor::bytes("t0", vec![n, T0], &self.t0),
            int("marker_family", self.markers.iter().map(|m| m.0).collect()),
            // A marker's value is any number it prints (a year, a docket number): 64 bits.
            Tensor::i64("marker_value", vec![n], self.markers.iter().map(|m| m.1)),
            int("marker_prefix", self.markers.iter().map(|m| m.2).collect()),
        ];
        if let Some((tokens, counts)) = &self.tokens {
            out.push(Tensor::i32(
                "tokens",
                vec![tokens.len()],
                tokens.iter().map(|&t| i64::from(t)),
            ));
            out.push(int(
                "token_counts",
                counts.iter().map(|&c| c as i64).collect(),
            ));
        }
        out
    }

    /// As the dump wrote them (`tensors`), from its entry (`Entry`'s `inputs`).
    pub fn read(all: &HashMap<String, Tensor>) -> Result<Self, String> {
        let get = |name: &str| {
            all.get(name)
                .ok_or_else(|| format!("its entry has no input {name}"))
        };
        let ints = |name: &str| get(name).map(Tensor::i64s);
        let (family, value, prefix) = (
            ints("marker_family")?,
            ints("marker_value")?,
            ints("marker_prefix")?,
        );
        let tokens = match (all.get("tokens"), all.get("token_counts")) {
            (Some(tokens), Some(counts)) => Some((
                tokens.i64s().into_iter().map(|t| t as u32).collect(),
                counts.i64s().into_iter().map(|c| c as usize).collect(),
            )),
            _ => None,
        };
        Ok(Self {
            own: get("own")?.bytes.iter().map(|&b| b as i8).collect(),
            shape: get("shape")?.bytes.iter().map(|&b| b as i8).collect(),
            boxes: get("boxes")?.f32s(),
            size: get("size")?.f32s(),
            pages: ints("pages")?.into_iter().map(|p| p as usize).collect(),
            looks: ints("looks")?,
            templates: ints("templates")?,
            anchors: get("anchors")?.bytes.clone(),
            rows: get("template_rows")?.f32s(),
            t0: get("t0")?.bytes.clone(),
            markers: family
                .into_iter()
                .zip(value)
                .zip(prefix)
                .map(|((f, v), p)| (f, v, p))
                .collect(),
            tokens,
        })
    }
}
