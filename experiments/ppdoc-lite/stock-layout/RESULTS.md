# Stock layout gate — laptop CPU

Date: 2026-08-14. This was a bounded 87-page evaluation on the retained legal
test split, using the same rendered images and exact-label overlap against the
25-class legal annotations. The laptop exposed eight logical processors. No
Docker or desktop inference was used.

These numbers compare stock ontologies only where their label meaning exactly
matches the legal ontology. They are product-selection evidence, not claims
about the vendors' private evaluation sets.

| Model | Runtime | Inference/page | Pages/s | Legal AP | AP50 | AP75 |
|---|---|---:|---:|---:|---:|---:|
| PP-DocLayout-S | Paddle 3.0, oneDNN | 0.0416 s | 24.04 | 0.236 | 0.435 | 0.171 |
| PicoDet-S layout-17 | Paddle 3.0, oneDNN | 0.0444 s | 22.54 | 0.167 | 0.336 | 0.125 |
| Docling Heron INT8 | ONNX Runtime, 4 threads | 0.8268 s | 1.21 | 0.359 | 0.506 | 0.387 |

Temporary Python image preparation measured 0.0628, 0.0820, and 0.0490 seconds
per page respectively; it is not included in the inference column. Rust's
production postprocessor is about 0.35 ms/page and is not material here.

Important AP50 slices:

- PP-DocLayout-S: heading 0.613, text 0.862, header 0.882, image 0.575,
  footnote 0.224, table 0.040.
- PicoDet-S: heading 0.415, text 0.769, image 0.311, footnote 0.095,
  table 0.000.
- Heron INT8: heading 0.621, text 0.879, header 0.702, image 0.548,
  footnote 0.826, table 0.004.

Decision:

- Retain the custom large PP-DocLayout model as the local quality incumbent.
- Offer Heron INT8 as the portable stock/balanced fallback because it is a
  standard ONNX graph with useful heading and footnote behavior, while clearly
  documenting that it remains a layout bottleneck and is not a table detector
  on this corpus.
- Do not promote PicoDet-S or PP-DocLayout-S as quality routes. Their speed is
  real, but their legal-layout accuracy is only suitable for an explicitly
  rough/turbo tier. Paddle's Windows runtime also makes either pack much less
  thin than ONNX/OpenVINO.
- Provide a PPDoc-free vision-model route for callers that need stock semantic
  coverage without accepting Heron's table failure.

The resulting product ladder is native/no-model turbo (51.83 pages/s in the
end-to-end gate), Heron INT8 balanced (1.09 pages/s), and the existing custom
PPDoc pack as the local quality tier. A caller selects the local pack rather
than a model name embedded in Rust, so later quantized or distilled packs reuse
the exact same preprocessing, postprocessing, CPU/GPU dispatch, cache identity,
and benchmark gates. The MLLM route is an opt-in fourth route for broader stock
semantics; it is not represented as local or zero-cost inference.

The production Rust gate subsequently ran the untouched release
`layout_heron_int8.onnx` (input name `pixel_values`) with manifest-selected
bilinear preprocessing. One bounded page produced the same class counts as the
Python reference: 18 Text, 1 List-item, 1 Section-header, 4 Page-footer, and 3
Picture detections above 0.3. This proves the production path no longer depends
on renaming the ONNX graph.

Production end-to-end gate on a separate 24-page legal PDF (release Rust,
OpenVINO, four threads, cold one-shot process):

| Route | Total | Seconds/page | Pages/s |
|---|---:|---:|---:|
| Native parser, no layout model | 0.463 s | 0.0193 | 51.83 |
| Native parser + Heron INT8 | 22.113 s | 0.9214 | 1.09 |

Both runs were `ready` with 24 pages, 1,133 lines, 128 paragraphs, 150
footnotes, one image, no tables, and zero diagnostics. Every page's multiset of
source line text was identical. This confirms text preservation and also proves
that Heron, not Rust extraction or postprocessing, is the end-to-end bottleneck
on this laptop.

The standalone Windows bundle gate also passed from inside the produced bundle
directory, using only `legalpdf.exe`, the model pack, and its copied OpenVINO
DLLs. The checksum inventory contained 11 files totalling 156,133,399 bytes
(148.9 MiB). A first one-page invocation, including process startup and model
compilation, took 4.21 seconds; reported warm inference was 0.713 seconds. The
gate caught and fixed a missing `openvino_onnx_frontend.dll` in the bundler.
