# Legal PDF Parser

Local PDF extraction, OCR routing and exact legal-document navigation. Digital-born
pages use native extraction; scanned pages require an explicitly enabled OCR
profile and its runtime assets. Results retain physical pages, geometry, reading
order and source witnesses, with content-addressed reuse of completed work.

## Build and use

From this repository's root, with the Rust toolchain installed:

```sh
cargo build --release --locked --package legal-pdf-parser --no-default-features --features pdf --bin legalpdf
./target/release/legalpdf --version
```

On Windows, the executable is `target\release\legalpdf.exe`; the configured
`rust-lld.exe` linker must be available. `cargo build` does not install the command
on PATH. Build/cache settings are in [.cargo/config.toml](.cargo/config.toml).

No features are enabled by default. [Cargo.toml](Cargo.toml) defines the profiles:

| Profile | Capability |
| --- | --- |
| `pdf` | Native PDF extraction, structure and queries; no OCR or layout models |
| `kraken` | PDF plus Kraken OCR |
| `ppdoc-openvino` / `ppdoc-full` | PDF with the corresponding layout runtime |
| `full` | Language support, Kraken OCR and the full layout profile |

Enabling a feature does not provide its separately licensed model/runtime pack.
Do not describe a source-only build as a complete OCR distribution. Browser OCR
and its single-file HTML application live in
[Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr), not this crate.

## Architecture and integration

This repository owns PDF extraction, geometry, reading order, OCR and PDF-specific
witnesses. Provider-neutral structure, citations, text coordinates and document
queries come from the pinned
[Legal Structure Parser](https://github.com/eliziff/legal-structure-parser).
Its README owns the shared profile, quotation-containment and query-lifetime
contract; Beaver owns application policy and its Node adapter.

A document's primary paragraphs/sections must not be replaced by numbering inside
quotations. Compound records require separately reviewed constituent boundaries;
a tab, numbering restart or style change alone is not enough. Ambiguous structure
must retain exact page/text access. This is a constraint, not a claim that compound
records are completely supported.

[Beaver](https://github.com/eliziff/Beaver) consumes this repository through a pinned
submodule. Its adapter can override the structure dependency with a local path;
that does not validate the Git revision shipped by a standalone parser. Publish
only an explicitly validated parser/structure combination. The PDF extraction
backbone is the single `pdf-inspector` branch selected by the Cargo manifest;
do not create another private fork to bypass its gate.

## Validation and evidence

Follow [AGENTS.md](AGENTS.md) for focused checks and reuse of warm builds. Select
the feature profile being changed, batch source edits, and run the appropriate
behavioral gate on the final candidate. Quotation ownership, constituent boundaries,
cached-extraction structure replay and the full extraction/OCR lifecycle need
separate evidence; passing unit tests or old throughput numbers does not certify
all of them.

[Documentation and benchmark receipts](docs/README.md) identify reproducible
measurements and historical records. Existing OCR results use a 153-page legal
print sample; the native benchmark uses eight documents/425 pages. Those results
are corpus- and machine-specific, not promises for every document or release.
Cross-project priorities belong to
[Beaver's structure plan](https://github.com/eliziff/Beaver/blob/main/docs/roadmap/document-structure.md).

## Credits and license

Recognition uses a legal-domain fine-tune of
[CATMuS Print Small](https://zenodo.org/records/10602357), trained with
[Kraken](https://kraken.re/) and the [CATMuS project](https://huggingface.co/CATMuS).
The PDF extraction backbone is the MIT-licensed `pdf-inspector`.

[MIT](LICENSE). Third-party components and model assets retain their own licenses
and notices.
