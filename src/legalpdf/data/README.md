# Bundled legal data

`mcgill_reporters.json` contains the 2,110 ordinal-distinct abbreviations from
columns A and B of the `Reporters & Journals` sheet in the McGill Guide (10th)
appendices workbook. Rows without both an abbreviation and a title are excluded;
duplicate abbreviations are collapsed, then entries are ordered by decreasing
character length and ordinal value so longer reporter names win prefix ties.

The UTF-8 JSON is 30,965 bytes with SHA-256
`946e7554e8e9134d9b148d244d825e999080dd900c666cc4cf43235fa5ec9e2f`.
The source workbook is not distributed with the engine.
