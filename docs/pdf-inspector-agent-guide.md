Firecrawl owns upstream PDF Inspector. Our modifications are maintained once in `vendor/pdf-inspector/`, ordinary files on parser main, consumed through a local Cargo path dependency. The initial source is byte-identical to our previously maintained `b2f3585def52d953bb41bfe19638ec35d090b0ee` patchset; upstream base/version are recorded in `.upstream-pdf-inspector`. Preserve source credit, MIT license and embedded CMaps.

Edit the vendored source directly and commit in this repository. To incorporate an upstream update, start with a clean vendored tree and run from the parser checkout:

```sh
git fetch --no-tags https://github.com/firecrawl/pdf-inspector.git main
python tools/sync-pdf-inspector.py FETCH_HEAD
```

The command applies upstream's delta from the recorded base into our source with Git's three-way merge. Conflicts stop the update; resolve them and record the fetched commit/version in `.upstream-pdf-inspector` before committing. Guidance stays local. Upstream history is not merged into parser history.

One CI gate checks real extraction products and source gold for Inspector library changes. There is no separate update workflow, automatic commit, Inspector publication, tag, bundle or dependency pin bump. A successful merge still needs behavior verification; Git cannot detect semantic incompatibility.
