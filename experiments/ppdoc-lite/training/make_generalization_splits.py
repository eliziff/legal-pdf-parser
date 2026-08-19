#!/usr/bin/env python3
"""Build leak-free PP-DocLayout COCO splits from the retained 661 pages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_FILES = ("instance_train.json", "instance_val.json", "instance_benchmark.json")
DEFAULT_TEST_JOURNALS = ("APPEAL", "CAN-US-LJ", "MCGILL-LJ-BACKCAT")
DROPPED_CLASSES = ("inline_formula",)
PAGE_RE = re.compile(
    r"__\d+_(?P<journal>.+?)_article-(?P<article>[^_]+)_pdf-page-(?P<page>\d+)\.png$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_page(file_name: str) -> tuple[str, str]:
    match = PAGE_RE.search(file_name)
    if not match:
        raise ValueError(f"Cannot recover journal/article identity from {file_name!r}")
    journal = match.group("journal")
    return journal, f"{journal}/article-{match.group('article')}"


def load_pages(source_dir: Path) -> tuple[list[dict], list[dict], dict[str, str]]:
    pages: list[dict] = []
    canonical_categories: list[dict] | None = None
    source_hashes: dict[str, str] = {}
    seen_files: set[str] = set()

    for source_name in SOURCE_FILES:
        path = source_dir / source_name
        data = json.loads(path.read_text(encoding="utf-8"))
        source_hashes[source_name] = sha256(path)
        categories = sorted(data["categories"], key=lambda category: int(category["id"]))
        if canonical_categories is None:
            canonical_categories = categories
        elif categories != canonical_categories:
            raise ValueError(f"Category contract differs in {source_name}")

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image in data["images"]:
            file_name = str(image["file_name"])
            if file_name in seen_files:
                raise ValueError(f"Duplicate page across source splits: {file_name}")
            seen_files.add(file_name)
            journal, article = parse_page(file_name)
            pages.append(
                {
                    "image": image,
                    "annotations": annotations_by_image[int(image["id"])],
                    "journal": journal,
                    "article": article,
                    "source_split": source_name,
                }
            )

    if canonical_categories is None:
        raise ValueError("No source annotations found")
    return pages, canonical_categories, source_hashes


def legal_ontology(pages: list[dict], categories: list[dict]) -> list[dict]:
    names = {int(category["id"]): str(category["name"]) for category in categories}
    dropped_ids = {category_id for category_id, name in names.items() if name in DROPPED_CLASSES}
    dropped_annotations = sum(
        int(annotation["category_id"]) in dropped_ids
        for page in pages
        for annotation in page["annotations"]
    )
    if dropped_annotations:
        raise ValueError(f"Refusing to discard {dropped_annotations} annotated dropped-class regions")

    kept = [category for category in categories if str(category["name"]) not in DROPPED_CLASSES]
    id_map = {int(category["id"]): new_id for new_id, category in enumerate(kept)}
    for page in pages:
        for annotation in page["annotations"]:
            annotation["category_id"] = id_map[int(annotation["category_id"])]
    return [{**category, "id": new_id} for new_id, category in enumerate(kept)]


def annotation_counts(pages: list[dict], categories: list[dict]) -> Counter[str]:
    names = {int(category["id"]): str(category["name"]) for category in categories}
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update(names[int(annotation["category_id"])] for annotation in page["annotations"])
    return counts


def choose_validation_articles(
    pages: list[dict], categories: list[dict], target_pages: int, candidates: int, salt: str
) -> tuple[set[str], int, float]:
    by_article: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_article[page["article"]].append(page)

    total_counts = annotation_counts(pages, categories)
    scored_classes = [name for name, count in total_counts.items() if count >= 20]
    article_counts = {
        article: annotation_counts(article_pages, categories)
        for article, article_pages in by_article.items()
    }
    total_pages = len(pages)
    best: tuple[float, int, set[str]] | None = None

    for candidate in range(candidates):
        order = sorted(
            by_article,
            key=lambda article: hashlib.sha256(
                f"{salt}:{candidate}:{article}".encode("utf-8")
            ).digest(),
        )
        chosen: set[str] = set()
        page_count = 0
        for article in order:
            size = len(by_article[article])
            if page_count >= target_pages:
                break
            if page_count and abs(page_count - target_pages) <= abs(page_count + size - target_pages):
                break
            chosen.add(article)
            page_count += size

        counts: Counter[str] = Counter()
        journals: set[str] = set()
        for article in chosen:
            counts.update(article_counts[article])
            journals.update(page["journal"] for page in by_article[article])

        score = ((page_count - target_pages) / 2.0) ** 2
        for name in scored_classes:
            expected = total_counts[name] * page_count / total_pages
            score += ((counts[name] - expected) / (math.sqrt(expected) + 2.0)) ** 2
            if counts[name] == 0:
                score += 20.0
        score += max(0, min(12, len({page["journal"] for page in pages})) - len(journals)) ** 2

        result = (score, candidate, chosen)
        if best is None or (result[0], result[1]) < (best[0], best[1]):
            best = result

    if best is None:
        raise ValueError("No validation split candidate was produced")
    return best[2], best[1], best[0]


def coco_document(pages: list[dict], categories: list[dict], description: str) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for image_id, page in enumerate(sorted(pages, key=lambda item: item["image"]["file_name"]), 1):
        image = copy.deepcopy(page["image"])
        old_image_id = int(image["id"])
        image["id"] = image_id
        images.append(image)
        for source_annotation in sorted(page["annotations"], key=lambda item: int(item["id"])):
            annotation = copy.deepcopy(source_annotation)
            if int(annotation["image_id"]) != old_image_id:
                raise ValueError("Annotation/image identity mismatch")
            annotation["id"] = annotation_id
            annotation["image_id"] = image_id
            annotations.append(annotation)
            annotation_id += 1
    return {
        "info": {"description": description},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": copy.deepcopy(categories),
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-journal", action="append", dest="test_journals")
    parser.add_argument("--val-pages", type=int, default=75)
    parser.add_argument("--search-candidates", type=int, default=10_000)
    parser.add_argument("--salt", default="legalpdf.ppdoc.generalization.v1")
    args = parser.parse_args()

    pages, categories, source_hashes = load_pages(args.source_dir)
    categories = legal_ontology(pages, categories)
    test_journals = set(args.test_journals or DEFAULT_TEST_JOURNALS)
    known_journals = {page["journal"] for page in pages}
    missing = test_journals - known_journals
    if missing:
        raise ValueError(f"Unknown held-out journal(s): {sorted(missing)}")

    test_pages = [page for page in pages if page["journal"] in test_journals]
    development_pages = [page for page in pages if page["journal"] not in test_journals]
    validation_articles, chosen_candidate, split_score = choose_validation_articles(
        development_pages,
        categories,
        args.val_pages,
        args.search_candidates,
        args.salt,
    )
    validation_pages = [page for page in development_pages if page["article"] in validation_articles]
    train_pages = [page for page in development_pages if page["article"] not in validation_articles]

    train_articles = {page["article"] for page in train_pages}
    validation_article_set = {page["article"] for page in validation_pages}
    test_articles = {page["article"] for page in test_pages}
    train_journals = {page["journal"] for page in train_pages}
    validation_journals = {page["journal"] for page in validation_pages}
    test_journal_set = {page["journal"] for page in test_pages}
    if train_articles & validation_article_set or train_articles & test_articles or validation_article_set & test_articles:
        raise AssertionError("Article leakage remains after splitting")
    if (train_journals | validation_journals) & test_journal_set:
        raise AssertionError("Journal leakage remains in the held-out test split")
    if len(train_pages) + len(validation_pages) + len(test_pages) != len(pages):
        raise AssertionError("Split is not exhaustive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "instance_train.json": coco_document(train_pages, categories, "PP-DocLayout generalization train"),
        "instance_val.json": coco_document(validation_pages, categories, "PP-DocLayout article-disjoint validation"),
        "instance_test.json": coco_document(test_pages, categories, "PP-DocLayout journal-disjoint test"),
    }
    for name, document in outputs.items():
        write_json(args.output_dir / name, document)

    category_names = [str(category["name"]) for category in categories]
    split_pages = {"train": train_pages, "val": validation_pages, "test": test_pages}
    manifest = {
        "schema_version": "legalpdf.ppdoc_generalization_split.v1",
        "source": {"pages": len(pages), "files": source_hashes},
        "policy": {
            "train_val_boundary": "article",
            "test_boundary": "journal",
            "test_journals": sorted(test_journals),
            "validation_target_pages": args.val_pages,
            "validation_search_candidates": args.search_candidates,
            "validation_candidate": chosen_candidate,
            "validation_score": split_score,
            "salt": args.salt,
            "dropped_classes": list(DROPPED_CLASSES),
        },
        "splits": {},
        "checks": {
            "article_overlap_train_val": 0,
            "article_overlap_train_test": 0,
            "article_overlap_val_test": 0,
            "journal_overlap_development_test": 0,
        },
    }
    for name, selected_pages in split_pages.items():
        counts = annotation_counts(selected_pages, categories)
        manifest["splits"][name] = {
            "pages": len(selected_pages),
            "articles": len({page["article"] for page in selected_pages}),
            "journals": sorted({page["journal"] for page in selected_pages}),
            "annotations": sum(counts.values()),
            "class_counts": {category: counts[category] for category in category_names},
        }
    manifest["outputs"] = {name: sha256(args.output_dir / name) for name in outputs}
    write_json(args.output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
