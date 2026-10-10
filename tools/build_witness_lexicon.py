"""Build the witness lexicon (legal-pdf-witnesses/data/lexicon.json) from the corpus store's index
of court forms: every form's title, every court's name, and the lines printed across at least three
distinct forms (a form's boilerplate: "Between", "REQUIRED", "S.C. No."). Lines are normalized as
the witnesses normalize them: lowercase letters and digits, single spaces.

  python legal-pdf-parser/tools/build_witness_lexicon.py
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/legal_pdf_corpus"))
import corpus_store  # noqa: E402

OUT = ROOT / "legal-pdf-parser/legal-pdf-witnesses/data/lexicon.json"


def normal(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def main():
    store = corpus_store.connect()
    titles = sorted({normal(title) for (title,) in store.execute("select title from legal_forms") if title and len(normal(title)) >= 4})
    courts = sorted({normal(court) for (court,) in store.execute("select court from legal_forms") if court and len(normal(court)) >= 6})
    printed = defaultdict(set)
    for form, metadata in store.execute("select form_id, metadata from legal_form_observations"):
        for sample in json.loads(metadata).get("heading_samples", []):
            for line in (sample if isinstance(sample, str) else sample.get("text", "")).splitlines():
                key = normal(line)
                if 2 <= len(key) <= 60 and re.search(r"[a-z]", key):
                    printed[key].add(form)
    boilerplate = sorted(key for key, forms in printed.items() if len(forms) >= 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"form_titles": titles, "courts": courts, "form_boilerplate": boilerplate}, indent=0), encoding="utf-8")
    print(f"{len(titles)} titles, {len(courts)} courts, {len(boilerplate)} boilerplate lines -> {OUT}")


if __name__ == "__main__":
    main()
