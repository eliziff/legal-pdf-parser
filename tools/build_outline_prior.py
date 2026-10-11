"""Build the outline prior (legal-pdf-witnesses/data/outline_prior.json) from the structure gold's
training half: for each unit signature (kind and numbering family), how often a unit of each other
signature is its parent. The held-out half stays unseen, so the gold's scores of what the prior
composes into remain honest.

  python legal-pdf-parser/tools/build_outline_prior.py
"""
import hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/legal-structure-gold"))
import evaluate, hierarchy_pass, score_all, worker  # noqa: E402

OUT = ROOT / "legal-pdf-parser/legal-pdf-witnesses/data/outline_prior.json"
GOLD = ROOT / "benchmarks/local-data/legal-structure-gold"
# The same families, in the same order, as the witnesses read (witness.rs).
FAMILIES = [
    ("bullet", r"^\s*[•·▪◦‣●○■□➢✓*–-]\s"), ("paren_num", r"^\s*\(\d{1,3}\)"),
    ("paren_letter", r"^\s*\([a-z]{1,2}\)"), ("paren_roman", r"^\s*\((?:[ivx]{1,5})\)"),
    ("decimal", r"^\s*\d{1,3}(?:\.\d{1,3})+\.?\s"), ("num", r"^\s*\d{1,3}[.)]\s"), ("num_bare", r"^\s*\d{1,3}\s+[A-Z]"),
    ("letter", r"^\s*[a-zA-Z][.)]\s"), ("roman", r"^\s*[IVXivx]{1,5}[.)]\s"),
    ("word", r"(?i)^\s*(part|section|article|chapter|schedule|division|rule|regulation)\s+[\dIVX]"),
]


def signature(kind, text):
    family = next((name for name, pattern in FAMILIES if re.search(pattern, text)), "none")
    return f"{'paragraph' if kind == 'prose' else kind}/{family}"


def main():
    counts = defaultdict(Counter)
    for mode, job, receipt in score_all.completed(GOLD):
        run = str(job.relative_to(GOLD))
        if int(hashlib.sha1(run.encode()).hexdigest(), 16) % 2 == 1:
            continue  # held out
        try:
            _, ref = evaluate.state_of(hierarchy_pass.apply(job, worker.read_json(job / "gold.json.gz")))
        except Exception:
            continue
        lines = {l["id"]: l for l in ref["lines"]}
        units = {g["line_ids"][0]: g for g in ref["groups"] if g["line_ids"] and (g["kind"] == "heading" or g["attributes"].get("marker"))}
        for first, g in units.items():
            child = signature(g["kind"], lines[first]["text"])
            parent = g["attributes"].get("parent")
            counts[child][signature(units[parent]["kind"], lines[parent]["text"]) if parent in units else "root"] += 1
    prior = {child: {parent: round(n / sum(c.values()), 4) for parent, n in c.items()} for child, c in counts.items() if sum(c.values()) >= 5}
    OUT.write_text(json.dumps(prior, indent=0, sort_keys=True), encoding="utf-8")
    print(f"{len(prior)} child signatures -> {OUT}")


if __name__ == "__main__":
    main()
