import argparse
import json
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

DB = Path(os.environ["LOCALAPPDATA"]) / "OpenLegalProducts/LegalData/providers/courtlistener/courtlistener.sqlite"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def draft_pages(cluster):
    db = sqlite3.connect(DB)
    rows = db.execute("select xml_harvard from opinion where cluster_id=? and coalesce(xml_harvard,'')<>'' order by length(xml_harvard) desc", (cluster,)).fetchall()
    db.close()
    if not rows:
        raise RuntimeError(f"no Harvard/CAP text for {cluster}")
    root = ET.fromstring(rows[0][0])
    footnotes = {}
    footnote_root = next((node for node in root.iter() if node.tag == "div" and node.get("class") == "footnotes"), None)
    if footnote_root is not None:
        for node in list(footnote_root):
            label = node.get("label") or ""
            footnotes[label] = re.sub(rf"^\s*{re.escape(label)}\s*\.\s*", "", clean("".join(node.itertext())))
        root.remove(footnote_root)
    parts, current = [], {"label": None, "pieces": [], "footnotes": []}

    def walk(node):
        nonlocal current
        if node.get("class") == "star-pagination":
            if current["pieces"]:
                parts.append(current)
            current = {"label": node.get("label"), "pieces": [], "footnotes": []}
        elif node.tag == "a" and node.get("class") == "footnote" and node.get("id", "").endswith("_ref"):
            label = clean("".join(node.itertext()))
            current["pieces"].append(label)
            if label in footnotes and label not in current["footnotes"]:
                current["footnotes"].append(label)
        elif node.text:
            current["pieces"].append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                current["pieces"].append(child.tail)

    walk(root)
    if current["pieces"]:
        parts.append(current)
    for part in parts:
        part["text"] = clean(" ".join(part.pop("pieces")))
        part["footnote_text"] = [f"{label}. {footnotes[label]}" for label in part["footnotes"]]
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(draft_pages(args.cluster), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
