"""Train native source-span roles or scoped attachment weights from frozen canonical gold (never the test split).

Usage: python legal-pdf-parser/tools/train_structure.py outline|spans|scope_openings|scopes|membership FAMILY_MANIFEST OUTPUT_MODEL
Requires the existing offline LightGBM environment and a built native addon.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/legal-structure-gold"))
import package
from partitions import validate_partitions

SCHEMAS = {"outline": "scoped-span-root-v6", "spans": "source-span-role-v7", "scope_openings": "source-scope-opening-v3", "scopes": "source-scope-identity-v8", "membership": "source-fragment-membership-v4"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def corrected_pages(gold, mode):
    """Use corrected physical source and observed style only; no parser-derived gold roles."""
    lines = package.line_table(gold)
    order = package.reading_order(gold)
    pages = []
    for physical in gold["pages"]:
        number = physical["page"]
        width, height = physical.get("width_points"), physical.get("height_points")
        require(isinstance(width, (int, float)) and width > 0 and isinstance(height, (int, float)) and height > 0,
                "Training source lacks physical page dimensions")
        page_lines = []
        for ident in order:
            line = lines[ident]
            if line["page"] != number:
                continue
            box = line.get("box")
            require(isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box),
                    "Training source lacks physical line geometry")
            bbox = [box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000]
            styles = line.get("type", [])
            spans = []
            # Retained character style evidence owns same-line boundary candidates.
            for index, style in enumerate(line.get("spans", [])):
                start, end = style.get("start"), style.get("end")
                if start is None or end is None:
                    continue
                require(type(start) is int and type(end) is int and 0 <= start <= end <= len(line["text"]), "Invalid corrected style span")
                style_box = style["bbox"]
                require(isinstance(style_box, list) and len(style_box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in style_box), "Invalid character style geometry")
                spans.append({"id": ident + "/style/" + str(index), "text": line["text"][start:end],
                    "bbox": [style_box[0] * width / 1000, style_box[1] * height / 1000, style_box[2] * width / 1000, style_box[3] * height / 1000],
                    "font": style.get("font", ""), "size": style.get("size", 0), "flags": style.get("flags", 0), "start": start, "end": end, "mcid": style.get("mcid"), "structure_role": style.get("structure_role")})
            # An inventory with several entries does not provide character-to-style positions.
            if not spans and len(styles) == 1:
                font, size, flags = styles[0]
                spans = [{"id": ident + "/style", "text": line["text"], "bbox": bbox,
                          "font": font, "size": size, "flags": flags, "start": 0, "end": len(line["text"])}]
            words = []
            for index, word in enumerate(line.get("words", [])):
                start, end = word.get("start"), word.get("end")
                if start is None or end is None:
                    continue  # Retained geometry without a declared character association is not an aligned word.
                require(type(start) is int and type(end) is int and 0 <= start <= end <= len(line["text"]) and line["text"][start:end] == word["text"], "Invalid corrected word alignment")
                word_box = word["bbox"]
                words.append({"id": word.get("id", ident + "/word/" + str(index)), "text": word["text"],
                    "start": start, "end": end, "underlined": word.get("underlined", False),
                    "bbox": [word_box[0] * width / 1000, word_box[1] * height / 1000, word_box[2] * width / 1000, word_box[3] * height / 1000]})
            ocr = line.get("ocr")
            if ocr is not None:
                ocr = {**ocr, **{key: [[point[0] * width / 1000, point[1] * height / 1000] for point in ocr.get(key, [])] for key in ("baseline", "boundary")}}
            slot = len(page_lines)
            page_lines.append({"id": ident, "page_index": len(pages), "page_number": number,
                "source_index": slot, "reading_order": slot, "block_index": slot, "text": line["text"],
                "bbox": bbox, "spans": spans, "words": words, "ocr": ocr, "region_id": line.get("region_id", ""),
                "region_type": line.get("region_type", "unknown"), "source": line.get("source", "native" if mode == "digitalborn" else "ocr")})
        pages.append({"id": "page-" + str(number), "index": len(pages), "number": number,
            "width": width, "height": height, "lines": page_lines, "regions": [{**region, "page_index": len(pages),
                "bbox": [region["bbox"][0] * width / 1000, region["bbox"][1] * height / 1000, region["bbox"][2] * width / 1000, region["bbox"][3] * height / 1000]} for region in physical.get("regions", [])],
            "source": "native" if mode == "digitalborn" else "ocr"})
    return pages


def packet(gold, mode, task):
    request = {"structure": package.parser_view(gold), "pages": corrected_pages(gold, mode)}
    witnesses = {item["id"]: item for item in request["structure"].get("native_navigation", [])}
    def depth(item):
        level, parent, seen = 1, item.get("parent_id"), {item["id"]}
        while parent is not None:
            require(parent in witnesses and parent not in seen, "Invalid native bookmark hierarchy")
            seen.add(parent); level += 1; parent = witnesses[parent].get("parent_id")
        return level
    request["source_outline"] = [{"title": item["title"], "level": depth(item), "page_index": item["page_index"]}
                                 for item in witnesses.values()]
    result = subprocess.run(["node", str(ROOT / "experiments/legal-structure-gold/extract.mjs"), {"outline": "--outline-training-rows", "spans": "--span-training-rows", "scope_openings": "--scope-opening-training-rows", "scopes": "--scope-training-rows", "membership": "--membership-training-rows"}[task]],
        input=json.dumps(request, ensure_ascii=False), capture_output=True, text=True, encoding="utf-8")
    require(result.returncode == 0, result.stderr.strip() or "Native training feature generation failed")
    value = json.loads(result.stdout)
    require(value.get("schema") == SCHEMAS[task], "Unexpected native training feature schema")
    return value


def main():
    require(len(sys.argv) == 4 and sys.argv[1] in SCHEMAS, "Expected outline|spans|scope_openings|scopes|membership, frozen family manifest and output model path")
    task = sys.argv[1]
    manifest_path, output = map(Path, sys.argv[2:])
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    require(manifest.get("schema_version") == "pdf-structure-gold-membership.v1", "Unexpected family manifest")
    require(manifest.get("status") == "evaluation_frozen" and not manifest.get("blockers"),
            "Training requires revised policy gold and an evaluation-frozen family manifest with no blockers")
    frozen = manifest.get("freeze")
    require(isinstance(frozen, dict), "Frozen manifest lacks its identity ledger")
    identity = frozen.get("identity")
    unsigned = {**manifest, "freeze": {key: value for key, value in frozen.items() if key != "identity"}}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    require(identity == hashlib.sha256(encoded).hexdigest(), "Frozen manifest identity differs")
    def sha256(value):
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
    require(all(sha256(frozen.get(key)) for key in ("selection_sha256", "review_sha256", "policy_sha256")),
            "Frozen manifest lacks prepared selection, review or policy identities")
    ledger = {}
    require(isinstance(frozen.get("gold"), list), "Frozen manifest lacks gold identities")
    for item in frozen["gold"]:
        require(isinstance(item, dict) and sha256(item.get("sha256")) and
                all(sha256(item.get(key)) for key in ("artifact_sha256", "structure_authoring_sha256", "text_identity")),
                "Invalid frozen gold identity")
        require(item["sha256"] not in ledger, "Duplicate source in frozen gold ledger")
        ledger[item["sha256"]] = item
    require(set(ledger) == {member["sha256"] for member in manifest["members"]},
            "Frozen gold ledger differs from selected membership")
    validate_partitions(manifest)
    for member in manifest["members"]:
        selected = member.get("selected_gold")
        require(isinstance(selected, dict) and selected.get("sha256") == member["sha256"] and
                selected.get("artifact_sha256") == ledger[member["sha256"]]["artifact_sha256"],
                "Selected artifact differs from frozen gold ledger")
    features, labels, groups, receipts = [], [], [], []
    feature_names = None
    for member in manifest["members"]:
        if member["split"] != "train":
            continue
        selected = member.get("selected_gold")
        require(selected and selected.get("status") == "complete", "Training source lacks selected complete gold")
        require(selected.get("sha256") == member["sha256"], "Selected gold source differs from frozen membership")
        require(selected.get("mode") in ("digitalborn", "ocr"), "Unknown selected gold source mode")
        path = Path(selected["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == selected["artifact_sha256"], "Frozen gold artifact changed")
        gold = json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)
        require(gold["source_sha256"] == member["sha256"], "Gold source differs from manifest")
        package.validate(gold)
        bound = ledger[member["sha256"]]
        proof = gold.get("provenance", {}).get("structure_authoring")
        require(isinstance(proof, dict), "Gold lacks structural authoring proof")
        proof_identity = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False).encode("utf-8")).hexdigest()
        require(proof_identity == bound["structure_authoring_sha256"],
                "Structural authoring proof differs from frozen gold")
        require(package.text_identity(gold) == bound["text_identity"],
                "Canonical text identity differs from frozen gold")
        rows = packet(gold, selected["mode"], task)
        if feature_names is None:
            feature_names = rows["feature_names"]
        require(rows["feature_names"] == feature_names, "Mixed native feature schemas")
        if task == "spans":
            require(rows.get("classes") == ["no_boundary", "text", "heading", "caption", "instruction", "headnote", "marginal_note", "furniture", "page_label", "close_unowned"], "Unexpected role classes")
            for row in rows["rows"]:
                require(len(row["features"]) == len(feature_names) and all(math.isfinite(v) for v in row["features"]), "Invalid span features")
                require(type(row["class"]) is int and 0 <= row["class"] < 10, "Invalid span role")
                features.append(row["features"])
                labels.append(row["class"])
            kept = len(rows["rows"])
            excluded_keys = sorted(set(rows["unavailable_gold_starts"] + rows["unavailable_gold_ends"]))
        elif task == "scope_openings":
            for row in rows["rows"]:
                require(len(row["features"]) == len(feature_names) and all(math.isfinite(v) for v in row["features"]), "Invalid scope opening features")
                require(type(row["count"]) is int and row["count"] >= 0, "Invalid scope opening count")
                features.append(row["features"])
                labels.append(row["count"])
            kept = len(rows["rows"])
            excluded_keys = rows.get("unavailable_targets", [])
        elif task in ("scopes", "membership"):
            by_unit = OrderedDict()
            for row in rows["rows"]:
                by_unit.setdefault((row["unit"], row.get("axis")), []).append(row)
            for candidates in by_unit.values():
                require(sum(row["positive"] is True for row in candidates) == 1, "Identity query lacks one gold choice")
                if task == "membership":
                    require(sum(row["candidate"] is None for row in candidates) == 1, "New-unit choice missing or duplicated")
                require(len({row["candidate"] for row in candidates}) == len(candidates), "Repeated scope candidate")
                for row in candidates:
                    require(len(row["features"]) == len(feature_names) and all(math.isfinite(v) for v in row["features"]), "Invalid scope features")
                    features.append(row["features"])
                    labels.append(int(row["positive"]))
                groups.append(len(candidates))
            kept = len(by_unit)
            excluded_keys = rows.get("unavailable_parent_targets", []) + rows.get("unavailable_opening_targets", [])
        else:
            by_unit = OrderedDict()
            for row in rows["rows"]:
                by_unit.setdefault(row["unit"], []).append(row)
            excluded = set(rows["unavailable_parent_units"])
            kept = 0
            for ident, candidates in by_unit.items():
                require(sum(row["candidate"] is None for row in candidates) == 1, "Root candidate missing or duplicated")
                if ident in excluded:
                    require(not any(row["positive"] for row in candidates), "Excluded gold parent also has a positive candidate")
                    continue
                require(sum(row["positive"] is True for row in candidates) == 1, "Training query lacks exactly one gold parent")
                # Root-only queries carry no attachment choice and no ranking gradient.
                if len(candidates) == 1:
                    continue
                for row in candidates:
                    vector = row["values"] + row["unit_values"] + row["candidate_values"]
                    require(len(vector) == len(feature_names) and all(math.isfinite(v) for v in vector), "Invalid feature vector")
                    features.append(vector)
                    labels.append(int(row["positive"]))
                groups.append(len(candidates))
                kept += 1
            excluded_keys = sorted(excluded)
        receipts.append({"sha256": member["sha256"], "family_id": member["family_id"],
            "gold_sha256": selected["artifact_sha256"], "examples": kept, "unavailable_targets": excluded_keys,
            "unannotated_starts": rows.get("unannotated_starts", [])})
    require(features and any(labels), "No supervised structural examples")
    import lightgbm as lgb
    import numpy as np
    dataset = lgb.Dataset(np.asarray(features, dtype=np.float32), label=labels, group=groups if task in ("outline", "scopes", "membership") else None, feature_name=feature_names)
    recipe = {"objective": "multiclass" if task == "spans" else "regression" if task == "scope_openings" else "lambdarank", "num_threads": 1,
        "num_leaves": 15, "learning_rate": 0.05, "seed": 0, "verbosity": -1}
    if task == "spans":
        recipe["num_class"] = 10
    model = lgb.train(recipe, dataset, num_boost_round=200)
    exported = model.dump_model()
    exported["feature_schema"] = SCHEMAS[task]
    exported["training_recipe"] = {**recipe, "num_boost_round": 200}
    exported["training_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    exported["training_sources"] = receipts
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".pending")
    try:
        pending.write_text(json.dumps(exported, separators=(",", ":")), encoding="utf-8")
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps({"model": str(output), "sources": len(receipts), "task": task, "examples": sum(row["examples"] for row in receipts),
        "unavailable_targets": sum(len(row["unavailable_targets"]) for row in receipts)}))


if __name__ == "__main__":
    main()
