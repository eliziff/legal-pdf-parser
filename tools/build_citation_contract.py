from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    table_root = arguments.oracle_root.resolve() / "data" / "grammar-tables"
    texts: list[str] = []
    for path in sorted(table_root.glob("*.json")):
        table = json.loads(path.read_text(encoding="utf-8"))
        for entry in table.get("entries", []):
            for vector in entry.get("vectors", []):
                value = vector.get("input")
                if isinstance(value, str) and value.strip():
                    texts.append(value)
    seeds = [
        "Groia v Law Society, 2018 SCC 27, [2018] 1 SCR 772 at paras 64–67.",
        "1068490 Ontario Ltd. V. Marlin Center Mobile Homes Inc. and Howard Geisler, 2001 CarswellOnt 4564, at para. 21 (Book of Authorities TAB 17)",
        "R v Jordan, 2016 SCC 27; R v Cody, 2017 SCC 31.",
        "See R v Jordan, 2016 SCC 27, at paras 5–7; but see R v Cody, 2017 SCC 31, at para 9.",
        "Kent Roach, ‘Constitutional Remedies’ (2020) 45 Queen's LJ 123 [Roach].",
        "Criminal Code, RSC 1985, c C-46, s 7; Canadian Charter of Rights and Freedoms, s 11(b).",
        "Ibid at para 4.",
        "Smith, supra note 3, at 12.",
        "https://example.org/a;b?x=1;2",
        "",
    ]
    seen: set[str] = set()
    ordered = []
    for text in [*texts, *seeds]:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    combinations = []
    citation_like = [text for text in ordered if any(char.isdigit() for char in text)][:40]
    for left, right in zip(citation_like, citation_like[1:]):
        combinations.extend((f"{left}; {right}", f"{left}. See also {right}"))
    cases = [
        {"mode": mode, "text": text}
        for text in [*ordered, *combinations]
        for mode in ("conservative", "recall_first")
    ]
    value = {
        "schema_version": "legalpdf.contract-input.v1",
        "operation": "citation_batch",
        "cases": cases,
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"citation contract: {len(cases)} cases; {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
