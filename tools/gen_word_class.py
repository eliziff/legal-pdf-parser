"""Generate the BMP word class Python's Unicode \\w matches.

The grammar tables compile with re.ASCII for \\d-stability, but the
source grammars were battle-tested under Unicode \\w/\\b. Both loaders
expand \\w/\\W/\\b/\\B against this generated class so compiled behavior
matches the source on every BMP script. Regenerate only deliberately —
the class is frozen so both runtimes stay identical regardless of their
own Unicode table versions.

    python -X utf8 tools/gen_word_class.py
"""

import re
import sys
import unicodedata


def main() -> int:
    word = re.compile(r"\w")
    ranges: list[tuple[int, int]] = []
    start = None
    prev = None
    for cp in range(0x10000):
        ch = chr(cp)
        if 0xD800 <= cp <= 0xDFFF:
            matches = False
        else:
            matches = bool(word.match(ch))
        if matches:
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            ranges.append((start, prev))
            start = None
    if start is not None:
        ranges.append((start, prev))

    def fmt(cp: int) -> str:
        if cp == 0x5F:
            return "_"
        if 0x30 <= cp <= 0x39 or 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
            return chr(cp)
        return f"\\u{cp:04x}"

    parts = []
    for lo, hi in ranges:
        if lo == hi:
            parts.append(fmt(lo))
        elif hi == lo + 1:
            parts.append(fmt(lo) + fmt(hi))
        else:
            parts.append(f"{fmt(lo)}-{fmt(hi)}")
    fragment = "".join(parts)
    sys.stdout.write(
        f"# python {sys.version.split()[0]}, unicodedata {unicodedata.unidata_version}\n"
    )
    sys.stdout.write(f"# ranges: {len(ranges)}, fragment chars: {len(fragment)}\n")
    sys.stdout.write(fragment + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
