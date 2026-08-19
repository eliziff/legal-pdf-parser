from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppdoc_lite.runtime import prepare_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an image for the native Paddle probe")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    feeds, (width, height) = prepare_image(args.image, backend="opencv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feeds["image"].tofile(args.output)
    print(json.dumps({"output": str(args.output), "width": width, "height": height}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
