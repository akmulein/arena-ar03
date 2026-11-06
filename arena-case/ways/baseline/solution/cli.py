import argparse
import json
import time
from pathlib import Path

import cv2

from solution.pipeline import solve
from solution.postprocess import to_html


def visualize(image, result):
    image = image.copy()
    for table in result["tables"]:
        for cell in table["cells"]:
            x0, y0, x1, y1 = map(round, cell["bbox"])
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 160, 0), 1)
        x0, y0, x1, y1 = map(round, table["bbox"])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        parser.error("cannot read input image")
    start = time.perf_counter()
    result = solve(image)
    elapsed = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if args.html:
        args.html.write_text(to_html(result))
    if args.visualization:
        if not cv2.imwrite(str(args.visualization), visualize(image, result)):
            raise OSError("cannot write visualization")
    print(json.dumps({"tables": len(result["tables"]), "seconds": elapsed}))


if __name__ == "__main__":
    main()
