"""Audit train augmentation geometry independently using OpenCV points."""

import json

import cv2
import numpy as np

from scripts.augment import distort
from scripts.diagnose import ROOT, frozen_hashes, sample
from solution.geometry import transform_box


def main():
    records = sample()
    assert frozen_hashes(records) == json.loads(
        (ROOT / "frozen.json").read_text()
    )
    max_roundtrip = 0.0
    checked = 0
    for record in records:
        page_id = record["id"]
        image = cv2.imread(f"data/train/images/{page_id}.jpg")
        height, width = image.shape[:2]
        with open(f"data/train/truth/{page_id}.json") as stream:
            truth = json.load(stream)
        matrix = np.array(record["distortion"]["matrix"])
        inverse = cv2.invertAffineTransform(matrix)
        for table in truth["tables"]:
            for item in [table] + table["cells"]:
                x0, y0, x1, y1 = item["bbox"]
                assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
                points = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
                original = cv2.transform(points, inverse)
                restored = cv2.transform(original, matrix)
                max_roundtrip = max(
                    max_roundtrip, float(abs(restored - points).max())
                )
                expected = cv2.transform(points, matrix)[0]
                bounds = [*expected.min(axis=0), *expected.max(axis=0)]
                bounds = np.clip(bounds, 0, [width, height, width, height])
                assert np.allclose(
                    transform_box(item["bbox"], matrix, width, height), bounds
                )
                checked += 1
    # Exercise the actual augmentation entry point against a visible marker.
    errors = []
    for index in range(30):
        image = np.full((400, 500, 3), 255, np.uint8)
        image[100:300, 120:380] = 0
        truth = {"tables": [{"bbox": [120, 100, 380, 300], "cells": []}]}
        encoded, transformed, _ = distort(image, truth, f"geometry_{index}")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        yy, xx = np.where(decoded < 100)
        observed = np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
        errors.append(
            float(abs(observed - transformed["tables"][0]["bbox"]).max())
        )
    assert max(errors) < 3
    result = dict(
        pages=len(records),
        boxes_checked=checked,
        point_roundtrip_max_error=max_roundtrip,
        marker_trials=30,
        marker_max_boundary_error_pixels=max(errors),
        frozen_files_unchanged=True,
        limitation="Original PDF annotations are not retained; this checks the affine implementation, stored bounds and pixel alignment, not a new source annotation audit.",
    )
    (ROOT / "geometry.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
