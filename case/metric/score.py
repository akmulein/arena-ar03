"""Standalone evaluator for prediction and truth JSON files."""

import json
import sys

from evaluator.metrics import score

if __name__ == "__main__":
    with open(sys.argv[1]) as predicted, open(sys.argv[2]) as truth:
        try:
            prediction = json.load(predicted)
        except json.JSONDecodeError:
            prediction = None
        print(json.dumps(score(prediction, json.load(truth)), indent=2))
