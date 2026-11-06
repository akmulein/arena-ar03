"""Arena adapter for the unchanged, private copy of AR-03 metrics."""

import json
import sys
from pathlib import Path

# This subprocess must not import Arena's identically named evaluator package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

from evaluator.metrics import score  # noqa: E402


def main():
    payload = json.load(sys.stdin)
    truth = json.loads(payload["example"]["reference"])
    try:
        prediction = json.loads(payload["answer"])
    except json.JSONDecodeError:
        prediction = None
    statistics = score(prediction, truth)
    print(json.dumps({"score": statistics["teds"], "statistics": statistics}))


if __name__ == "__main__":
    main()
