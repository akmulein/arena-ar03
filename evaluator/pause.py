import argparse
import json
import sys

from evaluator.service import Settings, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Pause private after active examples finish")
    parser.add_argument("--reason", required=True)
    arguments = parser.parse_args()
    try:
        result = create_app(Settings.from_env()).state.pause_private(arguments.reason)
    except Exception as error:
        print(json.dumps({"status": "pause_rejected", "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
