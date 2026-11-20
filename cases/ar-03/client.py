"""Research-side client: journal mutation only through the trusted operator API."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from evaluator.client import pack_candidate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("status", "propose", "evaluate", "result", "finalize", "progress")
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--parent", action="append", default=[])
    parser.add_argument("--hypothesis")
    parser.add_argument("--change")
    parser.add_argument(
        "--stage", choices=("detection", "structure", "ocr", "end-to-end"), default="end-to-end"
    )
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--split", choices=("train", "public"), default="train")
    parser.add_argument("--measurement")
    parser.add_argument("--example-id", action="append")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_request(args):
    body = None
    route, method = args.command, "GET"
    if args.command == "propose":
        body = {
            "candidate": pack_candidate("proposal", args.directory).model_dump(),
            "parents": args.parent,
            "hypothesis": args.hypothesis,
            "change": args.change,
            "stage": args.stage,
        }
        method = "POST"
    elif args.command == "evaluate":
        body, method = {"candidate_id": args.candidate[0], "split": args.split}, "POST"
        if args.example_id is not None:
            body["example_ids"] = args.example_id
    elif args.command == "finalize":
        body, method = {"candidate_ids": args.candidate}, "POST"
    elif args.command == "result":
        route = "result/" + args.measurement
    return method, route, body


def fetch_result(client, args, token):
    method, route, body = build_request(args)
    while True:
        response = client.request(
            method,
            os.environ["AR03_API"] + "/" + route,
            json=body,
            headers={"Authorization": "Bearer " + token},
        )
        if args.command == "evaluate" and response.status_code == 503:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(1, min(int(retry_after), 60))
                print(f"Evaluator busy; retrying the same candidate in {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
        response.raise_for_status()
        result = response.json()
        if not args.wait or result.get("status") not in ("RUNNING", "running"):
            return result
        time.sleep(2)


def output_result(args, result):
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False))
    if args.command == "result" and args.output:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "saved": str(args.output),
                    "score": result.get("score", result.get("metrics", {}).get("score")),
                }
            )
        )
    else:
        print(json.dumps(result, ensure_ascii=False))


def main():
    args = parse_args()
    token = Path("/run/secrets/submit-token").read_text().strip()
    with httpx.Client(trust_env=False, timeout=600) as client:
        result = fetch_result(client, args, token)
    output_result(args, result)


if __name__ == "__main__":
    main()
