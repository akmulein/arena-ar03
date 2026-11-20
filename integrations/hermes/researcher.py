import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from evaluator.client import search_readiness
from evaluator.reliability import request_timeout
from evaluator.research import SYSTEM_MESSAGE, ResearchError, private_status, run_search
from evaluator.search import Journal
from integrations.hermes import runtime


def main(
    argv: Sequence[str] | None = None,
    *,
    model: str | None = None,
    parameters: dict[str, Any] | None = None,
    default_prompt: Path | None = None,
    config_path: Path | None = None,
    require_private: bool = False,
) -> int:
    parser = argparse.ArgumentParser(description="Run an arena search with the Hermes adapter")
    configured_model = model or os.environ.get("ARENA_MODEL")
    parser.add_argument("--model", default=configured_model, required=not configured_model)
    parser.add_argument("--parameters-json")
    parser.add_argument("--config-file", type=Path, default=config_path)
    parser.add_argument("--credentials-file", type=Path, default=os.environ.get(
        "ARENA_MODEL_ACCESS_FILE", "/run/arena/model-access.json"
    ))
    parser.add_argument("--prompt-file", type=Path, default=os.environ.get(
        "ARENA_RESEARCH_PROMPT_FILE", default_prompt or "/workspace/prompts/start.md"
    ))
    parser.add_argument("--cwd", type=Path, default="/workspace")
    parser.add_argument("--journal", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--private-url")
    if not require_private:
        mode.add_argument("--no-private", action="store_true")
    for flag, kind in (
        ("--private-token-file", Path), ("--private-finalists", int),
        ("--private-output-count", int), ("--private-release-id", str),
    ):
        parser.add_argument(flag, type=kind, required=require_private)
    args = parser.parse_args(argv)
    for name in (
        "private_token_file", "private_finalists", "private_output_count", "private_release_id",
    ):
        value = getattr(args, name)
        flag = "--" + name.replace("_", "-")
        if args.private_url and value is None:
            parser.error(f"{flag} is required with --private-url")
        if not args.private_url and value is not None:
            parser.error(f"{flag} cannot be used with --no-private")
    if model is not None and args.model != model:
        parser.error("--model differs from the selected profile")
    try:
        access = runtime.credentials(args.credentials_file)
        if access.get("model", args.model) != args.model:
            raise ValueError("credentials model differs from the selected model")
        selected_parameters = (
            json.loads(args.parameters_json) if args.parameters_json is not None
            else parameters if parameters is not None else access.get("parameters", {})
        )
        if not isinstance(selected_parameters, dict):
            raise ValueError("model parameters must be an object")
        if parameters is not None and selected_parameters != parameters:
            raise ValueError("model parameters differ from the selected profile")
        if "parameters" in access and selected_parameters != access["parameters"]:
            raise ValueError("model parameters differ from credentials")
        timeout = access.get("model_request_timeout", request_timeout(
            access.get("model_timeout", 21600)
        ))
        prompt = args.prompt_file.read_text()
        home = Path(os.environ.get("HERMES_HOME", "/workspace/.hermes"))
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                runtime.prepare_home(
                    home, model=args.model, parameters=selected_parameters,
                    config_path=args.config_file,
                )
                state = runtime.install(
                    access["model_base_url"], timeout, access.get("api_key", "arena"),
                    model=args.model, parameters=selected_parameters,
                )
                agent_class = import_module("run_agent").AIAgent

                options: dict[str, Any] = {}
                if "reasoning_effort" in selected_parameters:
                    options["reasoning_config"] = {
                        "enabled": True, "effort": selected_parameters["reasoning_effort"],
                    }
                result = run_search(
                    lambda: agent_class(
                        model=args.model,
                        base_url=runtime.CONFIG_BASE_URL,
                        api_key="arena",
                        provider="custom",
                        api_mode="chat_completions",
                        enabled_toolsets=["terminal", "file", "delegation", "web", "todo"],
                        max_iterations=10000,
                        cwd=str(args.cwd.resolve()),
                        skip_memory=True,
                        skip_background_review=True,
                        skip_context_files=True,
                        save_trajectories=False,
                        quiet_mode=True,
                        **options,
                    ),
                    prompt, Journal(args.journal),
                    (lambda: private_status(args.private_url, args.private_token_file))
                    if args.private_url else None,
                    state.check,
                    check_readiness=(lambda maximum: search_readiness(
                        args.private_url, args.private_token_file, timeout=60,
                        require_campaign=True, max_solutions=maximum,
                        finalists=args.private_finalists, output_count=args.private_output_count,
                        release_id=args.private_release_id,
                    )) if args.private_url else None,
                    system_message=SYSTEM_MESSAGE + (
                        " For Hermes delegation use context_mode=fresh for independent ideas "
                        "and context_mode=fork for work inheriting the current method."
                    ),
                )
        print(json.dumps({"answer": result["final_response"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        reason = str(error) if isinstance(error, ResearchError) else "Hermes researcher failed"
        print(json.dumps({"status": "incomplete", "reason": reason}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
