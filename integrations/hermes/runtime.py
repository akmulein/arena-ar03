import importlib
import json
import logging
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any

REVISION = "86fa9e6ca1de7ee71132d2be0103f3e37d1fa872"
CONFIG_BASE_URL = "http://arena-gateway.invalid/v1"
REQUEST_TIMEOUT = 131460
RUNNER_TIMEOUT = 604800


def model_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(parameters or {})
    if {"model", "messages", "stream", "tools", "tool_choice"} & result.keys():
        raise ValueError("request fields cannot be supplied as model parameters")
    return result


def gateway_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: message[key] for key in (
        "role", "content", "tool_calls", "tool_call_id", "reasoning_content"
    ) if key in message}
    if isinstance(result.get("tool_calls"), list):
        result["tool_calls"] = [{
            "id": call["id"],
            "type": call["type"],
            "function": {key: call["function"][key] for key in ("name", "arguments")},
        } for call in result["tool_calls"]]
    return result


def gateway_request(
    request: dict[str, Any], *, model: str, parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        request = request | extra
    result = {
        key: request[key]
        for key in ("messages", "tools", "tool_choice")
        if key in request
    }
    if "messages" in result:
        result["messages"] = [gateway_message(message) for message in result["messages"]]
    result.update(model_parameters(parameters))
    result.update(model=model, stream=False)
    return result


class ModelState:
    def __init__(self) -> None:
        self.failed = threading.Event()
        self.diagnostics: list[dict[str, str]] = []

    def check(self) -> None:
        if self.failed.is_set():
            raise RuntimeError("model gateway failed; further requests disabled")


def prepare_home(
    home: Path, *, model: str, parameters: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("a campaign model is required")
    parameters = model_parameters(parameters)
    configuration = (config_path or Path(__file__).with_name("config.yaml")).read_text()
    configuration = configuration.replace("@MODEL@", json.dumps(model))
    configuration = configuration.replace(
        "@REASONING_EFFORT@", json.dumps(parameters.get("reasoning_effort")),
    )
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(configuration, encoding="utf-8")
    proxy = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
    if proxy:
        os.environ.setdefault("DDGS_PROXY", proxy)
    os.environ.update({
        "HERMES_HOME": str(home),
        "HERMES_SINGLE_QUERY_SESSION": "1",
        "OPENAI_BASE_URL": CONFIG_BASE_URL,
        "OPENAI_API_KEY": "arena-runtime-injects-credentials",
        "OPENAI_MODEL": model,
        "HERMES_API_TIMEOUT": str(REQUEST_TIMEOUT),
        "HERMES_API_CALL_STALE_TIMEOUT": str(REQUEST_TIMEOUT),
        "TERMINAL_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_MAX_FOREGROUND_TIMEOUT": str(RUNNER_TIMEOUT),
        "TERMINAL_LIFETIME_SECONDS": str(RUNNER_TIMEOUT),
    })

def install(
    base_url: str, timeout: float, api_key: str = "arena", *, model: str,
    parameters: dict[str, Any] | None = None,
) -> ModelState:
    parameters = model_parameters(parameters)
    logging.disable(logging.CRITICAL)
    import httpx
    openai: Any = importlib.import_module("openai")
    completions: Any = importlib.import_module("openai.resources.chat.completions")
    Completions = completions.Completions
    AsyncCompletions = completions.AsyncCompletions

    state = ModelState()
    original_init = openai.OpenAI.__init__
    original_async_init = openai.AsyncOpenAI.__init__
    original_create = Completions.create
    original_async_create = AsyncCompletions.create

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
        original_init(self, *args, **kwargs)

    @wraps(original_async_init)
    def initialize_async(self, *args, **kwargs):
        kwargs.update(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        kwargs["http_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout)
        original_async_init(self, *args, **kwargs)

    @wraps(original_create)
    def create(self, *args, **kwargs):
        state.check()
        try:
            return original_create(
                self, *args, **gateway_request(kwargs, model=model, parameters=parameters),
            )
        except Exception:
            state.failed.set()
            raise RuntimeError("model gateway request failed") from None

    @wraps(original_async_create)
    async def create_async(self, *args, **kwargs):
        state.check()
        try:
            return await original_async_create(
                self, *args, **gateway_request(kwargs, model=model, parameters=parameters),
            )
        except Exception:
            state.failed.set()
            raise RuntimeError("model gateway request failed") from None

    openai.OpenAI.__init__ = initialize
    openai.AsyncOpenAI.__init__ = initialize_async
    Completions.create = create
    AsyncCompletions.create = create_async

    async_delegate: Any = importlib.import_module("tools.async_delegation")
    delegate: Any = importlib.import_module("tools.delegate_tool")
    AIAgent = importlib.import_module("run_agent").AIAgent

    async_delegate._STALE_IDLE_SECONDS = RUNNER_TIMEOUT
    async_delegate._STALE_IN_TOOL_SECONDS = RUNNER_TIMEOUT
    delegate._HEARTBEAT_STALE_CYCLES_IDLE = RUNNER_TIMEOUT // 30
    delegate._HEARTBEAT_STALE_CYCLES_IN_TOOL = RUNNER_TIMEOUT // 30
    AIAgent._try_recover_primary_transport = lambda *args, **kwargs: False
    AIAgent._dump_api_request_debug = lambda *args, **kwargs: None
    from integrations.hermes import delegation

    delegation.install(AIAgent, state.diagnostics, model=model, parameters=parameters)
    return state


def credentials(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("model_base_url"), str):
        raise ValueError("invalid credentials")
    return data
