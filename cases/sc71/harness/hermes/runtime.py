import os as os
from pathlib import Path
from typing import Any

from integrations.hermes import runtime as shared
from integrations.hermes.runtime import ModelState

CONFIG_BASE_URL = shared.CONFIG_BASE_URL
REQUEST_TIMEOUT = shared.REQUEST_TIMEOUT
REVISION = shared.REVISION
RUNNER_TIMEOUT = shared.RUNNER_TIMEOUT
credentials = shared.credentials
gateway_message = shared.gateway_message

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
PARAMETERS = {"reasoning_effort": "max"}


def gateway_request(request: dict[str, Any]) -> dict[str, Any]:
    return shared.gateway_request(request, model=MODEL, parameters=PARAMETERS)


def prepare_home(home: Path) -> None:
    shared.prepare_home(home, model=MODEL, parameters=PARAMETERS)


def install(base_url: str, timeout: float, api_key: str = "arena") -> ModelState:
    return shared.install(base_url, timeout, api_key, model=MODEL, parameters=PARAMETERS)
