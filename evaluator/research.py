import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluator.client import ClientError, request
from evaluator.search import Journal

SYSTEM_MESSAGE = (
    "Follow the arena repository instructions and the supplied campaign. "
    "Internet access is only for general research ideas. Never send dataset "
    "examples or knowledge-base contents to external services. All model "
    "requests must use the fixed internal model route."
)


class ResearchError(ValueError):
    pass


def private_status(url: str, token_file: Path) -> str | None:
    try:
        result = request("GET", url, token_file, timeout=60)
    except ClientError as error:
        if error.status_code == 404 and error.error_code == "no_evaluation":
            return None
        raise ResearchError("private status unavailable; search was not continued") from None
    status = result.get("status")
    if (
        set(result) != {"status"} or not isinstance(status, str)
        or status not in {"running", "complete", "failed"}
    ):
        raise ResearchError("invalid private status; search was not continued")
    return status


def continuation(status: dict[str, Any], *, private_required: bool = True) -> str:
    if status["stopped"]:
        instruction = (
            "The journal records a stop. Do not propose new solutions and do not start "
            "new evaluations. Wait only for evaluations already running, then select and "
            "finalize the frozen selection using existing evidence. "
            "Do not invent a human instruction."
        )
    elif status["remaining_proposals"]:
        instruction = (
            f"The search has {status['remaining_proposals']} proposal slots remaining "
            f"({status['proposed']}/{status['max_solutions']} proposed). Continue the search "
            "under the existing campaign conditions. Do not finalize or submit private yet. "
            "Elapsed time, tokens and an adequate finalist count are not stop conditions. "
            "Before another local variation, delegate an independent search "
            "for an untested mechanism "
            "in a fresh context and register and test an alternative allowed by the campaign. "
            "A plateau or a claimed quality ceiling requires this change of direction, not a stop. "
            "Read the methods memory and keep deferred and untested "
            "distinct from measured_negative; "
            "a negative conclusion requires its snapshot, dataset and measurement evidence."
        )
    else:
        instruction = (
            "The proposal budget is exhausted. Do not propose new solutions. Finish planned "
            "evaluations of registered solutions, then select and finalize the frozen batch."
        )
    if status["running"]:
        instruction += (
            f" The journal still has running measurements: {json.dumps(status['running'])}. "
            "Collect train results with evaluator.train result; collect public status and "
            "record its permitted result with evaluator.search finish. Service completion "
            "does not update the journal automatically. If jobs are still running, use "
            "completion callbacks, native process waits or evaluator.wait --process ID=PID "
            "under common/long-runs.md, then collect the saved evaluation status. "
            "Do not end turns or repeatedly call the model just to poll status."
        )
    completion = (
        " Campaign completion requires its single private submission to complete; "
        "after authorized finalization, wait for its status only. "
        "The private service must authorize an early stop. "
        "Never request private scores or create a replacement private attempt. "
        if private_required else
        " The private split is empty. Save the final selection and finalize it in the journal. "
        "Do not submit to a private service or invent private scores. "
    )
    return (
        "Your conversation turn ended, but the campaign is not complete. " + instruction
        + completion
        + "If infrastructure prevents progress, report the exact blocker "
        "instead of claiming success."
    )


def wait_private(
    private: str | None,
    status: dict[str, Any],
    read_private: Callable[[], str | None],
    wait: Callable[[float], None],
) -> bool:
    if private is None:
        return False
    if (
        status["running"] or not status.get("finalization")
        or (status["remaining_proposals"] and not status["stopped"])
    ):
        raise ResearchError("private attempt already exists before valid search finalization")
    while private == "running":
        wait(60)
        private = read_private()
    if private != "complete":
        raise ResearchError(f"private evaluation is {private or 'unavailable'}; search incomplete")
    return True


def selection_complete(status: dict[str, Any]) -> bool:
    return bool(
        status.get("finalization") and not status["running"]
        and (not status["remaining_proposals"] or status["stopped"])
    )


def run_search(
    agent_factory: Callable[[], Any],
    prompt: str,
    journal: Journal,
    read_private: Callable[[], str | None] | None,
    check_gateway: Callable[[], None],
    wait: Callable[[float], None] = time.sleep,
    *,
    check_readiness: Callable[[int], object] | None = None,
    system_message: str = SYSTEM_MESSAGE,
) -> dict[str, Any]:
    previous = journal.status()
    if read_private is not None:
        if check_readiness is None:
            raise ResearchError("private search readiness is required")
        try:
            check_readiness(previous["max_solutions"])
        except ClientError:
            raise ResearchError(
                "private search admission is unavailable or mismatched; search was not continued"
            ) from None
        complete = wait_private(read_private(), previous, read_private, wait)
        message = "Private evaluation complete; no new researcher turn was started."
    else:
        complete = selection_complete(previous)
        message = "Search selection complete; no new researcher turn was started."
    if complete:
        return {"completed": True, "failed": False, "final_response": message}
    agent = agent_factory()
    history = None
    idle_turns = 0
    try:
        while True:
            result = agent.run_conversation(
                prompt, system_message=system_message, conversation_history=history,
            )
            check_gateway()
            if result.get("failed") or not result.get("completed"):
                raise ResearchError("Researcher failed; search remains incomplete")
            status = journal.status()
            if read_private is None:
                complete = selection_complete(status)
            else:
                complete = wait_private(read_private(), status, read_private, wait)
            if complete:
                return result
            idle_turns = idle_turns + 1 if status == previous else 0
            if idle_turns >= 2:
                raise ResearchError(
                    "two completed turns made no journal progress; search incomplete"
                )
            history = result.get("messages")
            if not isinstance(history, list) or not history:
                raise ResearchError(
                    "Researcher returned no conversation history; search incomplete"
                )
            prompt = continuation(status, private_required=read_private is not None)
            previous = status
    finally:
        agent.close()

