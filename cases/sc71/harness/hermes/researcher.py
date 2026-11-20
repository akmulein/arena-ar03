from pathlib import Path

from cases.sc71.harness.hermes import runtime
from evaluator.research import ResearchError as ResearchError
from evaluator.research import continuation as continuation
from evaluator.research import private_status as private_status
from evaluator.research import run_search as run_search
from evaluator.research import wait_private as wait_private
from integrations.hermes.researcher import main as launch


def main() -> int:
    return launch(
        model=runtime.MODEL,
        parameters={"reasoning_effort": "max"},
        default_prompt=Path("/workspace/cases/sc71/campaigns/sc71-first-20261019/start.md"),
        require_private=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
