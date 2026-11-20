import json
from pathlib import Path
from uuid import uuid4

import pytest

from evaluator.research_usage import collect_research_usage
from evaluator.trace_store import append_call
from evaluator.usage import add_usage, create_usage, seal_usage


def record(tmp_path: Path, tokens: int | None, *, run_id: str | None = None) -> tuple[str, str]:
    run_id = run_id or str(uuid4())
    usage_dir = tmp_path / "usage"
    if not (usage_dir / f"{run_id}.json").exists():
        create_usage(usage_dir, run_id)
    usage = None if tokens is None else {
        "prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens,
    }
    request_id = str(uuid4())
    add_usage(usage_dir, run_id, usage)
    append_call(tmp_path / "traces", run_id, {
        "attributes": {"request_id": request_id, "usage": usage},
        "start_time_ns": 1_000_000_000, "end_time_ns": 3_000_000_000,
        "inputs": {"secret_payload": "never export this"},
    })
    return run_id, request_id


def test_exact_calls_and_unknown_distribution_preserve_total(tmp_path: Path) -> None:
    run_id, first = record(tmp_path, 11)
    _, second = record(tmp_path, 17, run_id=run_id)
    _, third = record(tmp_path, 23, run_id=run_id)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [{"run_id": run_id, "request_id": first}],
        "S02": [{"run_id": run_id, "request_id": second}],
        "S03": [{"run_id": run_id, "request_id": second}],
    })
    assert result["total"]["usage"]["total_tokens"] == 51
    assert result["assigned"]["usage"]["total_tokens"] == 11
    assert result["shared"]["usage"]["total_tokens"] == 17
    assert result["unattributed"]["usage"]["total_tokens"] == 23
    assert result["candidates"]["S01"]["usage"]["total_tokens"] == 11
    assert result["candidates"]["S02"]["usage"]["total_tokens"] == 0
    assert not result["candidates"]["S02"]["tokens_complete"]
    assert not result["attribution_complete"]
    assert result["unattributed"]["refs"] == [{"run_id": run_id, "request_id": third}]
    assert "never export" not in json.dumps(result)


def test_scope_attribution_uses_ledger_and_deduplicates_refs(tmp_path: Path) -> None:
    run_id, request_id = record(tmp_path, 11)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id, run_id], {
        "S01": [{"run_id": run_id}, {"run_id": run_id},
                {"run_id": run_id, "request_id": request_id}],
    })
    assert result["total"]["usage"]["total_tokens"] == 11
    assert result["candidates"]["S01"]["usage"]["total_tokens"] == 11
    assert result["candidates"]["S01"]["model_seconds"] == 2
    assert result["attribution_complete"]
    assert result["total"]["tokens_complete"]


def test_missing_trace_keeps_trusted_scope_accounting(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "missing", [run_id], {
        "S01": [{"run_id": run_id}],
    })
    assert result["total"]["usage"]["total_tokens"] == 11
    assert result["candidates"]["S01"]["usage"]["total_tokens"] == 11
    assert not result["total"]["time_complete"]
    assert result["candidates"]["S01"]["tokens_complete"]


def test_unreported_and_active_scopes_cannot_claim_complete_usage(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, None)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [{"run_id": run_id}],
    })
    assert result["total"]["usage"]["unreported_attempts"] == 1
    assert not result["total"]["tokens_complete"]
    assert not result["candidates"]["S01"]["tokens_complete"]


def test_missing_ledger_is_unknown_not_measured_zero(tmp_path: Path) -> None:
    run_id = str(uuid4())
    result = collect_research_usage(tmp_path / "missing", tmp_path / "traces", [run_id], {})
    assert result["missing_run_ids"] == [run_id]
    assert not result["total"]["tokens_complete"]
    assert not result["attribution_complete"]


def test_unknown_explicit_request_and_outside_scope_are_rejected(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 1)
    for ref in ({"run_id": run_id, "request_id": str(uuid4())}, {"run_id": str(uuid4())}):
        with pytest.raises(ValueError):
            collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
                "S01": [ref],
            })


def test_duplicate_trace_delivery_is_not_an_extra_attempt(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    path = next((tmp_path / "traces" / run_id).glob("*.json"))
    duplicate = path.with_name(f"{uuid4()}.json")
    duplicate.write_bytes(path.read_bytes())
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {})
    assert result["total"]["usage"]["attempts"] == 1
    assert result["unattributed"]["usage"]["attempts"] == 1
    assert result["total"]["model_seconds"] == 2


def test_conflicting_trace_ids_fail_loud(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    path = next((tmp_path / "traces" / run_id).glob("*.json"))
    duplicate = json.loads(path.read_text())
    duplicate["end_time_ns"] += 10
    append_call(tmp_path / "traces", run_id, duplicate)
    with pytest.raises(ValueError, match="Conflicting"):
        collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {})


def test_manual_usage_in_assignment_is_rejected(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    with pytest.raises(ValueError):
        collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
            "S01": [{"run_id": run_id, "total_tokens": "900"}],
        })


def test_scope_and_request_overlap_remains_shared_once(tmp_path: Path) -> None:
    run_id, first = record(tmp_path, 11)
    record(tmp_path, 17, run_id=run_id)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [{"run_id": run_id}],
        "S02": [{"run_id": run_id, "request_id": first}],
    })
    assert result["total"]["usage"]["total_tokens"] == 28
    assert result["assigned"]["usage"]["total_tokens"] == 17
    assert result["shared"]["usage"]["total_tokens"] == 11
    assert result["unattributed"]["usage"]["total_tokens"] == 0
    assert result["candidates"]["S01"]["usage"]["total_tokens"] == 17
    assert not result["candidates"]["S01"]["tokens_complete"]
    assert not result["attribution_complete"]


def test_unknown_candidate_attribution_does_not_claim_measured_zero(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [],
    })
    assert not result["candidates"]["S01"]["tokens_complete"]
    assert result["unattributed"]["usage"]["total_tokens"] == 11


def test_mismatched_ledger_and_trace_are_not_silently_merged(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    path = next((tmp_path / "traces" / run_id).glob("*.json"))
    call = json.loads(path.read_text())
    call["attributes"]["usage"] = {
        "prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12,
    }
    path.write_text(json.dumps(call))
    with pytest.raises(ValueError, match="exceeds"):
        collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {})


def test_residual_tokens_without_attempts_are_a_ledger_mismatch(tmp_path: Path) -> None:
    run_id, request_id = record(tmp_path, 11)
    path = next((tmp_path / "traces" / run_id).glob("*.json"))
    call = json.loads(path.read_text())
    call["attributes"]["usage"] = {
        "prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1,
    }
    path.write_text(json.dumps(call))
    with pytest.raises(ValueError, match="Inconsistent"):
        collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
            "S01": [{"run_id": run_id, "request_id": request_id}],
        })


def test_no_ledger_ids_cannot_establish_measured_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        collect_research_usage(tmp_path / "usage", tmp_path / "traces", [], {})


def test_live_scope_is_incomplete_even_when_observed_calls_are_assigned(tmp_path: Path) -> None:
    run_id, _ = record(tmp_path, 11)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [{"run_id": run_id}],
    })
    assert not result["attribution_complete"]
    assert not result["assigned"]["tokens_complete"]
    assert not result["assigned"]["time_complete"]
    assert result["scope"] == "declared_research_runs"


def test_empty_sealed_scope_is_known_zero_with_scope_provenance(tmp_path: Path) -> None:
    run_id = str(uuid4())
    create_usage(tmp_path / "usage", run_id)
    seal_usage(tmp_path / "usage", run_id)
    result = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [{"run_id": run_id}],
    })
    assert result["total"]["tokens_complete"]
    assert result["attribution_complete"]
    assert result["candidates"]["S01"]["requested_refs"] == [{"run_id": run_id}]
    empty = collect_research_usage(tmp_path / "usage", tmp_path / "traces", [run_id], {
        "S01": [],
    })
    assert not empty["attribution_complete"]
