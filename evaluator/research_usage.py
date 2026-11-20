import fcntl
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from evaluator.reliability import add_usage, empty_usage
from evaluator.trace_store import iter_calls
from evaluator.usage import _load, _paths


def _uuid(value: str) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Research usage IDs must be canonical UUIDs")
    return value


def _bucket() -> dict[str, Any]:
    return {
        "usage": empty_usage(), "model_seconds": 0.0,
        "tokens_complete": True, "time_complete": True, "refs": [],
    }


def _add(
    target: dict[str, Any], usage: dict[str, int], seconds: float,
    ref: dict[str, str], *, tokens_complete: bool, time_complete: bool,
) -> None:
    for key, count in usage.items():
        target["usage"][key] += count
    target["model_seconds"] += seconds
    target["tokens_complete"] &= tokens_complete
    target["time_complete"] &= time_complete
    target["refs"].append(ref)


def _ledger(directory: Path, run_id: str) -> dict[str, Any]:
    path, lock_path = _paths(directory, run_id)
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        value = _load(path)
    return {
        "usage": {key: int(value[key]) for key in empty_usage()},
        "closed": value["closed"], "active_requests": value["active_requests"],
    }


def _calls(directory: Path, run_id: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for call in iter_calls(directory, run_id):
        attributes = call.get("attributes", {})
        request_id = _uuid(attributes.get("request_id"))
        usage = empty_usage()
        add_usage(usage, attributes.get("usage"))
        start, end = call.get("start_time_ns"), call.get("end_time_ns")
        if type(start) is int and type(end) is int and 0 <= start <= end:
            valid_time, seconds = True, (end - start) / 1e9
        else:
            valid_time, seconds = False, 0.0
        item = {
            "usage": usage, "model_seconds": seconds,
            "time_complete": valid_time,
        }
        if request_id in result and result[request_id] != item:
            raise ValueError("Conflicting research trace request IDs")
        result[request_id] = item
    return result


def collect_research_usage(
    usage_dir: Path,
    trace_dir: Path,
    run_ids: Sequence[str],
    assignments: Mapping[str, Sequence[dict[str, str]]],
) -> dict[str, Any]:
    runs = sorted({_uuid(run_id) for run_id in run_ids})
    if not runs:
        raise ValueError("Research usage needs at least one run ID")
    scope_owners: dict[str, set[str]] = {run_id: set() for run_id in runs}
    request_owners: dict[tuple[str, str], set[str]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_id, refs in assignments.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Research usage candidate ID must be nonempty text")
        if not isinstance(refs, (list, tuple)):
            raise ValueError("Research usage assignments must be reference lists")
        candidates[candidate_id] = {**_bucket(), "requested_refs": []}
        requested: set[tuple[str, str | None]] = set()
        if not refs:
            candidates[candidate_id]["tokens_complete"] = False
            candidates[candidate_id]["time_complete"] = False
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) not in (
                {"run_id"}, {"run_id", "request_id"},
            ):
                raise ValueError("Research usage references accept only run_id and request_id")
            run_id = _uuid(ref["run_id"])
            if run_id not in scope_owners:
                raise ValueError("Assigned run ID is outside the research runs")
            if "request_id" in ref:
                key = run_id, _uuid(ref["request_id"])
                request_owners.setdefault(key, set()).add(candidate_id)
            else:
                scope_owners[run_id].add(candidate_id)
            request_key = run_id, ref.get("request_id")
            if request_key not in requested:
                candidates[candidate_id]["requested_refs"].append(dict(ref))
                requested.add(request_key)
    total, assigned, shared, unattributed = (_bucket() for _ in range(4))
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    for run_id in runs:
        calls = _calls(trace_dir, run_id)
        for scope, request_id in request_owners:
            if scope == run_id and request_id not in calls:
                raise ValueError("Assigned research request ID was not found in the trace")
        try:
            ledger = _ledger(usage_dir, run_id)
        except FileNotFoundError:
            missing.append(run_id)
            total["tokens_complete"] = False
            total["time_complete"] = False
            owners = scope_owners[run_id] | set().union(*(
                values for (scope, _), values in request_owners.items() if scope == run_id
            ))
            for owner in owners:
                candidates[owner]["tokens_complete"] = False
                candidates[owner]["time_complete"] = False
            continue
        trace_usage = empty_usage()
        seconds = 0.0
        times_complete = True
        for call in calls.values():
            for key in trace_usage:
                trace_usage[key] += call["usage"][key]
            seconds += call["model_seconds"]
            times_complete &= call["time_complete"]
        residual = {key: ledger["usage"][key] - trace_usage[key] for key in trace_usage}
        if any(value < 0 for value in residual.values()):
            raise ValueError("Research trace accounting exceeds the trusted ledger")
        if residual["unreported_attempts"] > residual["attempts"] or (
            residual["attempts"] == residual["unreported_attempts"]
            and residual["total_tokens"]
        ):
            raise ValueError("Inconsistent research ledger and trace accounting")
        closed = bool(ledger["closed"] and not ledger["active_requests"])
        trace_complete = all(value == 0 for value in residual.values())
        tokens_complete = closed and not ledger["usage"]["unreported_attempts"]
        time_complete = closed and trace_complete and times_complete
        _add(total, ledger["usage"], seconds, {"run_id": run_id},
             tokens_complete=tokens_complete, time_complete=time_complete)
        run_records.append({
            "run_id": run_id, **ledger, "trace_complete": trace_complete,
            "model_seconds": seconds, "time_complete": time_complete,
        })
        pieces = [
            ({"run_id": run_id, "request_id": request_id}, call,
             scope_owners[run_id] | request_owners.get((run_id, request_id), set()))
            for request_id, call in sorted(calls.items())
        ]
        if any(residual.values()):
            pieces.append(({"run_id": run_id}, {
                "usage": residual, "model_seconds": 0.0, "time_complete": False,
            }, scope_owners[run_id]))
        for ref, call, owners in pieces:
            piece_complete = not call["usage"]["unreported_attempts"]
            if "request_id" not in ref:
                piece_complete &= closed
            target = assigned if len(owners) == 1 else shared if owners else unattributed
            _add(target, call["usage"], call["model_seconds"], ref,
                 tokens_complete=piece_complete, time_complete=call["time_complete"])
            if len(owners) == 1:
                owner = next(iter(owners))
                _add(candidates[owner], call["usage"], call["model_seconds"], ref,
                     tokens_complete=piece_complete, time_complete=call["time_complete"])
            elif owners:
                for owner in owners:
                    candidates[owner]["tokens_complete"] = False
                    candidates[owner]["time_complete"] = False
                    candidates[owner].setdefault("shared_refs", []).append(ref)
        if not closed:
            for owner in scope_owners[run_id]:
                candidates[owner]["tokens_complete"] = False
                candidates[owner]["time_complete"] = False
    research_closed = not missing and all(
        run["closed"] and not run["active_requests"] for run in run_records
    )
    if not research_closed:
        for bucket in (assigned, shared, unattributed):
            bucket["tokens_complete"] = False
            bucket["time_complete"] = False
    return {
        "source": "gateway_ledger", "scope": "declared_research_runs",
        "request_id_granularity": "attempt", "run_ids": runs,
        "total": total, "assigned": assigned, "shared": shared, "unattributed": unattributed,
        "candidates": candidates, "runs": run_records, "missing_run_ids": missing,
        "attribution_complete": research_closed and not any(shared["usage"].values())
        and not any(unattributed["usage"].values())
        and all(candidate["requested_refs"] for candidate in candidates.values()),
    }
