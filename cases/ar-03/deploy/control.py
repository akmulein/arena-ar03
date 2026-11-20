"""Operator-owned journal adapter; all scores come from stock Arena APIs."""

import importlib.util
import json
import re
import secrets
import tempfile
import threading
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx
from evaluator.candidates import materialize_candidate
from evaluator.client import pack_candidate
from evaluator.contracts import Candidate, Submission
from evaluator.search import Journal, Resources, SearchError
from evaluator.train import _digest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def validate_feedback(split, result):
    if split == "public":
        if set(result) - {"status", "scores", "evaluation_id"}:
            raise HTTPException(502, "invalid public feedback")
        for row in result.get("scores", []):
            if set(row) != {"candidate_id", "score"}:
                raise HTTPException(502, "invalid public feedback")
    if split == "private" and set(result) != {"status"}:
        raise HTTPException(502, "invalid private feedback")


def evaluation_ids(config, body):
    if set(body) - {"candidate_id", "split", "example_ids"}:
        raise HTTPException(422, "scores and evaluation metadata are operator-owned")
    split = body["split"]
    if split not in ("train", "public") or split not in config["services"]:
        raise HTTPException(409, "split is not enabled")
    if split == "public" and "example_ids" in body:
        raise HTTPException(422, "public subsets are forbidden")
    identifiers = config["train_ids"]
    if split == "public":
        return identifiers
    selected = body.get("example_ids", config["evaluation_ids"])
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(key, str) for key in selected)
        or len(set(selected)) != len(selected)
        or set(selected) - set(identifiers)
    ):
        raise HTTPException(422, "invalid train subset")
    return [key for key in identifiers if key in selected]


def measurement_outcome(binding, result):
    if binding["split"] == "train":
        if result["params"]["candidate_id"] != binding["candidate_id"]:
            raise HTTPException(502, "candidate identity mismatch")
        metrics = result["metrics"]
        resources = {
            role: Resources(
                metrics[f"{role}_tokens"],
                metrics[f"{role}_seconds"],
                metrics[f"{role}_resources_complete"]
                and metrics[f"{role}_unreported_attempts"] == 0,
            )
            for role in ("solver", "judge")
        }
        return metrics.get("score"), resources
    rows = result.get("scores", [])
    if result["status"] == "complete" and (
        len(rows) != 1 or rows[0]["candidate_id"] != binding["candidate_id"]
    ):
        raise HTTPException(502, "public candidate identity mismatch")
    return rows[0]["score"] if rows else None, {}


def research_feedback(split, result):
    if split == "public":
        return {
            "status": result["status"],
            **({"score": result["scores"][0]["score"]} if result.get("scores") else {}),
        }
    return result


def seal_human_stop(root, journal, submission):
    stop = root / "private/human-stop.json"
    if stop.exists():
        state = journal.status()
        write(
            root / "private/search-control/stop.json",
            {
                "batch_digest": submission.digest(),
                "stopped_at": state["stopped_at"],
                "reason": state["stop_reason"],
                "source_ref": json.loads(stop.read_text())["source_ref"],
            },
        )


def create_app(root):
    from security import SafetyGate

    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    journal = Journal(root / "search/search.sqlite3")
    token = (root / "research-token").read_text().strip()
    admin = (root / "private/admin-token").read_text().strip()
    lock = threading.RLock()
    security = SafetyGate(root, loaded_config=config)
    monitor_path = Path(__file__).resolve().parents[1] / "monitor.py"
    if Path(config["monitor"]).resolve() != monitor_path:
        raise ValueError("monitor must use the trusted case implementation")
    spec = importlib.util.spec_from_file_location("ar03_monitor", monitor_path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        if request.url.path == "/healthz":
            return JSONResponse({"status": "ok"})
        expected = admin if request.url.path.startswith("/admin/") else token
        if not secrets.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + expected
        ):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        if int(request.headers.get("Content-Length", "0")) > 72 * 1024 * 1024:
            return JSONResponse({"detail": "body too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def failure(request, error):
        return JSONResponse({"detail": "operator operation failed"}, status_code=503)

    def armed():
        security.require_permission()

    def reject_submission(path, binding, status_code):
        binding.update(status="rejected", submission_http_status=status_code)
        write(path, binding)
        measurement = next(
            row
            for row in journal.export()["measurements"]
            if row["measurement_id"] == binding["measurement"]
        )
        if measurement["state"] == "running":
            resources = (
                {"solver": Resources(0, 0), "judge": Resources(0, 0)}
                if binding["split"] == "train"
                else {}
            )
            journal.finish(binding["measurement"], error="evaluation_failed", **resources)
        binding["status"] = "finished"
        write(path, binding)

    def backend(split, method, payload=None, evaluation_id=None):
        endpoint = config["services"].get(split)
        if endpoint is None:
            raise HTTPException(409, "split is not enabled for this campaign")
        secret = (root / "private" / f"{split}-token").read_text().strip()
        suffix = "/" + evaluation_id if evaluation_id is not None else ""
        with httpx.Client(trust_env=False, timeout=600, follow_redirects=False) as client:
            response = client.request(
                method,
                endpoint + "/v1/evaluations" + suffix,
                json=payload,
                headers={"Authorization": "Bearer " + secret},
            )
        if not response.is_success:
            if response.status_code == 503:
                raise HTTPException(
                    503, "evaluation service is busy or unavailable", headers={"Retry-After": "60"}
                )
            raise HTTPException(response.status_code, "evaluation request rejected")
        result = response.json()
        validate_feedback(split, result)
        return result

    @contextmanager
    def restored(identifier):
        # Caller holds the lock; restore/materialize only, never execute candidate code here.
        with tempfile.TemporaryDirectory(dir=root / "private") as temporary:
            path = Path(temporary) / "candidate"
            journal.restore(identifier, path)
            yield pack_candidate(identifier, path)

    def publish_progress(comparison=None, train=None):
        if comparison is None:
            measurements = journal.export()["measurements"]
            if measurements:
                comparison = measurements[-1]["comparison"]
            else:
                return
        stage_file = root / "private/stage.json"
        stage = json.loads(stage_file.read_text()) if stage_file.exists() else "end-to-end"
        result = monitor.snapshot(journal, comparison, stage=stage, train_record=train)
        monitor.save_artifacts(result, root / "progress")

    @app.get("/status")
    def status():
        state = journal.status()
        retryable = []
        for measurement in state["running"]:
            path = root / "private/bindings" / f"{measurement}.json"
            if path.is_file() and json.loads(path.read_text())["status"] == "retryable":
                retryable.append(measurement)
        return {**state, **security.status(), "retryable_measurements": retryable}

    @app.post("/admin/verify")
    def verify():
        for measurement in journal.status()["running"]:
            path = root / "private/bindings" / f"{measurement}.json"
            binding = json.loads(path.read_text())
            if binding["status"] == "retryable":
                collect(measurement)  # GET only: a timed-out submit may have been accepted.
                if json.loads(path.read_text())["status"] == "retryable":
                    continue
            if binding["status"] == "rejected":
                reject_submission(path, binding, binding["submission_http_status"])
            if measurement in journal.status()["running"]:
                raise HTTPException(409, "wait for accepted evaluations before verification")
        return security.verify()

    @app.post("/admin/authorize")
    def authorize(body: dict):
        return security.authorize(body.get("source_ref"))

    @app.post("/admin/pause")
    def pause():
        return security.pause()

    @app.post("/propose")
    def propose(body: dict):
        armed()
        if set(body) - {"candidate", "parents", "hypothesis", "change", "stage"}:
            raise HTTPException(422, "unsupported proposal fields")
        if body.get("stage", "end-to-end") not in monitor.STAGES:
            raise HTTPException(422, "unknown pipeline stage")
        candidate = Candidate.model_validate(body["candidate"])
        with lock, tempfile.TemporaryDirectory(dir=root / "private") as temporary:
            directory = Path(temporary) / "candidate"
            materialize_candidate(candidate, directory)
            result = journal.propose(
                None,
                directory,
                parents=body.get("parents", []),
                hypothesis=body["hypothesis"],
                change=body["change"],
            )
            write(root / "private/stage.json", body.get("stage", "end-to-end"))
        return result

    @app.post("/evaluate")
    def evaluate(body: dict):
        armed()
        identifiers = evaluation_ids(config, body)
        split = body["split"]
        comparison = {
            "split": split,
            "dataset_id": config["dataset_id"],
            "subset_id": _digest(identifiers) if split == "train" else "public-fixed",
            "conditions_id": config["conditions_id"],
        }
        with lock, restored(body["candidate_id"]) as candidate:
            measurements = journal.export()["measurements"]
            previous = next(
                (
                    row
                    for row in reversed(measurements)
                    if row["candidate_id"] == candidate.id and row["comparison"] == comparison
                ),
                None,
            )
            if previous is not None:
                measurement = previous["measurement_id"]
                path = root / "private/bindings" / f"{measurement}.json"
                binding = json.loads(path.read_text())
                if "submission_http_status" in binding:
                    return {"measurement": measurement, "status": "FAILED"}
                if binding["status"] in ("running", "finished"):
                    return {
                        "measurement": measurement,
                        "evaluation_id": binding["evaluation_id"],
                        "status": (
                            "FINISHED"
                            if previous["state"] == "succeeded"
                            else "FAILED"
                            if previous["state"] == "failed"
                            else "RUNNING"
                        ),
                    }
                if binding["status"] != "retryable":
                    raise HTTPException(409, "submission needs operator recovery")
            else:
                measurement = "check-" + uuid.uuid4().hex
                path = root / "private/bindings" / f"{measurement}.json"
                binding = {
                    "measurement": measurement,
                    "candidate_id": candidate.id,
                    "comparison": comparison,
                    "split": split,
                    "status": "submitting",
                }
            for row in measurements:
                if row["state"] == "running" and row["measurement_id"] != measurement:
                    other = root / "private/bindings" / f"{row['measurement_id']}.json"
                    if json.loads(other.read_text())["status"] != "retryable":
                        raise HTTPException(
                            503, "evaluation capacity is busy", headers={"Retry-After": "60"}
                        )
            if previous is None:
                try:
                    journal.begin(
                        measurement, candidate.id, **comparison, job=config["services"][split]
                    )
                except SearchError:
                    raise HTTPException(409, "measurement cannot be registered") from None
            payload = (
                {
                    "candidate": candidate.model_dump(),
                    "example_ids": identifiers,
                    "metadata": {"measurement_id": measurement},
                }
                if split == "train"
                else {"candidates": [candidate.model_dump()]}
            )
            binding["status"] = "submitting"
            if split == "train":
                binding["expected_evaluation_id"] = _digest([candidate.digest(), identifiers])
            write(path, binding)
            try:
                result = backend(split, "POST", payload)
                binding.update(evaluation_id=result["evaluation_id"], status="running")
            except (HTTPException, httpx.RequestError, ValueError, KeyError) as error:
                if isinstance(error, HTTPException) and 400 <= error.status_code < 500:
                    reject_submission(path, binding, error.status_code)
                    raise
                binding.update(status="retryable", submission_error=type(error).__name__)
                write(path, binding)
                if isinstance(error, HTTPException):
                    raise
                raise HTTPException(
                    503,
                    "submission unconfirmed; retry the same candidate",
                    headers={"Retry-After": "60"},
                ) from None
            write(path, binding)
        return {
            "measurement": measurement,
            "evaluation_id": result["evaluation_id"],
            "status": result["status"],
        }

    def finish_measurement(path, saved, binding, result):
        score, resources = measurement_outcome(binding, result)
        write(saved, result)
        journal.finish(
            binding["measurement"],
            score=score,
            error="evaluation_failed" if result["status"] in ("FAILED", "failed") else None,
            **resources,
        )
        binding["status"] = "finished"
        write(path, binding)

    def collect(measurement):
        if not re.fullmatch(r"check-[a-f0-9]{32}", measurement):
            raise HTTPException(404, "unknown measurement")
        path = root / "private/bindings" / f"{measurement}.json"
        if not path.exists():
            raise HTTPException(404, "unknown measurement")
        with lock:
            binding = json.loads(path.read_text())
            if "submission_http_status" in binding:
                reject_submission(path, binding, binding["submission_http_status"])
                return {"status": "FAILED" if binding["split"] == "train" else "failed"}
            if "evaluation_id" not in binding:
                expected = binding.get("expected_evaluation_id")
                if binding["status"] != "retryable" or not expected:
                    raise HTTPException(409, "submission needs operator recovery")
                try:
                    backend(binding["split"], "GET", evaluation_id=expected)
                except HTTPException as error:
                    if error.status_code == 404:
                        return {
                            "status": "RETRYABLE",
                            "measurement": measurement,
                            "candidate_id": binding["candidate_id"],
                        }
                    raise
                binding.update(evaluation_id=expected, status="running")
                write(path, binding)
            saved = root / "private/results" / f"{measurement}.json"
            result = (
                json.loads(saved.read_text())
                if saved.exists()
                else backend(binding["split"], "GET", evaluation_id=binding["evaluation_id"])
            )
            terminal = result["status"] in ("FINISHED", "FAILED", "complete", "failed")
            if terminal and binding["status"] != "finished":
                finish_measurement(path, saved, binding, result)
            publish_progress(binding["comparison"], result if binding["split"] == "train" else None)
            return research_feedback(binding["split"], result)

    @app.get("/result/{measurement}")
    def result(measurement: str):
        if measurement == "private":
            return backend("private", "GET")
        return collect(measurement)

    @app.post("/finalize")
    def finalize(body: dict):
        security.require_live()
        if not (root / "private/human-stop.json").exists():
            armed()
        identifiers = body.get("candidate_ids", [])
        expected = config["finalists"] if "private" in config["services"] else 1
        if len(identifiers) != expected or len(set(identifiers)) != len(identifiers):
            raise HTTPException(422, "wrong finalist count")
        with lock, ExitStack() as stack:
            candidates = [stack.enter_context(restored(identifier)) for identifier in identifiers]
            submission = Submission(candidates=candidates)
            finalization = journal.finalize(submission)
            seal_human_stop(root, journal, submission)
            write(root / "private/finalists.json", submission.model_dump())
            if "private" in config["services"]:
                return backend("private", "POST", submission.model_dump())
            return {"status": "finalized", **finalization}

    @app.get("/progress")
    def progress():
        publish_progress()
        path = root / "progress/summary.json"
        return json.loads(path.read_text()) if path.exists() else journal.status()

    @app.get("/progress/{name}")
    def progress_file(name: str):
        if name not in {"summary.json", "teds.svg", "auc.svg", "lineage.svg"}:
            raise HTTPException(404, "unknown progress artifact")
        publish_progress()
        path = root / "progress" / name
        if not path.is_file():
            raise HTTPException(404, "progress artifact not ready")
        return FileResponse(path)

    @app.post("/admin/stop")
    def stop(body: dict):
        if not body.get("source_ref"):
            raise HTTPException(422, "human stop source reference is required")
        with lock:
            security.pause()
            write(root / "private/human-stop.json", {"source_ref": body["source_ref"]})
            return journal.stop(body["reason"])

    @app.on_event("startup")
    def reconcile():
        def work():
            while not closing.wait(2):
                if security.evidence.exists():
                    try:
                        security.require_live()
                    except HTTPException:
                        pass
                for path in (root / "private/bindings").glob("*.json"):
                    try:
                        if json.loads(path.read_text())["status"] in (
                            "running",
                            "retryable",
                            "rejected",
                        ):
                            collect(path.stem)
                    except Exception:
                        # Closed diagnostics stay with the operator, never the research response.
                        pass

        threading.Thread(target=work, daemon=True).start()

    closing = threading.Event()

    @app.on_event("shutdown")
    def shutdown():
        closing.set()

    return app
