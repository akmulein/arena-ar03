import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

from evaluator.capacity import CapacityError, acquire_slot
from evaluator.checkpoints import read_document
from evaluator.client import pack_candidate
from evaluator.contracts import Submission

TOKEN = "synthetic-private-parallel"
PROGRAM = """import json
import sys
import time
from pathlib import Path
from uuid import uuid4

task = json.load(sys.stdin)
name = NAME
gate = Path(task["assets_dir"]) / f"release-{name}-{task['input']}"
deadline = time.monotonic() + 120
while not gate.exists():
    if time.monotonic() > deadline:
        raise TimeoutError("synthetic barrier was not released")
    time.sleep(0.05)
print(json.dumps({"answer": f"{name}:{task['input']}:{uuid4()}"}))
"""


def docker(*arguments: str, timeout: float = 90, check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, timeout=timeout, check=check,
    )
    return result.stdout.strip()


def wait_for(predicate, description: str, timeout: float = 60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
        except (FileNotFoundError, urllib.error.URLError, ConnectionError, TimeoutError):
            value = None
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(description)


def request(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    query = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(query, timeout=10) as response:
        return response.status, json.load(response)


def runners(image: str) -> list[str]:
    return docker("ps", "--quiet", "--filter", f"ancestor={image}").split()


def occupied(slots: Path) -> bool:
    try:
        lease = acquire_slot(slots, 4)
    except CapacityError:
        return True
    if lease is not None:
        lease.close()
    return False


def start_server(image: str, runner: str, workspace: Path, split: str) -> dict:
    name = f"arena-private-parallel-{split}-{workspace.name}"
    state = workspace / split
    state.mkdir()
    settings = {
        "DATASET_PATH": str(workspace / "dataset.jsonl"),
        "METRIC_PATH": str(workspace / "metric.json"),
        "STATE_DIR": str(state),
        "TOKEN": TOKEN,
        "RELEASE_ID": workspace.name,
        "RUNNER_IMAGE": runner,
        "RUNNER_NETWORK": "none",
        "ASSETS_DIR": str(workspace / "assets"),
        "USES_MODEL": "false",
        "SPLIT": split,
        "FINALISTS": "6",
        "OUTPUT_COUNT": "3",
        "SLOT_DIR": str(workspace / "slots"),
        "EXECUTION_SLOTS": "4",
        "RUNNER_MEMORY": "128m",
        "RUNNER_CPUS": "0.25",
        "RUNNER_PIDS_LIMIT": "64",
        "RUNNER_TIMEOUT": "180",
    }
    environment = [
        item for key, value in settings.items() for item in ("--env", f"ARENA_{key}={value}")
    ]
    docker(
        "run", "--detach", "--name", name,
        "--user", "10001:10001", "--group-add", str(Path("/var/run/docker.sock").stat().st_gid),
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
        "--tmpfs", "/scratch:rw,nosuid,nodev,size=128m,mode=1777", "--env", "TMPDIR=/scratch",
        "--publish", "127.0.0.1::8080",
        "--mount", f"type=bind,src={workspace},dst={workspace}",
        "--mount", "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        *environment, image, "python", "-m", "evaluator",
    )
    port = docker("port", name, "8080/tcp").rsplit(":", 1)[1]
    url = "http://127.0.0.1:" + port
    try:
        wait_for(lambda: request(url + "/healthz")[1].get("status") == "ok", "server unhealthy")
    except Exception as error:
        logs = subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True, timeout=15, check=False,
        )
        docker("rm", "--force", name, check=False)
        raise RuntimeError(logs.stdout + logs.stderr) from error
    return {"name": name, "state": state, "url": url}


def record(server: dict) -> tuple[Path, dict]:
    path = next(server["state"].glob("*/record.json"))
    return path, json.loads(path.read_text())


def complete_candidate(server: dict, name: str) -> bool:
    found = list(server["state"].glob(f"*/details/{name}.json"))
    return bool(found and json.loads(found[0].read_text()).get("aggregation", {}).get("status")
                == "complete")


def active_examples(server: dict) -> list[tuple[str, str]]:
    return sorted(
        (document["candidate_id"], str(example["example_id"]))
        for path in server["state"].glob("*/details/*.json")
        for document in [read_document(path)]
        for example in document["examples"] if example.get("status") == "solving"
    )


def completed_example(server: dict, name: str, example_id: str) -> bool:
    return any(
        example.get("example_id") == example_id and "score" in example
        for path in server["state"].glob(f"*/details/{name}.json")
        for example in read_document(path)["examples"]
    )


def preserved(before: dict, after: dict) -> bool:
    by_id = {example["example_id"]: example for example in after["examples"]}
    return all(by_id.get(example["example_id"]) == example for example in before["examples"])


def cli(server: dict, module: str, *arguments: str) -> dict:
    return json.loads(docker("exec", server["name"], "python", "-m", module, *arguments))


def release(assets: Path, names: str, examples: tuple[str, ...]) -> None:
    for name in names:
        for example in examples:
            (assets / f"release-{name}-{example}").touch()


def check(image: str, runner_source: Path, root: Path) -> dict[str, Any]:
    started = time.monotonic()
    image_name = image
    image = docker("image", "inspect", "--format", "{{.Id}}", image)
    with tempfile.TemporaryDirectory(prefix="arena-private-smoke-", dir=root) as temporary:
        workspace = Path(temporary).resolve()
        runner = workspace.name + ":latest"
        servers = []
        pending = None
        occupied_lease = None
        try:
            assets = workspace / "assets"
            assets.mkdir()
            slots = workspace / "slots"
            slots.mkdir(mode=0o700)
            (workspace / "dataset.jsonl").write_text("\n".join(
                json.dumps({"id": str(index), "input": str(index)}) for index in range(3)
            ))
            metric_script = workspace / "metric.py"
            metric_script.write_text("print('{\"score\": 1}')\n")
            (workspace / "metric.json").write_text(json.dumps({
                "kind": "command", "version": "synthetic-parallel-v1", "command": {
                    "argv": [sys.executable, str(metric_script)], "uses_model": False,
                },
            }))
            context = workspace / "runner"
            context.mkdir()
            shutil.copyfile(runner_source, context / "main.py")
            (context / "Dockerfile").write_text(
                f"FROM {image}\nLABEL arena.private-smoke={workspace.name}\n"
                "COPY main.py /synthetic/main.py\n"
                'ENTRYPOINT ["python", "-I", "/synthetic/main.py"]\n'
            )
            docker("build", "--tag", runner, str(context), timeout=180)
            candidates = []
            for name in "abcdefp":
                path = workspace / ("candidate-" + name)
                path.mkdir()
                (path / "solve.py").write_text(PROGRAM.replace("NAME", repr(name)))
                candidates.append(pack_candidate(name, path))
            bundle = Submission(candidates=candidates[:6])
            body = bundle.model_dump()
            private = start_server(image, runner, workspace, "private")
            servers.append(private)
            occupied_lease = acquire_slot(slots, 4)
            assert occupied_lease is not None
            assert request(private["url"] + "/v1/evaluations", "POST", body)[0] == 202
            wait_for(lambda: len(runners(runner)) == 3, "three available slots were not used")
            assert occupied(slots)
            occupied_lease.close()
            occupied_lease = None
            wait_for(lambda: len(runners(runner)) == 4, "fourth slot was not used")
            assert occupied(slots)
            released = set()
            completed = []
            while not completed:
                work = wait_for(
                    lambda: ([name for name in "bcdef" if complete_candidate(private, name)]
                             or [pair for pair in active_examples(private)
                                 if pair[0] != "a" and pair not in released]),
                    "no runnable candidate can complete",
                )
                if isinstance(work[0], str):
                    completed = cast(list[str], work)
                    break
                name, example_id = cast(tuple[str, str], work[-1])
                release(assets, name, (example_id,))
                released.add((name, example_id))
                wait_for(lambda name=name, example_id=example_id:
                         completed_example(private, name, example_id),
                         "released example was not saved")
                completed = [name for name in "bcdef" if complete_candidate(private, name)]
            complete_name = completed[0]
            wait_for(lambda: len(runners(runner)) == 4, "queued examples were not started")
            assert cli(private, "evaluator.pause", "--reason", "synthetic handoff")["status"] in {
                "pause_requested", "paused",
            }
            for name, example_id in active_examples(private):
                release(assets, name, (example_id,))
            wait_for(lambda: record(private)[1]["status"] == "paused", "private did not drain")
            assert request(private["url"] + "/v1/evaluations")[1] == {"status": "running"}
            path, before_record = record(private)
            assert before_record["batch_digest"] == bundle.digest()
            assert not runners(runner)
            details = path.parent / "details"
            complete_bytes = (details / f"{complete_name}.json").read_bytes()
            before = {file.stem: read_document(file) for file in details.glob("*.json")}
            saved_count = sum(len(value["examples"]) for value in before.values())
            assert len(before[complete_name]["examples"]) == 3
            assert all("score" in example for value in before.values()
                       for example in value["examples"])
            assert 18 - saved_count >= 4
            leases = [acquire_slot(slots, 4) for _ in range(4)]
            for lease in leases:
                assert lease is not None
                lease.close()
            original_record = path.read_bytes()
            assert cli(private, "evaluator.resume", "--check", "--reason", "synthetic resume")[
                "status"
            ] == "resume_ready"
            assert path.read_bytes() == original_record
            public = start_server(image, runner, workspace, "public")
            servers.append(public)
            public_body = Submission(candidates=[candidates[-1]]).model_dump()
            public_response = request(public["url"] + "/v1/evaluations", "POST", public_body)[1]
            public_url = public["url"] + "/v1/evaluations/" + public_response["evaluation_id"]
            wait_for(lambda: len(runners(runner)) == 3,
                     "one public candidate did not use three example slots")
            release(assets, "p", ("2",))
            wait_for(
                lambda: any(
                    {example["example_id"] for example in read_document(path)["examples"]
                     if "score" in example} == {"2"}
                    for path in public["state"].glob("**/candidate-results/*/details.json")
                ),
                "out-of-order public example was not checkpointed",
            )
            release(assets, "p", ("0", "1"))
            wait_for(lambda: request(public_url)[1]["status"] == "complete", "public failed")
            assert request(public_url)[1]["scores"] == [{"candidate_id": "p", "score": 1.0}]
            pending = subprocess.Popen(
                ["docker", "exec", private["name"], "python", "-m", "evaluator.resume",
                 "--reason", "synthetic resume"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            wait_for(lambda: len(runners(runner)) == 4, "resumed private did not use four slots")
            assert occupied(slots)
            assert (details / f"{complete_name}.json").read_bytes() == complete_bytes
            for name, document in before.items():
                current = read_document(details / f"{name}.json")
                assert preserved(document, current)
            release(assets, "abcdef", ("0", "1", "2"))
            output, errors = pending.communicate(timeout=120)
            assert pending.returncode == 0, errors
            assert json.loads(output) == {"status": "complete"}
            final = record(private)[1]
            assert final["status"] == "complete" and final["batch_digest"] == bundle.digest()
            assert request(private["url"] + "/v1/evaluations")[1] == {"status": "complete"}
            assert [item["candidate_id"] for item in final["scores"]] == list("abcdef")
            assert len(final["winners"]) == 3 and len(set(final["winners"])) == 3
            assert (details / f"{complete_name}.json").read_bytes() == complete_bytes
            for name in "abcdef":
                current = json.loads((details / f"{name}.json").read_text())
                assert len(current["examples"]) == 3
                for index, detail in enumerate(current["examples"]):
                    assert detail["answer"].startswith(f"{name}:{index}:")
                    assert "failure" not in detail
                    assert detail["resources"]["solver"]["usage"]["total_tokens"] == 0
                    assert detail["resources"]["solver"]["wall_seconds"] > 0
                if name in before:
                    assert preserved(before[name], current)
            assert not runners(runner)
            return {"status": "ok", "image": image_name, "image_id": image,
                    "wall_seconds": round(time.monotonic() - started, 3), "global_slots": 4,
                    "parallel_workers": 4, "finalists": 6, "examples_per_finalist": 3,
                    "single_candidate_parallel_examples": 3,
                    "out_of_order_example_checkpoint": True,
                    "completed_examples_preserved": saved_count,
                    "completed_candidate_byte_exact": True,
                    "same_sealed_batch": True, "public_score": 1.0, "model_calls": 0}
        finally:
            if occupied_lease is not None:
                occupied_lease.close()
            if pending is not None and pending.poll() is None:
                pending.terminate()
                pending.communicate(timeout=10)
            for server in servers:
                docker("rm", "--force", server["name"], check=False)
            remaining = docker("ps", "--all", "--quiet", "--filter", f"ancestor={runner}",
                               check=False).split()
            if remaining:
                docker("rm", "--force", *remaining, check=False)
            docker("image", "rm", runner, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.image, arguments.runner_source, arguments.workspace_root)))


if __name__ == "__main__":
    main()
