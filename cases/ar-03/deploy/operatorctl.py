"""Operator-only setup and lifecycle; never starts evolution automatically."""

import argparse
import asyncio
import importlib.util
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

CASE = Path(__file__).resolve().parents[1]
ARENA = CASE.parents[1]
sys.path.insert(0, str(ARENA))

from evaluator.execution import make_runner  # noqa: E402
from evaluator.search import Journal  # noqa: E402
from evaluator.service import Settings, create_app  # noqa: E402
from evaluator.train import TrainSettings  # noqa: E402
from evaluator.train import create_app as train_app  # noqa: E402


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_preparation(args):
    if args.max_solutions is None:
        if args.mode == "main":
            raise ValueError("main campaign needs an explicitly agreed proposal limit")
        args.max_solutions = 4
    if args.mode == "main" and (
        not args.finalists
        or not args.output_count
        or args.output_count > args.finalists
        or args.closed_source is None
    ):
        raise ValueError("main requires agreed limit/K/N and operator-supplied closed source")


def prepare_workspace(root):
    workspace = root / "research-workspace"
    workspace.mkdir()
    for name in ("evaluator", "common", "prompts"):
        shutil.copytree(
            ARENA / name, workspace / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("AGENTS.md", "README.md", "pyproject.toml", "uv.lock", ".gitignore"):
        shutil.copyfile(ARENA / name, workspace / name)
    (workspace / "preparations").mkdir()
    shutil.copyfile(ARENA / "preparations/README.md", workspace / "preparations/README.md")
    research_case = workspace / "cases/ar-03"
    research_case.mkdir(parents=True)
    for name in ("client.py", "monitor.py", "provenance.json"):
        shutil.copyfile(CASE / name, research_case / name)
    shutil.copytree(
        CASE / "ways/baseline",
        research_case / "ways/baseline",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        CASE / "campaigns/pilot-01",
        research_case / "campaigns/pilot-01",
        ignore=shutil.ignore_patterns("_local"),
    )
    shutil.copyfile(CASE / "campaigns/pilot-01/preparation.md", research_case / "README.md")
    (research_case / "_local").mkdir()
    (research_case / "_local/train-500.jsonl").touch()
    shutil.copyfile(CASE / ".gitignore", research_case / ".gitignore")
    return research_case


def prepare_datasets(args, root, research_case):
    private = root / "private"
    (private / "data").mkdir()
    exported = args.resources / "data/arena"
    shutil.copyfile(exported / "train-500.jsonl", private / "data/train.jsonl")
    train_rows = [
        json.loads(line) for line in (exported / "train-500.jsonl").read_text().splitlines()
    ]
    identifiers = [row["id"] for row in train_rows]
    selected = json.loads((exported / "evaluation-ids.json").read_text())
    save(research_case / "_local/evaluation-ids.json", selected)
    services = {"train": f"http://127.0.0.1:{args.port + 1}"}
    if args.mode == "main":
        spec = importlib.util.spec_from_file_location("ar03_export", CASE / "data.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        for split in ("public", "private"):
            exporter.export_dataset(
                root=args.closed_source,
                manifest=args.closed_source / "subsets" / f"{split}-200.jsonl",
                split=split,
                output=private / "data" / f"{split}.jsonl",
                audit=args.closed_source / "subsets/arena-subsets-audit.json",
                operator=True,
            )
            services[split] = f"http://127.0.0.1:{args.port + (2 if split == 'public' else 3)}"
    return identifiers, selected, services


def prepare_tokens(root):
    private = root / "private"
    for name in ("admin", "train", "public", "private"):
        (private / f"{name}-token").write_text(secrets.token_hex(32))
        (private / f"{name}-token").chmod(0o600)
    (root / "research-token").write_text(secrets.token_hex(32))
    (root / "research-token").chmod(0o644)


def save_configuration(args, root, identifiers, selected, services):
    private = root / "private"
    metric = private / "metric"
    config = {
        "root": str(root),
        "campaign_id": root.name,
        "mode": args.mode,
        "max_solutions": args.max_solutions,
        "finalists": args.finalists,
        "output_count": args.output_count,
        "port": args.port,
        "services": services,
        "monitor": str(CASE / "monitor.py"),
        "image": "arena-ar03-runner:arm64-cpu-v1",
        "assets": str(args.resources / "runner-assets"),
        "metric": str(metric),
        "dataset_id": "ar03-train-500",
        "conditions_id": "ar03-teds-v1-docker-arm64-cpu-train-only",
        "train_data": str(private / "data/train.jsonl"),
        "train_ids": identifiers,
        "evaluation_ids": selected,
    }
    if args.mode == "main":
        config["conditions_id"] = "ar03-teds-v1-docker-arm64-cpu-main"
        config["dataset_id"] = "ar03-fixed-500-200-200"
    save(root / "config.json", config)
    sources = [
        {
            "split": split,
            "state_dir": str(private / "state" / split),
            "dataset_path": config["train_data"]
            if split == "train"
            else str(private / "data" / f"{split}.jsonl"),
        }
        for split in services
    ]
    save(
        private / "artifacts.json",
        {
            "output_root": str(private / "visualizations"),
            "metric_runtime": str(metric / "runtime"),
            "sources": sources,
        },
    )


def prepare(args):
    from bootstrap import external_resources

    validate_preparation(args)
    args.resources = external_resources(args.resources)
    root = args.root.resolve()
    if root.parent != args.resources / "runs":
        raise ValueError("campaign root must be <resources>/runs/<campaign>")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        raise ValueError("operator root already prepared; do not replace campaign state")
    (root / "private").mkdir(mode=0o700)
    (root / "private/slots").mkdir(mode=0o700)
    research_case = prepare_workspace(root)
    identifiers, selected, services = prepare_datasets(args, root, research_case)
    prepare_tokens(root)
    (root / "search").mkdir()
    Journal.create(root / "search/search.sqlite3", max_solutions=args.max_solutions)
    shutil.copytree(
        CASE / "campaigns/campaign-01/metric",
        root / "private/metric",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    save_configuration(args, root, identifiers, selected, services)
    print(json.dumps({"prepared": True, "mode": args.mode, "authorized": False}))


def settings(root, split, config=None):
    from security import image_pins

    images = image_pins(root)
    if config is None:
        config = json.loads((root / "config.json").read_text())
    private = root / "private"
    if Path(config["metric"]).resolve() != private / "metric":
        raise ValueError("metric must remain in the trusted operator directory")
    if Path(config["train_data"]).resolve() != private / "data/train.jsonl":
        raise ValueError("train must use the prepared operator dataset")
    return Settings(
        dataset_path=Path(config["train_data"])
        if split == "train"
        else private / "data" / f"{split}.jsonl",
        metric_path=Path(config["metric"]),
        state_dir=private / "state" / split,
        token=(private / f"{split}-token").read_text().strip(),
        runner_image=images["runner"],
        release_id=config["conditions_id"],
        uses_model=False,
        split="public" if split == "public" else "private",
        finalists=config["finalists"] or 1,
        output_count=config["output_count"] or 1,
        assets_dir=Path(config["assets"]),
        runner_memory="6g",
        runner_cpus="2",
        runner_tmpfs_size="512m",
        runner_timeout=600,
        harnesses={
            "ar03-program": {"image": images["runner"], "uses_model": False, "network": "none"}
        },
        search_mode="campaign",
        search_journal=root / "search/search.sqlite3",
        search_max_solutions=config["max_solutions"],
        search_stop_path=private / "search-control/stop.json",
        execution_slots=1,
        slot_dir=private / "slots",
        case_id="ar-03",
    )


def serve(args):
    import uvicorn

    sys.path.insert(0, str(Path(__file__).parent))
    from security import GuardedHarness, GuardedService, SafetyGate

    root = args.root.resolve()
    config = json.loads((root / "config.json").read_text())
    if args.role == "control":
        sys.path.insert(0, str(Path(__file__).parent))
        from control import create_app as control_app

        app, port = control_app(root), config["port"]
    else:
        gate = SafetyGate(root, loaded_config=config)
        configured = settings(root, args.role, config)
        runner = GuardedHarness(
            make_runner(
                configured, configured.usage_dir or configured.state_dir.resolve() / "usage"
            ),
            gate,
        )
        app = (
            train_app(
                TrainSettings(
                    configured,
                    config.get("campaign_id", root.name),
                    "ar-03",
                    config["dataset_id"],
                    config["conditions_id"],
                    tracking=False,
                ),
                solve=runner,
            )
            if args.role == "train"
            else create_app(configured, solve=runner)
        )
        app = GuardedService(app, gate)
        port = int(config["services"][args.role].rsplit(":", 1)[1])
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)


def configure_docker(private):
    spec = importlib.util.spec_from_file_location("ar03_train_config", CASE / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_local_docker(private)


def active_instructions(config):
    entry = (
        "Рабочий campaign_dir: cases/ar-03/campaigns/pilot-01. "
        "Каталог active содержит только текущие операторские условия. "
        "Сначала прочитай cases/ar-03/campaigns/pilot-01/start.md, "
        "затем prompts/start.md с уточнениями из кейсового start.md. "
        if config["mode"] == "pilot"
        else "Выполни prompts/start.md. "
    )
    return (
        "# Условия активной кампании\n\n"
        f"Режим: {config['mode']}; лимит предложений: {config['max_solutions']}; "
        f"активные API split: {', '.join(config['services'])}. "
        f"K={config['finalists']}, N={config['output_count']}; "
        "для pilot они не применяются.\n\n"
        "Score=mean(page TEDS), maximize, [0,1]. Данные 500/200/200 не меняются. "
        "Кандидаты исполняются только штатным DockerRunner на CPU. "
        "Метрика/веса неизменны; обучение запрещено. Замена моделей/существенные "
        "зависимости требуют согласования. Все агенты работают только здесь. "
        "Публикация результатов public — только общий TEDS; private — только статус. "
        "Результаты и журнал изменяет только операторский API.\n\n"
        f"{entry}Для main применяй те же client.py "
        "propose/evaluate/result/progress/finalize, но текущие M/K/N и активные split "
        "из этого файла; после завершения подай ровно K финалистов. "
        "Private не используется для обратной связи. "
        "Journal /search/search.sqlite3 read-only. "
        "Для client.py используй /opt/arena/.venv/bin/python: httpx уже установлен там; "
        "login-shell может подменить PATH и выбрать системный Python без зависимостей. "
        "Не вызывай evaluator.search для записи напрямую. "
        "Не повторяй аудит/benchmark baseline/прежние тесты. Skill не создавай.\n"
    )


def sync_workspace(compose, environment, config):
    files = {
        "/workspace/cases/ar-03/campaigns/active/campaign.md": active_instructions(config),
        "/workspace/preparations/README.md": (ARENA / "preparations/README.md").read_text(),
        "/workspace/cases/ar-03/README.md": (
            CASE / "campaigns/pilot-01/preparation.md"
        ).read_text(),
        "/workspace/cases/ar-03/client.py": (CASE / "client.py").read_text(),
        "/workspace/cases/ar-03/provenance.json": (CASE / "provenance.json").read_text(),
        "/workspace/cases/ar-03/campaigns/pilot-01/campaign.md": (
            CASE / "campaigns/pilot-01/campaign.md"
        ).read_text(),
        "/workspace/cases/ar-03/campaigns/pilot-01/start.md": (
            CASE / "campaigns/pilot-01/start.md"
        ).read_text(),
    }
    for name in (
        "preparation.md",
        "metric-description.json",
        "preparations/README.md",
        "board/readiness.md",
    ):
        files["/workspace/cases/ar-03/campaigns/pilot-01/" + name] = (
            CASE / "campaigns/pilot-01" / name
        ).read_text()
    if config.get("root"):
        reference = Path(config["root"]) / "baseline-reference.json"
        if reference.is_file():
            files["/workspace/cases/ar-03/campaigns/pilot-01/baseline-reference.json"] = (
                reference.read_text()
            )
    writer = (
        "import json,sys; from pathlib import Path; "
        "values=json.load(sys.stdin); "
        "[(Path(k).parent.mkdir(parents=True,exist_ok=True),Path(k).write_text(v)) "
        "for k,v in values.items()]"
    )
    subprocess.run(
        compose + ["exec", "-T", "--user", "10001:10001", "researcher", "python", "-c", writer],
        input=json.dumps(files),
        text=True,
        env=environment,
        check=True,
    )


def containers(args, root, config):
    from security import SafetyGate, image_pins

    images = image_pins(root)
    configure_docker(root / "private")
    environment = {
        **os.environ,
        "AR03_OPERATOR_ROOT": str(root),
        "AR03_OPERATOR_PORT": str(config["port"]),
        "AR03_RESEARCHER_IMAGE": images["researcher"],
    }
    compose = ["docker", "compose", "-p", "ar03-isolated", "-f", str(CASE / "deploy/compose.yaml")]
    if args.command == "containers-up":
        SafetyGate(root).revoke("container deployment requested")
        subprocess.run(compose + ["up", "-d"], env=environment, check=True)
        for source, target, recursive in (
            (str(root / "research-workspace"), "/workspace", True),
            ("ar03-isolated_codex-auth", "/auth", False),
        ):
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    "0",
                    "--cap-drop",
                    "ALL",
                    "--cap-add",
                    "CHOWN",
                    "--entrypoint",
                    "chown",
                    "--volume",
                    source + ":" + target,
                    images["researcher"],
                    *(["-R"] if recursive else []),
                    "10001:10001",
                    target,
                ],
                env=environment,
                check=True,
            )
        subprocess.run(
            compose
            + [
                "exec",
                "-T",
                "--user",
                "10001:10001",
                "researcher",
                "git",
                "init",
                "-q",
                "/workspace",
            ],
            env=environment,
            check=True,
        )
        from bootstrap import prepare_research_python

        prepare_research_python(compose, environment, root.parents[1])
        sync_workspace(compose, environment, config)
        cleanup = (
            "from pathlib import Path; "
            "p=Path('/workspace/cases/ar-03/campaigns/campaign-01/metric'); "
            "p.rmdir() if p.exists() else None"
        )
        subprocess.run(
            compose
            + ["exec", "-T", "--user", "10001:10001", "researcher", "python", "-I", "-c", cleanup],
            env=environment,
            check=True,
        )
    elif args.command == "containers-down":
        SafetyGate(root).revoke("container shutdown requested")
        subprocess.run(compose + ["down"], env=environment, check=True)
    else:
        if args.command == "research":
            SafetyGate(root).require_permission()
        command = (
            ["login", "--device-auth"]
            if args.command == "login"
            else [
                "--no-daemon",
                # Isolation is enforced by Docker; nested bwrap cannot create namespaces.
                "--dangerously-bypass-approvals-and-sandbox",
                "--enable",
                "multi_agent_v2",
                "-C",
                "/workspace",
                "Выполни cases/ar-03/campaigns/active/campaign.md",
            ]
        )
        subprocess.run(
            compose
            + ["exec", "--user", "10001:10001"]
            + (
                ["-e", "PYTHONPATH=/codex/research-packages:/opt/arena"]
                if args.command == "research"
                else []
            )
            + ["researcher", "codex"]
            + command,
            env=environment,
            check=True,
        )


def stop_services(root):
    for measurement in Journal(root / "search/search.sqlite3").status()["running"]:
        binding = json.loads((root / "private/bindings" / f"{measurement}.json").read_text())
        expected = binding.get("expected_evaluation_id")
        if binding["status"] == "retryable" and binding["split"] == "train" and expected:
            path = root / "private/state/train/evaluations" / expected / "record.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record.get("metadata", record)["status"] in ("FINISHED", "FAILED"):
                continue
        raise ValueError("wait for accepted evaluations before stopping services")
    processes = root / "private/processes.json"
    if processes.exists():
        for item in json.loads(processes.read_text()):
            try:
                command = subprocess.check_output(
                    ["ps", "-p", str(item["pid"]), "-o", "command="], text=True
                )
                if str(root) in command and (
                    "operatorctl.py" in command or "operator_artifacts.py" in command
                ):
                    os.kill(item["pid"], signal.SIGTERM)
            except (ProcessLookupError, subprocess.CalledProcessError):
                pass
        processes.unlink()
    print('{"services_stopped":true}')


def wait_for_services(config):
    deadline = time.monotonic() + 20
    with httpx.Client(trust_env=False, timeout=1) as client:
        for endpoint in [*config["services"].values(), f"http://127.0.0.1:{config['port']}"]:
            while True:
                try:
                    if client.get(endpoint + "/healthz").is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("service startup failed; inspect private logs")
                time.sleep(0.2)


def start_services(root, config):
    from security import process_identity, startup_fingerprint

    private = root / "private"
    processes = private / "processes.json"
    if processes.exists():
        raise ValueError("process record exists; stop old services first")
    configure_docker(private)
    started = []
    environment = {**os.environ, "PYTHONPATH": str(ARENA), "PYTHONDONTWRITEBYTECODE": "1"}
    fingerprint = startup_fingerprint(root)
    environment["AR03_BOOT_FINGERPRINT"] = fingerprint
    for key in ("MLFLOW_TRACKING_URI", "ARENA_OPERATOR_MLFLOW_TRACKING_URI"):
        environment.pop(key, None)
    for role in [*config["services"], "control", "artifacts"]:
        if role == "artifacts":
            command = [
                sys.executable,
                "-B",
                str(CASE / "operator_artifacts.py"),
                "--config",
                str(private / "artifacts.json"),
                "watch",
            ]
        else:
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "serve",
                "--root",
                str(root),
                "--role",
                role,
            ]
        with (private / f"{role}.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=ARENA,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env=environment,
            )
        started.append(
            {
                "role": role,
                "pid": process.pid,
                "started_at": process_identity(process.pid)["started_at"],
                "startup_fingerprint": fingerprint,
            }
        )
    save(processes, started)
    wait_for_services(config)
    print(
        json.dumps(
            {
                "started": [p["role"] for p in started],
                "evolution_authorized": (private / "authorized.json").exists(),
            }
        )
    )


def request_operator_action(args, root, config):
    administrative = args.command in ("stop", "verify", "authorize", "pause")
    secret = root / "private/admin-token" if administrative else root / "research-token"
    if args.command in ("stop", "authorize") and not args.source_ref:
        raise ValueError("human instruction reference is required")
    body = None
    endpoint = "/progress"
    if administrative:
        endpoint = "/admin/" + args.command
        body = {"source_ref": args.source_ref, "reason": args.reason}
    with httpx.Client(trust_env=False, timeout=60) as client:
        result = client.request(
            "POST" if administrative else "GET",
            f"http://127.0.0.1:{config['port']}" + endpoint,
            json=body,
            headers={"Authorization": "Bearer " + secret.read_text().strip()},
        )
        result.raise_for_status()
        value = result.json()
        print(json.dumps(value, ensure_ascii=False))
        return value


def run_campaign(args, root, config):
    if not args.source_ref:
        raise ValueError("run requires the source reference of the human launch command")
    if not (root / "private/processes.json").exists():
        start_services(root, config)
    for command in ("containers-up", "verify", "authorize", "research"):
        action = argparse.Namespace(**{**vars(args), "command": command})
        if command in ("containers-up", "research"):
            try:
                containers(action, root, config)
            finally:
                if command == "research":
                    request_operator_action(
                        argparse.Namespace(**{**vars(args), "command": "pause"}), root, config
                    )
        else:
            request_operator_action(action, root, config)


def reference_baseline(args, root, config):
    """An operator-authorized reference run, outside the proposal journal."""
    from evaluator.checkpoints import read_document
    from evaluator.client import pack_candidate
    from evaluator.train import TrainSubmission
    from fastapi import BackgroundTasks
    from security import GuardedHarness, SafetyGate

    if not args.source_ref:
        raise ValueError("reference requires the human instruction source")
    configure_docker(root / "private")
    gate = SafetyGate(root, loaded_config=config)
    gate.require_live()
    configured = replace(
        settings(root, "train", config),
        state_dir=root / "private/state/baseline-reference",
        search_mode="standalone",
        search_journal=None,
        search_max_solutions=None,
        search_stop_path=None,
    )
    runner = GuardedHarness(make_runner(configured, configured.state_dir / "usage"), gate)
    app = train_app(
        TrainSettings(
            configured,
            config.get("campaign_id", root.name),
            "ar-03",
            config["dataset_id"],
            config["conditions_id"],
            tracking=False,
        ),
        solve=runner,
    )
    candidate = pack_candidate("baseline-reference", CASE / "ways/baseline")
    submission = TrainSubmission(candidate=candidate, example_ids=config["evaluation_ids"])
    background = BackgroundTasks()
    endpoint = next(
        route.endpoint
        for route in app.routes
        if route.path == "/v1/evaluations" and "POST" in route.methods
    )
    response = json.loads(endpoint(submission, background).body)
    print(json.dumps({"reference_started": response["evaluation_id"]}), flush=True)
    asyncio.run(background())
    record = read_document(
        configured.state_dir / "evaluations" / response["evaluation_id"] / "record.json"
    )
    if record["status"] != "FINISHED":
        raise RuntimeError("baseline reference failed; retain its existing checkpoint")
    result = {
        "kind": "baseline_reference",
        "counts_toward_proposal_budget": False,
        "candidate_digest": candidate.digest(),
        "evaluation_id": response["evaluation_id"],
        "pages": len(record["example_ids"]),
        "status": record["status"],
        "metric_version": record["params"]["metric_version"],
        "dataset_id": config["dataset_id"],
        "subset_id": record["params"]["subset_id"],
        "conditions_id": config["conditions_id"],
        "device": "cpu",
        "metrics": record["metrics"],
        "source_ref": args.source_ref,
    }
    save(root / "baseline-reference.json", result)
    save(CASE / "campaigns/pilot-01/baseline-reference.json", result)
    print(json.dumps(result), flush=True)


def lifecycle(args):
    from security import SafetyGate

    root = args.root.resolve()
    config = json.loads((root / "config.json").read_text())
    if args.command in ("arm", "research", "authorize"):
        raise ValueError("separate authorization/launch is disabled; use run --source-ref ...")
    if args.command == "reference":
        reference_baseline(args, root, config)
    elif args.command == "run":
        run_campaign(args, root, config)
    elif args.command in ("containers-up", "containers-down", "login"):
        if args.command == "login":
            configure_docker(root / "private")
            SafetyGate(root).require_live()
        containers(args, root, config)
    elif args.command == "start":
        SafetyGate(root).revoke("operator services restarted")
        start_services(root, config)
    elif args.command == "services-down":
        stop_services(root)
    else:
        request_operator_action(args, root, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "bootstrap",
            "export",
            "reference",
            "prepare",
            "serve",
            "start",
            "arm",
            "run",
            "verify",
            "pause",
            "stop",
            "services-down",
            "progress",
            "containers-up",
            "containers-down",
            "login",
            "research",
        ),
    )
    parser.add_argument("--root", type=Path, help="Default: <resources>/runs/pilot-01")
    parser.add_argument("--mode", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--max-solutions", type=int)
    parser.add_argument("--finalists", type=int)
    parser.add_argument("--output-count", type=int)
    parser.add_argument("--closed-source", type=Path)
    parser.add_argument(
        "--resources",
        type=Path,
        default=ARENA.parent / f"{ARENA.name}-operator",
        help="External operator resources: data/arena and runner-assets",
    )
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--role", choices=("train", "public", "private", "control"))
    parser.add_argument("--reason", default="Operator requested stop")
    parser.add_argument("--source-ref")
    parser.add_argument("--step", choices=("data", "models", "images", "all"), default="all")
    parser.add_argument("--output", type=Path, help="New campaign publication directory in AR-3")
    parser.add_argument("--skill", type=Path, help="Existing operator-approved Skill directory")
    args = parser.parse_args()
    if args.root is None:
        args.root = args.resources / "runs/pilot-01"
    if args.command == "bootstrap":
        from bootstrap import bootstrap

        bootstrap(args)
    elif args.command == "export":
        from publication import export_campaign

        export_campaign(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "serve":
        serve(args)
    else:
        lifecycle(args)


if __name__ == "__main__":
    main()
