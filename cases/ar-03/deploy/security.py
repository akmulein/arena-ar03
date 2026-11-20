"""Host-side, instance-bound safety gate for the existing Arena services."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from evaluator.runner import DockerRunner, HarnessRunner, RunnerError
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from probes import RESEARCHER

DEPLOY = Path(__file__).resolve().parent
PINS = json.loads((DEPLOY / "images.json").read_text())
SERVICES = ("researcher", "network-guard", "api-relay", "egress-proxy")
APPROVED_MODEL_FILES = frozenset(
    {
        "models/DocLayout-YOLO/doclayout_yolo_docstructbench_imgsz1024.pt",
        "models/SLANet_plus/config.json",
        "models/SLANet_plus/inference.json",
        "models/SLANet_plus/inference.pdiparams",
        "models/SLANet_plus/inference.yml",
        "models/PP-OCRv6_small_rec/inference.json",
        "models/PP-OCRv6_small_rec/inference.pdiparams",
        "models/PP-OCRv6_small_rec/inference.yml",
    }
)


def image_pins(root):
    """Local immutable image identities, written only by operator bootstrap."""
    path = Path(root).resolve().parents[1] / "images.json" if root else DEPLOY / "images.json"
    images = json.loads(path.read_text()) if path.exists() else PINS
    if set(images) != {"researcher", "runner"} or any(
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(char not in "0123456789abcdef" for char in value[7:])
        for value in images.values()
    ):
        raise ValueError("operator image pins must contain immutable local image IDs")
    return images


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(threading.get_ident()) + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def file_signature(path, contents=False):
    info = path.stat()
    result = [info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode, info.st_uid]
    return result + [hashlib.sha256(path.read_bytes()).hexdigest()] if contents else result


def validate_assets(directory):
    directory = Path(directory)
    entries = list(directory.rglob("*"))
    files = {path.relative_to(directory).as_posix() for path in entries if path.is_file()}
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != {"models"}
        or files != APPROVED_MODEL_FILES
        or any(path.is_symlink() for path in entries)
        or any(not path.is_dir() and not path.is_file() for path in entries)
    ):
        raise ValueError("runner assets must contain exactly the approved model files, no symlinks")


def startup_files(root):
    repo = DEPLOY.parents[2]
    paths = {
        path
        for path in DEPLOY.iterdir()
        if path.is_file()
        and path.suffix in (".py", ".yaml", ".sh", ".json", ".conf", ".Dockerfile", ".dockerignore")
    }
    for directory in (repo / "evaluator", repo / "runner"):
        paths.update(directory.rglob("*.py"))
    paths.update(
        DEPLOY.parent / name for name in ("train.py", "monitor.py", "operator_artifacts.py")
    )
    paths.update((repo / "pyproject.toml", repo / "uv.lock", root / "config.json"))
    local_images = root.parents[1] / "images.json"
    if local_images.exists():
        paths.add(local_images)
    paths.update(path for path in (root / "private/metric").rglob("*") if path.is_file())
    return {str(path): file_signature(path, True) for path in sorted(paths)}


def startup_fingerprint(root):
    return digest(startup_files(Path(root)))


def process_identity(pid):
    environment = dict(os.environ, LC_ALL="C", TZ="UTC")
    started = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "lstart="], env=environment, text=True
    ).strip()
    command = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "command="], env=environment, text=True
    ).strip()
    if not started or not command:
        raise ValueError("operator process is unavailable")
    return {"started_at": started, "command": command}


def operator_processes(root, fingerprint):
    processes = json.loads((root / "private/processes.json").read_text())
    if not processes:
        raise ValueError("operator processes are not registered")
    for process in processes:
        if process.get("startup_fingerprint") != fingerprint:
            raise ValueError("operator process uses an old configuration or trusted code")
        actual = process_identity(process["pid"])
        if actual["started_at"] != process.get("started_at"):
            raise ValueError("operator process start time changed")
        if str(root) not in actual["command"]:
            raise ValueError("operator process identity changed")
    return processes


def clean_source(source):
    return source.removeprefix("/host_mnt")


def runtime_signature(row):
    result = {key: row[key] for key in ("Id", "Image", "Config", "HostConfig", "NetworkSettings")}
    result["Mounts"] = sorted(row["Mounts"], key=lambda mount: mount["Destination"])
    result["StartedAt"] = row["State"]["StartedAt"]
    for network in result["NetworkSettings"]["Networks"].values():
        for key in ("Aliases", "DNSNames"):
            if isinstance(network.get(key), list):
                network[key] = sorted(network[key])
    return result


def docker(*arguments, timeout=10):
    environment = dict(os.environ)
    for key in ("DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    if not environment.get("DOCKER_HOST", "").startswith("unix://"):
        raise ValueError("local Docker configuration is required")
    return subprocess.check_output(
        ["docker", *arguments], env=environment, text=True, timeout=timeout
    )


def owned_runners(root):
    names = docker("ps", "-q", "--filter", "name=arena-runner-").split()
    if not names:
        return []
    records = json.loads(docker("inspect", *names))
    state = root / "private/state"
    return [
        row
        for row in records
        if any(
            m["Destination"] == "/candidate"
            and Path(clean_source(m["Source"])).is_relative_to(state)
            for m in row["Mounts"]
        )
    ]


def hidden_paths(root):
    return [
        str(root / "private" / name)
        for name in (
            "admin-token",
            "train-token",
            "public-token",
            "private-token",
            "metric",
            "state",
            "visualizations",
            "security",
        )
    ]


def researcher_checks(record, root):
    host, config = record["HostConfig"], record["Config"]
    mounts = {
        m["Destination"]: (clean_source(m["Source"]), m["RW"])
        for m in record["Mounts"]
        if m["Type"] == "bind"
    }
    expected = {
        "/workspace": (str(root / "research-workspace"), True),
        "/search": (str(root / "search"), False),
        "/run/secrets/submit-token": (str(root / "research-token"), False),
        "/workspace/cases/ar-03/_local/train-500.jsonl": (
            str(root / "private/data/train.jsonl"),
            False,
        ),
    }
    volumes = [m for m in record["Mounts"] if m["Type"] == "volume"]
    return (
        record["State"]["Running"]
        and not record["State"]["Paused"]
        and record["Image"] == image_pins(root)["researcher"]
        and config["User"] == "10001:10001"
        and host["ReadonlyRootfs"]
        and not host["Privileged"]
        and host["CapDrop"] == ["ALL"]
        and not host["CapAdd"]
        and "no-new-privileges" in " ".join(host["SecurityOpt"])
        and host["PidMode"] != "host"
        and host["IpcMode"] != "host"
        and host["NetworkMode"].startswith("container:")
        and mounts == expected
        and len(volumes) == 1
        and volumes[0]["Destination"] == "/codex"
        and volumes[0]["Name"] == "ar03-isolated_codex-auth"
    )


def validate_runner(runner, root=None):
    if (
        type(runner) is not DockerRunner
        or runner.image != image_pins(root)["runner"]
        or runner.network != "none"
        or runner.uses_model
        or runner.supports_build
        or runner.supports_asset_selection
        or runner.build_output_dir is not None
        or runner.tmpfs_exec
        or runner.memory != "6g"
        or runner.cpus != "2"
        or runner.pids_limit != 128
        or runner.tmpfs_size != "512m"
    ):
        raise RunnerError("unsafe runner configuration; no native fallback")
    if root is not None and runner.assets_dir != root.parents[1] / "runner-assets":
        raise RunnerError("runner assets must be the approved model-only directory")
    validate_assets(runner.assets_dir)


def sidecar_checks(name, row):
    host = row["HostConfig"]
    caps = {
        "network-guard": ["NET_ADMIN"],
        "api-relay": [],
        "egress-proxy": ["SETUID", "SETGID", "CHOWN", "DAC_OVERRIDE"],
    }
    networks = {"ar03-isolated_research"}
    if name != "network-guard":
        networks.add("ar03-isolated_outbound")
    return (
        host["ReadonlyRootfs"]
        and host["CapDrop"] == ["ALL"]
        and sorted(cap.removeprefix("CAP_") for cap in host["CapAdd"] or []) == sorted(caps[name])
        and "no-new-privileges" in " ".join(host["SecurityOpt"])
        and host["PidMode"] != "host"
        and host["IpcMode"] != "host"
        and not row["Mounts"]
        and set(row["NetworkSettings"]["Networks"]) == networks
    )


def firewall_normalized(text):
    rules = []
    for line in text.splitlines():
        tokens = shlex.split(line)
        normalized = []
        index = 0
        while index < len(tokens):
            if tokens[index] == "-m" and tokens[index + 1] in ("tcp", "udp"):
                index += 2
            elif tokens[index] == "--reject-with":
                index += 2
            else:
                normalized.append(",".join(sorted(tokens[index].split(","))).removesuffix("/32"))
                index += 1
        rules.append(normalized)
    return rules


FIREWALL = """-P OUTPUT DROP
-A OUTPUT -d 127.0.0.11 -j REJECT
-A OUTPUT -p udp --dport 53 -j REJECT
-A OUTPUT -p tcp --dport 53 -j REJECT
-A OUTPUT -o lo -j ACCEPT
-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
-A OUTPUT -d 172.29.83.20 -p tcp --dport 8080 -j ACCEPT
-A OUTPUT -d 172.29.83.30 -p tcp --dport 3128 -j ACCEPT
-P OUTPUT DROP
-A OUTPUT -o lo -j ACCEPT
"""


class SafetyGate:
    def __init__(self, root, loaded_config=None):
        self.root = Path(root).resolve()
        self.private = self.root / "private"
        self.evidence = self.private / "security/evidence.json"
        self.permission = self.private / "security/permission.json"
        self.lock = threading.RLock()
        self.reason = "security verification required"
        self.startup_fingerprint = startup_fingerprint(self.root)
        expected = os.environ.get("AR03_BOOT_FINGERPRINT", self.startup_fingerprint)
        self.restart_required = expected != self.startup_fingerprint
        if loaded_config is not None:
            disk_config = json.loads((self.root / "config.json").read_text())
            self.restart_required |= digest(loaded_config) != digest(disk_config)

    def assert_startup(self):
        try:
            current = startup_files(self.root)
            if self.restart_required or digest(current) != self.startup_fingerprint:
                raise ValueError("loaded configuration or trusted code is stale")
            processes = operator_processes(self.root, self.startup_fingerprint)
        except Exception:
            self.restart_required = True
            self.revoke("operator restart required")
            raise HTTPException(423, "operator restart required") from None
        return current, processes

    def revoke(self, reason):
        with self.lock:
            had_evidence = self.evidence.exists() or self.permission.exists()
            self.evidence.unlink(missing_ok=True)
            self.permission.unlink(missing_ok=True)
            (self.private / "authorized.json").unlink(missing_ok=True)
            self.reason = reason
            atomic_json(
                self.private / "security/revoked.json",
                {"reason": reason, "at": datetime.now(UTC).isoformat()},
            )
        if not had_evidence:
            return
        try:
            for row in owned_runners(self.root):
                docker("rm", "--force", row["Id"])
        except Exception:
            pass  # Returned predictions are also gated before the judge can see them.

    def snapshot(self):
        startup, processes = self.assert_startup()
        images = image_pins(self.root)
        names = ["ar03-isolated-" + name + "-1" for name in SERVICES]
        records = json.loads(docker("inspect", *names))
        by_name = {name: row for name, row in zip(SERVICES, records, strict=True)}
        if not researcher_checks(by_name["researcher"], self.root):
            raise ValueError("unsafe researcher configuration")
        for name, row in by_name.items():
            if (
                not row["State"]["Running"]
                or row["State"]["Paused"]
                or row["Image"] != images["researcher"]
                or row["HostConfig"]["Privileged"]
            ):
                raise ValueError("deployment unavailable or image changed: " + name)
            if name != "researcher" and not sidecar_checks(name, row):
                raise ValueError("unsafe sidecar configuration: " + name)
        guard = by_name["network-guard"]["Id"]
        if by_name["researcher"]["HostConfig"]["NetworkMode"] != "container:" + guard:
            raise ValueError("researcher is outside its network guard")
        firewall = docker("exec", names[1], "sh", "-c", "iptables -S OUTPUT && ip6tables -S OUTPUT")
        if firewall_normalized(firewall) != firewall_normalized(FIREWALL):
            raise ValueError("network policy differs from the approved configuration")
        config = json.loads((self.root / "config.json").read_text())
        if (
            Path(config["metric"]).resolve() != self.private / "metric"
            or Path(config["train_data"]).resolve() != self.private / "data/train.jsonl"
        ):
            raise ValueError("datasets and metric must stay in the trusted operator directory")
        for split, endpoint in config["services"].items():
            offset = {"train": 1, "public": 2, "private": 3}[split]
            if endpoint != f"http://127.0.0.1:{config['port'] + offset}":
                raise ValueError("evaluation must use the configured local Arena service")
        assets = Path(config["assets"])
        if assets != self.root.parents[1] / "runner-assets":
            raise ValueError("unexpected runner assets directory")
        validate_assets(assets)
        resources = {str(path): file_signature(path) for path in assets.rglob("*")}
        datasets = {
            str(path): file_signature(path) for path in (self.private / "data").glob("*.jsonl")
        }
        runner_image = json.loads(docker("image", "inspect", images["runner"]))[0]["Id"]
        if runner_image != images["runner"]:
            raise ValueError("runner image is unavailable")
        state = {
            "containers": [runtime_signature(row) for row in records],
            "firewall": firewall,
            "startup": startup,
            "datasets": datasets,
            "resources": resources,
            "config": file_signature(self.root / "config.json", True),
            "processes": processes,
            "docker_cli": file_signature(Path(shutil.which("docker")).resolve()),
        }
        return digest(state), by_name

    def require_live(self):
        with self.lock:
            try:
                saved = json.loads(self.evidence.read_text())
                actual, _ = self.snapshot()
                if saved["fingerprint"] != actual:
                    raise ValueError("deployment instance or configuration changed")
            except Exception:
                self.revoke("deployment changed or isolation unavailable")
                raise HTTPException(423, "security verification required") from None

    def require_permission(self):
        self.require_live()
        try:
            permission = json.loads(self.permission.read_text())
            evidence = json.loads(self.evidence.read_text())
            valid = (
                permission["fingerprint"] == evidence["fingerprint"]
                and permission["generation"] == evidence["generation"]
            )
        except Exception:
            valid = False
        if not valid:
            if self.permission.exists():
                self.revoke("invalid deployment authorization")
            raise HTTPException(423, "evolution is not authorized")

    def authorize(self, source_ref):
        if not source_ref:
            raise HTTPException(422, "human permission reference required")
        with self.lock:
            self.require_live()
            evidence = json.loads(self.evidence.read_text())
            atomic_json(
                self.permission,
                {
                    "fingerprint": evidence["fingerprint"],
                    "generation": evidence["generation"],
                    "source_ref": source_ref,
                    "at": datetime.now(UTC).isoformat(),
                },
            )
        return self.status()

    def pause(self):
        self.permission.unlink(missing_ok=True)
        return self.status()

    def status(self):
        if self.evidence.exists():
            try:
                self.require_live()
            except HTTPException:
                pass
        try:
            permission = json.loads(self.permission.read_text())
            evidence = json.loads(self.evidence.read_text())
            authorized = (
                permission["fingerprint"] == evidence["fingerprint"]
                and permission["generation"] == evidence["generation"]
            )
        except Exception:
            authorized = False
        return {"verified": self.evidence.is_file(), "authorized": authorized}

    def verify(self):
        with self.lock:
            self.revoke("verification in progress")
            before, records = self.snapshot()
            paths = hidden_paths(self.root) + [
                "/workspace/cases/ar-03/campaigns/campaign-01/metric"
            ]
            paths += ["/host_mnt" + path for path in paths]
            output = docker(
                "exec",
                "ar03-isolated-researcher-1",
                "/opt/arena/.venv/bin/python",
                "-I",
                "-c",
                RESEARCHER,
                json.dumps(paths),
                timeout=30,
            )
            checks = json.loads(output)
            checks.update(
                rootfs_readonly=records["researcher"]["HostConfig"]["ReadonlyRootfs"],
                no_host_pid_or_privileged=records["researcher"]["HostConfig"]["PidMode"] != "host",
                only_approved_mounts=researcher_checks(records["researcher"], self.root),
                trusted_mounts_readonly=all(
                    not m["RW"]
                    for m in records["researcher"]["Mounts"]
                    if m["Destination"] not in ("/workspace", "/codex")
                ),
            )
            if len(checks) != 21 or not all(checks.values()):
                raise HTTPException(423, "researcher isolation failed")
            after, final = self.snapshot()
            if before != after:
                raise HTTPException(423, "deployment changed during verification")
            result = {
                "fingerprint": after,
                "generation": uuid.uuid4().hex,
                "checks": checks,
                "instances": {
                    name: {
                        "container_id": row["Id"],
                        "image_id": row["Image"],
                        "started_at": row["State"]["StartedAt"],
                    }
                    for name, row in final.items()
                },
                "checked_at": datetime.now(UTC).isoformat(),
            }
            atomic_json(self.evidence, result)
            return {"passed": 21, "total": 21, "authorized": False}


class GuardedHarness(HarnessRunner):
    def __init__(self, original, gate):
        validate_runner(original.default_runner, gate.root)
        for runner in original.registry.values():
            validate_runner(runner, gate.root)
        super().__init__(original.default_runner, dict(original.registry))
        self.gate = gate

    def __call__(self, candidate_dir, input_text):
        try:
            self.gate.require_live()
            result = super().__call__(candidate_dir, input_text)
            self.gate.require_live()
            return result
        except HTTPException:
            raise RunnerError("security verification required") from None


class GuardedService:
    def __init__(self, app, gate):
        self.app, self.gate = app, gate

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            try:
                self.gate.require_permission()
            except HTTPException as error:
                await JSONResponse({"detail": error.detail}, status_code=error.status_code)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)
