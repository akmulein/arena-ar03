import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evaluator.wait import WaitError, main, wait_any


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
@pytest.mark.parametrize("exit_code", [0, 3])
def test_wait_any_observes_first_exit_without_waiting_for_other_process(
    monkeypatch: pytest.MonkeyPatch, exit_code: int,
) -> None:
    def sleep(seconds: float) -> None:
        pytest.fail("Process events must not poll or sleep")

    monkeypatch.setattr("time.sleep", sleep)
    command = [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(int(sys.argv[1]))"]
    with subprocess.Popen([*command, "0"], stdin=subprocess.PIPE) as slow:
        with subprocess.Popen([*command, str(exit_code)], stdin=subprocess.PIPE) as fast:
            with ThreadPoolExecutor() as executor:
                try:
                    future = executor.submit(wait_any, {"slow": slow.pid, "fast": fast.pid})
                    assert fast.stdin is not None
                    fast.stdin.close()
                    assert future.result(5) == [{
                        "id": "fast", "pid": fast.pid, "process_status": "exited",
                        "evaluation_status": "unknown",
                    }]
                    assert slow.poll() is None
                finally:
                    assert slow.stdin is not None and fast.stdin is not None
                    slow.stdin.close()
                    fast.stdin.close()
                    fast.wait()
                    slow.wait()


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
def test_wait_any_reports_reaped_process_without_claiming_evaluation_success() -> None:
    with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
        process.wait()
    assert wait_any({"finished": process.pid}) == [{
        "id": "finished", "pid": process.pid, "process_status": "no_process",
        "evaluation_status": "unknown",
    }]


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
def test_wait_any_observes_an_exited_child_before_it_is_reaped() -> None:
    with subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"]) as process:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        assert wait_any({"failed-helper": process.pid}) == [{
            "id": "failed-helper", "pid": process.pid, "process_status": "exited",
            "evaluation_status": "unknown",
        }]
        assert process.wait() == 7


@pytest.mark.parametrize("processes", [
    {}, {"": 12}, {"name": 0}, {"name": -1}, {"name": True},
    {"a": 12, "b": 12}, {"self": os.getpid()},
])
def test_wait_any_rejects_invalid_processes_before_opening_descriptors(
    processes: dict[str, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def open_pid(pid: int, flags: int = 0) -> int:
        pytest.fail("Invalid process list must fail before attaching")

    monkeypatch.setattr(os, "pidfd_open", open_pid, raising=False)
    with pytest.raises(WaitError):
        wait_any(processes)


def test_wait_any_reports_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    with pytest.raises(WaitError, match="Linux pidfd"):
        wait_any({"job": 123})


def test_wait_any_closes_descriptors_when_another_process_cannot_be_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()

    def open_pid(pid: int, flags: int = 0) -> int:
        if pid == 12:
            return read_fd
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "pidfd_open", open_pid, raising=False)
    try:
        with pytest.raises(WaitError, match="attach"):
            wait_any({"first": 12, "second": 13})
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)


def test_wait_cli_saves_process_event_without_exposing_a_success_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def open_pid(pid: int, flags: int = 0) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(os, "pidfd_open", open_pid, raising=False)
    path = tmp_path / "event.json"
    assert main(["--process", "public=123", "--output", str(path)]) == 0
    event = json.loads(path.read_text())
    assert event == {"ready": [{
        "id": "public", "pid": 123, "process_status": "no_process",
        "evaluation_status": "unknown",
    }]}
    assert json.loads(capsys.readouterr().out) == event


@pytest.mark.parametrize("processes", [["job"], ["job=x"], ["job=1", "job=2"]])
def test_wait_cli_rejects_invalid_or_duplicate_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], processes: list[str],
) -> None:
    arguments = ["--output", str(tmp_path / "event.json")]
    for process in processes:
        arguments.extend(["--process", process])
    assert main(arguments) == 1
    assert capsys.readouterr().err.startswith("arena wait: ")
    assert not (tmp_path / "event.json").exists()
