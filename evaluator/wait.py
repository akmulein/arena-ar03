import argparse
import json
import os
import selectors
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


class WaitError(Exception):
    pass


def wait_any(processes: Mapping[str, int]) -> list[dict[str, object]]:
    if not processes or any(
        not isinstance(identifier, str) or not identifier.strip()
        or type(pid) is not int or pid <= 0 or pid == os.getpid()
        for identifier, pid in processes.items()
    ) or len(set(processes.values())) != len(processes):
        raise WaitError("provide distinct named positive PIDs, excluding this process")
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        raise WaitError("Linux pidfd support is required for event-driven waiting")
    descriptors = []
    ready: dict[str, str] = {}
    try:
        with selectors.DefaultSelector() as selector:
            for identifier, pid in processes.items():
                try:
                    descriptor = pidfd_open(pid, 0)
                except ProcessLookupError:
                    ready[identifier] = "no_process"
                    continue
                except OSError as error:
                    raise WaitError("cannot attach to process with Linux pidfd") from error
                descriptors.append(descriptor)
                selector.register(descriptor, selectors.EVENT_READ, identifier)
            while not ready:
                for key, _ in selector.select():
                    ready[key.data] = "exited"
        return [
            {"id": identifier, "pid": pid, "process_status": ready[identifier],
             "evaluation_status": "unknown"}
            for identifier, pid in processes.items() if identifier in ready
        ]
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wait for any local process exit without polling")
    parser.add_argument("--process", action="append", required=True, metavar="ID=PID")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        processes = {}
        for value in arguments.process:
            identifier, separator, pid = value.partition("=")
            if not separator or not identifier.strip() or identifier in processes:
                raise WaitError("process must use a distinct ID=PID")
            try:
                processes[identifier] = int(pid)
            except ValueError:
                raise WaitError("process PID must be an integer") from None
        result = {"ready": wait_any(processes)}
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        arguments.output.write_text(content, encoding="utf-8")
        print(content, end="")
        return 0
    except WaitError as error:
        print(f"arena wait: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("arena wait: cannot wait or save the process event", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
