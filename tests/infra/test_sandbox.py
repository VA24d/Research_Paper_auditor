import asyncio
from typing import Any

import pytest

from auditor.infra import sandbox as sandbox_module
from auditor.infra.sandbox import IMAGE, MathSandbox, SandboxLimits, docker_command


def test_the_container_has_no_network_and_tight_limits() -> None:
    command = docker_command("auditor-run-1", SandboxLimits())
    assert command[:3] == ["docker", "run", "--rm"]
    joined = " ".join(command)
    for flag in ("--network none", "--read-only", "--memory 1g", "--cpus 1", "--pids-limit 64"):
        assert flag in joined
    assert "--tmpfs /tmp:rw,size=64m" in joined
    assert "--user 65534:65534" in joined
    assert command[-4:] == [IMAGE, "python", "-I", "-"]


class _HangingProcess:
    """A docker client that never answers, as when the model's code loops."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


async def test_a_cancelled_run_stops_its_container(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _HangingProcess()
    commands: list[tuple[str, ...]] = []

    async def spawn(*args: Any, **kwargs: Any) -> _HangingProcess:
        return process

    async def docker(*args: str, **kwargs: Any) -> tuple[int, str]:
        commands.append(args)
        return 0, ""

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(sandbox_module, "_docker", docker)
    run = asyncio.create_task(MathSandbox(SandboxLimits(timeout_s=60)).run("while True: pass"))
    await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert commands
    assert commands[0][0] == "kill"
    assert commands[0][1].startswith("auditor-math-")
    assert process.killed


@pytest.fixture
async def sandbox() -> MathSandbox:
    box = MathSandbox(SandboxLimits(timeout_s=10))
    if not await box.available():
        pytest.skip("Docker or the auditor-math image is not available")
    return box


@pytest.mark.docker
async def test_sympy_runs_and_prints(sandbox: MathSandbox) -> None:
    result = await sandbox.run(
        "import sympy as sp\nn=sp.symbols('n')\nprint(sp.summation(n,(n,1,10)))"
    )
    assert result.ok
    assert result.stdout.strip() == "55"


@pytest.mark.docker
async def test_the_network_is_blocked(sandbox: MathSandbox) -> None:
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\nprint('reached')"
    result = await sandbox.run(code)
    assert not result.ok
    assert "reached" not in result.stdout


@pytest.mark.docker
async def test_a_runaway_loop_is_stopped(sandbox: MathSandbox) -> None:
    result = await sandbox.run("while True:\n    pass", timeout_s=3)
    assert result.timed_out
    assert not result.ok


@pytest.mark.docker
async def test_files_cannot_be_written_outside_tmp(sandbox: MathSandbox) -> None:
    result = await sandbox.run("open('/home/x.txt', 'w').write('x')")
    assert not result.ok
