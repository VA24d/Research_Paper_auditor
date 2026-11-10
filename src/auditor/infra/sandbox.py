"""The math sandbox: runs Python code written by the model, in a throwaway Docker container.

Checks of a paper's mathematics are better when the model can compute instead of guess:
evaluate a formula at small values, search for a counterexample, simplify an identity with
SymPy. That code comes from the model, which has read the paper, so it is treated as
untrusted. It runs in a container with no network, a read-only file system (only a small
`/tmp` is writable), an unprivileged user, and limits on memory, CPU, processes, time and
output. Repository code never runs here or anywhere else (decision D11).

The image is built from `docker/math/Dockerfile` (Python with SymPy, NumPy, mpmath,
NetworkX and z3). Without Docker, or without the image, the sandbox reports that it is
unavailable and the checks that would use it run without it.
"""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "auditor-math:1"
DOCKERFILE_DIR = Path(__file__).resolve().parents[3] / "docker" / "math"
MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float = 20.0
    memory: str = "1g"
    cpus: str = "1"
    pids: int = 64
    tmp_size: str = "64m"


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def docker_command(name: str, limits: SandboxLimits) -> list[str]:
    """The `docker run` command for one piece of code, read from standard input."""
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,size={limits.tmp_size}",
        "--memory",
        limits.memory,
        "--memory-swap",
        limits.memory,
        "--cpus",
        limits.cpus,
        "--pids-limit",
        str(limits.pids),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        IMAGE,
        "python",
        "-I",
        "-",
    ]


async def _docker(
    *args: str, stdin: bytes | None = None, timeout_s: float = 600
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(stdin), timeout_s)
    except BaseException:
        # A timeout or a cancelled task: end the docker client and reap it, so no process
        # outlives the call.
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(process.wait())
        raise
    return process.returncode or 0, out.decode("utf-8", "replace")


async def _stop(name: str, process: asyncio.subprocess.Process) -> None:
    """Stops the container itself, not only the docker client, and reaps the client."""
    with contextlib.suppress(OSError, TimeoutError):
        await _docker("kill", name, timeout_s=30)
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    await process.wait()


class MathSandbox:
    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    async def available(self) -> bool:
        """Docker answers and the image exists."""
        try:
            code, _ = await _docker("image", "inspect", IMAGE, timeout_s=30)
        except (OSError, TimeoutError):
            return False
        return code == 0

    async def build(self) -> None:
        code, out = await _docker("build", "-t", IMAGE, str(DOCKERFILE_DIR), timeout_s=1800)
        if code != 0:
            raise RuntimeError(f"building {IMAGE} failed:\n{out[-2000:]}")

    async def run(self, code: str, timeout_s: float | None = None) -> SandboxResult:
        name = f"auditor-math-{uuid.uuid4().hex[:12]}"
        limit = timeout_s if timeout_s is not None else self.limits.timeout_s
        process = await asyncio.create_subprocess_exec(
            *docker_command(name, self.limits),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(code.encode("utf-8")), limit)
        except TimeoutError:
            await _stop(name, process)
            return SandboxResult("", f"stopped after {limit:g} s", None, timed_out=True)
        except BaseException:
            # A cancelled check (the run stopped, a parent timeout) must not leave the
            # container running: it has no time limit of its own.
            await asyncio.shield(_stop(name, process))
            raise
        return SandboxResult(
            out.decode("utf-8", "replace")[:MAX_OUTPUT_CHARS],
            err.decode("utf-8", "replace")[:MAX_OUTPUT_CHARS],
            process.returncode,
        )
