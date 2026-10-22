"""Docker sandbox — isolated container for each solver with CTF tools."""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiodocker

from ctf_solver.sandbox import ExecResult

logger = logging.getLogger(__name__)

CONTAINER_LABEL = "ctf-agent"
MAX_CONCURRENT_CONTAINERS = 20
DEFAULT_MEMORY = 4 * 1024 * 1024 * 1024
_container_sem: asyncio.Semaphore | None = None


def get_container_semaphore() -> asyncio.Semaphore:
    global _container_sem
    if _container_sem is None:
        _container_sem = asyncio.Semaphore(MAX_CONCURRENT_CONTAINERS)
    return _container_sem


@dataclass
class DockerSandbox:
    image: str
    challenge_dir: str
    memory_limit: str = "4g"
    cpu_limit: int = 2
    workspace_dir: str = ""
    # Extra "host:container:ro" bind-mount strings, appended after the normal
    # workspace/distfiles binds. Used to confine a provider's own native agent
    # process (e.g. codex app-server) to this container instead of the host —
    # see providers/codex.py and solver/swarm.py's _locate_codex_native_bin_dir.
    extra_ro_binds: list[str] = field(default_factory=list)
    # Never silently grant external networking or a route back to the host.
    # "bridge" permits arbitrary egress and is intentionally opt-in.
    network_mode: str = "none"
    allow_host_gateway: bool = False
    _container: Any = field(default=None, repr=False)
    _docker: Any = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _sem_acquired: bool = field(default=False)

    @property
    def container_id(self) -> str:
        if not self._container:
            msg = "Sandbox not started"
            raise RuntimeError(msg)
        return self._container.id

    def _parse_memory_limit(self, s: str | None = None) -> int:
        s = (s or self.memory_limit).strip().lower()
        try:
            if s.endswith("g"):
                result = int(float(s[:-1]) * 1024 * 1024 * 1024)
            elif s.endswith("m"):
                result = int(float(s[:-1]) * 1024 * 1024)
            else:
                result = int(s)
            if result <= 0:
                return DEFAULT_MEMORY
            return result
        except (ValueError, IndexError):
            logger.warning("Invalid memory_limit %r, defaulting to 4GB", s)
            return DEFAULT_MEMORY

    async def start(self) -> None:
        if self.network_mode not in {"none", "bridge"}:
            raise ValueError("network_mode must be 'none' or 'bridge'")
        if self.allow_host_gateway and self.network_mode != "bridge":
            raise ValueError("host gateway requires explicit bridge networking")
        if self.network_mode == "bridge":
            logger.warning(
                "Sandbox uses unrestricted Docker bridge egress. "
                "This is NOT an outbound-domain allowlist."
            )
        sem = get_container_semaphore()
        await sem.acquire()
        self._sem_acquired = True
        try:
            self._docker = aiodocker.Docker()
            self.workspace_dir = tempfile.mkdtemp(prefix="ctf-workspace-")
            challenge_root = Path(self.challenge_dir).resolve()
            distfiles = str(challenge_root / "distfiles")
            binds: list[str] = [f"{self.workspace_dir}:/challenge/workspace:rw"]
            if Path(distfiles).exists():
                binds.append(f"{distfiles}:/challenge/distfiles:ro")
            binds.extend(self.extra_ro_binds)
            config = {
                "Image": self.image,
                "Cmd": ["sleep", "infinity"],
                "WorkingDir": "/challenge",
                "Tty": False,
                "Labels": {CONTAINER_LABEL: "true"},
                "NetworkDisabled": self.network_mode == "none",
                "HostConfig": {
                    "Binds": binds,
                    "NetworkMode": self.network_mode,
                    "ExtraHosts": (["host.docker.internal:host-gateway"]
                                   if self.allow_host_gateway else []),
                    "CapDrop": ["ALL"],
                    "CapAdd": ["SYS_PTRACE", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID", "CHOWN"],
                    "SecurityOpt": ["no-new-privileges:true"],
                    "Memory": self._parse_memory_limit(),
                    "NanoCpus": self.cpu_limit * 1_000_000_000,
                    "PidsLimit": 512,
                },
            }
            try:
                self._container = await self._docker.containers.create(config)
            except aiodocker.exceptions.DockerError as e:
                if "No such image" in str(e) or e.status == 404:
                    msg = f"Sandbox image '{self.image}' not found. Build it with:\n  docker build -t {self.image} sandbox/"
                    raise RuntimeError(msg) from e
                raise
            await self._container.start()
            logger.info("Sandbox started: %s", self._container.id[:12])
        except Exception:
            sem.release()
            self._sem_acquired = False
            raise

    async def exec(self, command: str, timeout_s: int = 300) -> ExecResult:
        if not self._container:
            msg = "Sandbox not started"
            raise RuntimeError(msg)
        async with self._lock:
            return await self._exec_inner(command, timeout_s)

    async def _exec_inner(self, command: str, timeout_s: int) -> ExecResult:
        wrapped = f"timeout --signal=KILL --kill-after=5 {timeout_s} bash -c {shlex.quote(command)}"
        exec_instance = await self._container.exec(
            cmd=["bash", "-c", wrapped], stdout=True, stderr=True, tty=False,
        )
        stream = exec_instance.start(detach=False)
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        async def _collect() -> None:
            async with stream:
                while True:
                    msg = await stream.read_out()
                    if msg is None:
                        break
                    data = msg.data
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    stream_type = getattr(msg, "stream", 1)
                    if stream_type == 1:
                        stdout_chunks.append(data)
                    else:
                        stderr_chunks.append(data)

        try:
            await asyncio.wait_for(_collect(), timeout=timeout_s + 30)
        except TimeoutError:
            return ExecResult(exit_code=-1, stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"), stderr="Command timed out")
        exec_info = await exec_instance.inspect()
        exit_code = exec_info.get("ExitCode", 0)
        return ExecResult(
            exit_code=exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )

    async def read_file(self, path: str) -> str | bytes:
        if not self._container:
            raise RuntimeError("Sandbox not started")
        try:
            tar = await asyncio.wait_for(self._container.get_archive(path), timeout=30)
        except aiodocker.exceptions.DockerError as e:
            raise FileNotFoundError(str(e)) from e
        with tar:
            for member in tar:
                if member.isfile():
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    file_data = f.read()
                    try:
                        return file_data.decode("utf-8")
                    except UnicodeDecodeError:
                        return file_data
        raise FileNotFoundError(f"No file found at {path}")

    async def read_file_bytes(self, path: str) -> bytes:
        result = await self.read_file(path)
        if isinstance(result, str):
            return result.encode("utf-8")
        return result

    async def write_file(self, path: str, content: str | bytes) -> None:
        if not self._container:
            raise RuntimeError("Sandbox not started")
        parent = str(Path(path).parent)
        await self.exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=10)
        if isinstance(content, str):
            content = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=Path(path).name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)
        await asyncio.wait_for(
            self._container.put_archive(parent, buf.getvalue()),
            timeout=30,
        )

    async def stop(self) -> None:
        if self._container:
            try:
                await self._container.delete(force=True)
            except Exception:
                pass
            self._container = None
        if self._docker:
            try:
                await self._docker.close()
            except Exception:
                pass
            self._docker = None
        if self.workspace_dir:
            import shutil

            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            self.workspace_dir = ""
        if self._sem_acquired:
            sem = get_container_semaphore()
            sem.release()
            self._sem_acquired = False
        logger.info("Sandbox stopped")


async def cleanup_orphan_containers() -> None:
    try:
        docker = aiodocker.Docker()
        try:
            containers = await docker.containers.list(all=True, filters={"label": [CONTAINER_LABEL]})
            for c in containers:
                try:
                    await c.delete(force=True)
                except Exception:
                    pass
            if containers:
                logger.info("Cleaned up %d orphan container(s)", len(containers))
        finally:
            await docker.close()
    except Exception as e:
        logger.warning("Orphan cleanup failed: %s", e)
