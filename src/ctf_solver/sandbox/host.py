"""Host sandbox — runs commands directly on the host (no isolation)."""

from __future__ import annotations

import asyncio
import re
import shlex
import tempfile
from pathlib import Path

from ctf_solver.sandbox import ExecResult

_BLOCKED_PATH_PREFIXES = ("/etc/", "/proc/", "/sys/", "/dev/", "/root/", "/boot/")


class HostSandbox:
    def __init__(self, challenge_dir: str = "") -> None:
        self.challenge_dir = str(Path(challenge_dir).resolve())
        self._workspace_dir = ""

    @property
    def container_id(self) -> str:
        return "host"

    @property
    def _distfiles_dir(self) -> Path:
        return Path(self.challenge_dir) / "distfiles"

    @property
    def _workspace_path(self) -> Path:
        return Path(self._workspace_dir) if self._workspace_dir else Path(self.challenge_dir) / "workspace"

    def _map_path(self, path: str) -> Path:
        if path == "/challenge":
            return Path(self.challenge_dir)
        if path.startswith("/challenge/distfiles/"):
            return self._distfiles_dir / path.removeprefix("/challenge/distfiles/")
        if path.startswith("/challenge/workspace/"):
            return self._workspace_path / path.removeprefix("/challenge/workspace/")
        if path == "/challenge/distfiles":
            return self._distfiles_dir
        if path == "/challenge/workspace":
            return self._workspace_path
        p = Path(path)
        if p.is_absolute():
            abs_str = str(p)
            for prefix in _BLOCKED_PATH_PREFIXES:
                if abs_str.startswith(prefix):
                    raise ValueError(f"Access to system path blocked: {path}")
            return p
        return Path(self.challenge_dir) / path

    def _rewrite_command_paths(self, command: str) -> str:
        replacements = [
            ("/challenge/distfiles", shlex.quote(str(self._distfiles_dir))),
            ("/challenge/workspace", shlex.quote(str(self._workspace_path))),
            ("/challenge", shlex.quote(self.challenge_dir)),
        ]
        def _replace_match(m: re.Match) -> str:
            matched = m.group(0)
            for old, new in replacements:
                if matched == old:
                    return new
            return matched
        return re.sub(r"/challenge(?:/distfiles|/workspace)?(?=/|\s|$)", _replace_match, command)

    async def start(self) -> None:
        self._workspace_dir = tempfile.mkdtemp(prefix="ctf-host-workspace-")

    async def exec(self, command: str, timeout_s: int = 300) -> ExecResult:
        command = self._rewrite_command_paths(command)
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.challenge_dir or None,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            return ExecResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        except TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return ExecResult(exit_code=-1, stdout="", stderr="Command timed out")
        except Exception as e:
            return ExecResult(exit_code=-1, stdout="", stderr=str(e))

    async def read_file(self, path: str) -> str | bytes:
        p = self._map_path(path)
        try:
            data = p.read_bytes()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data
        except FileNotFoundError:
            raise
        except Exception as e:
            raise FileNotFoundError(str(e)) from e

    async def read_file_bytes(self, path: str) -> bytes:
        p = self._map_path(path)
        return p.read_bytes()

    async def write_file(self, path: str, content: str | bytes) -> None:
        p = self._map_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode("utf-8")
        p.write_bytes(content)

    async def stop(self) -> None:
        if self._workspace_dir:
            import shutil
            shutil.rmtree(self._workspace_dir, ignore_errors=True)
            self._workspace_dir = ""
