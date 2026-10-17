"""Shared test fixtures: a scriptable mock provider/session and a fake sandbox.

These let swarm/coordinator tests drive the full LLM↔tool loop deterministically
with no Docker and no network. Wire them into the swarm with the ``patch_backends``
fixture, which repoints ``swarm.get_provider`` and replaces both sandbox classes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ctf_solver.providers.base import (
    ProviderBase,
    SolverResponse,
    SolverSession,
    TokenUsage,
    ToolCall,
)
from ctf_solver.sandbox import ExecResult


class FakeSandbox:
    """In-memory ``SandboxProtocol`` stub — no Docker, no subprocess."""

    def __init__(self, *args, **kwargs) -> None:
        self.started = False
        self.stopped = False
        self.files: dict[str, str | bytes] = {}
        # command substring -> ExecResult; falls back to ``default_exec``.
        self.exec_map: dict[str, ExecResult] = {}
        self.default_exec = ExecResult(exit_code=0, stdout="", stderr="")

    async def start(self) -> None:
        self.started = True

    async def exec(self, command: str, timeout_s: int = 300) -> ExecResult:
        for needle, res in self.exec_map.items():
            if needle in command:
                return res
        return self.default_exec

    async def read_file(self, path: str) -> str | bytes:
        return self.files.get(path, "")

    async def read_file_bytes(self, path: str) -> bytes:
        v = self.files.get(path, "")
        return v.encode() if isinstance(v, str) else v

    async def write_file(self, path: str, content: str | bytes) -> None:
        self.files[path] = content

    async def stop(self) -> None:
        self.stopped = True

    @property
    def container_id(self) -> str:
        return "fake"


def resp(
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
    done: bool | None = None,
    usage: TokenUsage | None = None,
) -> SolverResponse:
    """Build a SolverResponse. ``done`` defaults to True iff there are no tool calls."""
    tcs = tool_calls or []
    return SolverResponse(
        text=text,
        tool_calls=tcs,
        structured_output=None,
        usage=usage or TokenUsage(input_tokens=10, output_tokens=5),
        done=(not tcs) if done is None else done,
    )


def tool(name: str, **arguments) -> ToolCall:
    """Shorthand for a ToolCall with a deterministic call_id."""
    return ToolCall(name=name, arguments=arguments, call_id=f"call-{name}")


class MockSession(SolverSession):
    """Replays a scripted list of responses across ``send()`` calls.

    Script entries may be ``SolverResponse`` objects or zero-arg callables that
    return one (handy for branching on injected context). When the script is
    exhausted, returns a terminal ``done=True`` empty response so loops end.
    """

    def __init__(self, script: list | None = None) -> None:
        self._script = list(script or [])
        self.sent: list[tuple[str, list]] = []
        self.injected: list[str] = []
        self.closed = False

    async def send(self, message: str, tool_results=None) -> SolverResponse:
        self.sent.append((message, list(tool_results or [])))
        if self._script:
            item = self._script.pop(0)
            return item() if callable(item) else item
        return resp(text="", done=True)

    async def inject_context(self, text: str) -> None:
        self.injected.append(text)

    async def close(self) -> None:
        self.closed = True


class MockProvider(ProviderBase):
    """Provider that hands out ``MockSession`` objects from a per-solver script map."""

    name = "mock"

    def __init__(self, scripts: dict[str, list] | None = None) -> None:
        self.scripts = scripts or {}
        self.sessions: dict[str, MockSession] = {}

    def validate_config(self, config: dict) -> bool:
        return True

    async def create_session(self, solver_id, system_prompt, tools, config, sandbox=None) -> MockSession:
        session = MockSession(self.scripts.get(solver_id, []))
        session.system_prompt = system_prompt  # exposed for prompt-wiring assertions
        session.sandbox = sandbox  # exposed for sandbox-wiring assertions
        self.sessions[solver_id] = session
        return session


@pytest.fixture
def patch_backends(monkeypatch) -> Callable[[MockProvider], MockProvider]:
    """Return ``install(provider)`` to point the swarm at mock backends.

    It repoints ``swarm.get_provider`` at the given MockProvider (for every
    provider name) and replaces ``HostSandbox``/``DockerSandbox`` with
    ``FakeSandbox``. Tests should also set ``no_docker=True`` and a tmp
    ``log_dir`` on Settings to keep the run hermetic.
    """
    import ctf_solver.solver.swarm as swarm_mod

    def install(provider: MockProvider) -> MockProvider:
        monkeypatch.setattr(swarm_mod, "get_provider", lambda name: provider)
        monkeypatch.setattr(swarm_mod, "HostSandbox", FakeSandbox)
        monkeypatch.setattr(swarm_mod, "DockerSandbox", FakeSandbox)
        return provider

    return install
