"""Provider protocol and data models — the interface all AI providers implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""


@dataclass
class ToolResult:
    content: str | tuple[bytes, str]
    error: str | None = None
    tool_use_id: str = ""
    call_id: str = ""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class SolverResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured_output: dict | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    done: bool = False


class SolverSession(ABC):
    """Active session with an AI provider for one solver instance."""

    @abstractmethod
    async def send(self, message: str, tool_results: list[ToolResult] | None = None) -> SolverResponse: ...

    @abstractmethod
    async def inject_context(self, text: str) -> None: ...

    async def inject_image(self, data: bytes, mime_type: str) -> None:  # noqa: B027
        pass

    @abstractmethod
    async def close(self) -> None: ...


class ProviderBase(ABC):
    name: str

    @abstractmethod
    async def create_session(
        self,
        solver_id: str,
        system_prompt: str,
        tools: list[ToolDef],
        config: dict,
        sandbox: object | None = None,
    ) -> SolverSession:
        """`sandbox` is the solver's SandboxProtocol instance, when one is running.

        Only the Codex provider uses it (to run its own native agent process
        confined inside that same sandbox instead of on the bare host — see
        providers/codex.py). Every other provider ignores it; it is only ever
        a SandboxProtocol-shaped object, never a real tool for them to call."""
        ...

    @abstractmethod
    def validate_config(self, config: dict) -> bool: ...
