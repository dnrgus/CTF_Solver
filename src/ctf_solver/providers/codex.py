"""Codex provider — JSON-RPC over subprocess stdio."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from ctf_solver.providers.base import (
    ProviderBase,
    SolverResponse,
    SolverSession,
    TokenUsage,
    ToolCall,
    ToolDef,
    ToolResult,
)

logger = logging.getLogger(__name__)

def locate_codex_native_bin_dir() -> str | None:
    """Find the directory holding the `codex` CLI's native, statically-linked
    binaries (`codex` + its `codex-code-mode-host` sibling), so they can be
    bind-mounted read-only into a Docker sandbox and run confined to it
    instead of directly on the host.

    Without this, `codex`'s own agentic loop has native tool-use/sub-agent
    capabilities entirely separate from this project's dynamicTools — with
    read access to the whole real filesystem under its default sandbox modes
    — that this project's Docker sandbox never confines (confirmed against a
    live `codex app-server`: it read arbitrary files outside the intended
    sandbox on the first live run of this provider). Returns None if nothing
    matching the expected npm-package layout (or a plain standalone native
    binary) is found; the caller must NOT fall back to running codex
    unconfined on the host in that case.
    """
    exe = shutil.which("codex")
    if not exe:
        return None
    resolved = Path(exe).resolve()
    # npm's JS-shim layout: <pkg>/bin/codex.js execs a platform package's
    # node_modules/@openai/codex-<platform>/vendor/<rust-target-triple>/bin/codex.
    for parent in list(resolved.parents)[:4]:
        for candidate in sorted(parent.glob("node_modules/@openai/codex-*/vendor/*/bin/codex")):
            if (candidate.parent / "codex-code-mode-host").exists():
                return str(candidate.parent)
    # Already a native, statically-linked binary (non-npm install) — use its
    # own directory, but only if its sibling helper is right there too.
    if _is_elf(resolved) and (resolved.parent / "codex-code-mode-host").exists():
        return str(resolved.parent)
    return None


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


SANDBOX_TOOLS = [
    {
        "name": "bash",
        "description": "Execute a bash command in the Docker sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"},
                "timeout_seconds": {"type": "integer", "default": 60},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory in the sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "/challenge/distfiles"}},
        },
    },
    {
        "name": "submit_flag",
        "description": "Submit a flag candidate for validation.",
        "inputSchema": {
            "type": "object",
            "properties": {"flag": {"type": "string"}},
            "required": ["flag"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "body": {"type": "string", "default": ""},
            },
            "required": ["url"],
        },
    },
    {
        "name": "webhook_create",
        "description": "Create a webhook.site endpoint.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "webhook_get_requests",
        "description": "Get requests received by webhook.",
        "inputSchema": {
            "type": "object",
            "properties": {"uuid": {"type": "string"}},
            "required": ["uuid"],
        },
    },
    {
        "name": "view_image",
        "description": "View an image file.",
        "inputSchema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "notify_coordinator",
        "description": "Send a message to the coordinator.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "check_findings",
        "description": "Check for insights from other solvers.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class CodexSession(SolverSession):
    def __init__(
        self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict,
        sandbox: Any | None = None,
    ) -> None:
        self._solver_id = solver_id
        self._config = config
        self._system_prompt = system_prompt
        self._tools = tools
        self._sandbox = sandbox
        self._proc: asyncio.subprocess.Process | None = None
        self._thread_id: str | None = None
        self._msg_id = 0
        self._pending_responses: dict[int, asyncio.Future] = {}
        self._tool_request_ids: dict[str, int] = {}
        self._tool_call_queue: asyncio.Queue[ToolCall | None] = asyncio.Queue()
        self._text_buffer: str = ""
        self._usage: TokenUsage = TokenUsage()
        self._read_task: asyncio.Task | None = None
        self._turn_done_event = asyncio.Event()
        self._pending_context: str | None = None

    # In-container paths where swarm.py's extra_ro_binds mounts the native
    # codex binaries, and where the workspace bind gives us a writable spot
    # for an ephemeral CODEX_HOME (copied auth.json only — never the real
    # ~/.codex, which also holds unrelated session history/notes/etc).
    _CONTAINER_BIN = "/opt/codex-bin/codex"
    _CONTAINER_CODEX_HOME = "/challenge/workspace/.codex-home"

    def _sandboxed(self) -> bool:
        """True when a real (Docker) sandbox is attached, not HostSandbox's
        "host" sentinel or no sandbox at all."""
        if self._sandbox is None:
            return False
        return getattr(self._sandbox, "container_id", "host") not in ("", "host")

    async def _start_process(self) -> None:
        sandboxed = self._sandboxed()
        api_key = self._config.get("openai_api_key")

        if sandboxed:
            # Run codex app-server's own process INSIDE the same Docker
            # sandbox as the dynamicTools bash redirect, instead of on the
            # host. codex's native agent tooling (its own command execution,
            # sub-agents) is entirely separate from this project's
            # dynamicTools and is NOT confined by this project's sandbox on
            # its own — confirmed live: with no containment it read files
            # outside the intended sandbox (see locate_codex_native_bin_dir's
            # docstring). Inside the container, "danger-full-access" to that
            # process only ever means the container's own filesystem.
            cmd = ["docker", "exec", "-i"]
            if api_key:
                cmd += ["-e", "OPENAI_API_KEY"]  # pulled from this env=, not argv
            cmd += [
                "-e", f"CODEX_HOME={self._CONTAINER_CODEX_HOME}",
                self._sandbox.container_id, self._CONTAINER_BIN, "app-server",
            ]
            env = {**os.environ}
            if api_key:
                env["OPENAI_API_KEY"] = api_key
            thread_cwd = "/challenge"
            approval_policy = "never"
            sandbox_mode = "danger-full-access"
        else:
            cmd = ["codex", "app-server"]
            env = {**os.environ}
            if api_key:
                env["OPENAI_API_KEY"] = api_key
            thread_cwd = "."
            approval_policy = "on-request"
            sandbox_mode = "read-only"

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        await self._rpc(
            "initialize",
            {
                "clientInfo": {"name": "ctf-solver-agent", "version": "1.0.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self._send_notification("initialized", {})
        resp = await self._rpc(
            "thread/start",
            {
                "model": self._config.get("codex_model", "gpt-5.4"),
                "personality": "pragmatic",
                "baseInstructions": self._system_prompt,
                "cwd": thread_cwd,
                "approvalPolicy": approval_policy,
                "sandbox": sandbox_mode,
                # Respect an explicitly empty tools list (the coordinator's
                # text-only role) instead of always handing out bash etc. —
                # codex's own native agent tooling is a separate concern
                # (see this method's containment comments above) that this
                # does NOT disable, but there is no reason to also hand it
                # OUR dynamicTools when the caller asked for none.
                "dynamicTools": SANDBOX_TOOLS if self._tools else [],
            },
        )
        self._thread_id = resp.get("result", {}).get("thread", {}).get("id")

    async def _rpc(self, method: str, params: dict) -> dict:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Codex process is not running")
        self._msg_id += 1
        msg_id = self._msg_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_responses[msg_id] = future
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        try:
            self._proc.stdin.write((json.dumps(request) + "\n").encode())
            await self._proc.stdin.drain()
            return await asyncio.wait_for(future, timeout=300.0)
        except Exception:
            self._pending_responses.pop(msg_id, None)
            raise

    async def _send_notification(self, method: str, params: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write((json.dumps(notification) + "\n").encode())
        await self._proc.stdin.drain()

    async def _send_rpc_response(self, msg_id: int, result: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
        self._proc.stdin.write((json.dumps(response) + "\n").encode())
        await self._proc.stdin.drain()

    def _wake_pending_on_eof(self) -> None:
        for future in list(self._pending_responses.values()):
            if not future.done():
                future.set_exception(RuntimeError("Codex process exited unexpectedly"))
        self._pending_responses.clear()

    async def _read_loop(self) -> None:
        try:
            while self._proc and self._proc.stdout:
                line = await self._proc.stdout.readline()
                if not line:
                    self._wake_pending_on_eof()
                    break
                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                if "id" in msg and "result" in msg:
                    future = self._pending_responses.pop(msg["id"], None)
                    if future and not future.done():
                        future.set_result(msg)
                elif "id" in msg and "error" in msg:
                    future = self._pending_responses.pop(msg["id"], None)
                    if future and not future.done():
                        future.set_exception(
                            RuntimeError(json.dumps(msg["error"], ensure_ascii=False))
                        )
                elif "method" in msg:
                    method = msg.get("method", "")
                    params = msg.get("params", {})
                    msg_id = msg.get("id")

                    if method == "item/tool/call":
                        call_id = params.get("callId", "")
                        tool_name = params.get("tool", "")
                        if isinstance(tool_name, dict):
                            tool_name = tool_name.get("name", "")
                        arguments = params.get("arguments", {})
                        if not isinstance(arguments, dict):
                            arguments = {}

                        if call_id and msg_id is not None:
                            self._tool_request_ids[call_id] = msg_id

                        await self._tool_call_queue.put(
                            ToolCall(name=tool_name, arguments=arguments, call_id=call_id)
                        )
                    elif method == "item/completed":
                        item = params.get("item", {})
                        # Only the assistant's own reply updates the buffer —
                        # not the echoed userMessage item that completes first.
                        if item.get("type") == "agentMessage":
                            text = item.get("text")
                            if text is None:
                                # Older/alternate shape: content is a list of
                                # {"type": "text", "text": ...} parts.
                                text = "".join(
                                    c.get("text", "")
                                    for c in item.get("content", []) or []
                                    if isinstance(c, dict) and c.get("type") == "text"
                                )
                            self._text_buffer = text or ""
                    elif method == "turn/completed":
                        self._turn_done_event.set()
                        await self._tool_call_queue.put(None)
                    elif method == "thread/tokenUsage/updated":
                        usage = params.get("tokenUsage", {}).get("last", {})
                        self._usage = TokenUsage(
                            input_tokens=usage.get("inputTokens", 0),
                            output_tokens=usage.get("outputTokens", 0),
                            cache_read_tokens=usage.get("cachedInputTokens", 0),
                        )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Codex read loop error: %s", e)
            self._wake_pending_on_eof()

    async def send(self, message: str, tool_results: list[ToolResult] | None = None) -> SolverResponse:
        if not self._proc:
            await self._start_process()

        if tool_results:
            # Reply to the model's pending tool-call requests. The turn is still
            # active, so we fall through to the shared drain loop below to read
            # the model's next output. Returning here (as before) would hand the
            # swarm an empty tool_calls list and end the solver after one round.
            for tr in tool_results:
                content_items: list[dict] = []
                if isinstance(tr.content, tuple):
                    image_data, mime_type = tr.content
                    b64 = base64.b64encode(image_data).decode("ascii")
                    content_items.append({"type": "inputImage", "image": {"url": f"data:{mime_type};base64,{b64}"}})
                    content_items.append({"type": "inputText", "text": f"Image ({mime_type}, {len(image_data)} bytes)"})
                else:
                    content_items.append({"type": "inputText", "text": tr.content})
                request_id = self._tool_request_ids.pop(tr.call_id, None)
                if request_id is not None:
                    await self._send_rpc_response(request_id, {
                        "contentItems": content_items,
                        "success": tr.error is None,
                    })
                else:
                    logger.warning("No stored request id for tool call %s", tr.call_id)
            # The turn is still active: reset turn state so the shared drain waits
            # for the model's next batch instead of seeing the prior turn/completed.
            self._text_buffer = ""
            self._turn_done_event.clear()
        else:
            # New user input (or injected context) starts a fresh turn.
            prompt_text = message
            if self._pending_context:
                prompt_text += f"\n\n{self._pending_context}"
                self._pending_context = None

            self._text_buffer = ""
            self._turn_done_event.clear()
            await self._rpc(
                "turn/start",
                {
                    "threadId": self._thread_id,
                    "input": [{"type": "text", "text": prompt_text}],
                },
            )

        tool_calls = await self._drain_turn()
        done = not tool_calls
        return SolverResponse(text=self._text_buffer, tool_calls=tool_calls, usage=self._usage, done=done)

    async def _drain_turn(self) -> list[ToolCall]:
        """Read queued items until the turn completes or at least one tool
        call is ready to hand back. Shared by the new-turn and tool-result
        continuation paths.

        A live `codex app-server` was confirmed (by direct JSON-RPC trace) to
        emit exactly ONE `item/tool/call` and then pause — waiting for our
        RPC response — rather than batching several before pausing. The
        previous version of this method kept blocking on queue.get() after
        that first call, waiting for a second item that never arrives until
        we answer the first one: a real deadlock (the model waits on us, we
        wait on it), which in practice only ever resolved via the swarm's
        outer per-call timeout — burning the whole turn budget on every
        single tool call. So: return as soon as the queue has yielded at
        least one ToolCall, draining any more that happen to already be
        queued (non-blocking) in case a future/alternate codex build does
        batch them, but never blocking further for one that isn't there yet.
        """
        tool_calls: list[ToolCall] = []
        while not self._turn_done_event.is_set():
            item = await self._tool_call_queue.get()
            if item is None:
                return tool_calls
            tool_calls.append(item)
            # Give an already-in-flight completion signal for THIS SAME turn
            # (e.g. turn/completed landing right after this tool call) one
            # scheduler tick to arrive, so its None sentinel doesn't leak
            # into a future _drain_turn() call and get mistaken there for an
            # instant "nothing more to do" on an unrelated later turn.
            await asyncio.sleep(0)
            while not self._tool_call_queue.empty():
                nxt = self._tool_call_queue.get_nowait()
                if nxt is None:
                    return tool_calls
                tool_calls.append(nxt)
            return tool_calls
        # Turn already completed before we even started draining — sweep
        # whatever tool calls are queued ahead of the None sentinel so they
        # are not stranded (TOCTOU with _read_loop setting the event).
        while not self._tool_call_queue.empty():
            item = self._tool_call_queue.get_nowait()
            if item is not None:
                tool_calls.append(item)
        return tool_calls

    async def inject_context(self, text: str) -> None:
        self._pending_context = text

    async def close(self) -> None:
        if self._read_task:
            self._read_task.cancel()
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


class CodexProvider(ProviderBase):
    name = "codex"

    def validate_config(self, config: dict) -> bool:
        if config.get("openai_api_key"):
            return True
        # No API key configured — fall back to the `codex` CLI's own stored
        # login (ChatGPT subscription), the same auth `_start_process` already
        # inherits via `os.environ` when no key is injected. Best-effort: this
        # only checks that *something* is logged in, not that it is still valid.
        return shutil.which("codex") is not None and (Path.home() / ".codex" / "auth.json").exists()

    async def create_session(
        self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict,
        sandbox: Any | None = None,
    ) -> CodexSession:
        return CodexSession(solver_id, system_prompt, tools, config, sandbox=sandbox)
