"""Codex provider — JSON-RPC over subprocess stdio."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path

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
    def __init__(self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict) -> None:
        self._solver_id = solver_id
        self._config = config
        self._system_prompt = system_prompt
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

    async def _start_process(self) -> None:
        env = {**os.environ}
        if self._config.get("openai_api_key"):
            env["OPENAI_API_KEY"] = self._config["openai_api_key"]
        self._proc = await asyncio.create_subprocess_exec(
            "codex",
            "app-server",
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
                "cwd": ".",
                "approvalPolicy": "on-request",
                "sandbox": "read-only",
                "dynamicTools": SANDBOX_TOOLS,
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
                        text = ""
                        for item in params.get("item", {}).get("content", []):
                            if item.get("type") == "text":
                                text += item.get("text", "")
                        self._text_buffer = text
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
        """Read queued items until the model emits a tool-call batch or the turn
        completes. Shared by the new-turn and tool-result continuation paths."""
        tool_calls: list[ToolCall] = []
        while not self._turn_done_event.is_set():
            item = await self._tool_call_queue.get()
            if item is None:
                break
            tool_calls.append(item)
        # If turn/completed landed before/while we drained, sweep any tool calls
        # queued ahead of the None sentinel so they are not stranded (TOCTOU).
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

    async def create_session(self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict) -> CodexSession:
        return CodexSession(solver_id, system_prompt, tools, config)
