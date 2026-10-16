"""Tests for the Codex provider (JSON-RPC over a subprocess).

The Codex provider talks to a ``codex app-server`` child process over stdio
using JSON-RPC. We never launch a real process here: instead we monkeypatch
``asyncio.create_subprocess_exec`` (in the codex module's namespace) to return a
``FakeProcess`` whose stdin captures every written line and whose stdout feeds
queued JSON-RPC lines back to the provider's ``_read_loop``.

Determinism / no-hang strategy
------------------------------
``_start_process`` performs an ``initialize`` / ``initialized`` / ``thread/start``
handshake as ``_rpc`` calls that block on a response keyed by id, and ``send()``
issues a ``turn/start`` ``_rpc`` that must also be answered before it can drain
the turn. To keep tests from deadlocking, ``FakeStdin.write`` runs an
*auto-responder*: any outgoing JSON-RPC *request* (a line with both ``id`` and
``method``) immediately gets a matching ``{"id": N, "result": {...}}`` line
enqueued on stdout. That makes the handshake and ``turn/start`` self-completing
while leaving full control of turn *content* (notifications, ``turn/completed``)
to each test, which feeds those lines explicitly. The auto-responder can be
switched off (``auto_respond = False``) for the low-level ``_rpc`` / EOF tests.

To avoid the ``_drain_turn`` scheduling race (see
``test_drain_turn_strands_toolcall_when_completed_before_drain_starts``), the
``drive_turn`` helper feeds each content notification and then *pumps the event
loop until the read loop has consumed it* (the stdout queue drains back to
empty) before feeding the terminal ``turn/completed``. That guarantees the drain
loop is already blocked on ``queue.get()`` and picks up each ToolCall before the
turn-done event is set — the realistic streaming order.

Every ``send()`` / handshake await is wrapped in ``asyncio.wait_for(..., 5)`` so a
protocol mistake fails fast instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

import ctf_solver.providers.codex as codex_mod
from ctf_solver.providers.base import SolverResponse, ToolResult
from ctf_solver.providers.codex import CodexSession

TIMEOUT = 5.0


async def pump(n: int = 200) -> None:
    """Yield control to the event loop ``n`` times so background tasks run."""
    for _ in range(n):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------- #
# Fake subprocess plumbing
# --------------------------------------------------------------------------- #
class FakeStdout:
    """Async ``readline()`` that pops queued bytes; ``b""`` signals EOF."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed_line(self, obj: dict) -> None:
        self._queue.put_nowait((json.dumps(obj) + "\n").encode())

    def feed_eof(self) -> None:
        self._queue.put_nowait(b"")

    def pending(self) -> int:
        return self._queue.qsize()

    async def readline(self) -> bytes:
        return await self._queue.get()


class FakeStdin:
    """Captures every written line; auto-answers outgoing JSON-RPC requests."""

    def __init__(self, stdout: FakeStdout) -> None:
        self._stdout = stdout
        self.writes: list[bytes] = []
        self.auto_respond = True
        # Map outgoing-method -> result dict used to answer that request.
        # ``thread/start`` must hand back a thread id so ``_thread_id`` is set.
        self.auto_results: dict[str, dict] = {
            "initialize": {},
            "thread/start": {"thread": {"id": "thread-123"}},
            "turn/start": {},
        }
        self.drains = 0

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        if not self.auto_respond:
            return
        try:
            msg = json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        # Outgoing JSON-RPC *requests* carry both id and method; auto-respond so
        # the provider's ``_rpc`` future resolves without a real server.
        if isinstance(msg, dict) and "id" in msg and "method" in msg:
            result = self.auto_results.get(msg["method"], {})
            self._stdout.feed_line({"id": msg["id"], "result": result})

    async def drain(self) -> None:
        self.drains += 1

    # -- test helpers -------------------------------------------------------- #
    def written_messages(self) -> list[dict]:
        """All written lines parsed as JSON dicts (skips unparseable lines)."""
        out: list[dict] = []
        for raw in self.writes:
            try:
                out.append(json.loads(raw.decode()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return out

    def written_methods(self) -> list[str]:
        return [m["method"] for m in self.written_messages() if "method" in m]


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = FakeStdout()
        self.stdin = FakeStdin(self.stdout)
        self.terminated = False
        self.killed = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True
        # A real terminated process makes the read loop see EOF; mirror that so a
        # pending ``wait()`` and the read loop both unblock during ``close()``.
        self.stdout.feed_eof()

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waited = True
        return 0


@pytest.fixture
def fake_proc(monkeypatch) -> FakeProcess:
    """Install a FakeProcess as the result of ``create_subprocess_exec``."""
    proc = FakeProcess()

    async def fake_create(*args, **kwargs):
        return proc

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", fake_create)
    return proc


def make_session(**config) -> CodexSession:
    cfg = {"openai_api_key": "sk-test", **config}
    return CodexSession("solver-1", "system prompt", [], cfg)


async def wait_for_method(proc: FakeProcess, method: str) -> None:
    """Pump the loop until ``method`` has been written to stdin (or fail fast)."""
    for _ in range(200):
        await asyncio.sleep(0)
        if method in proc.stdin.written_methods():
            return
    raise AssertionError(f"{method!r} was never written; got {proc.stdin.written_methods()!r}")


async def feed_consumed(proc: FakeProcess, line: dict) -> None:
    """Feed one notification line and pump until the read loop has consumed it.

    Waiting for the stdout queue to drain back to empty guarantees the read loop
    processed the line (and, since ``_drain_turn`` is blocked on ``queue.get()``,
    that any emitted ToolCall was already appended) *before* the next line — in
    particular before any terminal ``turn/completed`` — is delivered.
    """
    proc.stdout.feed_line(line)
    for _ in range(200):
        await asyncio.sleep(0)
        if proc.stdout.pending() == 0:
            return
    raise AssertionError(f"read loop never consumed line {line!r}")


async def drive_turn(session: CodexSession, proc: FakeProcess, message: str, lines: list[dict]):
    """Run a single ``send(message)`` turn deterministically.

    Starts ``send()``, waits for ``turn/start`` to be written, then feeds each of
    ``lines`` (content notifications) one at a time — consuming each before the
    next — and finally a ``turn/completed`` to end the drain.
    """
    task = asyncio.create_task(session.send(message))
    await wait_for_method(proc, "turn/start")
    await pump(5)  # let _drain_turn reach its blocking queue.get()
    for line in lines:
        await feed_consumed(proc, line)
    await feed_consumed(proc, {"method": "turn/completed", "params": {}})
    return await asyncio.wait_for(task, timeout=TIMEOUT)


# --------------------------------------------------------------------------- #
# 1. _rpc id matching
# --------------------------------------------------------------------------- #
async def test_rpc_id_matching_resolves_future(fake_proc):
    """A response line carrying the matching id resolves the awaiting future."""
    session = make_session()
    # Drive the read loop directly, bypassing the handshake.
    session._proc = fake_proc
    session._read_task = asyncio.create_task(session._read_loop())
    # Turn off the auto-responder so we control exactly which id comes back.
    fake_proc.stdin.auto_respond = False

    task = asyncio.create_task(session._rpc("ping", {}))
    await pump(5)  # let the request get written / future registered

    sent = fake_proc.stdin.written_messages()
    assert sent and sent[0]["method"] == "ping"
    msg_id = sent[0]["id"]
    assert msg_id == 1  # first _msg_id
    assert msg_id in session._pending_responses

    fake_proc.stdout.feed_line({"id": msg_id, "result": {"pong": True}})
    result = await asyncio.wait_for(task, timeout=TIMEOUT)
    assert result == {"id": msg_id, "result": {"pong": True}}
    assert msg_id not in session._pending_responses  # future was popped

    await session.close()


# --------------------------------------------------------------------------- #
# 2. Handshake
# --------------------------------------------------------------------------- #
async def test_handshake_method_sequence(fake_proc):
    """First send issues initialize, initialized, thread/start, then turn/start."""
    session = make_session()
    resp = await drive_turn(session, fake_proc, "hi", [])

    methods = fake_proc.stdin.written_methods()
    # initialize + thread/start are RPC requests; initialized is a notification;
    # turn/start is the RPC that begins the conversation turn.
    for name in ("initialize", "initialized", "thread/start", "turn/start"):
        assert name in methods, f"{name} missing from {methods!r}"
    assert methods.index("initialize") < methods.index("initialized")
    assert methods.index("initialized") < methods.index("thread/start")
    assert methods.index("thread/start") < methods.index("turn/start")

    # thread id from the auto-responder's thread/start result was stored.
    assert session._thread_id == "thread-123"
    # No tool calls -> the turn completed cleanly.
    assert resp.done is True

    await session.close()


# --------------------------------------------------------------------------- #
# 3. Tool-call parsing
# --------------------------------------------------------------------------- #
async def test_tool_call_parsing_string_tool(fake_proc):
    """item/tool/call with ``tool`` as a string yields one ToolCall, done=False."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "go",
        [
            {
                "method": "item/tool/call",
                "id": 99,
                "params": {
                    "callId": "call-abc",
                    "tool": "bash",
                    "arguments": {"command": "ls"},
                },
            }
        ],
    )
    assert isinstance(resp, SolverResponse)
    assert resp.done is False
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls"}
    assert tc.call_id == "call-abc"
    # The RPC request id for this tool call was stored for the reply path.
    assert session._tool_request_ids.get("call-abc") == 99
    await session.close()


async def test_tool_call_parsing_dict_tool(fake_proc):
    """item/tool/call with ``tool`` as a dict ``{"name": ...}`` resolves the name."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "go",
        [
            {
                "method": "item/tool/call",
                "id": 7,
                "params": {
                    "callId": "call-dict",
                    "tool": {"name": "bash"},
                    "arguments": {"command": "id"},
                },
            }
        ],
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "bash"
    assert resp.tool_calls[0].call_id == "call-dict"
    assert resp.tool_calls[0].arguments == {"command": "id"}
    assert resp.done is False
    await session.close()


# --------------------------------------------------------------------------- #
# 4. Turn continuation (THE KEY FIX)
#
# SUSPECTED SRC BUG (see assertion (b) below): the ``tool_results`` branch of
# ``send()`` never clears ``self._turn_done_event`` before falling through to
# ``_drain_turn``. The event is still set from the *previous* turn's
# ``turn/completed``, so ``_drain_turn``'s ``while not self._turn_done_event.
# is_set()`` loop returns immediately with an EMPTY list — handing the swarm
# ``done=True`` with no tool calls and ending the solver after one tool round,
# which is precisely the regression this fix was meant to prevent. The new-turn
# branch DOES clear the event (and ``_text_buffer``); the continuation branch
# must do the same. FIXED: the continuation branch now clears _turn_done_event
# and resets _text_buffer, so this is a live regression test.
# --------------------------------------------------------------------------- #
async def test_turn_continuation_sends_rpc_response_and_drains_next_batch(fake_proc):
    """Regression test for the turn-continuation bug.

    After a first send() yields a tool call, replying via
    ``send("", tool_results=[...])`` must:
      (a) write a JSON-RPC *response* (id+result, NO "method") for that tool call,
      (b) FALL THROUGH to draining and return the *next* tool-call batch rather
          than an immediate empty ``tool_calls == []``.
    """
    session = make_session()

    # --- First turn: emit a tool call. -------------------------------------
    resp1 = await drive_turn(
        session,
        fake_proc,
        "start",
        [
            {
                "method": "item/tool/call",
                "id": 555,
                "params": {
                    "callId": "call-1",
                    "tool": "bash",
                    "arguments": {"command": "ls"},
                },
            }
        ],
    )
    assert [tc.call_id for tc in resp1.tool_calls] == ["call-1"]
    assert resp1.done is False
    assert session._tool_request_ids.get("call-1") == 555

    writes_before = len(fake_proc.stdin.writes)

    # --- Continuation: reply with the tool result. -------------------------
    task = asyncio.create_task(
        session.send("", tool_results=[ToolResult(content="output", call_id="call-1")])
    )
    # Let send() write the RPC response and reach the blocking drain.
    await pump(10)
    # Feed the NEXT tool-call batch (consumed) then complete the turn.
    await feed_consumed(
        fake_proc,
        {
            "method": "item/tool/call",
            "id": 556,
            "params": {
                "callId": "call-2",
                "tool": "bash",
                "arguments": {"command": "cat flag"},
            },
        },
    )
    await feed_consumed(fake_proc, {"method": "turn/completed", "params": {}})
    resp2 = await asyncio.wait_for(task, timeout=TIMEOUT)

    # (a) A JSON-RPC *response* (id+result, NOT a request with "method") was sent
    #     for the tool call, addressed to the stored request id (555).
    new_writes = [json.loads(raw.decode()) for raw in fake_proc.stdin.writes[writes_before:]]
    rpc_responses = [m for m in new_writes if "id" in m and "result" in m and "method" not in m]
    assert rpc_responses, f"expected an RPC response write, got {new_writes!r}"
    assert any(m["id"] == 555 for m in rpc_responses), (
        f"RPC response not addressed to stored id 555: {rpc_responses!r}"
    )
    # The reply content was relayed.
    reply = next(m for m in rpc_responses if m["id"] == 555)
    assert reply["result"]["success"] is True
    assert reply["result"]["contentItems"] == [{"type": "inputText", "text": "output"}]

    # (b) send() drained for the next batch and returned the NEXT tool call,
    #     NOT an immediate empty list. This is the core regression assertion.
    assert resp2.tool_calls, (
        "continuation send() returned an empty tool_calls list — the "
        "turn-continuation fix regressed (it did not fall through to _drain_turn)"
    )
    assert [tc.call_id for tc in resp2.tool_calls] == ["call-2"]
    assert resp2.done is False

    await session.close()


# --------------------------------------------------------------------------- #
# 5. turn/completed -> done with text
# --------------------------------------------------------------------------- #
async def test_turn_completed_agent_message_text_sets_done(fake_proc):
    """A real app-server agentMessage item (direct ``text`` field, as observed
    against a live `codex app-server`) is captured into response.text."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "summarize",
        [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "hello world"}},
            }
        ],
    )
    assert resp.done is True
    assert resp.tool_calls == []
    assert resp.text == "hello world"
    await session.close()


async def test_turn_completed_agent_message_content_array_fallback(fake_proc):
    """Fallback shape: agentMessage with a content-array instead of ``text``."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "summarize",
        [
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ],
                    }
                },
            }
        ],
    )
    assert resp.text == "hello world"
    await session.close()


async def test_user_message_echo_does_not_overwrite_text_buffer(fake_proc):
    """The echoed userMessage item/completed (which fires before the assistant's
    own agentMessage) must not be mistaken for the assistant's reply."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "go",
        [
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": "go"}],
                    }
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "real reply"}},
            },
        ],
    )
    assert resp.text == "real reply"
    await session.close()


# --------------------------------------------------------------------------- #
# 6. Token usage
# --------------------------------------------------------------------------- #
async def test_token_usage_reflected(fake_proc):
    """thread/tokenUsage/updated populates the returned SolverResponse.usage."""
    session = make_session()
    resp = await drive_turn(
        session,
        fake_proc,
        "work",
        [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 1200,
                            "outputTokens": 340,
                            "cachedInputTokens": 800,
                        }
                    }
                },
            }
        ],
    )
    assert resp.usage.input_tokens == 1200
    assert resp.usage.output_tokens == 340
    assert resp.usage.cache_read_tokens == 800
    await session.close()


# --------------------------------------------------------------------------- #
# 7. _wake_pending_on_eof
# --------------------------------------------------------------------------- #
async def test_eof_wakes_pending_rpc_with_runtimeerror(fake_proc):
    """stdout EOF while an _rpc future is pending fails it with RuntimeError."""
    session = make_session()
    session._proc = fake_proc
    session._read_task = asyncio.create_task(session._read_loop())
    # Don't auto-respond — we want the request to stay pending until EOF.
    fake_proc.stdin.auto_respond = False

    task = asyncio.create_task(session._rpc("hangs", {}))
    await pump(5)
    assert session._pending_responses, "request should be registered as pending"

    fake_proc.stdout.feed_eof()  # read loop sees b"" -> _wake_pending_on_eof

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(task, timeout=TIMEOUT)

    assert session._pending_responses == {}
    await session.close()


# --------------------------------------------------------------------------- #
# 8. close()
# --------------------------------------------------------------------------- #
async def test_close_terminates_process_and_cancels_read_task(fake_proc):
    """close() terminates the process and cancels the read task without raising."""
    session = make_session()
    session._proc = fake_proc
    read_task = asyncio.create_task(session._read_loop())
    session._read_task = read_task

    await asyncio.wait_for(session.close(), timeout=TIMEOUT)

    assert fake_proc.terminated is True
    assert fake_proc.waited is True

    # close() cancels the read task but does not await it; settle the cancellation
    # here and confirm it neither hangs nor surfaces an exception.
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(read_task, timeout=TIMEOUT)
    assert read_task.done()
    if not read_task.cancelled():
        assert read_task.exception() is None


# --------------------------------------------------------------------------- #
# Regression: _drain_turn scheduling race (FIXED — it now sweeps the queue
# after the event check so a fast turn/completed cannot strand a ToolCall).
# --------------------------------------------------------------------------- #
async def test_drain_turn_strands_toolcall_when_completed_before_drain_starts(fake_proc):
    """The tool call must survive even if turn/completed is processed first."""
    session = make_session()
    task = asyncio.create_task(session.send("go"))
    await wait_for_method(fake_proc, "turn/start")

    # Feed BOTH the tool call and the completion together, WITHOUT pumping in
    # between, so the read loop drains them into the queue (and sets the
    # turn-done event) before _drain_turn begins. This is the racy ordering.
    fake_proc.stdout.feed_line(
        {
            "method": "item/tool/call",
            "id": 42,
            "params": {"callId": "call-x", "tool": "bash", "arguments": {"command": "ls"}},
        }
    )
    fake_proc.stdout.feed_line({"method": "turn/completed", "params": {}})

    resp = await asyncio.wait_for(task, timeout=TIMEOUT)
    # Correct expectation: the tool call is NOT dropped.
    assert [tc.call_id for tc in resp.tool_calls] == ["call-x"]
    assert resp.done is False
    await session.close()
