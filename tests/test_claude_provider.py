"""HTTP-level tests for the Claude (Anthropic Messages API) provider.

Everything is mocked — no real network. We swap the session's lazily-built
``httpx.AsyncClient`` (``session._client``) for a fake whose async ``post(...)``
returns a canned response object exposing ``.json()``, ``.raise_for_status()``
and ``.status_code``.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from ctf_solver.providers.base import ToolDef, ToolResult
from ctf_solver.providers.claude import ClaudeProvider, ClaudeSession


class FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, payload: dict, *, status_code: int = 200, raise_exc: Exception | None = None):
        self._payload = payload
        self.status_code = status_code
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self) -> dict:
        return self._payload


class FakeClient:
    """Fake httpx.AsyncClient: records the last request, returns a canned response."""

    def __init__(self, response: FakeResponse | None = None, *, post_exc: Exception | None = None):
        self._response = response
        self._post_exc = post_exc
        self.calls: list[dict] = []

    async def post(self, url, *, headers=None, json=None):  # noqa: A002 - mirror httpx kwarg name
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self._post_exc is not None:
            raise self._post_exc
        return self._response

    async def aclose(self) -> None:
        pass


def make_session(**config_overrides) -> ClaudeSession:
    config = {"anthropic_api_key": "sk-test", "claude_model": "claude-test"}
    config.update(config_overrides)
    tools = config_overrides.pop("_tools", None) or [
        ToolDef(name="bash", description="run a shell command", parameters={"type": "object"})
    ]
    return ClaudeSession("solver-1", "system prompt", tools, config)


def install_client(session: ClaudeSession, client: FakeClient) -> FakeClient:
    session._client = client
    return client


async def test_text_and_done():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 11, "output_tokens": 3},
                }
            )
        ),
    )

    out = await session.send("hello")

    assert out.text == "hi"
    assert out.tool_calls == []
    assert out.done is True
    assert out.usage.input_tokens == 11
    assert out.usage.output_tokens == 3


async def test_tool_call_parsing():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "content": [
                        {"type": "text", "text": "running"},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "bash",
                            "input": {"command": "ls"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )
        ),
    )

    out = await session.send("do something")

    assert out.done is False
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls"}
    assert tc.call_id == "tu_1"


async def test_tool_result_is_error_flag():
    session = make_session()
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}
            )
        ),
    )

    await session.send(
        "",
        tool_results=[
            ToolResult(content="boom", error="bad", call_id="tu_1", tool_use_id="tu_1")
        ],
    )

    sent_body = client.calls[0]["json"]
    user_msgs = [m for m in sent_body["messages"] if m["role"] == "user"]
    assert user_msgs, "expected a user message carrying the tool_result"
    blocks = user_msgs[-1]["content"]
    tr_block = next(b for b in blocks if b.get("type") == "tool_result")
    assert tr_block["is_error"] is True
    assert tr_block["tool_use_id"] == "tu_1"
    assert tr_block["content"] == "boom"


async def test_tool_result_no_error_omits_flag():
    session = make_session()
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}
            )
        ),
    )

    await session.send(
        "",
        tool_results=[ToolResult(content="all good", call_id="tu_2", tool_use_id="tu_2")],
    )

    blocks = client.calls[0]["json"]["messages"][-1]["content"]
    tr_block = next(b for b in blocks if b.get("type") == "tool_result")
    assert "is_error" not in tr_block


async def test_multimodal_image_block():
    session = make_session()
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}
            )
        ),
    )

    raw = b"\x89PNG\r\n\x1a\nfake-bytes"
    await session.send(
        "",
        tool_results=[
            ToolResult(content=(raw, "image/png"), call_id="tu_img", tool_use_id="tu_img")
        ],
    )

    blocks = client.calls[0]["json"]["messages"][-1]["content"]
    tr_block = next(b for b in blocks if b.get("type") == "tool_result")
    # tool_result.content is itself a list of blocks containing the image.
    inner = tr_block["content"]
    assert isinstance(inner, list)
    image_block = next(b for b in inner if b.get("type") == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == base64.b64encode(raw).decode("ascii")


async def test_message_rollback_on_http_error():
    session = make_session()
    err = httpx.HTTPStatusError(
        "500", request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        response=httpx.Response(500),
    )
    # raise_for_status raises; verify the user message we appended is rolled back.
    install_client(session, FakeClient(FakeResponse({}, status_code=500, raise_exc=err)))

    n_before = len(session._messages)
    with pytest.raises(httpx.HTTPStatusError):
        await session.send("this should be rolled back")

    assert len(session._messages) == n_before


async def test_message_rollback_on_post_exception():
    session = make_session()
    install_client(session, FakeClient(post_exc=httpx.ConnectError("boom")))

    n_before = len(session._messages)
    with pytest.raises(httpx.ConnectError):
        await session.send("dropped")

    assert len(session._messages) == n_before


async def test_usage_cache_read_tokens():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "content": [{"type": "text", "text": "cached"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 80,
                    },
                }
            )
        ),
    )

    out = await session.send("hi")

    assert out.usage.input_tokens == 100
    assert out.usage.output_tokens == 20
    assert out.usage.cache_read_tokens == 80


async def test_tool_use_keeps_done_false_even_with_end_turn():
    """done requires end_turn AND no tool_calls; a tool_use block forces done=False."""
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "content": [
                        {"type": "tool_use", "id": "tu_9", "name": "bash", "input": {}},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {},
                }
            )
        ),
    )

    out = await session.send("go")
    assert out.tool_calls and out.done is False


async def test_create_session_returns_claude_session():
    provider = ClaudeProvider()
    session = await provider.create_session(
        "s", "sys", [], {"anthropic_api_key": "sk-test"}
    )
    assert isinstance(session, ClaudeSession)
    await session.close()
