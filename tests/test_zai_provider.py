"""HTTP-level tests for the z.ai (OpenAI-compatible chat completions) provider.

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
from ctf_solver.providers.zai import VISION_MODELS, ZAIProvider, ZAISession


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


def make_session(model: str = "glm-5.1", **config_overrides) -> ZAISession:
    config = {"zai_api_key": "sk-test", "zai_model": model}
    config.update(config_overrides)
    tools = [ToolDef(name="bash", description="run a shell command", parameters={"type": "object"})]
    return ZAISession("solver-1", "system prompt", tools, config)


def install_client(session: ZAISession, client: FakeClient) -> FakeClient:
    session._client = client
    return client


async def test_text_and_done():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }
            )
        ),
    )

    out = await session.send("hello")

    assert out.text == "hi"
    assert out.tool_calls == []
    assert out.done is True
    assert out.usage.input_tokens == 7
    assert out.usage.output_tokens == 2


async def test_done_on_end_turn_finish_reason():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "choices": [{"message": {"content": "bye"}, "finish_reason": "end_turn"}],
                    "usage": {},
                }
            )
        ),
    )

    out = await session.send("hi")
    assert out.done is True


async def test_tool_call_parsing():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"ls"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {},
                }
            )
        ),
    )

    out = await session.send("go")

    assert out.done is False
    assert out.text == ""  # None content normalised to ""
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls"}
    assert tc.call_id == "c1"


async def test_bad_json_args_fallback():
    session = make_session()
    install_client(
        session,
        FakeClient(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c2",
                                        "function": {"name": "bash", "arguments": "not json"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {},
                }
            )
        ),
    )

    out = await session.send("go")

    assert out.tool_calls[0].arguments == {"raw": "not json"}


async def test_supports_vision_true_for_vision_model():
    session = make_session(model="gpt-4o")
    assert session._supports_vision() is True


async def test_supports_vision_false_for_non_vision_model():
    session = make_session(model="glm-5.1")
    assert session._supports_vision() is False


def test_vision_models_constant_contains_gpt4o():
    assert "gpt-4o" in VISION_MODELS


async def test_multimodal_vision_path_emits_image_url():
    session = make_session(model="gpt-4o")
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            )
        ),
    )

    raw = b"\x89PNG\r\n\x1a\nbytes"
    await session.send(
        "",
        tool_results=[ToolResult(content=(raw, "image/png"), call_id="c_img")],
    )

    messages = client.calls[0]["json"]["messages"]
    # A tool message acknowledging the image...
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert any(m.get("tool_call_id") == "c_img" for m in tool_msgs)
    # ...followed by a user message carrying the image_url data URL.
    user_list_msgs = [
        m for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert user_list_msgs, "expected a follow-up user message with image parts"
    parts = user_list_msgs[-1]["content"]
    img_part = next(p for p in parts if p.get("type") == "image_url")
    url = img_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64encode(raw).decode("ascii") in url


async def test_multimodal_non_vision_path_uses_text_fallback():
    session = make_session(model="glm-5.1")
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            )
        ),
    )

    raw = b"\x89PNGdata"
    await session.send(
        "",
        tool_results=[ToolResult(content=(raw, "image/png"), call_id="c_img2")],
    )

    messages = client.calls[0]["json"]["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert "vision not supported" in content
    # No follow-up multimodal user message in the fallback path.
    assert not any(
        m.get("role") == "user" and isinstance(m.get("content"), list) for m in messages
    )


async def test_text_tool_result_passed_through():
    session = make_session()
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            )
        ),
    )

    await session.send("", tool_results=[ToolResult(content="stdout here", call_id="c3")])

    messages = client.calls[0]["json"]["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"] == "stdout here"
    assert tool_msgs[-1]["tool_call_id"] == "c3"


async def test_message_rollback_on_http_error():
    session = make_session()
    err = httpx.HTTPStatusError(
        "500", request=httpx.Request("POST", "https://api.z.ai/chat/completions"),
        response=httpx.Response(500),
    )
    install_client(session, FakeClient(FakeResponse({}, status_code=500, raise_exc=err)))

    n_before = len(session._messages)
    with pytest.raises(httpx.HTTPStatusError):
        await session.send("rolled back")

    assert len(session._messages) == n_before


async def test_message_rollback_on_post_exception():
    session = make_session()
    install_client(session, FakeClient(post_exc=httpx.ConnectError("boom")))

    n_before = len(session._messages)
    with pytest.raises(httpx.ConnectError):
        await session.send("dropped")

    assert len(session._messages) == n_before


async def test_endpoint_url_and_trailing_slash_handling():
    session = make_session(zai_endpoint="https://example.test/v1/")
    client = install_client(
        session,
        FakeClient(
            FakeResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            )
        ),
    )

    await session.send("hi")

    assert client.calls[0]["url"] == "https://example.test/v1/chat/completions"


async def test_create_session_returns_zai_session():
    provider = ZAIProvider()
    session = await provider.create_session("s", "sys", [], {"zai_api_key": "sk-test"})
    assert isinstance(session, ZAISession)
    await session.close()
