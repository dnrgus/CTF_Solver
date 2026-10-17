"""z.ai provider — REST API (OpenAI-compatible chat completions)."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from ctf_solver.providers.base import (
    ProviderBase,
    SolverResponse,
    SolverSession,
    TokenUsage,
    ToolCall,
    ToolDef,
    ToolResult,
)


def _tools_to_openai(tools: list[ToolDef]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision"}


class ZAISession(SolverSession):
    def __init__(self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict) -> None:
        self._solver_id = solver_id
        self._api_key = config.get("zai_api_key", "")
        self._endpoint = config.get("zai_endpoint", "https://api.z.ai/api/paas/v4/").rstrip("/")
        self._model = config.get("zai_model", "glm-5.1")
        self._tools = _tools_to_openai(tools)
        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        self._pending_context: str | None = None
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    def _supports_vision(self) -> bool:
        model_lower = self._model.lower()
        return any(vm in model_lower for vm in VISION_MODELS)

    async def send(self, message: str, tool_results: list[ToolResult] | None = None) -> SolverResponse:
        n_before = len(self._messages)
        if tool_results:
            pending_image_parts: list[dict] = []
            for tr in tool_results:
                if isinstance(tr.content, tuple):
                    if self._supports_vision():
                        image_data, mime_type = tr.content
                        b64 = base64.b64encode(image_data).decode("ascii")
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tr.call_id,
                            "content": "Image loaded. See next message.",
                        })
                        pending_image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        })
                    else:
                        image_data, mime_type = tr.content
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tr.call_id,
                            "content": f"Image ({mime_type}, {len(image_data)} bytes) — vision not supported by {self._model}. Use bash: file, xxd, strings.",
                        })
                else:
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tr.call_id,
                        "content": tr.content,
                    })
            if pending_image_parts:
                self._messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the attached image from the tool result."},
                        *pending_image_parts,
                    ],
                })
        elif message:
            self._messages.append({"role": "user", "content": message})

        context_injected = False
        if self._pending_context:
            if self._messages:
                last = self._messages[-1]
                if last.get("role") == "user":
                    if isinstance(last["content"], str):
                        last["content"] += f"\n\n{self._pending_context}"
                    elif isinstance(last["content"], list):
                        last["content"].append({"type": "text", "text": self._pending_context})
            else:
                self._messages.append({"role": "user", "content": self._pending_context})
            self._pending_context = None
            context_injected = True

        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._messages),
            "max_tokens": 16384,
        }
        if self._tools:
            body["tools"] = self._tools

        client = self._get_client()
        try:
            resp = await client.post(
                f"{self._endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
        except Exception:
            del self._messages[n_before:]
            if context_injected:
                self._pending_context = None
            raise
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        text = msg.get("content", "") or ""
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append(ToolCall(
                name=fn.get("name", ""),
                arguments=args,
                call_id=tc.get("id", ""),
            ))

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )

        done = choice.get("finish_reason") in ("stop", "end_turn") and not tool_calls

        self._messages.append(msg)

        return SolverResponse(text=text, tool_calls=tool_calls, usage=usage, done=done)

    async def inject_context(self, text: str) -> None:
        self._pending_context = text

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class ZAIProvider(ProviderBase):
    name = "zai"

    def validate_config(self, config: dict) -> bool:
        return bool(config.get("zai_api_key", ""))

    async def create_session(
        self, solver_id: str, system_prompt: str, tools: list[ToolDef], config: dict,
        sandbox: object | None = None,
    ) -> ZAISession:
        return ZAISession(solver_id, system_prompt, tools, config)
