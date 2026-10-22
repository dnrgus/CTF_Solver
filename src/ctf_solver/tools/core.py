"""SDK-agnostic tool logic — pure async functions."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import shlex
from urllib.parse import urlparse

import httpx

from ctf_solver.collaboration.message_bus import ChallengeMessageBus
from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.sandbox import SandboxProtocol

MAX_OUTPUT = 24_000
WEB_FETCH_MAX = 20_000
WEBHOOK_MAX = 8000


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    lines = text.split("\n")
    head = "\n".join(lines[:200])
    return head[:limit] + f"\n... [truncated — {len(text)} total chars, {len(lines)} lines]"


async def do_bash(sandbox: SandboxProtocol, command: str, timeout_seconds: int = 60) -> str:
    result = await sandbox.exec(command, timeout_s=timeout_seconds)
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    if result.exit_code != 0:
        parts.append(f"[exit {result.exit_code}]")
    out = "\n".join(parts).strip() or "(no output)"
    return _truncate(out)


async def do_read_file(sandbox: SandboxProtocol, path: str) -> str:
    try:
        data = await sandbox.read_file(path)
    except Exception as e:
        return f"Error reading file: {e}"
    if isinstance(data, bytes):
        return (
            f"Binary file ({len(data)} bytes) — use bash to inspect:\n"
            f"  file {path}\n  xxd {path} | head -40\n  strings {path}"
        )
    return _truncate(data) if isinstance(data, str) else str(data)


async def do_write_file(sandbox: SandboxProtocol, path: str, content: str) -> str:
    try:
        await sandbox.write_file(path, content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def do_list_files(sandbox: SandboxProtocol, path: str = "/challenge/distfiles") -> str:
    result = await sandbox.exec(f"ls -la {shlex.quote(path)}")
    if result.exit_code != 0:
        return result.stderr.strip() or f"Error listing {path}"
    return result.stdout.strip() or f"{path} is empty."


async def do_web_fetch(url: str, method: str = "GET", body: str = "") -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "Fetch error: only http/https URLs are allowed."
    host = parsed.hostname or ""
    if not host:
        return "Fetch error: missing hostname."

    def _blocked_ip(ip) -> bool:
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        )

    try:
        ip = ipaddress.ip_address(host)
        if _blocked_ip(ip):
            return "Fetch error: access to private/local IP addresses is blocked."
    except ValueError:
        pass
    try:
        import socket
        infos = await asyncio.to_thread(
            socket.getaddrinfo, host, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_rest, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if _blocked_ip(ip):
                return "Fetch error: hostname resolves to blocked IP address."
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(verify=True, timeout=30.0, follow_redirects=False) as client:
            resp = await client.request(method, url, content=body or None, headers={"User-Agent": "ctf-solver-agent/1.0"})
            text = resp.text
            prefix = f"HTTP {resp.status_code} {resp.reason_phrase}\n{'─' * 40}\n"
            if len(text) > WEB_FETCH_MAX:
                text = text[:WEB_FETCH_MAX] + f"\n... [truncated, total {len(resp.text)} bytes]"
            return prefix + text
    except Exception as e:
        return f"Fetch error: {e}"


async def do_webhook_create() -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post("https://webhook.site/token")
            if resp.status_code != 200:
                return f"webhook.site error: HTTP {resp.status_code}"
            data = resp.json()
            return json.dumps({"uuid": data["uuid"], "url": f"https://webhook.site/{data['uuid']}"})
    except Exception as e:
        return f"webhook_create error: {e}"


async def do_webhook_get_requests(uuid: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"https://webhook.site/token/{uuid}/requests")
            if resp.status_code != 200:
                return f"webhook.site error: HTTP {resp.status_code}"
            data = resp.json()
            if not data.get("data"):
                return "No requests received yet."
            out = json.dumps(data["data"], indent=2)
            return out[:WEBHOOK_MAX] if len(out) > WEBHOOK_MAX else out
    except Exception as e:
        return f"webhook_get_requests error: {e}"


async def do_submit_flag(
    flag: str,
    flag_pattern: str,
    submitted_flags: set[str],
) -> tuple[str, bool]:
    """Format-check one candidate. Returns ``(message, accepted_as_candidate)``.

    This never verifies anything: a format match only makes the string a
    CANDIDATE. Verification (verifier command / reproduction) is decided by the
    swarm, which owns the flag records.
    """
    from ctf_solver.tools.flag import is_exact_flag

    flag = flag.strip()
    if not flag:
        return "Empty flag — nothing to validate.", False
    if flag in submitted_flags:
        return "Already tried this flag.", False
    if not is_exact_flag(flag, flag_pattern):
        return f"Flag '{flag}' does not match expected pattern.", False
    submitted_flags.add(flag)
    return f"Flag candidate recorded (format ok, not yet verified): {flag}", True


async def do_notify_coordinator(message: str, event_bus: EventBus | None = None, solver_id: str = "") -> str:
    """Send a message to the coordinator via the event bus."""
    if not event_bus:
        return "No event bus available."
    event_bus.publish(SolverEvent(type="user_hint", solver_id=solver_id, data={"message": message}))
    return "Message sent to coordinator."


async def do_check_findings(message_bus: ChallengeMessageBus | None, model_spec: str) -> str:
    """Get unread findings from sibling solvers."""
    if not message_bus:
        return "No message bus available."
    findings = await message_bus.check(model_spec)
    if not findings:
        return "No new findings from other agents."
    return message_bus.format_unread(findings)
