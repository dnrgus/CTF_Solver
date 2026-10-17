"""ChallengeSwarm — parallel solver orchestration for a single challenge."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiodocker

from ctf_solver.collaboration.loop_detect import LOOP_WARNING_MESSAGE, LoopDetector
from ctf_solver.collaboration.message_bus import ChallengeMessageBus, InsightMessage
from ctf_solver.config import Settings, get_active_providers
from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.prompts import ChallengeMeta, build_prompt, list_distfiles
from ctf_solver.providers import get_provider
from ctf_solver.providers.base import SolverSession, ToolCall, ToolDef, ToolResult
from ctf_solver.providers.codex import locate_codex_native_bin_dir
from ctf_solver.sandbox.docker import DockerSandbox
from ctf_solver.sandbox.host import HostSandbox
from ctf_solver.session_state import SessionState, SolverState
from ctf_solver.solver.solver_base import ResultStatus, SolverResult
from ctf_solver.tools.core import (
    do_bash,
    do_check_findings,
    do_list_files,
    do_notify_coordinator,
    do_read_file,
    do_submit_flag,
    do_web_fetch,
    do_webhook_create,
    do_webhook_get_requests,
    do_write_file,
)
from ctf_solver.tools.flag import extract_flags
from ctf_solver.tracing import SolverTracer
from ctf_solver.tracking.circuit_breaker import CircuitBreaker
from ctf_solver.tracking.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 3

SUBMISSION_COOLDOWNS = [0, 30, 120, 300, 600]

# Re-engagement (bump) of solvers that stopped without a flag: cooldown grows
# linearly with the bump count, capped. Module-level so tests can shrink it.
BUMP_COOLDOWN_STEP = 30.0
BUMP_COOLDOWN_MAX = 300.0
INSIGHT_SHARE_INTERVAL = 5
MIN_STEPS_BEFORE_UNCONTACTED_FLAG = 3

_PLACEHOLDER_MARKERS = ("placeholder", "example", "fake", "xxxx", "redacted")
_PLACEHOLDER_BODIES = {"flag", "...", "", "test", "here", "your_flag_here"}
# Bait/decoy flags planted in challenge files or sandbox images to catch naive
# strings/grep solving are, empirically, short human-readable phrases with a
# "this isn't real" word as a distinct token (underscore/dot/space-bounded) --
# "local_safelua_popen_test", "LOCAL_ALIAS_BYPASS", "dev.flag.local.testing.only",
# "local_cache_poisoning_success" all match; a real flag's thematic content
# essentially never does.
_PLACEHOLDER_WORD_RE = re.compile(
    r"(?:^|[_.\s])(local|dev|dummy|sample|placeholder|example|fake|redacted|test(?:ing)?|staging)(?:[_.\s]|$)",
    re.IGNORECASE,
)


def _is_placeholder_flag(flag: str) -> bool:
    """True for template flags like CTF{flag} or known decoy/bait strings
    planted in challenge sandboxes to catch naive strings/grep solving."""
    body = flag[flag.find("{") + 1 : flag.rfind("}")].strip().lower() if "{" in flag else ""
    return (
        body in _PLACEHOLDER_BODIES
        or any(m in body for m in _PLACEHOLDER_MARKERS)
        or bool(_PLACEHOLDER_WORD_RE.search(body))
    )


def _parse_connection_info(connection_info: str) -> tuple[str | None, str | None]:
    """Best-effort host/port extraction from a free-form connection hint like
    "nc 169.58.126.107 5007", "169.58.126.107:5007", or a web challenge's
    "https://foo.example.com/" (no explicit port -> scheme default)."""
    if not connection_info:
        return None, None
    s = connection_info.strip()
    m = re.match(r"^(https?)://([^/:\s]+)(?::(\d+))?", s, re.IGNORECASE)
    if m:
        scheme, host, port = m.group(1).lower(), m.group(2), m.group(3)
        return host, port or ("443" if scheme == "https" else "80")
    tokens = re.split(r"[\s:]+", s)
    host: str | None = None
    port: str | None = None
    for tok in tokens:
        if re.fullmatch(r"\d{1,5}", tok) and port is None:
            port = tok
        elif host is None and tok.lower() not in ("nc", "ncat", "telnet", "connect", "to", "at"):
            host = tok
    return host, port


def _load_harness(path: str) -> str:
    """Read the optional external harness file (empty string when unset)."""
    if not path:
        return ""
    return Path(path).expanduser().read_text(encoding="utf-8")


def _bump_cooldown(bump_no: int) -> float:
    return min(bump_no * BUMP_COOLDOWN_STEP, BUMP_COOLDOWN_MAX)


def _build_tool_defs() -> list[ToolDef]:
    return [
        ToolDef(
            name="bash",
            description="Execute a bash command in the Docker sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
        ),
        ToolDef(
            name="read_file",
            description="Read a file from the sandbox.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        ToolDef(
            name="write_file",
            description="Write content to a file in the sandbox.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        ),
        ToolDef(
            name="list_files",
            description="List files in a directory in the sandbox.",
            parameters={"type": "object", "properties": {"path": {"type": "string", "default": "/challenge/distfiles"}}},
        ),
        ToolDef(
            name="submit_flag",
            description="Submit a flag candidate for validation.",
            parameters={"type": "object", "properties": {"flag": {"type": "string"}}, "required": ["flag"]},
        ),
        ToolDef(
            name="web_fetch",
            description="Fetch a URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "GET"},
                    "body": {"type": "string", "default": ""},
                },
                "required": ["url"],
            },
        ),
        ToolDef(
            name="webhook_create",
            description="Create a webhook.site endpoint.",
            parameters={"type": "object", "properties": {}},
        ),
        ToolDef(
            name="webhook_get_requests",
            description="Get requests received by webhook.",
            parameters={"type": "object", "properties": {"uuid": {"type": "string"}}, "required": ["uuid"]},
        ),
        ToolDef(
            name="view_image",
            description="View an image file.",
            parameters={"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]},
        ),
        ToolDef(
            name="notify_coordinator",
            description="Send a message to the coordinator.",
            parameters={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        ),
        ToolDef(
            name="check_findings",
            description="Check for insights from other solvers.",
            parameters={"type": "object", "properties": {}},
        ),
    ]


async def retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)


LONG_CALL_WARN_INTERVAL = 60  # seconds


async def _warn_if_long_running(solver_id: str, started: float) -> None:
    """Background watchdog for a single LLM call: a provider's own native
    tool/code execution (e.g. Codex's code mode) can run for a long time
    without this system ever seeing a tool_call — there's nothing to log
    until the call returns or the outer per-challenge timeout cuts it off.
    This at least surfaces that something is still running, so a long turn
    shows up in the log instead of only being discoverable by inspecting the
    sandbox container live."""
    elapsed = 0
    while True:
        await asyncio.sleep(LONG_CALL_WARN_INTERVAL)
        elapsed += LONG_CALL_WARN_INTERVAL
        logger.warning(
            "[%s] LLM call still running after %ds (started %.0fs ago) — likely waiting "
            "on something outside this system's view (e.g. a network read with no "
            "timeout in the model's own script, or a provider's own native tool use "
            "this system can't observe).",
            solver_id, elapsed, time.monotonic() - started,
        )


@dataclass
class SolverInstance:
    solver_id: str
    provider_name: str
    model_spec: str
    sandbox: DockerSandbox | HostSandbox
    session: SolverSession | None = None
    tracer: SolverTracer | None = None
    loop_detector: LoopDetector = field(default_factory=LoopDetector)
    step_count: int = 0
    consecutive_errors: int = 0
    flag: str | None = None
    confirmed: bool = False
    contacted_remote: bool = False
    findings: str = ""
    cost_usd: float = 0.0
    _wrong_submit_count: int = 0
    _last_submit_time: float = 0.0
    attempt_start_step: int = 0
    last_share_step: int = 0
    pending_results: list[ToolResult] | None = None


@dataclass
class ChallengeSwarm:
    challenge_dir: str
    challenge_name: str
    description: str
    category: str
    settings: Settings
    event_bus: EventBus
    connection_info: str = ""
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    message_bus: ChallengeMessageBus = field(default_factory=ChallengeMessageBus)
    solvers: dict[str, SolverInstance] = field(default_factory=dict)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    confirmed_flag: str | None = None
    winner_id: str | None = None
    _flag_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _submitted_flags: set[str] = field(default_factory=set)
    _circuit_breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    session_state: SessionState | None = None
    resume_state: SessionState | None = None
    _results: dict[str, ResultStatus] = field(default_factory=dict)
    _remote_reachable: bool | None = field(default=None, repr=False)

    async def _remote_is_reachable(self) -> bool:
        """Probed once per run and cached. A remote target that is down (or
        has no parseable host/port) must not block an otherwise-plausible
        flag forever -- the contact requirement only makes sense when there
        is actually something to contact."""
        if self._remote_reachable is not None:
            return self._remote_reachable
        host, port = _parse_connection_info(self.connection_info)
        if not host or not port:
            self._remote_reachable = False
            return False
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port)), timeout=5.0
            )
            writer.close()
            self._remote_reachable = True
        except Exception:
            self._remote_reachable = False
        return self._remote_reachable

    def _check_cooldown(self, inst: SolverInstance) -> tuple[bool, str]:
        """Check if solver is in cooldown. Returns (is_cooled_down, message)."""
        wrong_count = inst._wrong_submit_count
        cooldown_idx = min(wrong_count, len(SUBMISSION_COOLDOWNS) - 1)
        cooldown = SUBMISSION_COOLDOWNS[cooldown_idx]
        if cooldown > 0:
            elapsed = time.monotonic() - inst._last_submit_time
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                return False, f"COOLDOWN — wait {remaining}s. {wrong_count} wrong submissions."
        return True, ""

    def _record_wrong_submit(self, inst: SolverInstance) -> None:
        inst._wrong_submit_count += 1
        inst._last_submit_time = time.monotonic()

    async def _handle_flag_candidate(
        self, inst: SolverInstance, flag: str, *, penalize: bool = True
    ) -> tuple[str, bool]:
        """Validate and record one flag candidate.

        Shared by the explicit ``submit_flag`` tool and the regex text-scan path
        so dedup, winner-lock, event publication and message-bus sharing behave
        identically regardless of how the candidate surfaced. ``penalize`` is
        True for deliberate tool submissions (apply the escalating cooldown and
        count wrong tries) and False for passive scans of assistant text — a flag
        merely *mentioned* while reasoning must not block the next real submit.
        Returns ``(message, is_valid)``; the regex path ignores the message.
        """
        flag = flag.strip()
        if not flag:
            return "Empty flag — nothing to validate.", False
        if _is_placeholder_flag(flag):
            return f"Flag '{flag}' looks like a placeholder — not a real flag.", False
        if (
            self.connection_info
            and not inst.contacted_remote
            and inst.step_count < MIN_STEPS_BEFORE_UNCONTACTED_FLAG
            and await self._remote_is_reachable()
        ):
            return (
                f"Flag '{flag}' rejected — found after only {inst.step_count} tool call(s), "
                f"this challenge has a reachable remote target ({self.connection_info}), and "
                "there is no evidence this solver has connected to it yet (no bash command "
                "referenced that host/port). This looks like a decoy/placeholder planted in "
                "local files rather than the real flag; connect to and exploit the remote "
                "target, or do substantially more local analysis before resubmitting."
            ), False

        # Dedup, cooldown and verification are decided atomically so two solvers
        # racing on the same flag cannot both pass the "not yet tried" check.
        async with self._flag_lock:
            if flag in self._submitted_flags:
                return "Already tried this flag.", False
            if not extract_flags(flag, self.settings.flag_pattern):
                return f"Flag '{flag}' does not match expected pattern.", False

            # Deliberate submissions are subject to the escalating cooldown.
            if penalize:
                cooled, cooldown_msg = self._check_cooldown(inst)
                if not cooled:
                    return cooldown_msg, False

            result, is_valid = await do_submit_flag(
                flag, self.settings.flag_pattern, self._submitted_flags,
                accept_regex_as_verified=self.settings.accept_regex_flag_as_verified,
            )
            if is_valid:
                inst.flag = flag
                inst.confirmed = True
                if not self.confirmed_flag:
                    self.confirmed_flag = flag
                    self.winner_id = inst.solver_id
            elif penalize:
                self._record_wrong_submit(inst)

        self.event_bus.publish(SolverEvent(
            type="flag_candidate", solver_id=inst.solver_id,
            data={"flag": flag, "verified": is_valid},
        ))
        await self.message_bus.post(InsightMessage(
            solver_id=inst.solver_id, step=inst.step_count, category="flag_candidate",
            content=f"Candidate flag: {flag}", confidence=0.9 if is_valid else 0.4,
        ))
        return result, is_valid

    def _create_circuit_breakers(self) -> dict[str, CircuitBreaker]:
        return {name: CircuitBreaker() for name, _ in get_active_providers(self.settings)}

    def _model_spec_for(self, provider_name: str) -> str:
        if provider_name == "claude":
            return self.settings.claude_model
        if provider_name == "codex":
            return self.settings.codex_model
        if provider_name == "zai":
            return self.settings.zai_model
        return provider_name

    def _create_solvers(self) -> list[SolverInstance]:
        instances = []
        for provider_name, count in get_active_providers(self.settings):
            for i in range(count):
                solver_id = f"{provider_name}-{i}"
                if self.settings.no_docker:
                    sandbox: DockerSandbox | HostSandbox = HostSandbox(challenge_dir=self.challenge_dir)
                else:
                    extra_ro_binds = []
                    if provider_name == "codex":
                        bin_dir = locate_codex_native_bin_dir()
                        if bin_dir:
                            extra_ro_binds = [f"{bin_dir}:/opt/codex-bin:ro"]
                        # If not found, the Codex session's own docker-exec
                        # attempt below fails loudly (missing binary) instead
                        # of silently running codex unconfined on the host.
                    sandbox = DockerSandbox(
                        image=self.settings.sandbox_image,
                        challenge_dir=self.challenge_dir,
                        memory_limit=self.settings.sandbox_memory,
                        cpu_limit=self.settings.sandbox_cpus,
                        extra_ro_binds=extra_ro_binds,
                    )
                instance = SolverInstance(
                    solver_id=solver_id,
                    provider_name=provider_name,
                    model_spec=self._model_spec_for(provider_name),
                    sandbox=sandbox,
                )
                instances.append(instance)
        return instances

    def _apply_resume_state(self, instances: list[SolverInstance]) -> list[SolverInstance]:
        """Carry the winner/flag forward and drop solvers that already solved.

        A prior SOLVED solver means the challenge was already cracked, so its
        flag is kept and it is not re-run; every other solver is retried."""
        prev = self.resume_state
        if not prev:
            return instances
        self.confirmed_flag = prev.confirmed_flag
        self.winner_id = prev.winner_id
        solved = {s.solver_id for s in prev.solvers if s.status == ResultStatus.SOLVED.value}
        remaining: list[SolverInstance] = []
        for inst in instances:
            if inst.solver_id in solved:
                prior = next(s for s in prev.solvers if s.solver_id == inst.solver_id)
                inst.flag = prior.flag
                inst.confirmed = True
                self._results[inst.solver_id] = ResultStatus.SOLVED
                self.solvers[inst.solver_id] = inst
                logger.info("[%s] Skipping — already solved in prior session", inst.solver_id)
            else:
                remaining.append(inst)
        return remaining

    async def run(self) -> SolverResult | None:
        self._circuit_breakers = self._create_circuit_breakers()
        instances = self._apply_resume_state(self._create_solvers())
        for inst in instances:
            self.solvers[inst.solver_id] = inst
            self.event_bus.publish(SolverEvent(type="solver_started", solver_id=inst.solver_id, data={"provider": inst.provider_name}))

        self.session_state = SessionState(
            challenge_name=self.challenge_name,
            challenge_dir=self.challenge_dir,
            description=self.description,
            category=self.category,
        )

        tasks = [asyncio.create_task(self._run_solver(inst), name=f"solver-{inst.solver_id}") for inst in instances]

        try:
            while tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                    except Exception:
                        logger.warning("Solver task failed", exc_info=True)
                        continue
                    if result:
                        self._results[result.solver_id] = result.status
                    self._update_session_state()
                    if result and result.status == ResultStatus.SOLVED:
                        self._update_session_state()
                        self.cancel_event.set()
                        for p in pending:
                            p.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        return result
                tasks = list(pending)

            self.cancel_event.set()
            self._update_session_state()
            return None
        except Exception as e:
            logger.error("Swarm error: %s", e, exc_info=True)
            self.cancel_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._update_session_state()
            return None
        finally:
            self._save_session_state()

    async def _init_solver(self, inst: SolverInstance, t0: float) -> SolverResult | None:
        provider = get_provider(inst.provider_name)
        config = self.settings.to_provider_config()

        if not provider.validate_config(config):
            return SolverResult(
                solver_id=inst.solver_id,
                status=ResultStatus.ERROR,
                flag=None,
                steps=0,
                duration=time.monotonic() - t0,
                error=f"Provider {inst.provider_name} not configured",
            )

        if not self.settings.no_docker:
            docker = aiodocker.Docker()
            try:
                await docker.version()
            except Exception:
                return SolverResult(
                    solver_id=inst.solver_id,
                    status=ResultStatus.ERROR,
                    flag=None,
                    steps=0,
                    duration=time.monotonic() - t0,
                    error="Docker is not running. Start Docker and try again.",
                )
            finally:
                try:
                    await docker.close()
                except Exception:
                    pass

        await inst.sandbox.start()
        self.event_bus.publish(SolverEvent(type="state_change", solver_id=inst.solver_id, data={"state": "running"}))

        if inst.provider_name == "codex" and not self.settings.openai_api_key:
            self._prepare_codex_subscription_home(inst)

        log_dir = self.settings.log_dir or "logs"
        inst.tracer = SolverTracer(self.challenge_name, inst.provider_name, log_dir=log_dir)

        meta = ChallengeMeta(
            name=self.challenge_name,
            category=self.category,
            description=self.description,
            connection_info=self.connection_info,
        )
        distfiles = list_distfiles(self.challenge_dir)
        container_arch = "linux/amd64"
        system_prompt = build_prompt(
            meta, distfiles, container_arch, hint=self.settings.hint,
            harness=_load_harness(self.settings.harness_file),
        )
        tools = _build_tool_defs()
        inst.session = await provider.create_session(
            inst.solver_id, system_prompt, tools, config, sandbox=inst.sandbox
        )
        return None

    @staticmethod
    def _prepare_codex_subscription_home(inst: SolverInstance) -> None:
        """Copy only auth.json (never the whole ~/.codex — that also holds
        session history, local notes, etc.) into the sandbox's already
        read-write workspace bind, so the containerized codex process (see
        providers/codex.py) can authenticate with the user's ChatGPT/Codex
        subscription without ever seeing the real ~/.codex directory."""
        workspace_dir = getattr(inst.sandbox, "workspace_dir", "")
        if not workspace_dir:
            return
        host_auth = Path.home() / ".codex" / "auth.json"
        codex_home = Path(workspace_dir) / ".codex-home"
        codex_home.mkdir(exist_ok=True)
        if host_auth.exists():
            import shutil as _shutil

            dest = codex_home / "auth.json"
            _shutil.copy2(host_auth, dest)
            dest.chmod(0o600)

    async def _run_loop(
        self, inst: SolverInstance, t0: float, message: str = "Start solving."
    ) -> SolverResult:

        while not self.cancel_event.is_set():
            if inst.step_count - inst.attempt_start_step >= self.settings.max_steps:
                break
            elapsed = time.monotonic() - t0
            if elapsed >= self.settings.timeout:
                return SolverResult(
                    solver_id=inst.solver_id,
                    status=ResultStatus.TIMEOUT,
                    flag=inst.flag,
                    steps=inst.step_count,
                    duration=elapsed,
                    cost_usd=inst.cost_usd,
                    trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
                    findings_summary=inst.findings,
                )
            if self.cost_tracker.is_over_budget(self.settings.max_cost):
                break

            cb = self._circuit_breakers.get(inst.provider_name)
            if cb and not cb.is_available():
                break

            if inst.step_count - inst.last_share_step >= INSIGHT_SHARE_INTERVAL:
                inst.last_share_step = inst.step_count
                # Share this solver's latest reasoning with siblings...
                if inst.findings:
                    await self.message_bus.post(InsightMessage(
                        solver_id=inst.solver_id, step=inst.step_count,
                        category="finding", content=inst.findings[:500],
                    ))
                    self.event_bus.publish(SolverEvent(
                        type="insight_shared", solver_id=inst.solver_id,
                        data={"content": inst.findings[:200]},
                    ))
                # ...and pull in what the others have shared.
                insights_text = await do_check_findings(self.message_bus, inst.solver_id)
                if insights_text and "No new findings" not in insights_text:
                    await inst.session.inject_context(insights_text)
                    inst.loop_detector.reset()
                    self.event_bus.publish(SolverEvent(
                        type="insight_received", solver_id=inst.solver_id,
                        data={"summary": insights_text[:200]},
                    ))

            remaining = self.settings.timeout - (time.monotonic() - t0)
            if remaining <= 0:
                break
            watchdog = asyncio.create_task(_warn_if_long_running(inst.solver_id, time.monotonic()))
            try:
                _msg, _presults = message, inst.pending_results
                response = await asyncio.wait_for(
                    retry_with_backoff(
                        lambda m=_msg, p=_presults: inst.session.send(m, tool_results=p),
                        max_retries=3,
                        base_delay=1.0,
                    ),
                    timeout=max(remaining, 1.0),
                )
            except Exception as e:
                if cb:
                    cb.record_failure()
                # str(e) is empty for asyncio.TimeoutError — name the type too,
                # or a timeout silently logs as "LLM call failed: " with nothing
                # useful after the colon.
                logger.warning("[%s] LLM call failed: %s: %s", inst.solver_id, type(e).__name__, e)
                return SolverResult(
                    solver_id=inst.solver_id,
                    status=ResultStatus.ERROR,
                    flag=inst.flag,
                    steps=inst.step_count,
                    duration=time.monotonic() - t0,
                    cost_usd=inst.cost_usd,
                    trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
                    findings_summary=inst.findings,
                    error=str(e),
                )
            finally:
                watchdog.cancel()

            if cb:
                cb.record_success()
            inst.pending_results = None

            self.cost_tracker.record_tokens(
                inst.solver_id,
                inst.model_spec,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                provider_spec=inst.provider_name,
            )
            step_cost = self.cost_tracker.by_agent.get(inst.solver_id)
            if step_cost:
                inst.cost_usd = step_cost.cost_usd
            self.event_bus.publish(SolverEvent(type="cost_update", solver_id=inst.solver_id, data={"cost": inst.cost_usd}))

            if inst.tracer:
                inst.tracer.usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.usage.cache_read_tokens,
                    inst.cost_usd,
                )

            if response.text:
                inst.findings = response.text.strip()[:2000]

            for flag in extract_flags(response.text, self.settings.flag_pattern):
                await self._handle_flag_candidate(inst, flag, penalize=False)

            if response.done or not response.tool_calls:
                break

            tool_results: list[ToolResult] = []
            # Every tool_use needs a tool_result before the next send, even if we
            # stop mid-batch (loop break, error cap) and later resume via a bump.
            inst.pending_results = [
                ToolResult(content="Not executed (solver paused).", tool_use_id=c.call_id, call_id=c.call_id)
                for c in response.tool_calls
            ]
            for tc in response.tool_calls:
                inst.step_count += 1
                if inst.tracer:
                    inst.tracer.tool_call(tc.name, tc.arguments, inst.step_count)
                tr = await self._execute_tool(inst, tc)
                tool_results.append(tr)
                inst.pending_results[len(tool_results) - 1] = tr

                trace_content = tr.content if isinstance(tr.content, str) else f"<image {tr.content[1]}>"
                if inst.tracer:
                    inst.tracer.tool_result(tc.name, trace_content, inst.step_count)

                if tr.error:
                    inst.consecutive_errors += 1
                    if inst.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.warning("[%s] %d consecutive tool errors, stopping", inst.solver_id, MAX_CONSECUTIVE_ERRORS)
                        return SolverResult(
                            solver_id=inst.solver_id,
                            status=ResultStatus.ERROR,
                            flag=inst.flag,
                            steps=inst.step_count,
                            duration=time.monotonic() - t0,
                            cost_usd=inst.cost_usd,
                            trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
                            findings_summary=inst.findings,
                            error=f"{MAX_CONSECUTIVE_ERRORS} consecutive tool errors",
                        )
                else:
                    inst.consecutive_errors = 0

                loop_status = inst.loop_detector.check(tc.name, tc.arguments)
                if loop_status == "warn":
                    await inst.session.inject_context(LOOP_WARNING_MESSAGE)
                elif loop_status == "break":
                    return SolverResult(
                        solver_id=inst.solver_id,
                        status=ResultStatus.LOOP_DETECTED,
                        flag=inst.flag,
                        steps=inst.step_count,
                        duration=time.monotonic() - t0,
                        cost_usd=inst.cost_usd,
                        trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
                        findings_summary=inst.findings,
                    )

            message = ""

        if inst.confirmed and inst.flag:
            return SolverResult(
                solver_id=inst.solver_id,
                status=ResultStatus.SOLVED,
                flag=inst.flag,
                steps=inst.step_count,
                duration=time.monotonic() - t0,
                cost_usd=inst.cost_usd,
                trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
                findings_summary=inst.findings,
            )

        result = SolverResult(
            solver_id=inst.solver_id,
            status=ResultStatus.FAILED,
            flag=inst.flag,
            steps=inst.step_count,
            duration=time.monotonic() - t0,
            cost_usd=inst.cost_usd,
            trace_path=Path(inst.tracer.path) if inst.tracer else Path(),
            findings_summary=inst.findings,
        )

        if inst.step_count == 0 and inst.cost_usd == 0.0:
            logger.warning("[%s] Broken solver — 0 steps, $0 cost", inst.solver_id)

        return result

    async def _should_bump(
        self, inst: SolverInstance, result: SolverResult, t0: float, bumps: int
    ) -> bool:
        """Decide whether a solver that stopped without a flag gets another attempt."""
        if bumps >= self.settings.max_bumps or self.cancel_event.is_set():
            return False
        if result.status not in (ResultStatus.FAILED, ResultStatus.LOOP_DETECTED, ResultStatus.ERROR):
            return False
        if inst.confirmed or self.confirmed_flag:
            return False
        if result.steps == 0 and result.cost_usd == 0.0:
            return False  # broken solver — do not burn time re-running it
        if result.status == ResultStatus.ERROR and inst.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return False
        if self.cost_tracker.is_over_budget(self.settings.max_cost):
            return False
        cb = self._circuit_breakers.get(inst.provider_name)
        if cb and not cb.is_available():
            return False
        return self.settings.timeout - (time.monotonic() - t0) > _bump_cooldown(bumps + 1)

    async def _bump(
        self, inst: SolverInstance, prev: SolverResult, t0: float, bump_no: int
    ) -> SolverResult:
        """Wait out the cooldown, hand the solver sibling insights, and resume it."""
        try:
            await asyncio.wait_for(self.cancel_event.wait(), timeout=_bump_cooldown(bump_no))
            return prev  # cancelled during cooldown
        except TimeoutError:
            pass
        insights = await do_check_findings(self.message_bus, inst.solver_id)
        if "No new findings" in insights or "No message bus" in insights:
            insights = "No sibling insights available yet."
        self.event_bus.publish(SolverEvent(
            type="solver_bumped", solver_id=inst.solver_id, data={"bump": bump_no},
        ))
        logger.info("[%s] Bumped (%d), resuming", inst.solver_id, bump_no)
        inst.attempt_start_step = inst.step_count
        inst.consecutive_errors = 0
        inst.loop_detector.reset()
        message = (
            "You stopped without a confirmed flag. Do not repeat earlier dead ends — "
            "try a different approach.\n\n" + insights
        )
        return await self._run_loop(inst, t0, message=message)

    async def _run_solver(self, inst: SolverInstance) -> SolverResult | None:
        t0 = time.monotonic()
        try:
            err = await self._init_solver(inst, t0)
            if err:
                return err
            result = await self._run_loop(inst, t0)
            bumps = 0
            while await self._should_bump(inst, result, t0, bumps):
                bumps += 1
                result = await self._bump(inst, result, t0, bumps)
            return result
        except NotImplementedError as e:
            logger.warning("[%s] Provider stub: %s", inst.solver_id, e)
            return SolverResult(
                solver_id=inst.solver_id,
                status=ResultStatus.ERROR,
                flag=None,
                steps=0,
                duration=time.monotonic() - t0,
                error=str(e),
            )
        except asyncio.CancelledError:
            return SolverResult(
                solver_id=inst.solver_id,
                status=ResultStatus.CANCELLED,
                flag=inst.flag,
                steps=inst.step_count,
                duration=time.monotonic() - t0,
            )
        except Exception as e:
            logger.error("[%s] Fatal: %s", inst.solver_id, e, exc_info=True)
            return SolverResult(
                solver_id=inst.solver_id,
                status=ResultStatus.ERROR,
                flag=inst.flag,
                steps=inst.step_count,
                duration=time.monotonic() - t0,
                error=str(e),
            )
        finally:
            if inst.session:
                try:
                    await inst.session.close()
                except Exception:
                    logger.warning("[%s] Failed to close provider session", inst.solver_id, exc_info=True)
            try:
                await inst.sandbox.stop()
            except Exception:
                logger.warning("[%s] Failed to stop sandbox", inst.solver_id, exc_info=True)
            if inst.tracer:
                try:
                    inst.tracer.close()
                except Exception:
                    pass
            self.event_bus.publish(SolverEvent(type="solver_done", solver_id=inst.solver_id, data={}))

    async def _execute_tool(self, inst: SolverInstance, tc: ToolCall) -> ToolResult:
        try:
            raw = await self._execute_tool_inner(inst, tc)
            return ToolResult(content=raw, tool_use_id=tc.call_id, call_id=tc.call_id)
        except Exception as e:
            logger.warning("[%s] Tool %s raised: %s", inst.solver_id, tc.name, e, exc_info=True)
            return ToolResult(
                content=f"Tool error: {e}",
                error=str(e),
                tool_use_id=tc.call_id,
                call_id=tc.call_id,
            )

    async def _execute_tool_inner(self, inst: SolverInstance, tc: ToolCall) -> str | tuple[bytes, str]:
        name = tc.name
        args = tc.arguments

        if name == "bash":
            command = args.get("command", "")
            if self.connection_info and not inst.contacted_remote:
                host, port = _parse_connection_info(self.connection_info)
                if host and host in command and (not port or port in command):
                    inst.contacted_remote = True
            return await do_bash(inst.sandbox, command, int(args.get("timeout_seconds", args.get("timeout", 60))))
        elif name == "read_file":
            return await do_read_file(inst.sandbox, args.get("path", ""))
        elif name == "write_file":
            return await do_write_file(inst.sandbox, args.get("path", ""), args.get("content", ""))
        elif name == "list_files":
            return await do_list_files(inst.sandbox, args.get("path", "/challenge/distfiles"))
        elif name == "web_fetch":
            return await do_web_fetch(args.get("url", ""), args.get("method", "GET"), args.get("body", ""))
        elif name == "webhook_create":
            return await do_webhook_create()
        elif name == "webhook_get_requests":
            return await do_webhook_get_requests(args.get("uuid", ""))
        elif name == "submit_flag":
            msg_text, _ = await self._handle_flag_candidate(inst, args.get("flag", ""))
            return msg_text
        elif name == "check_findings":
            return await do_check_findings(self.message_bus, inst.solver_id)
        elif name == "notify_coordinator":
            return await do_notify_coordinator(args.get("message", ""), self.event_bus, inst.solver_id)
        elif name == "view_image":
            from ctf_solver.tools.vision import do_view_image

            return await do_view_image(inst.sandbox, args.get("filename", ""), use_vision=True)
        else:
            return f"Unknown tool: {name}"

    def _update_session_state(self) -> None:
        if not self.session_state:
            return
        solver_states = []
        for sid, inst in self.solvers.items():
            if sid in self._results:
                status = self._results[sid].value
            else:
                status = "running" if not self.cancel_event.is_set() else "done"
            solver_states.append(SolverState(
                solver_id=sid,
                provider=inst.provider_name,
                status=status,
                steps=inst.step_count,
                cost_usd=inst.cost_usd,
                flag=inst.flag,
            ))
        self.session_state.solvers = solver_states
        self.session_state.total_cost_usd = self.cost_tracker.total_cost_usd
        self.session_state.confirmed_flag = self.confirmed_flag
        self.session_state.winner_id = self.winner_id

    def _save_session_state(self) -> None:
        if self.session_state:
            log_dir = self.settings.log_dir or "logs"
            self.session_state.save(log_dir)

    def kill(self) -> None:
        self.cancel_event.set()

    def get_status(self) -> dict:
        return {
            "challenge": self.challenge_name,
            "cancelled": self.cancel_event.is_set(),
            "winner": self.winner_id,
            "confirmed_flag": self.confirmed_flag,
            "solvers": {
                sid: {
                    "steps": s.step_count,
                    "cost": s.cost_usd,
                    "findings": s.findings[:200],
                    "flag": s.flag,
                }
                for sid, s in self.solvers.items()
            },
        }
