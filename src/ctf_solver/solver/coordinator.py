"""CoordinatorAgent — reads solver traces, injects strategic guidance."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ctf_solver.collaboration.message_bus import ChallengeMessageBus
from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.providers import get_provider
from ctf_solver.providers.base import SolverSession
from ctf_solver.providers.codex import locate_codex_native_bin_dir
from ctf_solver.sandbox.docker import DockerSandbox
from ctf_solver.tracking.cost_tracker import CostTracker

if TYPE_CHECKING:
    from ctf_solver.config import Settings
    from ctf_solver.solver.swarm import ChallengeSwarm

logger = logging.getLogger(__name__)


COORDINATION_INTERVAL = 30


@dataclass
class CoordinatorAgent:
    provider_name: str
    settings: Settings
    event_bus: EventBus
    cost_tracker: CostTracker
    message_bus: ChallengeMessageBus
    swarm: ChallengeSwarm | None = None
    session: SolverSession | None = None
    _task: asyncio.Task | None = None
    _hint_queue: asyncio.Queue | None = None
    _hint_task: asyncio.Task | None = None
    _sandbox: DockerSandbox | None = None

    async def start(self) -> None:
        provider = get_provider(self.provider_name)
        config = self.settings.to_provider_config()
        if not provider.validate_config(config):
            logger.warning("Coordinator provider %s not configured — coordinator disabled", self.provider_name)
            return
        # The coordinator passes tools=[] (it only ever sends plain-text trace
        # summaries, never a tool_result) — but for Codex that does NOT mean
        # "no execution capability": its own native agent tooling (separate
        # from this project's dynamicTools) is available regardless of the
        # dynamicTools list, confirmed live by this project's solver path
        # having the exact same issue. So the coordinator needs its own
        # confining sandbox exactly like a solver instance does, not just an
        # empty tool list, whenever Docker is available to provide one.
        if self.provider_name == "codex" and not self.settings.no_docker:
            await self._start_codex_sandbox()
        self.session = await provider.create_session(
            solver_id="coordinator",
            system_prompt="You are a CTF coordinator. Read solver traces and provide strategic guidance.",
            tools=[],
            config=config,
            sandbox=self._sandbox,
        )
        self._task = asyncio.create_task(self._coordination_loop())
        self._hint_queue = self.event_bus.subscribe()
        self._hint_task = asyncio.create_task(self._relay_hints())
        logger.info("Coordinator started with %s", self.provider_name)

    async def _start_codex_sandbox(self) -> None:
        """Give the coordinator's own codex session a confining container,
        mirroring swarm.py's per-solver setup (see providers/codex.py's
        containment comments for why this is required at all)."""
        challenge_dir = self.swarm.challenge_dir if self.swarm else "."
        bin_dir = locate_codex_native_bin_dir()
        extra_ro_binds = [f"{bin_dir}:/opt/codex-bin:ro"] if bin_dir else []
        sandbox = DockerSandbox(
            image=self.settings.sandbox_image,
            challenge_dir=challenge_dir,
            memory_limit=self.settings.sandbox_memory,
            cpu_limit=self.settings.sandbox_cpus,
            extra_ro_binds=extra_ro_binds,
        )
        try:
            await sandbox.start()
        except Exception as e:
            logger.warning("Coordinator sandbox failed to start (%s) — coordinator disabled", e)
            return
        if not self.settings.openai_api_key and sandbox.workspace_dir:
            host_auth = Path.home() / ".codex" / "auth.json"
            codex_home = Path(sandbox.workspace_dir) / ".codex-home"
            codex_home.mkdir(exist_ok=True)
            if host_auth.exists():
                import shutil

                dest = codex_home / "auth.json"
                shutil.copy2(host_auth, dest)
                dest.chmod(0o600)
        self._sandbox = sandbox

    async def _coordination_loop(self) -> None:
        while True:
            await asyncio.sleep(COORDINATION_INTERVAL)
            if not self.swarm or not self.session:
                continue
            status = self.swarm.get_status()
            for solver_id, info in status.get("solvers", {}).items():
                solver = self.swarm.solvers.get(solver_id)
                if not solver or not solver.tracer:
                    continue
                try:
                    trace_path = Path(solver.tracer.path)
                    if not trace_path.exists():
                        continue
                    lines = trace_path.read_text().strip().split("\n")
                    recent = lines[-10:]
                    summary = self._summarize_trace(recent)
                    guidance_prompt = f"Solver {solver_id} (steps: {info.get('steps', 0)}) recent activity:\n{summary}\n\nProvide brief strategic guidance."
                    response = await self.session.send(guidance_prompt)
                    if response.text:
                        await self.message_bus.broadcast("coordinator", response.text[:500])
                        self.event_bus.publish(SolverEvent(
                            type="coordinator_guidance",
                            solver_id="coordinator",
                            data={"guidance": response.text[:500]},
                        ))
                    model_name = self.swarm._model_spec_for(self.provider_name) if self.swarm else self.provider_name
                    self.cost_tracker.record_tokens(
                        "coordinator",
                        model_name,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        cache_read_tokens=response.usage.cache_read_tokens,
                        provider_spec=self.provider_name,
                    )
                except NotImplementedError:
                    continue
                except Exception as e:
                    logger.warning("Coordinator error: %s", e)

    async def _relay_hints(self) -> None:
        while True:
            try:
                event = await asyncio.wait_for(self._hint_queue.get(), timeout=1.0)
                if event.type == "user_hint":
                    hint = event.data.get("message", "")
                    if hint:
                        await self.message_bus.broadcast("operator", f"Operator hint: {hint}")
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _summarize_trace(self, lines: list[str]) -> str:
        parts = []
        for line in lines:
            try:
                d = json.loads(line)
                t = d.get("type", "?")
                if t == "tool_call":
                    parts.append(f"Step {d.get('step', '?')}: called {d.get('tool', '?')}")
                elif t == "tool_result":
                    parts.append(f"Step {d.get('step', '?')}: got result from {d.get('tool', '?')}")
                elif t in ("finish", "error", "bump"):
                    parts.append(f"{t}: {str(d)[:100]}")
            except (json.JSONDecodeError, Exception):
                parts.append(line[:80])
        return "\n".join(parts) if parts else "No recent activity."

    async def stop(self) -> None:
        if self._hint_queue:
            try:
                self.event_bus.unsubscribe(self._hint_queue)
            except Exception:
                pass
            self._hint_queue = None
        if self._hint_task:
            self._hint_task.cancel()
            try:
                await self._hint_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.session:
            await self.session.close()
        if self._sandbox:
            try:
                await self._sandbox.stop()
            except Exception:
                pass
            self._sandbox = None
        logger.info("Coordinator stopped")
