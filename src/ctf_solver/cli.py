"""Click CLI entry point — ctf-solve and ctf-msg commands."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console

from ctf_solver.config import (
    Settings,
    get_coordinator_provider,
    load_toml_config,
    validate_provider_config,
)
from ctf_solver.events import EventBus
from ctf_solver.session_state import SessionState
from ctf_solver.solver.swarm import ChallengeSwarm

console = Console()
logger = logging.getLogger(__name__)


_SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}


def _auto_discover_files(cwd: Path) -> list[str]:
    """Bare-invocation file discovery: every regular file directly in `cwd`,
    plus one level into a conventional `distfiles/` subdirectory if present.
    Hidden files/dirs and common tooling noise are skipped."""
    out: list[str] = []
    for p in sorted(cwd.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_file():
            out.append(str(p))
        elif p.is_dir() and p.name == "distfiles":
            out.extend(str(f) for f in sorted(p.iterdir()) if f.is_file() and not f.name.startswith("."))
        elif p.is_dir() and p.name in _SKIP_DIR_NAMES:
            continue
    return out


_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]")


def _normalize_path_input(raw: str) -> str:
    """Make a pasted path usable, however Windows/WSL mangled it on the way in.

    Handles: wrapping quotes (Explorer's "Copy as path" adds them), a Windows
    drive path (``C:\\...`` -> ``/mnt/c/...``, the WSL mount convention), and
    stray backslashes as path separators. A plain WSL/Linux path passes
    through unchanged.
    """
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    m = _WINDOWS_DRIVE_RE.match(raw)
    if m:
        drive = m.group(1).lower()
        rest = raw[m.end():].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return raw.replace("\\", "/")


def _prompt_or_x(label: str) -> str:
    """Prompt for one line; typing X (any case) means "none" -> empty string.

    Used for both 문제설명 (no description) and 원격 서버 (no remote target —
    solve using only the given files, locally)."""
    raw = click.prompt(f"{label} — 없으면 X", default="", show_default=False)
    raw = raw.strip()
    return "" if raw.upper() == "X" else raw


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiodocker").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%X"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


@click.command()
@click.option("--challenge-dir", default=None, help="Challenge directory")
@click.option("--files", multiple=True, help="Individual challenge files")
@click.option("--desc", default=None, help="Challenge description (required with --files)")
@click.option("--remote", default=None, help="Connection info, e.g. 'nc host port' or a URL")
@click.option("--category", default=None, help="Category hint")
@click.option("--claude", "claude_count", default=None, type=int, help="Number of Claude solvers")
@click.option("--codex", "codex_count", default=None, type=int, help="Number of Codex solvers")
@click.option("--zai", "zai_count", default=None, type=int, help="Number of z.ai solvers")
@click.option("--coordinator", default=None, help="Coordinator provider")
@click.option("--no-coordinator", is_flag=True, help="Disable coordinator")
@click.option("--timeout", default=None, type=int, help="Max seconds per challenge")
@click.option("--max-steps", default=None, type=int, help="Max tool calls per solver")
@click.option("--max-cost", default=None, type=float, help="Max USD cost")
@click.option("--flag-pattern", default=None, help="Regex for flag extraction")
@click.option("--hint", default=None, help="Pre-inject hint")
@click.option("--interactive", is_flag=True, help="Enable stdin hint input")
@click.option("--harness-file", default=None, type=click.Path(exists=True, dir_okay=False), help="Extra harness text appended to every solver's system prompt (for A/B runs)")
@click.option("--max-bumps", default=None, type=int, help="Re-engage a stopped solver up to N times with sibling insights (0 = off)")
@click.option("--strict-flags", is_flag=True, help="Do not auto-confirm regex-matching flags (candidates are only recorded)")
@click.option("--sandbox-image", default=None, help="Docker image")
@click.option("--sandbox-memory", default=None, help="Memory limit per container")
@click.option("--sandbox-cpus", default=None, type=int, help="CPU limit per container")
@click.option("--no-docker", is_flag=True, help="Run on host (debug)")
@click.option("--output-dir", default=None, help="Output directory")
@click.option("--log-dir", default=None, help="Log directory")
@click.option("--no-tui", is_flag=True, help="CLI mode only")
@click.option("--dry-run", is_flag=True, help="Show config without executing")
@click.option("--port", default=None, type=int, help="Hint endpoint port (0=auto)")
@click.option("--verbose", is_flag=True, help="Debug logging")
@click.option("--resume", default=None, help="Resume from previous session log dir")
@click.option("--config", default=None, help="TOML config file")
def main(
    challenge_dir: str | None,
    files: tuple[str, ...],
    desc: str | None,
    remote: str | None,
    category: str | None,
    claude_count: int | None,
    codex_count: int | None,
    zai_count: int | None,
    coordinator: str | None,
    no_coordinator: bool,
    timeout: int | None,
    max_steps: int | None,
    max_cost: float | None,
    flag_pattern: str | None,
    hint: str | None,
    interactive: bool,
    harness_file: str | None,
    max_bumps: int | None,
    strict_flags: bool,
    sandbox_image: str | None,
    sandbox_memory: str | None,
    sandbox_cpus: int | None,
    no_docker: bool,
    output_dir: str | None,
    log_dir: str | None,
    no_tui: bool,
    dry_run: bool,
    port: int | None,
    verbose: bool,
    resume: str | None,
    config: str | None,
) -> None:
    """CTF Solver Agent — multi-model solver swarm."""
    _setup_logging(verbose)

    defaults = Settings()

    if config:
        toml_data = load_toml_config(config)
        env_overrides: dict[str, str] = {}
        providers = toml_data.get("providers", {})
        if "anthropic_api_key" in providers:
            env_overrides["ANTHROPIC_API_KEY"] = providers["anthropic_api_key"]
        if "openai_api_key" in providers:
            env_overrides["OPENAI_API_KEY"] = providers["openai_api_key"]
        if "zai_api_key" in providers:
            env_overrides["ZAI_API_KEY"] = providers["zai_api_key"]
        if "zai_endpoint" in providers:
            env_overrides["ZAI_ENDPOINT"] = providers["zai_endpoint"]
        models = toml_data.get("models", {})
        if "claude_model" in models:
            env_overrides["CLAUDE_MODEL"] = models["claude_model"]
        if "codex_model" in models:
            env_overrides["CODEX_MODEL"] = models["codex_model"]
        if "zai_model" in models:
            env_overrides["ZAI_MODEL"] = models["zai_model"]
        if "zai_endpoint" in models:
            env_overrides["ZAI_ENDPOINT"] = models["zai_endpoint"]
        toml_settings = toml_data.get("settings", {})
        import os

        for k, v in env_overrides.items():
            os.environ.setdefault(k, v)

        defaults = Settings()

    _timeout = timeout if timeout is not None else int(toml_settings.get("timeout", defaults.timeout)) if config else (timeout if timeout is not None else defaults.timeout)
    _max_steps = max_steps if max_steps is not None else int(toml_settings.get("max_steps", defaults.max_steps)) if config else (max_steps if max_steps is not None else defaults.max_steps)
    _max_cost = max_cost if max_cost is not None else float(toml_settings.get("max_cost", defaults.max_cost)) if config else (max_cost if max_cost is not None else defaults.max_cost)
    _claude_count = claude_count if claude_count is not None else int(toml_settings.get("claude_count", defaults.claude_count)) if config else (claude_count if claude_count is not None else defaults.claude_count)
    _codex_count = codex_count if codex_count is not None else int(toml_settings.get("codex_count", defaults.codex_count)) if config else (codex_count if codex_count is not None else defaults.codex_count)
    _zai_count = zai_count if zai_count is not None else int(toml_settings.get("zai_count", defaults.zai_count)) if config else (zai_count if zai_count is not None else defaults.zai_count)

    resume_state: SessionState | None = None
    if resume:
        try:
            prev = SessionState.load(resume)
        except Exception as e:
            console.print(f"[red]Error loading session state: {e}[/red]")
            sys.exit(1)
        console.print(f"[bold]Previous session: {prev.challenge_name}[/bold]")
        console.print(f"  Started: {prev.started_at}")
        for s in prev.solvers:
            console.print(f"  {s.solver_id}: {s.status}, {s.steps} steps, ${s.cost_usd:.4f}, flag={s.flag}")
        if prev.confirmed_flag:
            console.print(f"\n[bold green]FLAG ALREADY FOUND: {prev.confirmed_flag} (by {prev.winner_id})[/bold green]")
            return
        console.print("\n[yellow]No flag found previously. Resuming the same challenge...[/yellow]")
        # Make --resume self-sufficient: reuse the prior challenge unless overridden.
        if not challenge_dir and not files:
            if Path(prev.challenge_dir).is_dir():
                challenge_dir = prev.challenge_dir
            else:
                console.print(
                    f"[red]Prior challenge dir missing: {prev.challenge_dir}. "
                    "Pass --challenge-dir to resume.[/red]"
                )
                sys.exit(1)
        if not category:
            category = prev.category
        resume_state = prev

    if not no_tui and not sys.stdout.isatty():
        no_tui = True

    if challenge_dir and files:
        console.print("[red]Error: --challenge-dir and --files are mutually exclusive[/red]")
        sys.exit(1)

    # Bare invocation (no --challenge-dir / --files / --desc / --resume): ask where
    # the challenge actually lives — you run this from wherever the solver is
    # installed, not from inside the challenge folder — then auto-pick up its
    # files and ask the two things that can't be guessed.
    bare_mode = not challenge_dir and not files and not desc and not resume_state
    if bare_mode:
        raw_dir = click.prompt("분석할 문제 폴더 경로", default=str(Path.cwd()), show_default=True)
        normalized_dir = _normalize_path_input(raw_dir)
        target_dir = Path(normalized_dir).expanduser().resolve()
        if not target_dir.is_dir():
            hint = f" (입력: {raw_dir!r} -> {normalized_dir!r})" if normalized_dir != raw_dir.strip() else ""
            console.print(f"[red]Error: 존재하지 않는 폴더입니다: {target_dir}{hint}[/red]")
            sys.exit(1)
        auto_files = _auto_discover_files(target_dir)
        names = ", ".join(Path(f).name for f in auto_files) or "(없음)"
        console.print(f"[dim]{target_dir} 에서 파일 {len(auto_files)}개를 찾았습니다: {names}[/dim]")
        files = tuple(auto_files)
        desc = _prompt_or_x("문제설명")
        remote = _prompt_or_x("원격 서버 (예: nc host port)")

    if files and desc is None:
        console.print("[red]Error: --desc is required when using --files[/red]")
        sys.exit(1)

    if not challenge_dir and not files and not bare_mode:
        console.print("[red]Error: specify --challenge-dir or --files[/red]")
        sys.exit(1)

    settings = Settings(
        claude_count=_claude_count,
        codex_count=_codex_count,
        zai_count=_zai_count,
        coordinator=coordinator or "",
        no_coordinator=no_coordinator,
        timeout=_timeout,
        max_steps=_max_steps,
        max_cost=_max_cost,
        flag_pattern=flag_pattern or Settings.model_fields["flag_pattern"].default,
        hint=hint or "",
        interactive=interactive,
        harness_file=harness_file or "",
        max_bumps=max_bumps if max_bumps is not None else defaults.max_bumps,
        accept_regex_flag_as_verified=not strict_flags,
        sandbox_image=sandbox_image or defaults.sandbox_image,
        sandbox_memory=sandbox_memory or defaults.sandbox_memory,
        sandbox_cpus=sandbox_cpus if sandbox_cpus is not None else defaults.sandbox_cpus,
        no_docker=no_docker,
        output_dir=output_dir or "",
        log_dir=log_dir or "",
        no_tui=no_tui,
        dry_run=dry_run,
        port=port if port is not None else defaults.port,
        verbose=verbose,
    )

    try:
        validate_provider_config(settings)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    try:
        coord = get_coordinator_provider(settings)
    except ValueError:
        coord = None

    console.print("[bold]CTF Solver Agent[/bold]")
    console.print(f"  Providers: claude={settings.claude_count}, codex={settings.codex_count}, zai={settings.zai_count}")
    console.print(f"  Coordinator: {coord or 'disabled'}")
    console.print(f"  Sandbox: {settings.sandbox_image} ({settings.sandbox_memory}, {settings.sandbox_cpus} CPUs)")
    console.print(f"  Limits: {settings.timeout}s, {settings.max_steps} steps, ${settings.max_cost:.2f}")

    if dry_run:
        console.print("\n[green]DRY RUN — configuration valid.[/green]")
        return

    if no_docker:
        confirmed = click.confirm("Running without Docker sandbox is dangerous. Continue?", default=False)
        if not confirmed:
            return

    if challenge_dir:
        chall_dir = challenge_dir
    else:
        chall_dir = tempfile.mkdtemp(prefix="ctf-chall-")
        distfiles_dir = Path(chall_dir) / "distfiles"
        distfiles_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, str(distfiles_dir))
        (Path(chall_dir) / "description.txt").write_text(desc or "")

    if not desc and challenge_dir:
        desc_path = Path(chall_dir)
        for candidate in ["description.txt", "challenge.yml", "challenge.yaml"]:
            if (desc_path / candidate).exists():
                if candidate.endswith(".yml") or candidate.endswith(".yaml"):
                    from ctf_solver.prompts import ChallengeMeta
                    meta = ChallengeMeta.from_yaml(str(desc_path / candidate))
                    desc = meta.description
                    if not category:
                        category = meta.category
                    if not remote:
                        remote = meta.connection_info
                else:
                    desc = (desc_path / candidate).read_text().strip()
                break

    event_bus = EventBus()
    swarm = ChallengeSwarm(
        challenge_dir=chall_dir,
        challenge_name=Path(chall_dir).name,
        description=desc or "",
        category=category or "",
        settings=settings,
        event_bus=event_bus,
        connection_info=remote or "",
        resume_state=resume_state,
    )

    swarm_ref = [swarm]
    sigint_triggered = [False]

    def handle_sigint(signum: int, frame: object) -> None:
        console.print("\n[yellow]SIGINT received, shutting down gracefully...[/yellow]")
        swarm_ref[0].kill()
        sigint_triggered[0] = True

    signal.signal(signal.SIGINT, handle_sigint)

    if no_tui:
        result = asyncio.run(_run_cli(swarm, event_bus, settings, interactive, port, sigint_triggered, coordinator_provider=coord))
        from ctf_solver.writeup import generate_writeup

        if result and result.flag:
            console.print(f"\n[bold green]FLAG FOUND: {result.flag}[/bold green]")
            console.print(f"  Solver: {result.solver_id}")
            console.print(f"  Steps: {result.steps}, Duration: {result.duration:.1f}s, Cost: ${result.cost_usd:.4f}")
        elif sigint_triggered[0]:
            console.print("\n[yellow]Interrupted — generating partial writeup.[/yellow]")
        else:
            console.print("\n[red]No flag found.[/red]")
        if result:
            generate_writeup(
                output_dir or chall_dir, Path(chall_dir).name, category, result,
                total_cost_usd=swarm.cost_tracker.total_cost_usd,
                total_solvers=len(swarm.solvers),
            )
    else:
        from ctf_solver.tui.app import CTFApp

        app = CTFApp(event_bus=event_bus, challenge_name=Path(chall_dir).name, max_cost=settings.max_cost, category=category)

        async def run_with_tui() -> object:
            from ctf_solver.hint_server import start_hint_server
            from ctf_solver.writeup import generate_writeup

            hint_server, hint_port = await start_hint_server(event_bus, settings.port)
            console.print(f"  Hint endpoint: port {hint_port}")
            console.print(f"  Use: ctf-msg --port {hint_port} \"your hint\"")

            coordinator = None
            coord_provider = coord
            if coord_provider:
                from ctf_solver.solver.coordinator import CoordinatorAgent
                coordinator = CoordinatorAgent(
                    provider_name=coord_provider,
                    settings=settings,
                    event_bus=event_bus,
                    cost_tracker=swarm.cost_tracker,
                    message_bus=swarm.message_bus,
                    swarm=swarm,
                )
                await coordinator.start()

            swarm_task = asyncio.create_task(swarm.run())
            tui_task = asyncio.create_task(app.run_async())

            done, pending = await asyncio.wait(
                [swarm_task, tui_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            swarm.kill()
            if coordinator:
                await coordinator.stop()
            hint_server.close()
            await hint_server.wait_closed()
            result = None
            if swarm_task in done:
                try:
                    result = swarm_task.result()
                except Exception:
                    result = None
            if result:
                generate_writeup(
                    settings.output_dir or chall_dir, Path(chall_dir).name, category or "", result,
                    total_cost_usd=swarm.cost_tracker.total_cost_usd,
                    total_solvers=len(swarm.solvers),
                )
            return result

        result = asyncio.run(run_with_tui())
        if result and result.flag:
            console.print(f"\n[bold green]FLAG: {result.flag}[/bold green]")


async def _run_cli(
    swarm: ChallengeSwarm,
    event_bus: EventBus,
    settings: Settings,
    interactive: bool,
    port: int,
    sigint_triggered: list[bool],
    coordinator_provider: str | None = None,
) -> object:
    from ctf_solver.events import SolverEvent
    from ctf_solver.hint_server import start_hint_server

    hint_server = None
    coordinator = None
    queue = None
    events_task = None
    stdin_task = None

    try:
        hint_server, hint_port = await start_hint_server(event_bus, port)
        console.print(f"  Hint endpoint: port {hint_port}")
        console.print(f"  Use: ctf-msg --port {hint_port} \"your hint\"")

        if coordinator_provider:
            from ctf_solver.solver.coordinator import CoordinatorAgent

            coordinator = CoordinatorAgent(
                provider_name=coordinator_provider,
                settings=settings,
                event_bus=event_bus,
                cost_tracker=swarm.cost_tracker,
                message_bus=swarm.message_bus,
                swarm=swarm,
            )
            await coordinator.start()

        queue = event_bus.subscribe()

        async def print_events() -> None:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if event.type == "solver_started":
                        console.print(f"[green]\u25b6 {event.solver_id} started[/green]")
                    elif event.type == "solver_done":
                        console.print(f"[blue]\u25cb {event.solver_id} done[/blue]")
                    elif event.type == "flag_found":
                        console.print(f"[bold yellow]\u2691 {event.solver_id}: {event.data}[/bold yellow]")
                    elif event.type == "flag_candidate":
                        flag = event.data.get("flag", "")
                        verified = event.data.get("verified", False)
                        color = "green" if verified else "yellow"
                        console.print(f"[{color}]\u2691 {event.solver_id}: {flag} (verified={verified})[/{color}]")
                    elif event.type == "cost_update":
                        pass
                    else:
                        console.print(f"  [{event.solver_id}] {event.type}")
                except TimeoutError:
                    continue

        events_task = asyncio.create_task(print_events())

        if interactive:
            loop = asyncio.get_running_loop()

            async def read_stdin() -> None:
                while True:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    hint_text = line.strip()
                    if hint_text:
                        event_bus.publish(SolverEvent(type="user_hint", solver_id="operator", data={"message": hint_text}))

            stdin_task = asyncio.create_task(read_stdin())

        result = await swarm.run()

        if sigint_triggered[0] and result is None:
            for sid, inst in swarm.solvers.items():
                if inst.flag:
                    from ctf_solver.solver.solver_base import ResultStatus, SolverResult

                    result = SolverResult(
                        solver_id=sid,
                        status=ResultStatus.CANCELLED,
                        flag=inst.flag,
                        steps=inst.step_count,
                        duration=0.0,
                        cost_usd=inst.cost_usd,
                        findings_summary=inst.findings,
                    )
                    break

        return result
    finally:
        if coordinator:
            await coordinator.stop()
        for task in (events_task, stdin_task):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (events_task, stdin_task) if t), return_exceptions=True)
        if queue:
            event_bus.unsubscribe(queue)
        if hint_server:
            hint_server.close()
            await hint_server.wait_closed()


@click.command()
@click.argument("message")
@click.option("--port", default=9400, type=int, help="Coordinator port")
@click.option("--host", default="127.0.0.1", help="Coordinator host")
def send_message(message: str, port: int, host: str) -> None:
    """Send a hint to a running ctf-solve instance."""
    import urllib.request

    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/hint",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            console.print(f"[green]Sent:[/green] {data.get('queued', message[:200])}")
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    # So `python3 -m ctf_solver.cli` (or running this file directly) also
    # works, not just the installed `ctf-solve` console script.
    main()
