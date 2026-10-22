"""Container hardening and the tool list the Codex provider exposes."""

from __future__ import annotations

import pytest
from test_codex_provider import FakeDockerSandbox, drive_turn, fake_proc  # noqa: F401  (fixture re-export)

from ctf_solver.providers.base import ToolDef
from ctf_solver.providers.codex import CodexSession
from ctf_solver.sandbox import docker as docker_mod


async def test_codex_thread_start_exposes_exactly_the_tools_the_swarm_passes(fake_proc):  # noqa: F811
    tools = [
        ToolDef(name="record_experiment", description="record", parameters={"type": "object", "properties": {}}),
        ToolDef(name="bash", description="run", parameters={"type": "object", "properties": {"command": {"type": "string"}}}),
    ]
    session = CodexSession(
        "solver-1", "system", tools,
        {"openai_api_key": "sk-test", "allow_container_credentials": True},
        sandbox=FakeDockerSandbox(),
    )

    await drive_turn(session, fake_proc, "hi", [])

    start = next(m for m in fake_proc.stdin.written_messages() if m.get("method") == "thread/start")
    dynamic = start["params"]["dynamicTools"]
    assert [t["name"] for t in dynamic] == ["record_experiment", "bash"]
    assert dynamic[1]["inputSchema"] == tools[1].parameters
    await session.close()


async def test_codex_coordinator_with_no_tools_gets_none(fake_proc):  # noqa: F811
    session = CodexSession(
        "coord", "system", [],
        {"openai_api_key": "sk-test", "allow_container_credentials": True},
        sandbox=FakeDockerSandbox(),
    )

    await drive_turn(session, fake_proc, "hi", [])

    start = next(m for m in fake_proc.stdin.written_messages() if m.get("method") == "thread/start")
    assert start["params"]["dynamicTools"] == []
    await session.close()


class _FakeContainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.id = "abcdef123456"

    async def start(self) -> None:
        return None


class _FakeContainers:
    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create(self, config: dict) -> _FakeContainer:
        self.created.append(config)
        return _FakeContainer(config)


class _FakeDocker:
    instances: list[_FakeDocker] = []

    def __init__(self) -> None:
        self.containers = _FakeContainers()
        _FakeDocker.instances.append(self)


@pytest.fixture
def fake_docker(monkeypatch):
    _FakeDocker.instances = []
    monkeypatch.setattr(docker_mod.aiodocker, "Docker", _FakeDocker)
    return _FakeDocker


async def test_solver_container_drops_all_capabilities_and_limits_resources(tmp_path, fake_docker):
    sandbox = docker_mod.DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path))
    await sandbox.start()
    host = sandbox._container.config["HostConfig"]

    assert host["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in host["SecurityOpt"]
    assert host["PidsLimit"] > 0
    assert host["Memory"] > 0
    assert host["NanoCpus"] > 0
    # MKNOD lets a solver create device nodes; no solver step needs it.
    assert "MKNOD" not in host["CapAdd"]
    assert "SYS_ADMIN" not in host["CapAdd"]
    assert "NET_ADMIN" not in host["CapAdd"]


async def test_workspace_is_the_only_writable_bind(tmp_path, fake_docker):
    (tmp_path / "distfiles").mkdir()
    sandbox = docker_mod.DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path))
    await sandbox.start()
    binds = sandbox._container.config["HostConfig"]["Binds"]

    rw = [b for b in binds if b.endswith(":rw")]
    assert len(rw) == 1 and rw[0].endswith(":/challenge/workspace:rw")
    assert any(b.endswith(":/challenge/distfiles:ro") for b in binds)
