"""Tests for DockerSandbox — uses mock Docker daemon when available."""

import subprocess

import pytest

from ctf_solver.sandbox import ExecResult
from ctf_solver.sandbox.docker import DockerSandbox


def _docker_available():
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def _image_available(image):
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def test_exec_result_dataclass():
    r = ExecResult(exit_code=0, stdout="hello", stderr="")
    assert r.exit_code == 0
    assert r.stdout == "hello"


def test_parse_memory_limit():
    sandbox = DockerSandbox(image="test", challenge_dir="/tmp")
    assert sandbox._parse_memory_limit("4g") == 4 * 1024 * 1024 * 1024
    assert sandbox._parse_memory_limit("512m") == 512 * 1024 * 1024
    assert sandbox._parse_memory_limit("2048") == 2048


def test_parse_memory_limit_float():
    sandbox = DockerSandbox(image="test", challenge_dir="/tmp")
    assert sandbox._parse_memory_limit("2.5g") == int(2.5 * 1024 * 1024 * 1024)
    assert sandbox._parse_memory_limit("1.5m") == int(1.5 * 1024 * 1024)


def test_parse_memory_limit_negative():
    sandbox = DockerSandbox(image="test", challenge_dir="/tmp")
    assert sandbox._parse_memory_limit("-1g") == 4 * 1024 * 1024 * 1024
    assert sandbox._parse_memory_limit("0") == 4 * 1024 * 1024 * 1024


@pytest.mark.skipif(
    "not _docker_available() or not _image_available('alpine:latest')",
    reason="Docker not available or alpine:latest not pulled",
)
@pytest.mark.asyncio
async def test_sandbox_lifecycle():
    sandbox = DockerSandbox(image="alpine:latest", challenge_dir="/tmp")
    await sandbox.start()
    result = await sandbox.exec("echo hello", timeout_s=10)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    await sandbox.stop()


@pytest.mark.skipif("not _docker_available()", reason="Docker not available")
@pytest.mark.asyncio
async def test_missing_image_error():
    sandbox = DockerSandbox(image="nonexistent-image-xyz:latest", challenge_dir="/tmp")
    with pytest.raises(RuntimeError, match="not found"):
        await sandbox.start()


# --------------------------------------------------------------------------- #
# extra_ro_binds — used to confine a provider's own native process (codex)
# inside the container instead of the host; see providers/codex.py.
# --------------------------------------------------------------------------- #
class _FakeContainer:
    def __init__(self, container_id: str = "fakecontainerid") -> None:
        self.id = container_id

    async def start(self) -> None:
        pass


class _FakeContainers:
    def __init__(self) -> None:
        self.created_config: dict | None = None

    async def create(self, config: dict) -> _FakeContainer:
        self.created_config = config
        return _FakeContainer()


class _FakeDocker:
    def __init__(self) -> None:
        self.containers = _FakeContainers()

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_extra_ro_binds_included_in_container_config(monkeypatch, tmp_path):
    import ctf_solver.sandbox.docker as docker_mod

    fake_docker = _FakeDocker()
    monkeypatch.setattr(docker_mod.aiodocker, "Docker", lambda: fake_docker)

    sandbox = DockerSandbox(
        image="ctf-sandbox",
        challenge_dir=str(tmp_path),
        extra_ro_binds=["/host/codex-bin:/opt/codex-bin:ro"],
    )
    await sandbox.start()

    binds = fake_docker.containers.created_config["HostConfig"]["Binds"]
    assert "/host/codex-bin:/opt/codex-bin:ro" in binds
    # Still after the normal workspace bind, not replacing it.
    assert any(b.endswith(":/challenge/workspace:rw") for b in binds)


@pytest.mark.asyncio
async def test_no_extra_ro_binds_by_default(monkeypatch, tmp_path):
    import ctf_solver.sandbox.docker as docker_mod

    fake_docker = _FakeDocker()
    monkeypatch.setattr(docker_mod.aiodocker, "Docker", lambda: fake_docker)

    sandbox = DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path))
    await sandbox.start()

    binds = fake_docker.containers.created_config["HostConfig"]["Binds"]
    assert not any("codex" in b for b in binds)
