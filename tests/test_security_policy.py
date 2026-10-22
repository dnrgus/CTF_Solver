"""Fail-closed sandbox network and credential policy tests.

These tests use a fake Docker daemon; they do not need a running container.
Real networking behaviour still requires Docker integration tests.
"""

from __future__ import annotations

import pytest

from ctf_solver.config import Settings
from ctf_solver.sandbox import docker as docker_mod
from ctf_solver.sandbox.docker import DockerSandbox


class FakeContainer:
    id = "test-container-id"

    async def start(self) -> None:
        pass

    async def delete(self, force: bool = False) -> None:
        pass


class FakeContainers:
    def __init__(self) -> None:
        self.config = None

    async def create(self, config: dict) -> FakeContainer:
        self.config = config
        return FakeContainer()


class FakeDocker:
    def __init__(self) -> None:
        self.containers = FakeContainers()

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_default_sandbox_cannot_egress_or_use_host_gateway(monkeypatch, tmp_path):
    client = FakeDocker()
    monkeypatch.setattr(docker_mod.aiodocker, "Docker", lambda: client)
    sandbox = DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path))
    try:
        await sandbox.start()
        cfg = client.containers.config
        assert cfg["NetworkDisabled"] is True
        assert cfg["HostConfig"]["NetworkMode"] == "none"
        assert cfg["HostConfig"]["ExtraHosts"] == []
        assert cfg["HostConfig"]["CapDrop"] == ["ALL"]
        assert "MKNOD" not in cfg["HostConfig"]["CapAdd"]
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_bridge_egress_and_host_gateway_are_explicit(monkeypatch, tmp_path):
    client = FakeDocker()
    monkeypatch.setattr(docker_mod.aiodocker, "Docker", lambda: client)
    sandbox = DockerSandbox(
        image="ctf-sandbox", challenge_dir=str(tmp_path),
        network_mode="bridge", allow_host_gateway=True,
    )
    try:
        await sandbox.start()
        cfg = client.containers.config
        assert cfg["NetworkDisabled"] is False
        assert cfg["HostConfig"]["NetworkMode"] == "bridge"
        assert cfg["HostConfig"]["ExtraHosts"] == ["host.docker.internal:host-gateway"]
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_invalid_network_configuration_fails_before_container_start(tmp_path):
    sandbox = DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path), network_mode="host")
    with pytest.raises(ValueError, match="network_mode"):
        await sandbox.start()
    sandbox = DockerSandbox(image="ctf-sandbox", challenge_dir=str(tmp_path), allow_host_gateway=True)
    with pytest.raises(ValueError, match="host gateway"):
        await sandbox.start()


def test_credentials_default_to_denied(monkeypatch):
    monkeypatch.delenv("ALLOW_CONTAINER_CREDENTIALS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.allow_container_credentials is False
    assert settings.sandbox_network_mode == "none"
    assert settings.sandbox_allow_host_gateway is False
    assert settings.to_provider_config()["allow_container_credentials"] is False
