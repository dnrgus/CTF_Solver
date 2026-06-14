"""Tests for HostSandbox — real bash/tmp dirs on Linux (no isolation)."""

from pathlib import Path

import pytest

from ctf_solver.sandbox.host import HostSandbox


@pytest.fixture
def challenge_dir(tmp_path):
    (tmp_path / "distfiles").mkdir()
    return tmp_path


def test_map_path_distfiles_and_workspace(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))

    # distfiles maps under the challenge dir (no start() needed for distfiles).
    mapped_dist = sandbox._map_path("/challenge/distfiles/x")
    assert mapped_dist == challenge_dir.resolve() / "distfiles" / "x"

    # /challenge itself maps to the challenge dir.
    assert sandbox._map_path("/challenge") == Path(str(challenge_dir.resolve()))


async def test_map_path_workspace_uses_tempdir_after_start(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    await sandbox.start()
    try:
        workspace_root = Path(sandbox._workspace_dir)
        assert workspace_root.is_dir()
        # The workspace tempdir is created by start() and is NOT inside the
        # challenge dir (it lives under the system temp prefix).
        assert workspace_root.name.startswith("ctf-host-workspace-")

        mapped = sandbox._map_path("/challenge/workspace/y")
        assert mapped == workspace_root / "y"
    finally:
        await sandbox.stop()


@pytest.mark.parametrize("blocked", ["/etc/passwd", "/proc/self/maps", "/root/.ssh/id_rsa"])
def test_blocked_paths_map_raises(challenge_dir, blocked):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    with pytest.raises(ValueError):
        sandbox._map_path(blocked)


@pytest.mark.parametrize("blocked", ["/etc/passwd", "/proc/self/maps", "/root/.ssh/id_rsa"])
async def test_blocked_paths_read_write_raise(challenge_dir, blocked):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    with pytest.raises(ValueError):
        await sandbox.read_file(blocked)
    with pytest.raises(ValueError):
        await sandbox.write_file(blocked, "nope")


async def test_write_then_read_round_trip_in_workspace(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    await sandbox.start()
    try:
        await sandbox.write_file("/challenge/workspace/note.txt", "hello-workspace")
        content = await sandbox.read_file("/challenge/workspace/note.txt")
        assert content == "hello-workspace"
        # File physically landed under the workspace tempdir.
        assert (Path(sandbox._workspace_dir) / "note.txt").read_text() == "hello-workspace"
    finally:
        await sandbox.stop()


async def test_exec_runs_real_command(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    await sandbox.start()
    try:
        result = await sandbox.exec("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
    finally:
        await sandbox.stop()


async def test_exec_rewrites_challenge_tokens(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    await sandbox.start()
    try:
        await sandbox.write_file("/challenge/workspace/data.txt", "rewritten-content")
        # The /challenge/workspace token must be rewritten to the real tempdir
        # so a plain `cat` finds the file.
        result = await sandbox.exec("cat /challenge/workspace/data.txt")
        assert result.exit_code == 0
        assert "rewritten-content" in result.stdout
    finally:
        await sandbox.stop()


def test_container_id_is_host(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    assert sandbox.container_id == "host"


async def test_stop_is_idempotent(challenge_dir):
    sandbox = HostSandbox(challenge_dir=str(challenge_dir))
    await sandbox.start()
    workspace = Path(sandbox._workspace_dir)
    assert workspace.is_dir()

    await sandbox.stop()
    assert not workspace.exists()
    # Second stop must not raise even though the tempdir is already gone.
    await sandbox.stop()
