"""Tests for tools/core.py — sandbox tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctf_solver.tools.core import (
    _truncate,
    do_bash,
    do_check_findings,
    do_list_files,
    do_notify_coordinator,
    do_read_file,
    do_submit_flag,
    do_web_fetch,
    do_write_file,
)


def test_truncate_under_limit():
    assert _truncate("hello") == "hello"


def test_truncate_over_limit():
    long_text = "a" * 30_000
    result = _truncate(long_text)
    assert len(result) < 30_000
    assert "truncated" in result


@pytest.mark.asyncio
async def test_do_bash_success():
    sandbox = MagicMock()
    result = AsyncMock()
    result.stdout = "hello"
    result.stderr = ""
    result.exit_code = 0
    sandbox.exec = AsyncMock(return_value=result)
    out = await do_bash(sandbox, "echo hello")
    assert "hello" in out


@pytest.mark.asyncio
async def test_do_bash_nonzero_exit():
    sandbox = MagicMock()
    result = AsyncMock()
    result.stdout = ""
    result.stderr = "error"
    result.exit_code = 1
    sandbox.exec = AsyncMock(return_value=result)
    out = await do_bash(sandbox, "bad_cmd")
    assert "exit 1" in out


@pytest.mark.asyncio
async def test_do_read_file_text():
    sandbox = MagicMock()
    sandbox.read_file = AsyncMock(return_value="file content")
    out = await do_read_file(sandbox, "/challenge/test.txt")
    assert "file content" in out


@pytest.mark.asyncio
async def test_do_read_file_binary():
    sandbox = MagicMock()
    sandbox.read_file = AsyncMock(return_value=b"\x00\x01")
    out = await do_read_file(sandbox, "/challenge/binary.bin")
    assert "Binary file" in out


@pytest.mark.asyncio
async def test_do_read_file_error():
    sandbox = MagicMock()
    sandbox.read_file = AsyncMock(side_effect=Exception("not found"))
    out = await do_read_file(sandbox, "/missing")
    assert "Error" in out


@pytest.mark.asyncio
async def test_do_write_file_success():
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    out = await do_write_file(sandbox, "/challenge/out.txt", "data")
    assert "Written" in out


@pytest.mark.asyncio
async def test_do_write_file_error():
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock(side_effect=Exception("no space"))
    out = await do_write_file(sandbox, "/out.txt", "data")
    assert "Error" in out


@pytest.mark.asyncio
async def test_do_list_files_success():
    sandbox = MagicMock()
    result = MagicMock()
    result.stdout = "file1.txt\nfile2.py"
    result.stderr = ""
    result.exit_code = 0
    sandbox.exec = AsyncMock(return_value=result)
    out = await do_list_files(sandbox, "/challenge/distfiles")
    assert "file1.txt" in out


@pytest.mark.asyncio
async def test_do_web_fetch_blocks_localhost():
    out = await do_web_fetch("http://localhost/secret")
    assert "blocked" in out.lower() or "error" in out.lower()


@pytest.mark.asyncio
async def test_do_web_fetch_blocks_127():
    out = await do_web_fetch("http://127.0.0.1/secret")
    assert "blocked" in out.lower() or "error" in out.lower()


@pytest.mark.asyncio
async def test_do_web_fetch_blocks_private_ip():
    out = await do_web_fetch("http://10.0.0.1/secret")
    assert "blocked" in out.lower() or "error" in out.lower()


@pytest.mark.asyncio
async def test_do_submit_flag_format_match_is_only_a_candidate():
    """A format match is accepted as a candidate; it never reports verification."""
    flags: set[str] = set()
    msg, valid = await do_submit_flag(
        "FLAG{test}", r"([Ff][Ll][Aa][Gg]|[Cc][Tt][Ff])\{[^}]+\}", flags
    )
    assert valid is True
    assert "FLAG{test}" in flags
    assert "not yet verified" in msg


@pytest.mark.asyncio
async def test_do_submit_flag_rejects_embedded_flag_in_surrounding_text():
    """'junk FLAG{x} junk' contains a flag but is not one; it must not become a candidate."""
    flags: set[str] = set()
    msg, valid = await do_submit_flag(
        "result: FLAG{x} done", r"([Ff][Ll][Aa][Gg]|[Cc][Tt][Ff])\{[^}]+\}", flags
    )
    assert valid is False
    assert flags == set()


@pytest.mark.asyncio
async def test_do_submit_flag_empty():
    flags: set[str] = set()
    msg, valid = await do_submit_flag("", r"FLAG\{[^}]+\}", flags)
    assert valid is False


@pytest.mark.asyncio
async def test_do_submit_flag_duplicate():
    flags: set[str] = {"FLAG{same}"}
    msg, valid = await do_submit_flag("FLAG{same}", r"FLAG\{[^}]+\}", flags)
    assert valid is False


@pytest.mark.asyncio
async def test_do_submit_flag_no_match():
    flags: set[str] = set()
    msg, valid = await do_submit_flag("not_a_flag", r"FLAG\{[^}]+\}", flags)
    assert valid is False


@pytest.mark.asyncio
async def test_do_check_findings_no_bus():
    out = await do_check_findings(None, "solver-1")
    assert "No message bus" in out


@pytest.mark.asyncio
async def test_do_check_findings_with_bus():
    bus = MagicMock()
    bus.check = AsyncMock(return_value=[])
    out = await do_check_findings(bus, "solver-1")
    assert "No new findings" in out


@pytest.mark.asyncio
async def test_do_notify_coordinator_no_bus():
    out = await do_notify_coordinator("test", None, "solver-1")
    assert "No event bus" in out
