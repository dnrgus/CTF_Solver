"""Tests for CLI."""

from click.testing import CliRunner

from ctf_solver.cli import _auto_discover_files, _prompt_or_x, main


def test_no_args_prompts_then_aborts_on_eof():
    """Bare invocation no longer errors immediately — it tries to prompt for
    문제설명/원격 서버, and (with no simulated stdin) aborts on EOF instead."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "문제설명" in result.output


def test_bare_invocation_x_sentinels_reach_dry_run(tmp_path, monkeypatch):
    """Typing X for both prompts must NOT be treated as 'missing --desc' (an
    empty string from X is a valid, deliberate answer, not an absent one)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chall.bin").write_bytes(b"\x00")
    runner = CliRunner()
    result = runner.invoke(main, ["--claude", "1", "--dry-run"], input="X\nX\n")
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "chall.bin" in result.output


def test_bare_invocation_with_answers_reaches_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["--claude", "1", "--dry-run"], input="pwn remote, get the flag\nnc 1.2.3.4 1337\n"
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output


def test_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--claude" in result.output
    assert "--codex" in result.output
    assert "--zai" in result.output
    assert "--remote" in result.output


def test_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, [
        "--files", "/dev/null",
        "--desc", "test challenge",
        "--claude", "1",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output


def test_mutual_exclusion():
    runner = CliRunner()
    result = runner.invoke(main, [
        "--challenge-dir", "/tmp",
        "--files", "/dev/null",
        "--desc", "test",
        "--claude", "1",
    ])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# _auto_discover_files
# --------------------------------------------------------------------------- #
def test_auto_discover_files_top_level_and_distfiles(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / ".hidden").write_bytes(b"x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"x")
    dist = tmp_path / "distfiles"
    dist.mkdir()
    (dist / "b.bin").write_bytes(b"x")

    found = {p.split("/")[-1] for p in _auto_discover_files(tmp_path)}
    assert found == {"a.bin", "b.bin"}


def test_auto_discover_files_empty_dir(tmp_path):
    assert _auto_discover_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# _prompt_or_x
# --------------------------------------------------------------------------- #
def test_prompt_or_x_literal_x_means_empty(monkeypatch):
    import click

    monkeypatch.setattr(click, "prompt", lambda *a, **k: "X")
    assert _prompt_or_x("문제설명") == ""


def test_prompt_or_x_lowercase_x_means_empty(monkeypatch):
    import click

    monkeypatch.setattr(click, "prompt", lambda *a, **k: "x")
    assert _prompt_or_x("원격 서버") == ""


def test_prompt_or_x_real_text_passes_through(monkeypatch):
    import click

    monkeypatch.setattr(click, "prompt", lambda *a, **k: "  nc 1.2.3.4 1337  ")
    assert _prompt_or_x("원격 서버") == "nc 1.2.3.4 1337"
