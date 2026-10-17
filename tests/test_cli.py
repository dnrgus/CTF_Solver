"""Tests for CLI."""

from click.testing import CliRunner

from ctf_solver.cli import _auto_discover_files, _prompt_or_x, main


def test_no_args_prompts_then_aborts_on_eof():
    """Bare invocation no longer errors immediately — it asks for the
    challenge folder path (so you can run this from wherever the solver is
    installed, not necessarily from inside the challenge folder itself) and,
    with no simulated stdin, aborts on EOF instead."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "분석할 문제 폴더 경로" in result.output


def test_bare_invocation_default_path_is_cwd(tmp_path, monkeypatch):
    """An empty first answer (just Enter) falls back to the current directory,
    same as before this folder-path prompt existed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chall.bin").write_bytes(b"\x00")
    runner = CliRunner()
    result = runner.invoke(main, ["--claude", "1", "--dry-run"], input="\nX\nX\n")
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "chall.bin" in result.output


def test_bare_invocation_explicit_folder_path_is_analyzed(tmp_path, monkeypatch):
    """Running from one directory but pointing at a different challenge
    folder — the actual point of asking for a path instead of always using
    cwd — picks up THAT folder's files, not the cwd's."""
    solver_dir = tmp_path / "solver"
    solver_dir.mkdir()
    chall_dir = tmp_path / "some_challenge"
    chall_dir.mkdir()
    (chall_dir / "pwn_bin").write_bytes(b"\x7fELF")
    monkeypatch.chdir(solver_dir)  # nothing relevant here
    runner = CliRunner()
    result = runner.invoke(
        main, ["--claude", "1", "--dry-run"], input=f"{chall_dir}\nX\nX\n"
    )
    assert result.exit_code == 0, result.output
    assert "pwn_bin" in result.output
    assert str(chall_dir) in result.output


def test_bare_invocation_nonexistent_folder_path_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["--claude", "1"], input=f"{tmp_path / 'nope'}\n")
    assert result.exit_code != 0
    assert "존재하지 않는 폴더" in result.output


def test_bare_invocation_x_sentinels_reach_dry_run(tmp_path, monkeypatch):
    """Typing X for both prompts must NOT be treated as 'missing --desc' (an
    empty string from X is a valid, deliberate answer, not an absent one)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chall.bin").write_bytes(b"\x00")
    runner = CliRunner()
    result = runner.invoke(main, ["--claude", "1", "--dry-run"], input=f"{tmp_path}\nX\nX\n")
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "chall.bin" in result.output


def test_bare_invocation_with_answers_reaches_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["--claude", "1", "--dry-run"],
        input=f"{tmp_path}\npwn remote, get the flag\nnc 1.2.3.4 1337\n",
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


# --------------------------------------------------------------------------- #
# _normalize_path_input — pasted Windows/WSL paths
# --------------------------------------------------------------------------- #
def test_normalize_path_strips_double_quotes():
    from ctf_solver.cli import _normalize_path_input

    assert _normalize_path_input('"C:\\Users\\x\\chall"') == "/mnt/c/Users/x/chall"


def test_normalize_path_strips_single_quotes():
    from ctf_solver.cli import _normalize_path_input

    assert _normalize_path_input("'/home/x/chall'") == "/home/x/chall"


def test_normalize_path_windows_drive_to_wsl_mount():
    from ctf_solver.cli import _normalize_path_input

    assert _normalize_path_input(r"C:\Users\dnrgu\Desktop\chall") == "/mnt/c/Users/dnrgu/Desktop/chall"


def test_normalize_path_lowercase_drive_letter():
    from ctf_solver.cli import _normalize_path_input

    assert _normalize_path_input(r"c:\foo") == "/mnt/c/foo"


def test_normalize_path_plain_wsl_path_passthrough():
    from ctf_solver.cli import _normalize_path_input

    assert _normalize_path_input("/mnt/c/Users/x/chall") == "/mnt/c/Users/x/chall"


def test_bare_invocation_pasted_windows_path_with_quotes(tmp_path, monkeypatch):
    """Reproduces the exact failure: a Windows path pasted with quotes and
    backslashes, from a WSL prompt, must resolve to the real /mnt/c/... dir."""
    monkeypatch.chdir(tmp_path)
    # Simulate "C:\...\battle_ctf\20_PWN_003" by using whatever drive /mnt/c
    # actually maps to in this environment, so the directory genuinely exists.
    import pathlib

    mnt_c = pathlib.Path("/mnt/c")
    if not mnt_c.is_dir():
        import pytest

        pytest.skip("no /mnt/c in this environment")
    chall = mnt_c / "ctf_solver_test_tmp_20_PWN_003"
    chall.mkdir(exist_ok=True)
    (chall / "notes").write_bytes(b"x")
    try:
        windows_style = '"C:\\ctf_solver_test_tmp_20_PWN_003"'
        runner = CliRunner()
        result = runner.invoke(
            main, ["--claude", "1", "--dry-run"], input=f"{windows_style}\nX\nX\n"
        )
        assert result.exit_code == 0, result.output
        assert "notes" in result.output
    finally:
        (chall / "notes").unlink(missing_ok=True)
        chall.rmdir()
