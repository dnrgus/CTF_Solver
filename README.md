# CTF Solver Agent

Multi-model CTF solver: for one challenge, it races several LLM "solver"
instances in parallel, each in its own Docker sandbox, sharing findings and
racing to the first confirmed flag.

This repo is a fork-and-merge of two projects:

- **[yakisoba0728/ctf-solver-agent](https://github.com/yakisoba0728/ctf-solver-agent)**
  — the base this repo is built on: provider-agnostic (Claude/Codex/z.ai),
  manual challenge input (no CTFd auto-fetch), Docker-sandboxed, Textual TUI.
- **[verialabs/ctf-agent](https://github.com/verialabs/ctf-agent)** — the
  original design this one is "inspired by" (per its own spec); 1st place,
  BSidesSF 2026 CTF (52/52). Some of its logic (retry/bump, service-connection
  prompt hints) was ported back in here — see `MERGE_NOTES.md`.

## Branches

| Branch | What it is |
|---|---|
| `merge-v-y` *(default)* | This fork's own work: verialabs' bump/retry + prompt hints ported in, several yakisoba-side bugs fixed (see `MERGE_NOTES.md`), plus the Codex-subscription fixes below. **Use this one.** |
| `main` | Unmodified `yakisoba0728/ctf-solver-agent`, kept as a baseline to diff/benchmark against. |
| `bench/codex-subscription-auth` | `main` + only the Codex-subscription-auth fixes (no bump/retry, no prompt changes) — an apples-to-apples baseline for A/B runs against `merge-v-y`. |

## How it works

- `ChallengeSwarm` (`src/ctf_solver/solver/swarm.py`) owns one challenge: it
  spins up N `SolverInstance`s (one per requested model), each with its own
  Docker sandbox, and runs them as parallel `asyncio` tasks.
- Each instance calls a small fixed tool set — `bash`, `read_file`,
  `write_file`, `list_files`, `view_image`, `web_fetch`, `webhook_create`,
  `submit_flag`, `check_findings`, `notify_coordinator` — against its own
  container. The *tools* are generic; the actual capability (pwntools,
  radare2, SageMath, RsaCtfTool, volatility3, steghide, zsteg, ...) is
  whatever `sandbox/Dockerfile` installs, and the system prompt is what tells
  the model which one fits the challenge's category.
- Every 5 steps, instances exchange findings over an in-process message bus.
  The first instance to get a flag confirmed takes a lock, and every other
  instance for that challenge is cancelled.
- Safety rails: per-run `--timeout`, `--max-steps`, `--max-cost`, a
  per-provider circuit breaker, and a loop detector that interrupts (then
  kills) a solver repeating the same tool call.

Docker isolation matters here specifically because the model is an
*autonomous* agent generating and running its own shell commands against
unknown binaries and live remote services — that should never happen
directly on the host. `asyncio` is what lets N instances' network-bound
work (LLM calls, `docker exec`) overlap in one process without needing N
OS threads.

## Install

```bash
# dependencies (editable install; works with uv or plain pip)
uv sync                      # or: pip install -e .

# sandbox image (shared across branches — the Dockerfile hasn't diverged)
docker build -t ctf-sandbox -f sandbox/Dockerfile .
```

Requires Docker, and Python 3.12+.

## Authentication

| Provider | API key | Or: CLI subscription login |
|---|---|---|
| `--claude N` | `ANTHROPIC_API_KEY` | not supported yet — `providers/claude.py` calls the raw Messages API directly |
| `--codex N` | `OPENAI_API_KEY` | **yes** — if `OPENAI_API_KEY` is unset, it inherits whatever the `codex` CLI is already logged into (ChatGPT/Codex Plus/Pro/Team), no key needed |
| `--zai N` | `ZAI_API_KEY` | not applicable |

The Codex subscription path needed three fixes on top of yakisoba's code
(all in `merge-v-y` and `bench/codex-subscription-auth`), found by driving a
live `codex app-server` by hand:

1. `CodexProvider.validate_config()` required `openai_api_key` even though
   `_start_process()` already falls back to the CLI's own stored login when
   no key is injected — it just never let that path run.
2. The hardcoded default `codex_model` (`gpt-5.4`) is rejected outright by a
   ChatGPT-plan login (`"model is not supported when using Codex with a
   ChatGPT account"`). There's no single model name that works for every
   account — set env var `CODEX_MODEL` (there's no `--codex-model` CLI flag)
   to whatever your own `~/.codex/config.toml`'s `model =` line already uses.
3. A live app-server's plain-text reply (no tool call) arrives as
   `item.text` on an `agentMessage` item — not as the `item.content[]` array
   the original parser expected, which only ever happened to populate from
   nothing. Any Codex turn that replied in text only (as opposed to a tool
   call) always returned an empty `response.text`, silently breaking the
   text-based flag scan for that provider.

Set credentials in `.env` (loaded via `pydantic-settings`) or the matching
environment variable.

## Usage

```bash
# a directory challenge (reads description.md, mounts distfiles/)
ctf-solve --challenge-dir ./chall/ --claude 1 --codex 1 --timeout 600 --no-tui

# loose files + description, Codex on a ChatGPT/Codex subscription (no API key) —
# CODEX_MODEL must match a model your account's plan actually accepts, e.g.
# whatever ~/.codex/config.toml already uses day-to-day
CODEX_MODEL=gpt-5.6-sol ctf-solve --files chall.bin --desc "pwn, remote at 1.2.3.4:1337" \
  --category pwn --codex 2 --no-tui

# re-engage a stalled solver up to twice, with sibling insights, before giving up
ctf-solve --challenge-dir ./chall/ --codex 1 --max-bumps 2

# inject extra process-discipline text (e.g. your own harness doc) into every
# solver's system prompt, for A/B comparisons
ctf-solve --challenge-dir ./chall/ --codex 1 --harness-file ./my-harness.md

# never auto-confirm a flag found by regex alone (only trust an external judge)
ctf-solve --challenge-dir ./chall/ --codex 1 --strict-flags
```

`--no-tui` runs plain CLI logging instead of the Textual dashboard — use it
for CI/scripted runs. `ctf-msg --port <N> "hint"` sends a live hint to a
running `ctf-solve` instance (the port is printed on startup).

Full option list: `ctf-solve --help`.

## Testing

```bash
uv run pytest -q --ignore=tests/test_docker_integration.py   # no Docker needed
uv run ruff check src tests
```

`test_docker_integration.py` needs a real Docker daemon and the
`ctf-sandbox` image built.

## What's not here

- CTFd (or any platform) auto-fetch/auto-submit — by design, same as upstream
  yakisoba. Challenges are supplied manually.
- verialabs' quota-fallback (swap to an API-backed provider mid-run when a
  subscription-backed one is rate-limited) and its CTFd client — not ported.
- Claude subscription auth (riding the `claude` CLI / Claude Agent SDK the
  way verialabs' `ClaudeSolver` does, instead of a raw API key) — the Codex
  side now has this, Claude doesn't yet.

See `MERGE_NOTES.md` for the full list of what was ported from verialabs and
what was fixed on the yakisoba side.
