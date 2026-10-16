# Merge notes (verialabs/ctf-agent -> ctf-solver-agent)

Branch `merge-v-y`. Baseline = `main` (original yakisoba code).

## Ported from verialabs
- Bump/retry: a solver that stops without a flag is resumed up to `--max-bumps` (default 2)
  after an escalating cooldown, with sibling insights; fresh step budget per attempt.
  Broken solvers (0 steps, $0), budget exhaustion, open circuit breaker and timeout are never bumped.
- Prompt: strict FIRST ACTION for service challenges, web/nc(heredoc) connection hints,
  pyghidra block for rev/pwn/misc, "don't guess, cover surface" + `stty raw -echo` tip.

## Fixes to the yakisoba side
- Image files now tell the model to call `view_image` (prompt contradicted the tool set).
- Flag dedup + cooldown + confirmation happen under one lock.
- Insight sharing fires every 5 steps even if a step count skips a multiple of 5.
- Placeholder flags (`CTF{flag}`, `FLAG{placeholder}`) are rejected.
- `accept_regex_flag_as_verified` defaulted to False with no CLI switch, so a run could never reach
  SOLVED. Default is now True; `--strict-flags` restores the old behaviour.
- Pending tool results are kept so a resumed session never has an unanswered tool_use.

## Experiment switches (for A/B runs)
| Switch | Effect |
|---|---|
| `--harness-file FILE` | appends FILE under `## Harness` at the end of every solver prompt |
| `--max-bumps 0` | disables the retry loop (baseline behaviour) |
| `--strict-flags` | never auto-confirm regex matches |
| `--hint` | operator hint section (unchanged) |

Not ported: verialabs quota fallback (needs an API-backed provider mapping), CTFd client/poller.
