# Example Challenges

This directory holds bundled CTF challenges you can run the solver against
without sourcing your own. Each challenge is a self-contained **directory** that
the CLI loads with `--challenge-dir`.

## Directory layout

A challenge directory follows the structure the CLI expects (see
`src/ctf_solver/cli.py` and `src/ctf_solver/prompts.py`):

```
challenges/example/
├── challenge.yml        # metadata: name, category, value, description, tags, hints, connection_info
├── description.txt      # the prompt the solver actually reads
└── distfiles/           # artifacts mounted into the sandbox at /challenge/distfiles/
    └── cipher.txt
```

How the CLI loads it (`--challenge-dir <dir>` with no `--desc`): it looks for
`description.txt`, `challenge.yml`, then `challenge.yaml`, **in that order**, and
uses the first it finds for the description. When `description.txt` is present it
wins; `challenge.yml` still supplies the `--category` hint if you don't pass
`--category` yourself. Everything under `distfiles/` is mounted read-only into
the sandbox at `/challenge/distfiles/`.

---

## `example/` — "Layer Cake" (misc, 50 pts)

A beginner multi-layer **encoding** challenge. The artifact `cipher.txt` is a
flag that has been encoded twice:

```
FLAG{...}  --(hex encode)-->  hex string  --(base64 encode)-->  cipher.txt
```

So solving it means undoing the layers in reverse: **base64-decode** the blob to
get a hex string, then **hex-decode** that to recover the ASCII flag. It is pure
encoding (reversible, no key) — exactly what the description hints at. Solvable
with `base64`, `xxd`, and `python3`, all of which exist in the sandbox image.

### Run it

From the repository root, with one Claude solver and no TUI (plain stdout):

```bash
ctf-solve --challenge-dir challenges/example/ --claude 1 --no-tui
```

Other useful variants:

```bash
# Dry run — validate config / providers without executing
ctf-solve --challenge-dir challenges/example/ --claude 1 --dry-run

# Mix providers (1 Claude + 1 Codex), cap cost
ctf-solve --challenge-dir challenges/example/ --claude 1 --codex 1 --max-cost 1.00

# Debug on the host without Docker (prompts for confirmation; needs base64 + xxd locally)
ctf-solve --challenge-dir challenges/example/ --claude 1 --no-docker --no-tui
```

A correct run ends with the line:

```
FLAG FOUND: FLAG{l4y3r3d_3nc0d1ng_15_n0t_3ncryp710n}
```

---

<details>
<summary><strong>⚠️ SPOILER — Solution (click to expand)</strong></summary>

The artifact is `base64( hex( FLAG{...} ) )`. Undo it from the outside in.

**Two-step (clearest):**

```bash
# 1) base64-decode the blob -> a hex string (only 0-9 a-f)
base64 -d challenges/example/distfiles/cipher.txt
# -> 464c41477b6c3479337233645f336e633064316e675f31355f6e30745f336e637279703731306e7d

# 2) hex-decode that -> the flag
base64 -d challenges/example/distfiles/cipher.txt | xxd -r -p
# -> FLAG{l4y3r3d_3nc0d1ng_15_n0t_3ncryp710n}
```

**One-liner:**

```bash
base64 -d challenges/example/distfiles/cipher.txt | xxd -r -p; echo
```

**Pure Python (no xxd):**

```bash
python3 -c "import base64,sys; print(bytes.fromhex(base64.b64decode(open('challenges/example/distfiles/cipher.txt').read()).decode()).decode())"
```

All three print the flag:

```
FLAG{l4y3r3d_3nc0d1ng_15_n0t_3ncryp710n}
```

</details>
