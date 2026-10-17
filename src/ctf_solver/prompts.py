"""System prompt builder — category-aware CTF solver prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ctf_solver.tools.vision import IMAGE_EXTS


@dataclass
class ChallengeMeta:
    name: str = "Unknown"
    category: str = ""
    value: int = 0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    connection_info: str = ""
    hints: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChallengeMeta:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            name=data.get("name", "Unknown"),
            category=data.get("category", ""),
            value=data.get("value", 0),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            connection_info=data.get("connection_info", ""),
            hints=data.get("hints", []),
        )


def list_distfiles(challenge_dir: str) -> list[str]:
    dist = Path(challenge_dir) / "distfiles"
    if not dist.exists():
        return []
    return sorted(f.name for f in dist.iterdir() if f.is_file())


def _rewrite_connection_info(conn: str) -> str:
    if not conn:
        return conn
    conn = re.sub(r"\blocalhost\b", "host.docker.internal", conn)
    conn = re.sub(r"\b127\.0\.0\.1\b", "host.docker.internal", conn)
    return conn


# Tactical first-moves per category — injected to prime the right approach.
CATEGORY_PLAYBOOK: dict[str, list[str]] = {
    "pwn": [
        "Run `checksec` and `file`; note arch and RELRO/canary/NX/PIE.",
        "Disassemble (objdump -d, radare2, pwntools) and find the vulnerable function.",
        "Build the exploit with pwntools and test locally before the remote service.",
        "Usual bugs: stack overflow, format string, off-by-one, UAF, integer overflow.",
    ],
    "rev": [
        "Triage with `file`, `strings`, `nm`; spot packers, language, embedded data.",
        "Decompile (radare2, objdump, pyghidra) and locate the check gating the flag.",
        "Look for hardcoded comparisons, XOR/encode routines, or a serial/keygen check.",
        "Use ltrace/strace or a debugger to observe runtime behavior.",
    ],
    "crypto": [
        "Identify the scheme (RSA, AES, XOR, classical) from the source/params first.",
        "RSA: check small e, shared modulus, Wiener, factorable n (RsaCtfTool, sympy).",
        "Hunt for nonce/key reuse, ECB patterns, and weak/predictable randomness.",
        "Implement the attack with python3 + pycryptodome / sympy / z3.",
    ],
    "web": [
        "Map the app: `curl -i` endpoints, read HTML/JS, inspect cookies and headers.",
        "Test SQLi, SSTI, XSS, IDOR, path traversal, auth bypass, SSRF.",
        "Use webhook_create for out-of-band (blind SSRF/XXE/OOB) confirmation.",
        "Check robots.txt, /.git, backups, and source-disclosure leaks.",
    ],
    "forensics": [
        "Identify the artifact: `file`, `binwalk`, `exiftool`, `strings`.",
        "pcap → tshark; memory → volatility3; disk → sleuthkit/foremost.",
        "Carve embedded files, follow streams, reassemble fragments.",
    ],
    "stego": [
        "Start broad: `strings`, `exiftool`, `binwalk`, `xxd` on the media.",
        "Images: zsteg (PNG/BMP), steghide/stegseek (JPG), LSB and channel splits.",
        "Audio: inspect spectrograms; always check metadata and appended data.",
    ],
    "misc": [
        "Enumerate first: `file`, `strings`, `xxd`; identify encodings (base64/hex/rot).",
        "Decode layer by layer with python3; chained encodings are common.",
        "Watch for jails/sandboxes, esolangs, QR/barcodes, and protocol puzzles.",
    ],
}

_CATEGORY_ALIASES = {
    "reversing": "rev",
    "reverse": "rev",
    "re": "rev",
    "binary": "pwn",
    "binexp": "pwn",
    "exploitation": "pwn",
    "cryptography": "crypto",
    "steganography": "stego",
    "forensic": "forensics",
}


def _category_tips(category: str) -> list[str]:
    cat = (category or "").lower()
    for key, tips in CATEGORY_PLAYBOOK.items():
        if key and (key in cat or cat == key):
            return tips
    for alias, key in _CATEGORY_ALIASES.items():
        if alias == cat:
            return CATEGORY_PLAYBOOK[key]
    return []


def build_prompt(
    meta: ChallengeMeta,
    distfile_names: list[str],
    container_arch: str = "unknown",
    hint: str = "",
) -> str:
    conn_info = _rewrite_connection_info(meta.connection_info.strip())
    lines: list[str] = [
        "You are an expert CTF solver. Find the real flag for the challenge below.",
        "",
    ]

    if conn_info:
        lines += [
            "> FIRST ACTION: Connect to the service now.",
            f"> Run: `{conn_info}`",
            "",
        ]

    lines += [
        "## Challenge",
        f"**Name**: {meta.name}",
        f"**Category**: {meta.category or 'Unknown'}",
        f"**Points**: {meta.value or '?'}",
        f"**Arch**: {container_arch}",
    ]
    lines += ["", "## Description", meta.description or "No description provided.", ""]

    if distfile_names:
        lines.append("## Attached Files")
        for name in distfile_names:
            ext = Path(name).suffix.lower()
            is_img = ext in IMAGE_EXTS
            suffix = " <- IMAGE: use steghide/exiftool/strings via bash" if is_img else ""
            lines.append(f"- `/challenge/distfiles/{name}`{suffix}")
        lines.append("")

    if meta.hints:
        lines.append("## Hints")
        for h in meta.hints:
            lines.append(f"- {h}")
        lines.append("")

    if hint:
        lines += ["## Operator Hint", hint, ""]

    lines += [
        "",
        "## Environment",
        "- You act through tools; `bash` runs inside an isolated sandbox container.",
        "- Files: `/challenge/distfiles/` (read-only). Scratch space: `/challenge/workspace/` (writable).",
        "- Other solver agents run in parallel — call `check_findings` to pull their insights.",
        "",
    ]

    tips = _category_tips(meta.category)
    if tips:
        lines.append(f"## Strategy ({meta.category})")
        lines += [f"- {t}" for t in tips]
        lines.append("")

    lines += [
        "## Method",
        "**Act now — use tools instead of describing what you would do.**",
        "",
        "1. **Enumerate first** — inspect files/service before theorizing (`file`, `strings`, `ls`).",
        "2. Form a hypothesis, then test it with one concrete tool call at a time; read each result.",
        "3. If a technique fails twice, **change approach** — do not repeat identical commands.",
        "4. Recover the flag, then verify it with `submit_flag` (it checks the pattern locally).",
        "5. Ignore placeholder flags like `CTF{flag}` or `FLAG{placeholder}`.",
        "6. Once you have the real flag, output `FLAG: <value>` on its own line.",
        "7. Always run commands through the `bash` tool, not any other execution mechanism —",
        "   that is the only path whose output and timing this system can see and bound.",
        "8. Any script that touches the network MUST set an explicit timeout on every blocking",
        "   call (e.g. pwntools `context.timeout=10` or per-call `timeout=`, socket `settimeout()`).",
        "   A remote that never responds must not hang your script forever.",
    ]

    return "\n".join(lines)
