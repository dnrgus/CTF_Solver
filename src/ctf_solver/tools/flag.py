"""Flag pattern matching — extract and validate CTF flags from text."""

from __future__ import annotations

import re

# The lookbehind keeps a prefix glued to the word ("xflag{...}", "myCTF{...}")
# from being read as a flag. Competition formats that differ (e.g. POCTF{...})
# must be passed explicitly via --flag-pattern or challenge.yml.
DEFAULT_FLAG_PATTERN = r"(?<![A-Za-z0-9_])([Ff][Ll][Aa][Gg]|[Cc][Tt][Ff])\{([^}]+)\}"


def extract_flags(text: str, pattern: str = DEFAULT_FLAG_PATTERN) -> list[str]:
    """Extract unique flags from text using regex pattern. Preserves order."""
    seen: set[str] = set()
    results: list[str] = []
    for match in re.finditer(pattern, text):
        flag = match.group(0)
        if flag not in seen:
            seen.add(flag)
            results.append(flag)
    return results


def is_exact_flag(candidate: str, pattern: str = DEFAULT_FLAG_PATTERN) -> bool:
    """True only when the whole candidate is one flag that matches the pattern.

    Substring matches are not enough: "junk FLAG{x} junk" contains a flag but
    is not a flag, and recording it would make the verifier judge the wrong string.
    """
    return re.fullmatch(pattern, candidate) is not None
