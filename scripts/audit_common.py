"""Shared helpers for the pre-commit source audits.

The audit scripts are line/token heuristics over Rust source; before scanning
they strip the surfaces that would otherwise produce false matches — comments
and `#[cfg(test)]` modules. These two helpers are shared so the stripping
behavior stays identical across audits and is fixed in one place.

String literals are deliberately NOT stripped: doing so robustly would need a
raw-string-aware lexer, and a literal enum token inside a string is an
obfuscated form that the audits' honest scope leaves to review.
"""

import re

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_CFG_TEST_ATTR = re.compile(r"#\[cfg\(test\)\]")
_MOD_OPEN = re.compile(r"\s*mod\s+\w+\s*\{")


def strip_comments(src: str) -> str:
    """Remove block and line comments."""
    src = _BLOCK_COMMENT.sub("", src)
    src = _LINE_COMMENT.sub("", src)
    return src


def _match_block(src: str, brace_start: int) -> int:
    """Return the index just past the `}` matching the `{` at brace_start."""
    depth = 0
    k = brace_start
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return k


def strip_cfg_test_mods(src: str) -> str:
    """Remove `#[cfg(test)] mod name { ... }` blocks (brace-matched).

    A bare `#[cfg(test)]` on something other than a `mod` only has its
    attribute dropped; the item itself stays (the audits do not need to
    excise inline test fns, only whole test modules).
    """
    out = []
    i = 0
    while i < len(src):
        m = _CFG_TEST_ATTR.search(src, i)
        if not m:
            out.append(src[i:])
            break
        out.append(src[i : m.start()])
        mm = _MOD_OPEN.match(src, m.end())
        if mm:
            i = _match_block(src, mm.end() - 1)
        else:
            i = m.end()
    return "".join(out)
