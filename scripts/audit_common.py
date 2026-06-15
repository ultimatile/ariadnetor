"""Shared helpers for the pre-commit source audits.

The audit scripts are line/token heuristics over Rust source; before scanning
they neutralize the surfaces that would otherwise produce false matches or
corrupt brace counting — comments, string / char literal contents, and
`#[cfg(test)]` modules. These helpers are shared so the behavior stays
identical across audits and is fixed in one place.

[`strip_noncode`] blanks comment and string / char-literal *content* (keeping
newlines so line numbers stay stable, and keeping structural braces that live
in real code) in a single left-to-right pass, so a `//` inside a string or a
`{` inside a `"{"` literal no longer derails the downstream scans. Raw strings
with hashes (`r#"..."#`) are the one residual: their inner `"` can close the
scan early, an obfuscated form the audits' honest scope leaves to review.
"""

import re

_CFG_TEST_ATTR = re.compile(r"#\[cfg\(test\)\]")
# `#[cfg(test)] mod name {` — allow an optional visibility (`pub`, `pub(crate)`)
# and a raw identifier (`r#name`) before the brace.
_MOD_OPEN = re.compile(r"\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(?:r#)?\w+\s*\{")


def _blank(span: str) -> str:
    """Replace a span with spaces, preserving newlines for line numbering."""
    return "".join(c if c == "\n" else " " for c in span)


def strip_noncode(src: str) -> str:
    """Blank comment and string / char-literal content; keep code structure.

    Handles line comments, nested block comments, double-quoted strings (incl.
    `r"..."` without hashes, since the delimiters coincide), and char literals
    distinguished from lifetimes. `#[cfg(test)] mod` excision runs separately
    via [`strip_cfg_test_mods`] on the output of this pass.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(_blank(src[i:j]))
            i = j
        elif c == "/" and nxt == "*":
            depth, j = 0, i
            while j < n:
                if src[j] == "/" and j + 1 < n and src[j + 1] == "*":
                    depth += 1
                    j += 2
                elif src[j] == "*" and j + 1 < n and src[j + 1] == "/":
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            out.append(_blank(src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(_blank(src[i:j]))
            i = j
        elif c == "'":
            # Char literal vs lifetime: `'\n'` / `'x'` are literals (blank their
            # content so an enclosed `{`/`"` cannot leak); `'a` is a lifetime.
            if nxt == "\\":
                j = src.find("'", i + 2)
                j = n if j == -1 else j + 1
                out.append(_blank(src[i:j]))
                i = j
            elif i + 2 < n and src[i + 2] == "'":
                out.append(_blank(src[i : i + 3]))
                i += 3
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


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

    Run on the output of [`strip_noncode`] so braces inside string / char
    literals cannot unbalance the match. A bare `#[cfg(test)]` on something
    other than a `mod` only has its attribute dropped; the item itself stays.
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
