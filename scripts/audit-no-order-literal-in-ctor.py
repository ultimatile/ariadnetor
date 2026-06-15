#!/usr/bin/env python3
"""Audit: no hard-coded memory-order literal in tensor constructors.

Host-convenience constructors must take their `MemoryOrder` from the
substrate via `host_order()` / `preferred_order()`, never a hard-coded
`MemoryOrder::ColumnMajor` / `RowMajor` literal. Keeping that indirection
intact is what lets the `Host` substrate be repointed in one line; a
constructor that bakes the order in would silently defeat the swap (and,
being column-major, would also slip past the deliberately column-major
pluggability litmus).

Scope is `ariadnetor-tensor` source — where the host constructors live.
Legitimate non-constructor literals are excluded:
  - order-dispatch match arms (`MemoryOrder::X =>`);
  - `reordered(MemoryOrder::X)` reinterpretation calls;
  - test code (`tests/` dirs, `tests.rs` / `test_util.rs` files, and inline
    `#[cfg(test)]` modules).
The linalg faer-interop scratch buffers legitimately pin an explicit order
and live outside this scope. This is a line-level heuristic for accidental
regressions; obfuscated forms remain review territory.
"""

import re
import sys
from pathlib import Path

from audit_common import strip_cfg_test_mods, strip_comments

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "crates/ariadnetor-tensor/src"
EXCLUDED_NAMES = {"tests.rs", "test_util.rs"}

LITERAL = re.compile(r"MemoryOrder::(?:ColumnMajor|RowMajor)")
DISPATCH_ARM = re.compile(r"\s*=>")
# A `reordered(...)` argument, allowing a qualified path before the literal,
# e.g. `self.reordered(arnet_core::backend::MemoryOrder::RowMajor)`. The 96-char
# look-back below comfortably spans such a prefix.
REORDERED_ARG = re.compile(r"\.reordered\(\s*(?:\w+\s*::\s*)*$")


def main() -> int:
    if not SRC.exists():
        print(f"audit-no-order-literal-in-ctor: missing {SRC}", file=sys.stderr)
        return 2

    violations = []
    for path in sorted(SRC.rglob("*.rs")):
        if path.name in EXCLUDED_NAMES or "tests" in path.parts:
            continue
        raw = path.read_text()
        src = strip_cfg_test_mods(strip_comments(raw))
        for m in LITERAL.finditer(src):
            after = src[m.end() : m.end() + 8]
            if DISPATCH_ARM.match(after):
                continue
            before = src[max(0, m.start() - 96) : m.start()]
            if REORDERED_ARG.search(before):
                continue
            line = src.count("\n", 0, m.start()) + 1
            rel = path.relative_to(REPO)
            violations.append(f"{rel}:~{line}: hard-coded {m.group(0)} in a constructor path")

    if violations:
        print("Constructors must take memory order from host_order()/preferred_order():")
        for v in violations:
            print(f"  {v}")
        print("(line numbers are approximate: comments and #[cfg(test)] blocks are stripped first)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
