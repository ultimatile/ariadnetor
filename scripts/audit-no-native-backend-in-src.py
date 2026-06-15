#!/usr/bin/env python3
"""Audit: no concrete `NativeBackend` in mps / algorithms / linalg source.

Signatures and host paths must name the substrate through the `Host` alias
or a generic `OpsFor<…>` bound, never the concrete `NativeBackend` type.
A bare `NativeBackend` in a signature, or a `NativeBackend::shared()` value
in a body, pins code to the native backend and defeats the one-line `Host`
swap the pluggability litmus relies on.

Scope is the source of the three consumer crates. The `Host = NativeBackend`
alias definition (and the litmus stub) live in `ariadnetor-tensor`, which is
out of scope here. Test code is excluded — `RecordingBackend` and the
numeric-agreement tests legitimately instantiate `NativeBackend`:
  - any path under a `tests/` directory;
  - files named `tests.rs` or `test_util.rs`;
  - `#[cfg(test)]` modules and comments.
This is a token-level heuristic for accidental regressions.
"""

import re
import sys
from pathlib import Path

from audit_common import strip_cfg_test_mods, strip_comments

REPO = Path(__file__).resolve().parent.parent
SRC_DIRS = [
    REPO / "crates/ariadnetor-mps/src",
    REPO / "crates/ariadnetor-algorithms/src",
    REPO / "crates/ariadnetor-linalg/src",
]
EXCLUDED_NAMES = {"tests.rs", "test_util.rs"}
TOKEN = re.compile(r"\bNativeBackend\b")


def excluded(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or "tests" in path.parts


def main() -> int:
    violations = []
    for src_dir in SRC_DIRS:
        if not src_dir.exists():
            print(f"audit-no-native-backend-in-src: missing {src_dir}", file=sys.stderr)
            return 2
        for path in sorted(src_dir.rglob("*.rs")):
            if excluded(path):
                continue
            src = strip_cfg_test_mods(strip_comments(path.read_text()))
            for m in TOKEN.finditer(src):
                line = src.count("\n", 0, m.start()) + 1
                rel = path.relative_to(REPO)
                violations.append(f"{rel}:~{line}: concrete NativeBackend (use the Host alias or OpsFor<…>)")

    if violations:
        print("Source must name the substrate via the Host alias or a generic OpsFor<…> bound:")
        for v in violations:
            print(f"  {v}")
        print("(line numbers are approximate: comments and #[cfg(test)] blocks are stripped first)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
