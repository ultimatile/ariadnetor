#!/usr/bin/env python3
"""Audit: host-extension methods must delegate to a `*_with_backend` twin.

The `host_ops` extension traits are the ergonomic, host-defaulting face of
the call-site-backend surface: each method must be a thin delegation to its
generic `*_with_backend` twin, passing `Host::shared()`. A method that
instead reproduces kernel dispatch — building a descriptor inline, or
calling a crate-private `*_dense` kernel helper directly — would re-introduce
a host-baking path that bypasses the call-site-backend design.

This check enforces the positive invariant: every method body in the two
host-ext files must contain a `*_with_backend(` call. A descriptor-only
negative check is insufficient (a `svd_dense(Host::shared().as_ref(), ..)`
bypass contains no descriptor token), so the body must be inspected per
method. Scope is honestly limited to these two files; structurally
obfuscated forms remain review territory.
"""

import re
import sys
from pathlib import Path

from audit_common import strip_comments

REPO = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO / "crates/ariadnetor-linalg/src/host_ops/mod.rs",
    REPO / "crates/ariadnetor-linalg/src/host_ops/block_sparse.rs",
]

TWIN_CALL = re.compile(r"\w+_with_backend\s*\(")


def method_bodies(src: str):
    """Yield (name, body) for every `fn name(...) { ... }` with a real body.

    Abstract trait methods (`fn name(...) -> T;`) have no body and are
    skipped: after the `fn` token the first top-level `;` precedes any `{`.
    """
    for m in re.finditer(r"\bfn\s+(\w+)", src):
        j = m.end()
        while j < len(src) and src[j] not in "{;":
            j += 1
        if j >= len(src) or src[j] == ";":
            continue
        depth = 0
        k = j
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        yield m.group(1), src[j : k + 1]


def main() -> int:
    violations = []
    for path in TARGETS:
        if not path.exists():
            print(f"audit-host-twin-delegation: missing target {path}", file=sys.stderr)
            return 2
        src = strip_comments(path.read_text())
        for name, body in method_bodies(src):
            if not TWIN_CALL.search(body):
                rel = path.relative_to(REPO)
                violations.append(f"{rel}: `{name}` does not delegate to a `*_with_backend` twin")

    if violations:
        print("Host-extension methods must delegate to their `*_with_backend` twin:")
        for v in violations:
            print(f"  {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
