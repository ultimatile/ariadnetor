#!/usr/bin/env python3
"""Audit: host-extension methods must delegate to a `*_with_backend` twin.

The `host_ops` extension traits are the ergonomic, host-defaulting face of
the call-site-backend surface: each method must be a thin delegation to its
generic `*_with_backend` twin, passing `Host::shared()`. A method that
instead reproduces kernel dispatch — building a descriptor inline, or
calling a crate-private `*_dense` kernel helper directly — would re-introduce
a host-baking path that bypasses the call-site-backend design.

This check enforces the positive invariant: every method body in the two
host-ext files must contain a call to its own twin — the `*_with_backend(`
form for most ops, or the unified bare name for the layout-dispatched ops
(see below). A descriptor-only negative check is insufficient
(a `svd_dense(Host::shared().as_ref(), ..)` bypass contains no descriptor
token), so the body must be inspected per method. Scope is honestly limited
to these two files; structurally obfuscated forms remain review territory.

The four decompositions (`svd` / `trunc_svd` / `qr` / `lq`) dispatch over
layout via `LinalgDecompose` (issue #299), and `contract` dispatches over
layout via `LinalgContract` (issue #372), so their host methods delegate to
the unified bare-name free fns (`svd(`, `contract(`, …) rather than a
`*_with_backend` twin. The generic free fn is the call-site-backend twin; only
its name differs, so the invariant is unchanged and the name pattern matches
the bare name (not `*_with_backend`) for this set alone.
"""

import re
import sys
from pathlib import Path

from audit_common import strip_noncode

REPO = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO / "crates/ariadnetor-linalg/src/host_ops/mod.rs",
    REPO / "crates/ariadnetor-linalg/src/host_ops/block_sparse.rs",
]


# Layout-dispatched ops delegate to the unified bare-name free fns, not a
# `*_with_backend` twin: the four decompositions via `LinalgDecompose`
# (issue #299) — `svd` / `trunc_svd` / `qr` / `lq` — and `contract` via
# `LinalgContract` (issue #372).
LAYOUT_DISPATCH_OPS = frozenset({"svd", "trunc_svd", "qr", "lq", "contract"})


def twin_pattern(name: str) -> re.Pattern:
    """The method's own twin: dense `<name>_with_backend` or block-sparse
    `<name>_block_sparse_with_backend`. The layout-dispatched ops
    ([`LAYOUT_DISPATCH_OPS`]) instead accept *only* the unified bare name
    `<name>(` — their `*_with_backend` forms were removed, so accepting them
    would bless a delegation to a symbol that no longer exists. Anchoring to the
    method name rejects a body that delegates to some *other* method's twin
    (e.g. `svd` calling `qr_with_backend`), which a generic `_with_backend`
    match would accept; the `\\b{name}\\s*\\(` boundary also rejects inline
    kernel dispatch such as `svd_dense(` (the `_` after the name blocks the
    bare-name match).
    """
    if name in LAYOUT_DISPATCH_OPS:
        return re.compile(rf"\b{re.escape(name)}\s*\(")
    return re.compile(rf"\b{re.escape(name)}(?:_block_sparse)?_with_backend\s*\(")


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
        src = strip_noncode(path.read_text())
        for name, body in method_bodies(src):
            if not twin_pattern(name).search(body):
                rel = path.relative_to(REPO)
                violations.append(f"{rel}: `{name}` does not delegate to its `*_with_backend` twin")

    if violations:
        print("Host-extension methods must delegate to their `*_with_backend` twin:")
        for v in violations:
            print(f"  {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
