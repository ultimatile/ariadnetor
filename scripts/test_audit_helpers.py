#!/usr/bin/env python3
"""Contract tests for the pre-commit source-audit helpers.

These lock the robustness contracts that review (PR #349) found broken in the
first cut of the audit tooling. Each assertion fails on the original buggy
implementation:

- absolute-path test exclusion silently disabled every audit when the checkout
  itself sat under a directory named "tests";
- the host-twin audit accepted delegation to an unrelated method's twin;
- naive regex comment-stripping mis-ate `//` inside strings and let braces in
  string / char literals corrupt the `#[cfg(test)]` brace matching;
- the cfg(test) matcher missed visibility-qualified / raw-identifier modules.

Runner-free: plain asserts, non-zero exit on failure. Wired as a pre-commit
hook on the audit scripts so a regression in the (now non-trivial) scanner is
caught locally rather than silently weakening the audits.
"""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import audit_common as ac  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_line_comment_not_eaten_inside_string():
    out = ac.strip_noncode('let u = "a//b"; let x = NativeBackend;\n')
    assert "NativeBackend" in out, "// inside a string must not consume the rest of the line"


def test_braces_in_literals_do_not_unbalance_cfg_test_match():
    src = '#[cfg(test)]\nmod tests { let s = "}"; let c = ' "'{'" "; fn f(){ let _ = NativeBackend; } }\nlet real = NativeBackend;\n"
    out = ac.strip_cfg_test_mods(ac.strip_noncode(src))
    assert out.count("NativeBackend") == 1, "cfg(test) mod with brace literals must be fully stripped"
    assert "let real = NativeBackend" in out, "real code after the test mod must survive"


def test_lifetimes_preserved():
    out = ac.strip_noncode("fn f<'a>(x: &'a str) -> &'a str { x }\n")
    assert "'a" in out, "lifetimes must not be treated as char literals"


def test_nested_block_comment_removed():
    out = ac.strip_noncode("let a=0; /* outer /* inner */ still */ let b=NativeBackend;\n")
    assert "NativeBackend" in out and "still" not in out, "nested block comments must be fully removed"


def test_cfg_test_mod_visibility_and_raw_ident():
    for decl in ("pub mod tests", "pub(crate) mod tests", "mod r#tests"):
        src = f"#[cfg(test)]\n{decl} {{ let _ = NativeBackend; }}\nlet real = 1;\n"
        out = ac.strip_cfg_test_mods(ac.strip_noncode(src))
        assert "NativeBackend" not in out, f"cfg(test) `{decl}` must be stripped"


def test_twin_pattern_requires_own_twin():
    twin = _load("audit-host-twin-delegation")
    # dense and block-sparse twins of the SAME method are accepted
    assert twin.twin_pattern("svd").search("svd_with_backend(b, x)")
    assert twin.twin_pattern("svd").search("svd_block_sparse_with_backend(b, x)")
    # delegation to a DIFFERENT method's twin is rejected
    assert not twin.twin_pattern("svd").search("qr_with_backend(b, x)")
    # prefix overlap must not leak (eig must not accept eigvals' twin)
    assert not twin.twin_pattern("eig").search("eigvals_with_backend(b, x)")


def test_exclusion_is_relative_not_absolute():
    anb = _load("audit-no-native-backend-in-src")
    src_dir = Path("/x/tests/repo/crates/ariadnetor-mps/src")
    prod = src_dir / "foo.rs"  # ancestor dir is named "tests"
    genuine = src_dir / "bar" / "tests" / "mod.rs"
    assert not anb.excluded(prod, src_dir), "an ancestor 'tests' dir must not disable the audit"
    assert anb.excluded(genuine, src_dir), "a genuine tests/ dir under the root must be excluded"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        print(f"{failures}/{len(tests)} audit-helper contract tests failed")
        return 1
    print(f"{len(tests)} audit-helper contract tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
