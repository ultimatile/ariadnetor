#!/bin/sh
# Contract tests for check-packaged-targets.sh, against a stand-in `cargo`
# that fakes packaging and records builds instead of compiling anything.
#
# The script under test runs from a copy inside a fixture workspace, since it
# finds the workspace root from its own location. The stand-in reads, from
# $FAKE:
#   packages      the package names `cargo package` archives, one per line
#   package-exit  if present, `cargo package`'s exit status
#   fail-<crate>  if present, a build of that crate whose arguments contain
#                 the file's content fails (an empty file fails every build)
# and appends `<crate> <args>` to $FAKE/calls for every build.
set -eu

unset GIT_DIR GIT_INDEX_FILE GIT_WORK_TREE GIT_PREFIX GIT_OBJECT_DIRECTORY \
    GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG_PARAMETERS GIT_CONFIG_COUNT
GIT_CONFIG_GLOBAL=/dev/null
GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL GIT_CONFIG_NOSYSTEM

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
FAKE="$work/fake"
export FAKE
mkdir -p "$FAKE/bin"

cat > "$FAKE/bin/cargo" <<'EOF'
#!/bin/sh
set -eu
case "$1" in
package)
    target=""
    while [ $# -gt 0 ]; do
        [ "$1" = --target-dir ] && target="$2"
        shift
    done
    if [ -e "$FAKE/package-exit" ]; then
        exit "$(cat "$FAKE/package-exit")"
    fi
    mkdir -p "$target/package" "$target/src"
    while read -r name; do
        mkdir -p "$target/src/$name-0.1.0"
        printf '[package]\nname = "%s"\nversion = "0.1.0"\n' "$name" \
            > "$target/src/$name-0.1.0/Cargo.toml"
        tar czf "$target/package/$name-0.1.0.crate" -C "$target/src" "$name-0.1.0"
    done < "$FAKE/packages"
    ;;
build | test)
    crate=$(basename "$PWD")
    crate=${crate%-*}
    echo "$crate $*" >> "$FAKE/calls"
    if [ -e "$FAKE/fail-$crate" ]; then
        pattern=$(cat "$FAKE/fail-$crate")
        case " $* " in *"$pattern"*) exit 101 ;; esac
    fi
    ;;
*)
    echo "unexpected: cargo $*" >&2
    exit 99
    ;;
esac
EOF
chmod +x "$FAKE/bin/cargo"
PATH="$FAKE/bin:$PATH"
export PATH

failures=0
fail() {
    echo "FAIL $name: $*"
    failures=$((failures + 1))
}

# Fresh fixture workspace and stand-in state for one test.
setup() {
    name=$1
    rm -rf "$work/ws" "$FAKE/calls" "$FAKE"/package-exit "$FAKE"/fail-*
    mkdir -p "$work/ws/scripts"
    cp "$here/check-packaged-targets.sh" "$work/ws/scripts/"
    echo '[toolchain]' > "$work/ws/rust-toolchain.toml"
    printf 'alpha\nbeta\n' > "$FAKE/packages"
    : > "$FAKE/calls"
}

# run [VAR=value ...]: run the script with those variables set, setting
# $out and $status. run_args <args...>: run it with those arguments.
run() {
    set +e
    out=$(cd "$work" && env "$@" "$work/ws/scripts/check-packaged-targets.sh" 2>&1)
    status=$?
    set -e
}
run_args() {
    set +e
    out=$(cd "$work" && "$work/ws/scripts/check-packaged-targets.sh" "$@" 2>&1)
    status=$?
    set -e
}

expect_status() { [ "$status" = "$1" ] || fail "exit $status, expected $1; output: $out"; }
expect_out() { printf '%s\n' "$out" | grep -qF -- "$1" || fail "missing '$1' in: $out"; }
expect_no_out() { ! printf '%s\n' "$out" | grep -qF -- "$1" || fail "unexpected '$1' in: $out"; }
calls_matching() { grep -cF -- "$1" "$FAKE/calls" || true; }

setup every-package-passes
run
expect_status 0
for crate in alpha beta; do
    for config in default all-features; do expect_out "ok $crate $config"; done
done
[ "$(wc -l < "$FAKE/calls")" -eq 12 ] || fail "expected 12 builds: $(cat "$FAKE/calls")"
for cmd in "build --lib" "build --all-targets" "test --all-targets --no-run"; do
    [ "$(calls_matching "$cmd")" -eq 4 ] || fail "expected 4 '$cmd' builds"
done
[ "$(calls_matching "--all-features")" -eq 6 ] || fail "expected 6 all-features builds"
# Every build sees every package's patch.
[ "$(calls_matching "patch.crates-io.alpha.path=")" -eq 12 ] || fail "alpha patch missing"
[ "$(calls_matching "patch.crates-io.beta.path=")" -eq 12 ] || fail "beta patch missing"

setup refuses-arguments
run_args --foo
expect_status 2
expect_out "takes no arguments"
[ ! -s "$FAKE/calls" ] || fail "built despite refusing"

setup crate-name-selects-one-package
run CRATE_NAME=beta
expect_status 0
expect_out "ok beta default"
expect_no_out "ok alpha"
[ "$(grep -c '^alpha ' "$FAKE/calls" || true)" -eq 0 ] || fail "built alpha"

setup unknown-crate-name-fails
run CRATE_NAME=gamma
expect_status 1
expect_out "no published package is named 'gamma'"

setup failing-build-is-reported
echo "--all-features" > "$FAKE/fail-alpha"
run
expect_status 1
expect_out "FAIL alpha all-features: cargo build --lib --all-features"
expect_out "ok alpha default"
expect_out "ok beta all-features"
expect_no_out "ok alpha all-features"
# The configuration stops at its first failing build.
[ "$(grep -c '^alpha .*--all-features' "$FAKE/calls")" -eq 1 ] || fail "alpha all-features kept building"
expect_no_out "left uncommitted"

setup execute-run-failure-reports-the-bump
git -C "$work/ws" init -q
echo "--features=" > "$FAKE/fail-beta"
run DRY_RUN=false
expect_status 1
expect_out "FAIL beta default"
expect_out "left uncommitted in $work/ws"
expect_out "git -C $work/ws restore ."

setup dry-run-failure-does-not-report-the-bump
echo "" > "$FAKE/fail-beta"
run DRY_RUN=true
expect_status 1
expect_no_out "left uncommitted"

setup package-failure-fails
echo 101 > "$FAKE/package-exit"
run DRY_RUN=false
expect_status 1
expect_out "cargo package failed"
expect_out "left uncommitted"

setup no-archive-fails
: > "$FAKE/packages"
run
expect_status 1
expect_out "produced no .crate archive"

if [ "$failures" -gt 0 ]; then
    echo "$failures check-packaged-targets contract checks failed"
    exit 1
fi
echo "check-packaged-targets contract tests passed"
