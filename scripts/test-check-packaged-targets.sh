#!/bin/sh
# Contract tests for check-packaged-targets.sh, against a stand-in `cargo`
# that fakes packaging and records builds instead of compiling anything.
#
# The script under test runs from a copy inside a fixture workspace, since it
# finds the workspace root from its own location. The stand-in reads, from
# $FAKE:
#   packages      the package names `cargo package` archives, one per line
#   package-exit  if present, `cargo package`'s exit status
#   nameless      if present, the archived manifests carry no `name =` line
#   misnamed      if present, each archived manifest names its package
#                 without the last letter
#   corrupt       if present, `cargo package` also writes an archive that is
#                 not a tarball
#   fail-<crate>  if present, a build of that crate whose arguments contain
#                 the file's content fails (an empty file fails every build)
# and appends `<crate> <args>` to $FAKE/calls for every build, and writes the
# arguments of `cargo package` to $FAKE/package-args.
set -eu

unset GIT_DIR GIT_INDEX_FILE GIT_WORK_TREE GIT_PREFIX GIT_OBJECT_DIRECTORY \
    GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG_PARAMETERS GIT_CONFIG_COUNT
GIT_CONFIG_GLOBAL=/dev/null
GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL GIT_CONFIG_NOSYSTEM

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# The script's mktemp honours TMPDIR; keep that inside this test's directory.
TMPDIR="$work/tmp"
export TMPDIR
mkdir "$TMPDIR"
FAKE="$work/fake"
export FAKE
mkdir -p "$FAKE/bin"

cat > "$FAKE/bin/cargo" <<'EOF'
#!/bin/sh
set -eu
case "$1" in
package)
    echo "$*" > "$FAKE/package-args"
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
        if [ -e "$FAKE/nameless" ]; then
            printf '[package]\nversion = "0.1.0"\n' > "$target/src/$name-0.1.0/Cargo.toml"
        elif [ -e "$FAKE/misnamed" ]; then
            # A prefix of the directory's name, which a prefix match accepts.
            printf '[package]\nname = "%s"\nversion = "0.1.0"\n' "${name%?}" \
                > "$target/src/$name-0.1.0/Cargo.toml"
        else
            # Shaped like a normalized manifest: the target and dependency
            # tables after [package] carry `name =` / `version =` lines of
            # their own.
            printf '[package]\nname = "%s"\nversion = "0.1.0"\n\n[lib]\nname = "%s_lib"\n\n[[test]]\nname = "%s_test"\n\n[dependencies.dep]\nversion = "9.9.9"\n' \
                "$name" "$name" "$name" > "$target/src/$name-0.1.0/Cargo.toml"
        fi
        touch -t 200001010000 "$target/src/$name-0.1.0/Cargo.toml"
        tar czf "$target/package/$name-0.1.0.crate" -C "$target/src" "$name-0.1.0"
    done < "$FAKE/packages"
    if [ -e "$FAKE/corrupt" ]; then
        echo "not an archive" > "$target/package/corrupt-0.1.0.crate"
    fi
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
    rm -rf "$work/ws" "$FAKE/calls" "$FAKE/package-args" "$FAKE"/package-exit "$FAKE"/fail-* \
        "$FAKE/nameless" "$FAKE/misnamed" "$FAKE/corrupt"
    mkdir -p "$work/ws/scripts"
    cp "$here/check-packaged-targets.sh" "$work/ws/scripts/"
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
# count_calls <regex>: how many recorded builds match the basic regex.
count_calls() { grep -c -- "$1" "$FAKE/calls" || true; }
# expect_calls <n> <regex> <message>: exactly n recorded builds match.
expect_calls() { [ "$(count_calls "$2")" -eq "$1" ] || fail "$3"; }
# expect_fixed_calls <n> <string> <message>: the same, for a fixed string.
expect_fixed_calls() { [ "$(grep -cF -- "$2" "$FAKE/calls" || true)" -eq "$1" ] || fail "$3"; }
expect_no_builds() { [ ! -s "$FAKE/calls" ] || fail "$1"; }
# Only the script's own temporary directories: other tools (Apple's git) may
# write to TMPDIR too.
expect_tmp_clean() {
    left=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -name 'check-packaged-targets.*')
    [ -z "$left" ] || fail "temporary directory left behind: $left"
}
extracted="$work/ws/target/package/packaged-targets"
touch -t 200701010000 "$work/older-than-extraction"

setup every-package-passes
run
expect_status 0
for crate in alpha beta; do
    for config in default all-features; do expect_out "ok $crate $config"; done
done
[ "$(wc -l < "$FAKE/calls")" -eq 12 ] || fail "expected 12 builds: $(cat "$FAKE/calls")"
for cmd in "build --lib" "build --all-targets" "test --all-targets --no-run"; do
    expect_calls 4 "$cmd" "expected 4 '$cmd' builds"
done
expect_calls 6 "--all-features" "expected 6 all-features builds"
expect_fixed_calls 12 "--target-dir $work/ws/target/package-check" "builds outside the stable build dir"
# Packaging covers the whole workspace, skips cargo's lib-only verification,
# and accepts the uncommitted version bump of an --execute run.
for flag in --workspace --no-verify --allow-dirty; do
    grep -qF -- "$flag" "$FAKE/package-args" || fail "cargo package without $flag"
done
# The temporary directory holding the archives is removed on exit.
expect_tmp_clean
# Every build sees every package's patch, pointing at that package in the
# extraction directory under target/package/.
for crate in alpha beta; do
    expect_fixed_calls 12 "patch.crates-io.$crate.path=\"$extracted/$crate-0.1.0\"" \
        "$crate patch missing or misdirected"
done
# The extracted files carry the extraction time, not the archive's mtime.
[ "$extracted/alpha-0.1.0/Cargo.toml" -nt "$work/older-than-extraction" ] ||
    fail "extracted files kept the archive's mtime"

setup refuses-arguments
run_args --foo
expect_status 2
expect_out "takes no arguments"
expect_no_builds "built despite refusing"

# A second run reuses the extraction directory and clears what an earlier run
# left there.
setup rerun-clears-the-extraction-directory
run
expect_status 0
mkdir "$extracted/stale-0.1.0"
printf '[package]\nname = "stale"\nversion = "0.1.0"\n' > "$extracted/stale-0.1.0/Cargo.toml"
: > "$FAKE/calls"
run
expect_status 0
expect_calls 0 '^stale ' "built a package left from an earlier run"
expect_no_out "ok stale"

setup crate-name-selects-one-package
run CRATE_NAME=beta
expect_status 0
expect_out "ok beta default"
expect_no_out "ok alpha"
expect_calls 0 '^alpha ' "built alpha"

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
expect_calls 1 '^alpha .*--all-features' "alpha all-features kept building"
expect_no_out "left uncommitted"

# The case the check exists for: the lib builds, a test or example does not.
setup failing-target-build-is-reported
echo "test --all-targets --no-run --features=" > "$FAKE/fail-beta"
run
expect_status 1
expect_out "FAIL beta default: cargo test --all-targets --no-run --features="
expect_tmp_clean
expect_out "ok beta all-features"
expect_no_out "ok beta default"
expect_calls 1 '^beta build --lib --features=' "beta lib not built"

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
expect_tmp_clean

setup no-archive-fails
: > "$FAKE/packages"
run
expect_status 1
expect_out "produced no .crate archive"

setup corrupt-archive-fails
touch "$FAKE/corrupt"
run
expect_status 1
expect_out "cannot extract"
expect_no_builds "built despite the failed extraction"

setup uncreatable-extraction-directory-fails
mkdir -p "$work/ws/target"
: > "$work/ws/target/package"
run
expect_status 1
expect_out "cannot create $extracted"
expect_no_builds "built despite the missing extraction directory"

setup unclearable-extraction-directory-fails
mkdir -p "$extracted/left"
chmod a-w "$extracted"
run
chmod u+w "$extracted"
expect_status 1
expect_out "cannot clear $extracted"
expect_no_builds "built despite the uncleared extraction directory"

setup nameless-manifest-fails
touch "$FAKE/nameless"
run
expect_status 1
expect_out "package '' version '0.1.0' does not match"

setup missing-temporary-directory-fails
run TMPDIR="$work/absent"
expect_status 1
expect_out "cannot create a temporary directory"
expect_no_builds "built without a temporary directory"

setup misnamed-manifest-fails
touch "$FAKE/misnamed"
run
expect_status 1
expect_out "package 'alph' version '0.1.0' does not match"
expect_tmp_clean
expect_no_builds "built despite the mismatch"

if [ "$failures" -gt 0 ]; then
    echo "$failures check-packaged-targets contract checks failed"
    exit 1
fi
echo "check-packaged-targets contract tests passed"
