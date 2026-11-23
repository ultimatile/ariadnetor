#!/usr/bin/env bash
# Builds every target of the published packages from the packages
# themselves, as crates.io will hold them. `cargo package` drops a
# dev-dependency that has a `path` and no `version`, while the packaged
# manifest still declares the targets that use it, and its own verification
# builds the lib and binaries only. A test, bench or example that needs a
# dropped dev-dependency would otherwise be published unbuildable.
# cargo-release runs this as its `pre-release-hook` (release.toml), once per
# crate before anything is committed or published; `cargo make package-check`
# runs it for every crate.
#
# The extracted packages are built against each other: one
# `[patch.crates-io]` entry per package makes every `ariadnetor-*`
# requirement, which names the workspace's one shared version as the
# extracted packages do, resolve to the extracted package rather than to
# crates.io. In an --execute run that version is not on crates.io yet; in a
# dry run it is the last published one, whose code is older than the
# workspace.
#
# Each package is built with default features and with --all-features (which
# needs the system ARPACK-NG library), by three commands:
#   cargo build --lib                  the build a dependent gets, without
#                                      dev-dependency features unified in
#   cargo build --all-targets          examples as runnable executables
#   cargo test --all-targets --no-run  the test harnesses, examples' included
# Doctests, which are not a build target, are not compiled.
#
# Environment, both set by cargo-release for a hook:
#   CRATE_NAME  check only this package; unset, check every one
#   DRY_RUN     `false` in an --execute run, where a failure also reports the
#               uncommitted version bump cargo-release has written by then
#
# Takes no arguments. Runs one at a time per checkout, as a release does: two
# concurrent runs in one checkout share the extraction directory.
set -eu

if [ $# -gt 0 ]; then
    echo "check-packaged-targets: takes no arguments" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
build_dir="$root/target/package-check"

report_bump() {
    [ "${DRY_RUN:-}" = false ] || return 0
    echo "check-packaged-targets: any version bump cargo-release wrote is left uncommitted in $root:"
    git -C "$root" status --short || true
    echo "check-packaged-targets: revert only the version bump, in the manifests and Cargo.lock; other edits listed above are not the release's"
}

die() {
    echo "check-packaged-targets: $*" >&2
    report_bump
    exit 1
}

# The archives go to a fresh temporary directory, so that the set is exactly
# this run's. They are extracted under target/package/, where cargo's
# workspace discovery stops, so that each extracted package is its own root,
# as it is when built from the registry. The directory is the same on every
# run and cleared first: cargo keys a path package's build artifacts on its
# path, so a new path per run would rebuild every package into a new slot of
# build_dir. `tar -m` stamps the files with the extraction time; cargo writes
# fixed, old mtimes into the archive entries, and restoring them would let a
# changed source look older than the artifacts built from the previous one.
# An explicit template, since macOS mktemp ignores TMPDIR without one.
work=$(mktemp -d "${TMPDIR:-/tmp}/check-packaged-targets.XXXXXX") ||
    die "cannot create a temporary directory"
trap 'rm -rf "$work" || true' EXIT
extract_dir="$root/target/package/packaged-targets"

(cd "$root" && cargo package --workspace --no-verify --allow-dirty --target-dir "$work/target") ||
    die "cargo package failed"

rm -rf "$extract_dir" || die "cannot clear $extract_dir"
mkdir -p "$extract_dir" || die "cannot create $extract_dir"
for archive in "$work"/target/package/*.crate; do
    [ -e "$archive" ] || die "cargo package produced no .crate archive"
    tar xmzf "$archive" -C "$extract_dir" || die "cannot extract $archive"
done

# The package name and version are the first `name =` and `version =` lines
# of the normalized manifest, the ones in its [package] table; the archive's
# `<name>-<version>` directory must match them.
dirs=()
names=()
for dir in "$extract_dir"/*/; do
    dir=${dir%/}
    name=$(sed -n 's/^name = "\(.*\)"$/\1/p' "$dir/Cargo.toml" | head -n 1)
    version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$dir/Cargo.toml" | head -n 1)
    [ "$(basename "$dir")" = "$name-$version" ] ||
        die "package '$name' version '$version' does not match $dir"
    dirs+=("$dir")
    names+=("$name")
done

patches=()
for i in "${!dirs[@]}"; do
    patches+=(--config "patch.crates-io.${names[$i]}.path=\"${dirs[$i]}\"")
done

found=false
failed=false

# run_step <dir> <crate> <config> <cargo args...>
run_step() {
    local dir=$1 crate=$2 config=$3
    shift 3
    if (cd "$dir" && cargo "$@" --quiet --target-dir "$build_dir" "${patches[@]}"); then
        return 0
    fi
    echo "FAIL $crate $config: cargo $*"
    failed=true
    return 1
}

for i in "${!dirs[@]}"; do
    dir=${dirs[$i]}
    crate=${names[$i]}
    if [ -n "${CRATE_NAME:-}" ] && [ "$crate" != "$CRATE_NAME" ]; then
        continue
    fi
    found=true
    for config in default all-features; do
        if [ "$config" = all-features ]; then
            features=--all-features
        else
            features=--features=
        fi
        if run_step "$dir" "$crate" "$config" build --lib "$features" &&
            run_step "$dir" "$crate" "$config" build --all-targets "$features" &&
            run_step "$dir" "$crate" "$config" test --all-targets --no-run "$features"; then
            echo "ok $crate $config"
        fi
    done
done

if [ "$found" = false ]; then
    die "no published package is named '${CRATE_NAME:-}'"
fi
if [ "$failed" = true ]; then
    report_bump
    exit 1
fi
