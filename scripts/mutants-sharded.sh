#!/bin/sh
# The shipped mutation pass as MUTANTS_SHARDS consecutive `cargo mutants
# --shard k/n` runs, resumable across invocations (`cargo make
# mutants-sharded`). Makefile.toml says when to prefer this over the
# single-run pass.
#
# `--sharding round-robin` is REQUIRED for a faithful result. cargo-mutants'
# baseline tests only the packages that own the shard's mutants, and the test
# timeout is derived from that baseline, while test_workspace = true runs the
# whole workspace against every mutant. The default `slice` sharding hands
# most shards a single package, whose baseline is far shorter than the
# workspace suite, so a large share of the mutants there is reported as a
# timeout. Round-robin spreads each package over the shards.
#
# Refused, each before a shard runs:
#   - any argument: cargo-make would otherwise drop a forwarded flag in
#     silence, and a flag that reached a resumed pass would differ from the
#     one its finished shards ran with;
#   - a MUTANTS_SHARDS that is not a positive integer of at most six digits
#     written without a leading zero;
#   - a mutant listing that names no package at all;
#   - an n so large that some shard holds no mutant of some package that
#     has mutants, which is where the round-robin argument above stops
#     holding;
#   - resuming once the tree or its mutant list differs from the one the
#     pass started from: shards divide that list by position, and a shard
#     run against other code (an edited test, say) would not be comparable.
#     The tree is compared by HEAD and `git diff HEAD`, so untracked files
#     are not covered.
#
# State lives in mutants-shards/<n>/ at the workspace root — outside target/,
# like `cargo make mutants`' mutants.out, so `cargo clean` does not take days
# of results with it:
#   <k>/mutants.out/  shard k's cargo-mutants output
#   <k>/complete      shard k ran to a verdict; skipped on the next invocation
#   pass.id           a checksum of the tree and mutant list the pass
#                     started from
#   {caught,missed,timeout,unviable}.txt  the merged lists, once all are done
# The markers and pass.id are removed after the merge, so they span one
# pass only; the merged lists stay until the next pass starts.
# cargo-mutants leaves every directory named mutants.out out of its scratch
# copies, so of the state above only the small files are copied along.
#
# Exit status: 1 for a refusal; a shard's own status when it stops without
# a verdict (that shard is redone on the next invocation); the failing
# command's status when anything else fails under `set -e` (a listing, git,
# the merge). Once every shard has a verdict, it follows cargo-mutants over
# the merged lists: 3 if any mutant timed out, else 2 if any was missed,
# else 0.
set -eu

if [ $# -gt 0 ]; then
    echo "mutants-sharded takes no arguments, got: $*" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

n="${MUTANTS_SHARDS-40}"
case "${n}" in
    '' | *[!0-9]* | 0* | ???????*)
        echo "MUTANTS_SHARDS must be a positive integer of at most six digits, no leading zero, got '${n}'" >&2
        exit 1
        ;;
esac

# Each listing is captured on its own first, so that a failing cargo stops
# the script under `set -e` instead of reaching `packages` as empty input.
packages() {
    printf '%s\n' "$1" | grep '^    "package": ' | sort -u
}
listing=$(cargo mutants --list --json)
all=$(packages "${listing}")
if [ -z "${all}" ]; then
    echo "found no package in \`cargo mutants --list --json\`" >&2
    exit 1
fi
k=0
while [ "${k}" -lt "${n}" ]; do
    listing=$(cargo mutants --list --json --shard "${k}/${n}" --sharding round-robin)
    held=$(packages "${listing}")
    if [ "${held}" != "${all}" ]; then
        echo "shard ${k}/${n} does not hold a mutant of every mutated package, so its baseline would not cover them; lower MUTANTS_SHARDS" >&2
        exit 1
    fi
    k=$((k + 1))
done

root="mutants-shards/${n}"
head=$(git rev-parse HEAD)
changes=$(git diff HEAD)
mutant_list=$(cargo mutants --list)
id=$(printf '%s\n' "${head}" "${changes}" "${mutant_list}" | cksum)
if [ -e "${root}/pass.id" ]; then
    if [ "$(cat "${root}/pass.id")" != "${id}" ]; then
        echo "the tree or its mutant list changed since this pass started; remove ${root} and start over" >&2
        exit 1
    fi
else
    mkdir -p "${root}"
    for list in caught missed timeout unviable; do
        rm -f "${root}/${list}.txt"
    done
    printf '%s\n' "${id}" > "${root}/pass.id"
fi

k=0
while [ "${k}" -lt "${n}" ]; do
    dir="${root}/${k}"
    if [ -e "${dir}/complete" ]; then
        echo "== mutants shard ${k}/${n}: complete, skipping =="
    else
        echo "== mutants shard ${k}/${n} =="
        mkdir -p "${dir}"
        status=0
        cargo mutants --shard "${k}/${n}" --sharding round-robin --output "${dir}" || status=$?
        # 0, 2 and 3 are cargo-mutants' verdicts (nothing missed or timed
        # out, some missed, some timed out); anything else means the shard
        # did not finish.
        case "${status}" in
            0 | 2 | 3) touch "${dir}/complete" ;;
            *)
                echo "mutants shard ${k}/${n} exited with ${status}; rerun to resume from it" >&2
                exit "${status}"
                ;;
        esac
    fi
    k=$((k + 1))
done

for list in caught missed timeout unviable; do
    cat "${root}"/[0-9]*/mutants.out/"${list}.txt" > "${root}/${list}.txt"
done
rm -f "${root}"/[0-9]*/complete "${root}/pass.id"
wc -l "${root}"/*.txt
if [ -s "${root}/timeout.txt" ]; then exit 3; fi
if [ -s "${root}/missed.txt" ]; then exit 2; fi
