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
# A pass is only as consistent as the conditions it runs under: its shards
# are comparable when they all see the same tree and environment. Keeping
# those fixed for the length of a pass is the caller's job (a worktree of
# its own is the easy way); the checks below catch the common slips between
# invocations, and nothing more. In particular, untracked files, edits made
# while an invocation runs, and changes to the environment, the toolchain or
# cargo-mutants itself are not detected.
#
# Refused, each before a shard runs:
#   - any argument: a flag that reached a resumed pass would differ from
#     the one its finished shards ran with, and the task forwards arguments
#     here (cargo-make would drop them in silence otherwise) only so that
#     they are refused rather than ignored;
#   - a MUTANTS_SHARDS that is not a positive integer of at most six digits
#     written without a leading zero;
#   - a mutant listing that names no package at all;
#   - running while another invocation with the same n holds its lock,
#     since both would run the same shards into the same directories;
#   - resuming once HEAD, `git diff HEAD` or the mutant list differ from
#     what the pass started from;
#   - starting a pass with an n so large that some shard holds no mutant of
#     some package that has mutants, which is where the round-robin argument
#     above stops holding.
#
# State lives in mutants-shards/<n>/ at the workspace root — outside target/,
# like `cargo make mutants`' mutants.out, so `cargo clean` does not take days
# of results with it:
#   <k>/mutants.out/  shard k's cargo-mutants output
#   <k>/complete      shard k ran to a verdict; skipped on the next invocation
#   pass.id           a checksum of what the pass in progress started from
#   finished.id       the same, kept once the pass has finished
#   {caught,missed,timeout,unviable}.txt  the merged lists, once all are done
# The markers are removed after the merge and pass.id becomes finished.id.
# Invoked again over the same tree, a finished pass re-reports its merged
# lists instead of running again; over a changed tree, a new pass replaces
# them. Beside the directory, mutants-shards/<n>.lock is only the file the
# lock is taken on: its existence means nothing.
# cargo-mutants leaves every directory named mutants.out out of its scratch
# copies, so of the state above only the small files are copied along.
#
# Exit status: 1 for a refusal; a shard's own status when it stops without
# a verdict (that shard is redone on the next invocation; cargo-mutants
# exits 1 on most errors of its own, so a 1 alone does not say whether
# shards ran, and the message on stderr tells a refusal apart); the failing
# command's status when anything else fails under `set -e` (a listing, git,
# the merge). A signal that ends the script leaves the running shard
# unmarked. Once every shard has a verdict, it follows cargo-mutants over
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
root="mutants-shards/${n}"
lists="caught missed timeout unviable"

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

head=$(git rev-parse HEAD)
changes=$(git diff HEAD)
mutant_list=$(cargo mutants --list)
id=$(printf '%s\n' "${head}" "${changes}" "${mutant_list}" | cksum)

mkdir -p mutants-shards
# A kernel lock (flock(2), through flock(1) or BSD lockf(1)) on an open
# descriptor rather than a lock file: the kernel drops it once no process
# holds the descriptor, however they ended, so a killed pass or a reboot
# leaves nothing to clean up. The shards are started with the descriptor
# closed, so nothing they leave running (a build server, say) holds the lock.
lock="mutants-shards/${n}.lock"
exec 9> "${lock}"
if command -v flock > /dev/null; then
    take_lock() { flock -n 9; }
elif command -v lockf > /dev/null; then
    take_lock() { lockf -s -t 0 9; }
else
    take_lock() {
        echo "neither flock nor lockf found; running without a lock against concurrent passes" >&2
    }
fi
if ! take_lock; then
    echo "could not lock ${lock}: another pass with MUTANTS_SHARDS=${n} holds it, or the lock tool failed" >&2
    exit 1
fi

# Report the merged lists and exit with the verdict they carry.
report() {
    set --
    for list in ${lists}; do
        set -- "$@" "${root}/${list}.txt"
    done
    wc -l "$@"
    if [ -s "${root}/timeout.txt" ]; then exit 3; fi
    if [ -s "${root}/missed.txt" ]; then exit 2; fi
    exit 0
}

if [ -e "${root}/finished.id" ] && [ "$(cat "${root}/finished.id")" = "${id}" ]; then
    echo "== this pass already finished over the same tree; remove ${root} to run it again =="
    report
fi
if [ -e "${root}/pass.id" ]; then
    if [ "$(cat "${root}/pass.id")" != "${id}" ]; then
        echo "the tree or its mutant list changed since this pass started; remove ${root} and start over" >&2
        exit 1
    fi
else
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
    mkdir -p "${root}"
    rm -f "${root}/finished.id" "${root}"/[0-9]*/complete
    for list in ${lists}; do
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
        cargo mutants --shard "${k}/${n}" --sharding round-robin --output "${dir}" 9>&- || status=$?
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

for list in ${lists}; do
    cat "${root}"/[0-9]*/mutants.out/"${list}.txt" > "${root}/${list}.txt"
done
mv "${root}/pass.id" "${root}/finished.id"
rm -f "${root}"/[0-9]*/complete
report
