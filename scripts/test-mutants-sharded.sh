#!/bin/sh
# Contract tests for mutants-sharded.sh, against a stand-in `cargo` that
# answers `cargo mutants` from files instead of running a mutation pass.
#
# The stand-in reads, from $FAKE:
#   list      the mutant list, one `crates/<package>/...` line per mutant
#   exit-<k>  if present, shard k's exit status; consumed by that run
#   hang-<k>  if present, shard k's run records `started` in $FAKE and
#             waits (up to 30 s) for release-<k> before finishing
#   linger-<k>  if present, shard k's run leaves a process behind, its id
#             in $FAKE/linger-pid
#   list-fails  if present, the listings it names fail: `json` (the
#             whole-workspace JSON listing), `shard` (a per-shard listing)
#             or `plain` (the plain-text listing)
# and appends every shard run's index to $FAKE/runs. A shard run writes one
# caught and one unviable mutant, plus one missed (exit 2) or timed-out
# (exit 3) mutant when its status says so, then records done-<k>.
set -eu

# Git exports these to its hooks, and this test runs as one; left set, they
# would point the fixture repositories' git calls at the outer repository.
unset GIT_DIR GIT_INDEX_FILE GIT_WORK_TREE GIT_PREFIX GIT_OBJECT_DIRECTORY \
    GIT_ALTERNATE_OBJECT_DIRECTORIES
# Keep the contributor's own git config (commit signing, hooks) out of the
# fixture commits, including config passed to the outer git with `-c`.
unset GIT_CONFIG_PARAMETERS GIT_CONFIG_COUNT
GIT_CONFIG_GLOBAL=/dev/null
GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL GIT_CONFIG_NOSYSTEM

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
FAKE="${work}/fake"
export FAKE
mkdir -p "${FAKE}/bin"

cat > "${FAKE}/bin/cargo" <<'EOF'
#!/bin/sh
set -eu
[ "$1" = mutants ] || { echo "unexpected: cargo $*" >&2; exit 99; }
shift
list=0 json=0 shard="" sharding="" out=""
while [ $# -gt 0 ]; do
    case "$1" in
        --list) list=1 ;;
        --json) json=1 ;;
        --shard) shard="$2"; shift ;;
        --sharding) sharding="$2"; shift ;;
        --output) out="$2"; shift ;;
        *) echo "unexpected flag: $1" >&2; exit 99 ;;
    esac
    shift
done
# A shard is only ever taken round-robin, for the listing and the run alike.
if [ -n "${shard}" ] && [ "${sharding}" != round-robin ]; then
    echo "shard ${shard} requested with sharding '${sharding}'" >&2
    exit 99
fi
k="${shard%%/*}"
n="${shard##*/}"
if [ "${list}" = 1 ]; then
    if [ -n "${shard}" ]; then kind=shard
    elif [ "${json}" = 1 ]; then kind=json
    else kind=plain
    fi
    if [ -e "${FAKE}/list-fails" ] && [ "$(cat "${FAKE}/list-fails")" = "${kind}" ]; then
        echo "${kind} listing failed" >&2
        exit 101
    fi
    awk -v k="${k}" -v n="${n}" -v json="${json}" -F/ '
        n == "" || (NR - 1) % n == k {
            if (json == 1) print "    \"package\": \"" $2 "\","
            else print
        }' "${FAKE}/list"
    exit 0
fi
echo "${k}" >> "${FAKE}/runs"
if [ -e "${FAKE}/hang-${k}" ]; then
    touch "${FAKE}/started"
    tries=0
    while [ ! -e "${FAKE}/release-${k}" ] && [ "${tries}" -lt 300 ]; do
        sleep 0.1
        tries=$((tries + 1))
    done
fi
if [ -e "${FAKE}/linger-${k}" ]; then
    sleep 30 &
    echo "$!" > "${FAKE}/linger-pid"
fi
mkdir -p "${out}/mutants.out"
for l in missed timeout; do : > "${out}/mutants.out/${l}.txt"; done
echo "caught-${k}" > "${out}/mutants.out/caught.txt"
echo "unviable-${k}" > "${out}/mutants.out/unviable.txt"
status=0
if [ -e "${FAKE}/exit-${k}" ]; then
    status=$(cat "${FAKE}/exit-${k}")
    rm "${FAKE}/exit-${k}"
fi
case "${status}" in
    2) echo "missed-${k}" > "${out}/mutants.out/missed.txt" ;;
    3) echo "timeout-${k}" > "${out}/mutants.out/timeout.txt" ;;
esac
touch "${FAKE}/done-${k}"
exit "${status}"
EOF
chmod +x "${FAKE}/bin/cargo"
PATH="${FAKE}/bin:${PATH}"
real_flock=$(command -v flock || true)
real_lockf=$(command -v lockf || true)
export PATH

failures=0
fail() {
    echo "FAIL: ${case_name}: $1" >&2
    failures=$((failures + 1))
}

# Append $2 mutants of package $1 to the stand-in's mutant list.
mutants() {
    i=1
    while [ "${i}" -le "$2" ]; do
        echo "crates/$1/src/lib.rs:${i}:1: replace x" >> "${FAKE}/list"
        i=$((i + 1))
    done
}

# Start a case in a fresh one-commit git workspace holding its own copy of
# the script — the script resolves the workspace from its own location —
# with six mutants over two packages: three shards then hold one mutant of
# each.
begin() {
    case_name="$1"
    rm -rf "${work}/ws" "${FAKE}"/exit-* "${FAKE}"/hang-* "${FAKE}"/release-* \
        "${FAKE}"/done-* "${FAKE}"/linger-* "${FAKE}/started" \
        "${FAKE}/list-fails"
    mkdir -p "${work}/ws/scripts"
    cp "$(dirname "$0")/mutants-sharded.sh" "${work}/ws/scripts/"
    echo "fn main() {}" > "${work}/ws/main.rs"
    git -C "${work}/ws" init -q
    git -C "${work}/ws" add .
    git -C "${work}/ws" -c user.name=test -c user.email=test@example.invalid commit -q -m fixture
    state="${work}/ws/mutants-shards"
    : > "${FAKE}/runs"
    : > "${FAKE}/list"
    mutants a 3
    mutants b 3
}

# Start a three-shard pass in the background with shard $1 marked hang-<k>,
# its output in ${work}/bg-out and its process id in bg_pid, and wait (up to
# 10 s) until that shard has started.
start_in_background() {
    touch "${FAKE}/hang-$1"
    MUTANTS_SHARDS=3 "${work}/ws/scripts/mutants-sharded.sh" > "${work}/bg-out" 2>&1 &
    bg_pid=$!
    tries=0
    while [ ! -e "${FAKE}/started" ]; do
        tries=$((tries + 1))
        if [ "${tries}" -gt 100 ]; then
            fail "no shard started within 10 s: $(cat "${work}/bg-out")"
            return
        fi
        sleep 0.1
    done
}

# Wait for the background pass and expect it to exit with status $1.
expect_background_exit() {
    status=0
    wait "${bg_pid}" 2> /dev/null || status=$?
    [ "${status}" = "$1" ] || fail "background pass exited ${status}, expected $1: $(cat "${work}/bg-out")"
}

# Let the hanging shard $1 finish, and wait (up to 10 s) until it has, so that
# no shard outlives its case.
release_shard() {
    touch "${FAKE}/release-$1"
    tries=0
    while [ ! -e "${FAKE}/done-$1" ]; do
        tries=$((tries + 1))
        if [ "${tries}" -gt 100 ]; then
            fail "shard $1 did not finish within 10 s"
            return
        fi
        sleep 0.1
    done
}

# Switch PATH to a directory holding only what the script and the stand-ins
# call, plus, when $1 is `flock` or `lockf`, a stand-in for that lock tool;
# with $1 `none` neither lock tool is found. The stand-in checks that it is
# called the way the script calls the real tool, then locks descriptor 9
# with whichever real tool this host has. restore_path switches back.
use_minimal_path() {
    minbin="${work}/minbin-$1"
    mkdir -p "${minbin}"
    for tool in awk cat cksum dirname git grep mkdir mv rm sleep sort touch wc; do
        ln -sf "$(command -v "${tool}")" "${minbin}/${tool}"
    done
    ln -sf "${FAKE}/bin/cargo" "${minbin}/cargo"
    case "$1" in
        flock) expected_args="-n 9" ;;
        lockf) expected_args="-s -t 0 9" ;;
        none) expected_args="" ;;
    esac
    if [ -n "${expected_args}" ]; then
        if [ -n "${real_lockf}" ]; then
            lock_with="${real_lockf} -s -t 0 9"
        else
            lock_with="${real_flock} -n 9"
        fi
        cat > "${minbin}/$1" <<EOF
#!/bin/sh
if [ "\$*" != "${expected_args}" ]; then
    echo "unexpected: \$0 \$*" >&2
    exit 99
fi
exec ${lock_with}
EOF
        chmod +x "${minbin}/$1"
    fi
    saved_path="${PATH}"
    PATH="${minbin}"
}

restore_path() {
    PATH="${saved_path}"
}

# Run the script with MUTANTS_SHARDS=$1 (left unset when $1 is `unset`) and
# the remaining arguments from an unrelated directory, expecting exit status
# $2.
run() {
    shards="$1"
    expected="$2"
    shift 2
    status=0
    (
        cd /
        if [ "${shards}" = unset ]; then unset MUTANTS_SHARDS; else export MUTANTS_SHARDS="${shards}"; fi
        "${work}/ws/scripts/mutants-sharded.sh" "$@"
    ) > "${work}/out" 2>&1 || status=$?
    [ "${status}" = "${expected}" ] || fail "exit ${status}, expected ${expected}: $(cat "${work}/out")"
}

# The shard indices run so far in this case must be exactly $1.
expect_runs() {
    ran=$(tr '\n' ' ' < "${FAKE}/runs")
    [ "${ran}" = "$1" ] || fail "ran shards '${ran}', expected '$1'"
}

# Merged list $1 of the three-shard pass must hold exactly the lines $2.
expect_merged() {
    got=$(sort "${state}/3/$1.txt" | tr '\n' ' ')
    [ "${got}" = "$2" ] || fail "merged $1.txt is '${got}', expected '$2'"
}

# The run just made must have printed a message containing $1.
expect_output() {
    grep -q "$1" "${work}/out" || fail "no '$1' in the output: $(cat "${work}/out")"
}

# Start a three-shard pass that shard 1 stops with a non-verdict exit, so
# shard 0 is complete and the pass is left to resume.
interrupt_at_shard_1() {
    echo 70 > "${FAKE}/exit-1"
    run 3 70
}

# Resuming the pass interrupted at shard 1 must be refused for a changed
# tree or mutant list, without running another shard.
expect_resume_refused() {
    run 3 1
    expect_output "changed since this pass started"
    expect_runs "0 1 "
}

# The run just made must have been refused with a message containing $1,
# before any shard ran or any pass was recorded.
expect_clean_refusal() {
    expect_output "$1"
    expect_runs ""
    if [ -d "${state}" ] && [ -n "$(find "${state}" -name pass.id)" ]; then
        fail "recorded a pass"
    fi
}

# Change a tracked file, so that the next invocation sees another tree.
edit_tree() {
    echo "// edited" >> "${work}/ws/main.rs"
}

begin "a clean pass merges every shard and leaves no resume state"
run 3 0
expect_runs "0 1 2 "
expect_merged caught "caught-0 caught-1 caught-2 "
expect_merged unviable "unviable-0 unviable-1 unviable-2 "
expect_merged missed ""
expect_merged timeout ""
[ -z "$(find "${state}" -name complete -o -name pass.id)" ] || fail "resume state left behind"

begin "rerunning a finished pass over the same tree re-reports it without running a shard"
echo 2 > "${FAKE}/exit-0"
run 3 2
run 3 2
expect_output "already finished"
expect_runs "0 1 2 "
expect_merged missed "missed-0 "

begin "rerunning a finished pass over a changed tree runs a new pass"
run 3 0
edit_tree
run 3 0
expect_runs "0 1 2 0 1 2 "

begin "a new pass ignores completion markers a previous pass left behind"
run 3 0
mkdir -p "${state}/3/0"
touch "${state}/3/0/complete"
edit_tree
run 3 0
expect_runs "0 1 2 0 1 2 "

begin "a new pass drops the previous pass's merged lists while it is in progress"
run 3 0
edit_tree
interrupt_at_shard_1
[ -z "$(find "${state}/3" -maxdepth 1 -name '*.txt')" ] || fail "stale merged lists kept"
[ ! -e "${state}/3/finished.id" ] || fail "the previous pass still counts as finished"

begin "a shard that does not reach a verdict stops the pass and is redone on rerun"
# Shard 0 finishes with a missed mutant before shard 1 fails; the resumed
# pass must still count shard 0's verdict without running it again.
echo 2 > "${FAKE}/exit-0"
interrupt_at_shard_1
expect_runs "0 1 "
[ -e "${state}/3/0/complete" ] || fail "finished shard 0 unmarked"
[ ! -e "${state}/3/1/complete" ] || fail "unfinished shard 1 marked"
[ ! -e "${state}/3/caught.txt" ] || fail "merged before every shard finished"
run 3 2
expect_runs "0 1 1 2 "
expect_merged caught "caught-0 caught-1 caught-2 "
expect_merged missed "missed-0 "

begin "missed mutants in any shard exit 2 after the whole pass"
echo 2 > "${FAKE}/exit-0"
run 3 2
expect_runs "0 1 2 "
expect_merged missed "missed-0 "

begin "timeouts in any shard exit 3 after the whole pass"
echo 3 > "${FAKE}/exit-2"
run 3 3
expect_runs "0 1 2 "
expect_merged timeout "timeout-2 "

begin "timeout outranks missed, as in cargo-mutants"
echo 2 > "${FAKE}/exit-0"
echo 3 > "${FAKE}/exit-1"
run 3 3

begin "an unset MUTANTS_SHARDS runs 40 shards"
# Forty mutants of each package in turn: every shard holds one of each.
: > "${FAKE}/list"
mutants a 40
mutants b 40
run unset 0
[ "$(wc -l < "${FAKE}/runs" | tr -d ' ')" = 40 ] || fail "ran $(wc -l < "${FAKE}/runs") shards"
[ -e "${state}/40/caught.txt" ] || fail "no merged list under 40/"

begin "an argument is refused before anything runs"
run 3 1 --list
expect_clean_refusal "takes no arguments"

begin "a six-digit MUTANTS_SHARDS passes the format check"
# Six mutants cannot fill 100000 shards, so the package check refuses it next.
run 100000 1
expect_output "shard 0/100000 does not hold"

for bad in "" 0 03 -1 abc 1000000 99999999999999999999999; do
    begin "MUTANTS_SHARDS='${bad}' is rejected before anything runs"
    run "${bad}" 1
    expect_clean_refusal "MUTANTS_SHARDS must be"
done

begin "a mutant list that changed mid-pass refuses to resume"
interrupt_at_shard_1
mutants b 1
expect_resume_refused

begin "an edit that leaves the mutant list as it was refuses to resume too"
interrupt_at_shard_1
edit_tree
expect_resume_refused

begin "a resumed pass whose list no longer fits n is told the tree changed"
interrupt_at_shard_1
# Four mutants of a, then one of b: a new pass would be refused for n = 3.
: > "${FAKE}/list"
mutants a 4
mutants b 1
expect_resume_refused

begin "a terminated pass leaves the running shard unmarked"
start_in_background 0
kill -TERM "${bg_pid}"
expect_background_exit 143
release_shard 0
[ ! -e "${state}/3/0/complete" ] || fail "terminated shard marked complete"

begin "a second invocation is refused while a pass is running"
start_in_background 0
run 3 1
expect_output "holds"
release_shard 0
expect_background_exit 0
expect_runs "0 1 2 "

begin "a pass killed with SIGKILL leaves no lock behind, even with its shard still running"
start_in_background 0
kill -KILL "${bg_pid}"
wait "${bg_pid}" 2> /dev/null || true
# The orphaned shard was started without the lock's descriptor.
rm "${FAKE}/hang-0"
run 3 0
expect_runs "0 0 1 2 "
# The rerun's own shard 0 recorded done-0; wait for the orphan's instead.
rm "${FAKE}/done-0"
release_shard 0

begin "a process a shard leaves behind does not hold the lock"
touch "${FAKE}/linger-0"
run 3 0
run 3 0
expect_output "already finished"
kill "$(cat "${FAKE}/linger-pid")" 2> /dev/null || true

begin "without flock or lockf the pass runs unlocked, with a warning"
use_minimal_path none
run 3 0
restore_path
expect_output "running without a lock"
expect_runs "0 1 2 "

if [ -n "${real_lockf}${real_flock}" ]; then
    for tool in flock lockf; do
        begin "with only ${tool}, a second invocation is refused while a pass runs"
        use_minimal_path "${tool}"
        start_in_background 0
        run 3 1
        release_shard 0
        expect_background_exit 0
        restore_path
        expect_output "holds"
        expect_runs "0 1 2 "
    done
else
    echo "note: neither flock nor lockf is installed; the per-tool lock cases did not run" >&2
fi

begin "a lock file left on disk is not a lock"
mkdir -p "${state}"
: > "${state}/3.lock"
run 3 0
expect_runs "0 1 2 "

begin "a commit made mid-pass refuses to resume"
interrupt_at_shard_1
edit_tree
git -C "${work}/ws" -c user.name=test -c user.email=test@example.invalid commit -q -am edit
expect_resume_refused

begin "an n that leaves a shard without one of the packages is refused"
# Four mutants of a, then one of b: with n = 3 only shard 1 holds b.
: > "${FAKE}/list"
mutants a 4
mutants b 1
run 3 1
expect_clean_refusal "shard 0/3 does not hold"

begin "a listing that names no package is refused rather than passed vacuously"
: > "${FAKE}/list"
run 3 1
expect_clean_refusal "found no package"

begin "a workspace git cannot read stops the script before any shard runs"
rm -rf "${work}/ws/.git"
# git's own status for "not a git repository".
run 3 128
expect_runs ""
[ ! -e "${state}" ] || fail "state written"

for kind in json shard plain; do
    begin "a failing ${kind} listing stops the script with cargo's status"
    echo "${kind}" > "${FAKE}/list-fails"
    run 3 101
    expect_clean_refusal "${kind} listing failed"
    if grep -q "lower MUTANTS_SHARDS" "${work}/out"; then fail "blamed the shard count"; fi
done

if [ "${failures}" -gt 0 ]; then
    echo "${failures} failure(s)" >&2
    exit 1
fi
echo "mutants-sharded: all cases passed"
