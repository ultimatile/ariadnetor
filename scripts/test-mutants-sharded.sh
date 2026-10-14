#!/bin/sh
# Contract tests for mutants-sharded.sh, against a stand-in `cargo` that
# answers `cargo mutants` from files instead of running a mutation pass.
#
# The stand-in reads, from $FAKE:
#   list      the mutant list, one `crates/<package>/...` line per mutant
#   exit-<k>  if present, shard k's exit status; consumed by that run
#   list-fails  if present, the listings it names fail: `json` (the
#             whole-workspace JSON listing), `shard` (a per-shard listing)
#             or `plain` (the plain-text listing)
# and appends every shard run's index to $FAKE/runs. A shard run writes one
# caught and one unviable mutant, plus one missed (exit 2) or timed-out
# (exit 3) mutant when its status says so.
set -eu

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
exit "${status}"
EOF
chmod +x "${FAKE}/bin/cargo"
PATH="${FAKE}/bin:${PATH}"
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
    rm -rf "${work}/ws" "${FAKE}"/exit-* "${FAKE}/list-fails"
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

# The run just made must have been refused with a message containing $1.
expect_refusal() {
    grep -q "$1" "${work}/out" || fail "not refused with '$1': $(cat "${work}/out")"
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
    expect_refusal "changed since this pass started"
    expect_runs "0 1 "
}

# The run just made must have been refused with a message containing $1,
# before any shard ran or any pass state was written.
expect_clean_refusal() {
    expect_refusal "$1"
    expect_runs ""
    [ ! -e "${state}" ] || fail "state written"
}

begin "a clean pass merges every shard and leaves no resume state"
run 3 0
expect_runs "0 1 2 "
expect_merged caught "caught-0 caught-1 caught-2 "
expect_merged unviable "unviable-0 unviable-1 unviable-2 "
expect_merged missed ""
expect_merged timeout ""
[ -z "$(find "${state}" -name complete -o -name pass.id)" ] || fail "resume state left behind"

begin "the next pass after a finished one runs every shard again"
run 3 0
run 3 0
expect_runs "0 1 2 0 1 2 "

begin "a new pass drops the previous pass's merged lists while it is in progress"
run 3 0
interrupt_at_shard_1
[ -z "$(find "${state}/3" -maxdepth 1 -name '*.txt')" ] || fail "stale merged lists kept"

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
expect_refusal "shard 0/100000 does not hold"

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
echo "// edited" >> "${work}/ws/main.rs"
expect_resume_refused

begin "a commit made mid-pass refuses to resume"
interrupt_at_shard_1
echo "// edited" >> "${work}/ws/main.rs"
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
[ ! -e "${state}/3/pass.id" ] || fail "recorded a pass"

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
