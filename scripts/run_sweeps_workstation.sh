#!/usr/bin/env bash
# Build all five calibration sweep examples in release mode and run each,
# capturing output to sweep_*.log in the working directory. Designed for
# workstation-class CPUs (Xeon NUMA, 112 cores).
set -euo pipefail

# Resolve the working directory in this preference order:
#   1. WORKDIR (explicit override from the caller)
#   2. the repository root inferred from this script's location
# This keeps the script portable across checkouts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKDIR_RESOLVED="${WORKDIR:-$DEFAULT_WORKDIR}"

if [ ! -d "$WORKDIR_RESOLVED" ]; then
    echo "error: resolved workdir does not exist: $WORKDIR_RESOLVED" >&2
    echo "set WORKDIR to the repository checkout path before running this script" >&2
    exit 1
fi

cd "$WORKDIR_RESOLVED"

EXAMPLES=(
    sweep_decomp_par
    sweep_decomp_rect_par
    sweep_gemm_par
    sweep_solve_par
    sweep_transpose_par
)

BUILD_ARGS=()
for ex in "${EXAMPLES[@]}"; do
    BUILD_ARGS+=(--example "$ex")
done

echo "==> cargo build --release (all sweep examples)"
cargo build --release -p ariadnetor-linalg "${BUILD_ARGS[@]}" --quiet

echo "==> available cores: $(nproc)"
echo "==> rayon will read RAYON_NUM_THREADS / std::thread::available_parallelism"

for ex in "${EXAMPLES[@]}"; do
    log="sweep_${ex#sweep_}.log"
    {
        echo "================================================================"
        echo "Running ${ex}"
        echo "host: $(hostname), cores: $(nproc), date: $(date -Iseconds)"
        echo "================================================================"
    } > "$log"

    echo "==> ${ex} → ${log}"
    cargo run --release --quiet -p ariadnetor-linalg --example "${ex}" >> "$log" 2>&1
done

echo "==> all sweeps complete"
ls -la sweep_*.log
