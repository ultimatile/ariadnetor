# `ariadnetor_linalg` benches

Two classes of bench live here, distinguished by filename:

- **Criterion regression benches** (`scalar_ops`, `block_sparse_ops`) — statistical
  micro-benchmarks for catching performance regressions. These are the regression
  suite, run by `cargo make bench`.
- **Threshold-sweep instruments** (`sweep_*`) — plain `fn main()` binaries that sweep
  matrix size comparing `ExecPolicy::Sequential` against `ExecPolicy::Parallel(0)` to
  locate the crossover that feeds per-call `ExecPolicy` dispatch in `ariadnetor-native`.
  Run individually on demand, e.g. `cargo bench --bench sweep_decomp_par`. They
  build the naive transpose fallback by default; pass `--features hptt` to
  exercise the HPTT kernel instead.

`cargo make bench` runs only the criterion regression benches; the threshold sweeps
are excluded from it because each is an expensive matrix-size sweep. A bare
`cargo bench -p ariadnetor-linalg` (no `--bench` filter) still runs every target,
sweeps included — cargo cannot keep a bench target compiled under `--all-targets`
while excluding it from a default `cargo bench`, so the regression suite is scoped
at the `cargo make bench` task instead.
