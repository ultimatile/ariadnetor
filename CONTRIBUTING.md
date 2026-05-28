# Contributing to Ariadnetor

This document captures the coding conventions you are expected to
follow when adding or modifying code in this repository.

## Building

`cargo make` aliases defined in `Makefile.toml`:

```bash
cargo make build       # Build workspace
cargo make test        # Run unit tests
cargo make ci          # Full CI checks (fmt, clippy, test)
```

## Coding Conventions

### Public API surface taxonomy

Every workspace `pub` item placement is anchored to one of three
layers. New additions are placed by the rule, not by analogy with
neighbours.

| Layer | Visibility | Membership rule |
| --- | --- | --- |
| User-API | `pub` in member crate **and** re-exported by umbrella `arnet` | The umbrella surface. Covers inherent methods on `Tensor` / `DenseTensor` / `BlockSparseTensor`, free fns (e.g. `add_all`, `linear_combine`, `contract`, `eig`), trait extensions, error/result types, and traits re-exported as type-parameter shapes. |
| Mid-layer | `pub` in member crate, **not** re-exported by umbrella | Workspace-internal consumer API. Reachable only via a direct member-crate dependency (e.g. `arnet-mps` depending directly on `arnet-tensor`). The `*TensorData` joined-form bundle and storage-half basic accessors live here. |
| Internal | `pub(crate)` | Consumed only by in-crate forwarders. New items default here unless a Mid-layer or User-API caller exists. |

Re-exporting a type does not automatically promote its inherent
methods to User-API. Even when `arnet` re-exports a struct as a
generic type-parameter shape, classify each inherent method
independently: some may stay at Mid-layer or Internal. The
membership rule applies per-item, not per-type.

When adding a new `pub` item, check the rule in this order:

1. Is the umbrella `arnet` going to re-export it? → User-API.
2. Will a sibling workspace crate call it directly (bypassing the
   umbrella)? → Mid-layer.
3. Otherwise → Internal (`pub(crate)`).

If demoting to `pub(crate)` triggers a `dead_code` warning, the item
was already dead under the narrower visibility — remove it rather
than annotate.

### Naming

#### In-place vs out-of-place method pairs

When a method comes in two flavors — one that mutates through
`&mut self` and one that returns a new value from `&self` — name
them as a pair using the **`-ed` suffix** for the out-of-place
variant:

| in-place (`&mut self`)        | out-of-place (`&self`)         |
| ----------------------------- | ------------------------------ |
| `scale(&mut self, factor)`    | `scaled(&self, factor)`        |
| `normalize(&mut self)`        | `normalized(&self)`            |

Rationale:

- `&mut self` already conveys in-place mutation; an `_in_place`
  suffix is redundant.
- `-ed` reads naturally in English as "the X-ed version of self,"
  matching the semantic of "the value after applying X."
- Aligns with the standard library's `sort` (in-place) /
  `sorted` (out-of-place, on iterators) pattern.

## Profiling

The workspace `[profile.bench]` is configured for sampling-profiler
use (`debug = "line-tables-only"`, `split-debuginfo = "packed"`),
producing `.dSYM` (macOS) / `.dwp` (Linux) bundles next to bench
binaries without affecting optimization. `samply record` is
supported on Linux and macOS only.

Recipe (using `samply` and `profiler-cli` `pq`):

```sh
mkdir -p target/samply

cargo bench --bench block_sparse_ops --no-run
# → target/release/deps/block_sparse_ops-<hash>

samply record --save-only -o target/samply/<name>.json.gz -- \
  target/release/deps/block_sparse_ops-<hash> \
  --bench --profile-time=10 '<criterion-id-regex>'

# pq does not resolve symbols from a local .dSYM bundle directly.
# Run samply load as a local symbol-server sidecar and pass the
# token URL to pq:
samply load --no-open target/samply/<name>.json.gz &
# Extract http://127.0.0.1:<port>/<token> from samply output.

pq load target/samply/<name>.json.gz --symbol-server <token-url>
pq thread samples-bottom-up         # self-time ranking
pq thread functions --min-self 1    # hot functions
pq thread samples-bottom-up --json  # for jq pipelines
```

`--profile-time=N` runs the matched benchmark for ~N seconds
without statistical analysis — designed for sampling-profiler
integration.
