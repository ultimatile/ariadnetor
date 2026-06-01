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

### Error types

All error enums in the workspace derive their `Display` / `Error` /
`From` impls with [`thiserror`](https://docs.rs/thiserror) rather than
hand-writing them. The conventions below keep error chains free of
duplicated cause text under a `source()`-walking reporter (e.g.
`anyhow`'s `{:#}`).

#### `Display` describes this layer; `source()` links the cause

Per the [`std::error::Error` guideline][std-error-guideline], an inner
error is exposed through **either** the outer error's `source()` **or**
its `Display` — never both. Doing both makes a reporter that prints the
wrapper and then walks `source()` surface the same cause twice.

Concretely, when a variant wraps another error type:

- Expose the cause with `#[from]` (source is the only field) or
  `#[source]` (source plus self-layer context fields).
- Write the variant's `#[error("...")]` for **this layer only** — do
  not interpolate the wrapped error's `Display` (no `: {0}` /
  `: {source}` that re-renders the cause).

```rust
// Good: self-layer context in Display, cause reachable via source().
#[error("step failed at site {site}")]
Step {
    site: usize,
    #[source]
    source: HeffError,
},
```

#### Pure repackaging is `transparent`

A variant that adds no context of its own — it only re-tags a child
error into this layer's enum — uses `#[error(transparent)]` with
`#[from]`, delegating both `Display` and `source()` to the child:

```rust
#[error(transparent)]
Backend(#[from] BackendError),
```

A fixed self-layer string (e.g. `#[error("backend operation failed")]`)
still counts as context, so it is **not** transparent. Reserve
`transparent` for pure re-tagging with no distinguishing text.

#### Leaf errors

An error that carries all its context in its own fields (strings,
sizes, labels) and wraps no structured inner error is a leaf: its
`source()` is always `None`. Write the full message in `#[error("...")]`
and add nothing else. `BackendError`, `ContractionError`, and
`TensorError` are leaves.

#### Exception: mirroring an external leaf error

The wrap-and-expose pattern is the default for an inner error. One case
departs from it deliberately: `ArpackError` (`krylov/arpack.rs`)
**mirrors** the upstream `arpack::Error` variant-for-variant and
re-materializes its data into its own fields, instead of holding it via
`#[from]` / `#[source]`. The result is itself a leaf (`source()` is
`None`), and the `From<arpack::Error>` conversion is hand-written.

This is allowed only because all three conditions hold:

1. **No cause is lost.** `arpack::Error` is itself a leaf — its variants
   carry only primitive diagnostics (`&'static str`, `i32`, iparam
   counters) and its own `source()` is `None`. Re-materializing that
   data into typed fields drops no link in the chain; a `source()` link
   would preserve nothing the mirrored fields do not.
2. **It decouples the public surface from an unstable dependency.**
   `arpack` is a pre-1.0, `#[non_exhaustive]`, FFI-bound crate. Holding
   it via `#[from]` would put `arpack::Error` on this crate's public
   API, making every upstream bump a breaking change here. The mirror
   lets `ArpackError` own a stable surface and choose its own
   `#[non_exhaustive]` policy.
3. **`#[from]` cannot express the remap.** `#[from]` forwards one inner
   value whole; the per-variant remap — including the catch-all that
   absorbs future upstream variants — requires a hand-written `From`.

Do not generalize this to an inner error that has its own `source()`
chain: re-materializing such an error would drop causes, which is
exactly what the wrap-and-expose rule prevents. Mirroring is reserved
for an external leaf at a dependency boundary.

#### `#[from]` vs `#[source]`

Both expose the cause via `source()`; the choice is about field shape,
independent of the `Display` rules above:

- `#[from]` — source is the variant's **only** field; also generates
  the `From<Child>` conversion for `?`.
- `#[source]` — the variant carries additional context fields alongside
  the source, so `From` cannot be derived.

#### One-line cause display

If a one-line message that includes the cause is needed, build it at the
final display boundary by walking the `source()` chain — do not fold the
cause into an error type's `Display`. With `anyhow`, `{:#}` already does
this.

[std-error-guideline]: https://doc.rust-lang.org/std/error/trait.Error.html

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
