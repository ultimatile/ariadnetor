//! MPS serialization: lossless, deterministic `save_mps` / `load_mps` for the
//! current `Mps<St, L>` type.
//!
//! Restart of an iterative MPS algorithm reduces to *load an MPS, feed it back
//! in*: the state captures everything a warm continuation needs, and
//! environments are recomputed on re-entry. The reusable primitive is
//! therefore MPS serialization, not a DMRG-coupled "restart mode". See the
//! format specification below.
//!
//! # Scope
//!
//! Warm continuation from a saved state — not bit-exact replay of an
//! interrupted run. The energy baseline, sweep count / numbering, and sweep
//! budget live *outside* the MPS and reset on re-entry, so loading an MPS is a
//! correct warm continuation, not a reproduction of a run's stopping decision.
//! The run-state envelope, RNG state, environments, downstream-defined sector
//! types, a directory checkpoint format, and cross-version migration are all
//! out of scope (a future per-algorithm checkpoint manager owns them).
//!
//! # Format
//!
//! A single stream: `[8-byte magic] [u64 LE manifest length]
//! [CBOR manifest] [numeric data section]`.
//!
//! - **Metadata** is `serde` + CBOR: self-describing (field names preserved),
//!   so a checkpoint read back much later survives struct evolution.
//!   Compactness is irrelevant — the metadata is small.
//! - **Numeric bodies** are explicit little-endian scalar bytes, not pushed
//!   through the generic serde model: that keeps the complex representation
//!   layout-explicit and bit-exact (signed zero / infinities / NaN payloads
//!   round-trip).
//!
//! The manifest carries scalar / storage / sector type-identity tags checked
//! against the requested `T` / `St` / `S` before any value is decoded — this
//! rejects, for instance, a U(1) file loaded as Z₂ even though their stored
//! values are byte-compatible.
//!
//! # Version policy
//!
//! `format_version` is `1`. Pre-v0.1 there are no migration shims: a load
//! rejects an unknown (higher) version cleanly. A future MPS shape change
//! bumps the version rather than being speculatively accommodated now.
//!
//! # Decode safety
//!
//! Load never panics on crafted input. Every descriptor is validated and all
//! extent arithmetic is checked before the panicking reconstruction
//! constructors run; violations map to a typed [`MpsIoError`].

mod codec;
mod container;
mod error;
mod manifest;

#[cfg(test)]
mod tests;

pub use codec::MpsCodec;
pub use container::{load_mps, load_mps_from_path, save_mps, save_mps_to_path};
pub use error::MpsIoError;
pub use manifest::{MpsManifest, OrderTag, SiteMeta};
