//! Shared algorithms-level fixtures for the algorithms crate's tests and
//! benchmarks: U(1) MPS / MPO chains (`fixtures`) and Dense builders and
//! dense references (`dense_fixtures`). The tensor-layer densify / leg
//! helpers live in `ariadnetor_tensor::test_fixtures`.
//!
//! These modules live in a library crate so their `pub` API is rooted
//! and reachable, letting `dead_code = "forbid"` apply workspace-wide
//! without per-file suppressions: a `tests/*.rs` integration test
//! compiles as its own binary where `pub` does not exempt an item from
//! the dead-code lint, but a library crate's reachable `pub` items are
//! always live.

#![deny(missing_docs)]

pub mod dense_fixtures;
pub mod fixtures;
