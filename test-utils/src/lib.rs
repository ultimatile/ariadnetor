//! Shared fixtures and helpers for DMRG integration tests.
//!
//! These modules live in a library crate so their `pub` API is rooted
//! and reachable, letting `dead_code = "forbid"` apply workspace-wide
//! without per-file suppressions: a `tests/*.rs` integration test
//! compiles as its own binary where `pub` does not exempt an item from
//! the dead-code lint, but a library crate's reachable `pub` items are
//! always live.

pub mod dense_fixtures;
pub mod fixtures;
pub mod helpers;
