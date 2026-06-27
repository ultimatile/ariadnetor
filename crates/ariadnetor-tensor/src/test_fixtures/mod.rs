//! Shared BlockSparse test fixtures: `QNIndex` leg builders ([`legs`] and
//! friends) and densify / template helpers ([`densify_bsp_f64`] and friends).
//!
//! The module is gated on `cfg(any(test, feature = "test-fixtures"))`: this
//! crate's own in-lib unit tests reach it as `crate::test_fixtures` under
//! `cfg(test)`, while every other crate's tests enable the `test-fixtures`
//! feature in their dev-dependencies and reach it as
//! `arnet_tensor::test_fixtures`. A separate fixture crate cannot serve the
//! in-lib unit tests: a dev-dependency cycle would link the non-test build of
//! this crate, whose `Sector` / `QNIndex` types are a distinct instance from
//! the `cfg(test)` build under test.

mod densify;
mod legs;

pub use densify::*;
pub use legs::*;
