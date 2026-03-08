//! Backend-agnostic linear algebra API for Ariadnetor
//!
//! This crate will provide high-level operations (contract, transpose, svd, ...)
//! that delegate to a [`ComputeBackend`] for the actual computation.
//!
//! **Status**: Skeleton — API will be added in Phase 1.2+.

pub use arnet_core::backend::ComputeBackend;
