//! Core traits and types for Ariadnetor tensor framework
//!
//! This crate provides backend-agnostic abstractions:
//! - `Scalar`, `FloatCompute`: Element type traits
//! - `ComputeBackend`: Pluggable backend trait
//! - `LabelId`: Interned tensor index labels
//! - `EinsumExpr`, `ContractionPlan`: Einsum parsing and analysis

pub mod backend;
mod contraction_error;
mod einsum;
mod label;
mod scalar;

pub use backend::{ComputeBackend, MemoryOrder};
pub use contraction_error::ContractionError;
pub use einsum::{ContractionPlan, EinsumExpr, compute_permutation};
pub use label::LabelId;
pub use scalar::{FloatCompute, Scalar};

// Re-export num_complex for user convenience
pub use num_complex::Complex;
