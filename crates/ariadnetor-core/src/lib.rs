//! Core traits and types for Ariadnetor tensor framework
//!
//! This crate provides backend-agnostic abstractions:
//! - `Scalar`, `FloatCompute`: Element type traits
//! - `ComputeBackend`: Pluggable backend trait
//! - `LabelId`: Interned tensor index labels
//! - `EinsumExpr`, `ContractionPlan`: Einsum parsing and analysis

pub mod backend;
pub mod contraction_error;
pub mod einsum;
pub mod label;
pub mod scalar;

pub use backend::ComputeBackend;
pub use contraction_error::ContractionError;
pub use einsum::{compute_permutation, ContractionPlan, EinsumExpr};
pub use label::LabelId;
pub use scalar::{FloatCompute, Scalar};

// Re-export num_complex for user convenience
pub use num_complex::Complex;
