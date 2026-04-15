//! Core traits and types for Ariadnetor tensor framework
//!
//! This crate provides backend-agnostic abstractions:
//! - `Scalar`: Element type trait (sealed to f32, f64, Complex<f32>, Complex<f64>)
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
pub use scalar::Scalar;

// Re-export num_complex for user convenience
pub use num_complex::Complex;
