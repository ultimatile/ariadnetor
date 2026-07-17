//! Core traits and types for ariadnetor tensor library
//!
//! This crate provides backend-agnostic abstractions:
//! - `Scalar`: Element type trait (sealed to f32, f64, Complex<f32>, Complex<f64>)
//! - `ComputeBackend`: Pluggable backend trait
//! - `LabelId`: Interned tensor index labels
//! - `EinsumExpr`, `ContractionPlan`: Einsum parsing and analysis
//! - `scale_safe_norm`, `combine_norms`, `NormAccumulator`: Scale-safe
//!   sum-of-squares accumulation

#![deny(missing_docs)]

pub mod backend;
mod contraction_error;
mod einsum;
mod label;
mod norm;
mod scalar;

pub use backend::{ComputeBackend, ExecPolicy, MemoryOrder};
pub use contraction_error::ContractionError;
pub use einsum::{ContractionPlan, EinsumExpr, compute_permutation};
pub use label::LabelId;
pub use norm::{NormAccumulator, combine_norms, scale_safe_norm};
pub use scalar::Scalar;

// Re-export num_complex for user convenience
pub use num_complex::Complex;
