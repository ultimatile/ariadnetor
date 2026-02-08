//! Tensor library with two-layer architecture
//!
//! - **TensorStorage**: Storage layer (Dense, Sparse, BlockSparse)
//! - **Tensor**: Main API type wrapping TensorStorage
//!
//! # Type System
//!
//! All tensor types are generic over element type `T` with default `f64`:
//! - [`DenseTensor<T>`]: Dense storage layer
//! - [`TensorStorage<T>`]: Storage format enum
//! - [`Tensor<T>`]: Main tensor type
//!
//! # Type Aliases
//!
//! For convenience, type aliases are provided for common element types:
//! - `Tensor64`, `DenseTensor64`: `f64` (double precision)
//! - `Tensor32`, `DenseTensor32`: `f32` (single precision)
//! - `TensorC64`, `DenseTensorC64`: `Complex<f64>` (complex double)
//! - `TensorC32`, `DenseTensorC32`: `Complex<f32>` (complex single)

pub mod arithmetic;
pub mod dense;
pub mod tensor;
pub mod tensor_storage;
pub mod sector;

// Re-export from ariadnetor-core
pub use arnet_core::{
    Complex, ContractionError, ContractionPlan, EinsumExpr, FloatCompute, Scalar,
    compute_permutation, contraction_error, einsum, scalar,
};

pub use dense::DenseTensor;
pub use tensor::Tensor;
pub use tensor_storage::TensorStorage;

// Convenient type aliases for common numeric types
pub type Tensor64 = Tensor<f64>;
pub type Tensor32 = Tensor<f32>;
pub type TensorC64 = Tensor<Complex<f64>>;
pub type TensorC32 = Tensor<Complex<f32>>;

pub type DenseTensor64 = DenseTensor<f64>;
pub type DenseTensor32 = DenseTensor<f32>;
pub type DenseTensorC64 = DenseTensor<Complex<f64>>;
pub type DenseTensorC32 = DenseTensor<Complex<f32>>;

pub type TensorStorage64 = TensorStorage<f64>;
pub type TensorStorage32 = TensorStorage<f32>;
pub type TensorStorageC64 = TensorStorage<Complex<f64>>;
pub type TensorStorageC32 = TensorStorage<Complex<f32>>;
