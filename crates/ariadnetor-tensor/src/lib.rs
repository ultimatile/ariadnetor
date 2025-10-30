//! Tensor library with two-layer architecture
//!
//! - **RawTensor**: Storage layer (Dense, Sparse, BlockSparse)
//! - **FatTensor**: Metadata layer (storage + indices)
//!
//! # Type System
//!
//! All tensor types are generic over element type `T` with default `f64`:
//! - [`DenseTensor<T>`]: Dense storage layer
//! - [`RawTensor<T>`]: Storage format enum
//! - [`FatTensor<T>`]: Full tensor with metadata
//! - [`Tensor<T>`]: Public API alias for FatTensor
//!
//! # Type Aliases
//!
//! For convenience, type aliases are provided for common element types:
//! - `Tensor64`, `DenseTensor64`: `f64` (double precision)
//! - `Tensor32`, `DenseTensor32`: `f32` (single precision)
//! - `TensorC64`, `DenseTensorC64`: `Complex<f64>` (complex double)
//! - `TensorC32`, `DenseTensorC32`: `Complex<f32>` (complex single)

pub mod dense;
pub mod einsum;
pub mod fat_tensor;
pub mod index;
pub mod raw_tensor;
pub mod sector;

pub use dense::DenseTensor;
pub use fat_tensor::FatTensor;
pub use index::{Index, IndexSet};
pub use raw_tensor::RawTensor;

// Re-export num_complex for user convenience
pub use num_complex::Complex;

/// Public API alias for FatTensor
pub type Tensor<T = f64> = FatTensor<T>;

// Convenient type aliases for common numeric types
pub type Tensor64 = Tensor<f64>;
pub type Tensor32 = Tensor<f32>;
pub type TensorC64 = Tensor<Complex<f64>>;
pub type TensorC32 = Tensor<Complex<f32>>;

pub type DenseTensor64 = DenseTensor<f64>;
pub type DenseTensor32 = DenseTensor<f32>;
pub type DenseTensorC64 = DenseTensor<Complex<f64>>;
pub type DenseTensorC32 = DenseTensor<Complex<f32>>;

pub type RawTensor64 = RawTensor<f64>;
pub type RawTensor32 = RawTensor<f32>;
pub type RawTensorC64 = RawTensor<Complex<f64>>;
pub type RawTensorC32 = RawTensor<Complex<f32>>;

pub type FatTensor64 = FatTensor<f64>;
pub type FatTensor32 = FatTensor<f32>;
pub type FatTensorC64 = FatTensor<Complex<f64>>;
pub type FatTensorC32 = FatTensor<Complex<f32>>;
