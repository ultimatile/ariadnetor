//! Dense tensor storage with flat contiguous data.
//!
//! Provides the storage / layout split for the dense case:
//!
//! - [`DenseStorage<T>`] — pure-data half (flat 64-byte-aligned buffer
//!   with Arc-based Copy-on-Write).
//! - [`DenseLayout`] — interpretation half (shape + memory order).
//! - [`DenseTensorData<T>`] — joined bundle
//!   = `TensorData<DenseStorage<T>, DenseLayout>`.

mod layout;
mod storage;
mod tensor_data;

pub use layout::DenseLayout;
pub use storage::DenseStorage;
pub use tensor_data::DenseTensorData;
