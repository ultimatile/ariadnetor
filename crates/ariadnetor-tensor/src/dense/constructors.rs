//! Factory methods for `DenseTensorData<T>` taking explicit memory order.
//!
//! Backend-aware callers use these via `ComputeBackendTensorExt`;
//! explicit-order callers (tests, kernels) invoke them directly. The
//! `*_in_order` suffix names the explicit-order signature shape.

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;
use num_traits::{One, Zero};
use rand::RngExt;

use crate::{DenseLayout, DenseStorage, DenseTensorData, TensorData};

impl<T> DenseTensorData<T>
where
    T: Clone,
{
    /// Zero-filled tensor in the requested memory order.
    pub fn zeros_in_order(shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        data.resize(total, T::zero());
        let storage = DenseStorage::from_aligned(data);
        let layout = DenseLayout::new(shape, order);
        TensorData::new(storage, layout)
    }

    /// Ones-filled tensor in the requested memory order.
    pub fn ones_in_order(shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: One + Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        data.resize(total, T::one());
        let storage = DenseStorage::from_aligned(data);
        let layout = DenseLayout::new(shape, order);
        TensorData::new(storage, layout)
    }

    /// Tensor filled with `value` in the requested memory order.
    pub fn filled_in_order(shape: Vec<usize>, value: T, order: MemoryOrder) -> Self {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        data.resize(total, value);
        let storage = DenseStorage::from_aligned(data);
        let layout = DenseLayout::new(shape, order);
        TensorData::new(storage, layout)
    }

    /// `n × n` identity matrix in the requested memory order.
    ///
    /// The identity matrix is symmetric, so the flat buffer is the
    /// same under either memory order; only the layout's `order()`
    /// differs.
    pub fn eye_in_order(n: usize, order: MemoryOrder) -> Self
    where
        T: Zero + One,
    {
        let mut buf = vec![T::zero(); n * n];
        for i in 0..n {
            buf[i * n + i] = T::one();
        }
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, n * n);
        for elem in buf {
            data.push(elem);
        }
        let storage = DenseStorage::from_aligned(data);
        let layout = DenseLayout::new(vec![n, n], order);
        TensorData::new(storage, layout)
    }

    /// Random-filled tensor (standard distribution) in the requested
    /// memory order. Random values are layout-invariant; only the
    /// layout's `order()` tag matters for downstream interpretation.
    pub fn random_in_order<R: rand::Rng>(shape: Vec<usize>, order: MemoryOrder, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        for _ in 0..total {
            data.push(rng.random());
        }
        let storage = DenseStorage::from_aligned(data);
        let layout = DenseLayout::new(shape, order);
        TensorData::new(storage, layout)
    }
}
