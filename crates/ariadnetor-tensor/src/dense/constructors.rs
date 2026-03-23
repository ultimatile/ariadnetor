//! Factory methods and data initialization for DenseTensor.

use aligned_vec::{AVec, ConstAlign};
use num_traits::{One, Zero};
use std::sync::Arc;

use super::{Align64, DenseTensor, MemoryOrder, column_major_strides, row_major_strides};

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::zero());
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, T::one());
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create a tensor filled with a constant value
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total_elements: usize = shape.iter().product();
        let mut data: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total_elements);
        data.resize(total_elements, value);
        let strides = row_major_strides(&shape);

        Self {
            data: Arc::new(data),
            strides,
            shape,
            offset: 0,
            order: MemoryOrder::RowMajor,
        }
    }

    /// Create an n×n identity matrix in row-major order.
    pub fn eye(n: usize) -> Self
    where
        T: Zero + One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::from_data_with_order(data, vec![n, n], MemoryOrder::RowMajor)
    }

    /// Create a tensor from data with explicit strides and offset.
    ///
    /// Used by backends to produce tensors in non-row-major layouts.
    ///
    /// # Panics
    ///
    /// Panics if any logical index would address outside the data buffer,
    /// or if shape and strides ranks differ.
    pub fn from_data_with_strides(
        data: Vec<T>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
        order: MemoryOrder,
    ) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Shape rank {} doesn't match strides rank {}",
            shape.len(),
            strides.len()
        );

        // Validate that all reachable indices are within bounds.
        let data_len = data.len();

        // Empty tensors still need offset within buffer (for data()/as_ptr() safety)
        assert!(
            offset <= data_len,
            "from_data_with_strides: offset {offset} exceeds data buffer of length {data_len}"
        );

        if !shape.contains(&0) {
            let mut min_offset: isize = offset as isize;
            let mut max_offset: isize = offset as isize;
            for (&dim, &stride) in shape.iter().zip(&strides) {
                let end = stride * (dim as isize - 1);
                if end >= 0 {
                    max_offset += end;
                } else {
                    min_offset += end;
                }
            }
            assert!(
                min_offset >= 0 && (max_offset as usize) < data_len,
                "from_data_with_strides: reachable index range [{min_offset}, {max_offset}] \
                 exceeds data buffer of length {data_len}"
            );
        }

        let mut aligned_data: AVec<T, Align64> = AVec::with_capacity(64, data_len);
        for elem in data {
            aligned_data.push(elem);
        }

        Self {
            data: Arc::new(aligned_data),
            shape,
            strides,
            offset,
            order,
        }
    }

    /// Create a tensor from existing data with the specified memory order.
    ///
    /// This is the primary order-explicit constructor. Strides are computed
    /// automatically based on the memory order.
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the shape.
    pub fn from_data_with_order(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self {
        let strides = match order {
            MemoryOrder::RowMajor => row_major_strides(&shape),
            MemoryOrder::ColumnMajor => column_major_strides(&shape),
        };
        Self::from_data_with_strides(data, shape, strides, 0, order)
    }

    /// Create a tensor filled with random values from the standard distribution.
    #[cfg(feature = "random")]
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::from_data_with_order(data, shape, MemoryOrder::RowMajor)
    }
}
