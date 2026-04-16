//! Factory methods and data initialization for Dense.

use aligned_vec::{AVec, ConstAlign};
use num_traits::{One, Zero};
use std::sync::Arc;

use super::{Align64, Dense};

impl<T> Dense<T>
where
    T: Clone,
{
    /// Create a Dense tensor from flat data and shape.
    ///
    /// The caller is responsible for ensuring the data is laid out
    /// in the intended memory order. Dense itself stores no layout
    /// information — interpretation is the backend's responsibility.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total,
            "Dense::new: data length {} doesn't match shape {:?} (total {})",
            data.len(),
            shape,
            total
        );

        let mut aligned: AVec<T, ConstAlign<64>> = AVec::with_capacity(64, total);
        for elem in data {
            aligned.push(elem);
        }

        Self {
            data: Arc::new(aligned),
            shape,
        }
    }

    /// Create a new tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, T::zero());

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, T::one());

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total: usize = shape.iter().product();
        let mut data: AVec<T, Align64> = AVec::with_capacity(64, total);
        data.resize(total, value);

        Self {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create an n×n identity matrix.
    ///
    /// Data is laid out in row-major order: element (i,j) is at flat
    /// index `i*n + j`. The backend must interpret it accordingly.
    pub fn eye(n: usize) -> Self
    where
        T: Zero + One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::new(data, vec![n, n])
    }

    /// Create a tensor filled with random values from the standard distribution.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::new(data, shape)
    }
}
