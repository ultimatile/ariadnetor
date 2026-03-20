//! SiteOps trait and concrete operator dictionaries

use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{NumCast, Zero};

/// Trait for site-local operator dictionaries.
///
/// Provides the physical dimension of the local Hilbert space.
/// Concrete types (e.g., [`SpinHalf`]) supply type-safe methods
/// returning specific operators as dense matrices.
pub trait SiteOps {
    /// Physical dimension of the local Hilbert space.
    fn dim(&self) -> usize;
}

/// Spin-1/2 site operators.
///
/// Provides `sz`, `sp` (S+), `sm` (S-), and `id` operators as
/// 2×2 matrices, generic over any [`Scalar`] type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpinHalf;

impl SiteOps for SpinHalf {
    fn dim(&self) -> usize {
        2
    }
}

impl SpinHalf {
    /// S_z = diag(1/2, -1/2)
    pub fn sz<T: Scalar>(&self) -> DenseTensor<T> {
        let half = real::<T>(0.5);
        let neg_half = real::<T>(-0.5);
        let z = T::zero();
        DenseTensor::from_data(vec![half, z, z, neg_half], vec![2, 2])
    }

    /// S+ (raising operator) = [[0, 1], [0, 0]]
    pub fn sp<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data(vec![z, o, z, z], vec![2, 2])
    }

    /// S- (lowering operator) = [[0, 0], [1, 0]]
    pub fn sm<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data(vec![z, z, o, z], vec![2, 2])
    }

    /// Identity operator = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data(vec![o, z, z, o], vec![2, 2])
    }
}

/// Convert an f64 value to a Scalar type via its Real component.
fn real<T: Scalar>(val: f64) -> T {
    let r: T::Real = NumCast::from(val).unwrap();
    T::from_real_imag(r, T::Real::zero())
}
