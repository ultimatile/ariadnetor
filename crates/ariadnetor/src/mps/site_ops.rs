//! SiteOps trait and concrete operator dictionaries

use arnet_core::scalar::Scalar;
use arnet_tensor::{DenseTensor, MemoryOrder};
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
        DenseTensor::from_data_with_order(
            vec![half, z, z, neg_half],
            vec![2, 2],
            MemoryOrder::RowMajor,
        )
    }

    /// S+ (raising operator) = [[0, 1], [0, 0]]
    pub fn sp<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![z, o, z, z], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// S- (lowering operator) = [[0, 0], [1, 0]]
    pub fn sm<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![z, z, o, z], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// Identity operator = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![o, z, z, o], vec![2, 2], MemoryOrder::RowMajor)
    }
}

/// Qubit site operators for quantum computing.
///
/// Provides Pauli gates (X, Y, Z), Hadamard (H), phase gates (S, T),
/// identity, and projectors as 2×2 matrices, generic over any [`Scalar`] type.
///
/// Gates with imaginary components (Y, S, T) use `Scalar::from_real_imag`.
/// When `T` is a real type, imaginary parts are dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qubit;

impl SiteOps for Qubit {
    fn dim(&self) -> usize {
        2
    }
}

impl Qubit {
    /// Pauli X = [[0, 1], [1, 0]]
    pub fn x<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![z, o, o, z], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// Pauli Y = [[0, -i], [i, 0]]
    pub fn y<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let zi = T::Real::zero();
        let one = real_val::<T::Real>(1.0);
        let neg_i = T::from_real_imag(zi, real_val::<T::Real>(-1.0));
        let pos_i = T::from_real_imag(zi, one);
        DenseTensor::from_data_with_order(
            vec![z, neg_i, pos_i, z],
            vec![2, 2],
            MemoryOrder::RowMajor,
        )
    }

    /// Pauli Z = [[1, 0], [0, -1]]
    pub fn z<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        let neg = real::<T>(-1.0);
        DenseTensor::from_data_with_order(vec![o, z, z, neg], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// Hadamard = [[1, 1], [1, -1]] / sqrt(2)
    pub fn h<T: Scalar>(&self) -> DenseTensor<T> {
        let inv_sqrt2 = real::<T>(std::f64::consts::FRAC_1_SQRT_2);
        let neg_inv_sqrt2 = real::<T>(-std::f64::consts::FRAC_1_SQRT_2);
        DenseTensor::from_data_with_order(
            vec![inv_sqrt2, inv_sqrt2, inv_sqrt2, neg_inv_sqrt2],
            vec![2, 2],
            MemoryOrder::RowMajor,
        )
    }

    /// S gate = [[1, 0], [0, i]]
    pub fn s<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        let i = T::from_real_imag(T::Real::zero(), real_val::<T::Real>(1.0));
        DenseTensor::from_data_with_order(vec![o, z, z, i], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// T gate = [[1, 0], [0, exp(iπ/4)]]
    pub fn t<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        let angle = std::f64::consts::FRAC_PI_4;
        let t_val = T::from_real_imag(
            real_val::<T::Real>(angle.cos()),
            real_val::<T::Real>(angle.sin()),
        );
        DenseTensor::from_data_with_order(vec![o, z, z, t_val], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// Identity = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![o, z, z, o], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// |0⟩⟨0| = [[1, 0], [0, 0]]
    pub fn proj0<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![o, z, z, z], vec![2, 2], MemoryOrder::RowMajor)
    }

    /// |1⟩⟨1| = [[0, 0], [0, 1]]
    pub fn proj1<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        DenseTensor::from_data_with_order(vec![z, z, z, o], vec![2, 2], MemoryOrder::RowMajor)
    }
}

/// Convert an f64 value to a Scalar type via its Real component.
fn real<T: Scalar>(val: f64) -> T {
    let r: T::Real = NumCast::from(val).unwrap();
    T::from_real_imag(r, T::Real::zero())
}

/// Convert an f64 value to a FloatCompute (Real) type.
fn real_val<R: NumCast>(val: f64) -> R {
    NumCast::from(val).unwrap()
}
