//! SiteOps trait and concrete operator dictionaries
//!
//! Operator data is stored in column-major (Fortran) order to match
//! `NativeBackend::preferred_order()`. For a 2x2 matrix `[[a, b], [c, d]]`,
//! the flat layout is `[a, c, b, d]`.

use arnet_core::scalar::Scalar;
use arnet_tensor::Dense;
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
    pub fn sz<T: Scalar>(&self) -> Dense<T> {
        let half = real::<T>(0.5);
        let neg_half = real::<T>(-0.5);
        let z = T::zero();
        // CM layout of [[0.5, 0], [0, -0.5]]: col0=[0.5, 0], col1=[0, -0.5]
        Dense::new(vec![half, z, z, neg_half], vec![2, 2])
    }

    /// S+ (raising operator) = [[0, 1], [0, 0]]
    pub fn sp<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 1], [0, 0]]: col0=[0, 0], col1=[1, 0]
        Dense::new(vec![z, z, o, z], vec![2, 2])
    }

    /// S- (lowering operator) = [[0, 0], [1, 0]]
    pub fn sm<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 0], [1, 0]]: col0=[0, 1], col1=[0, 0]
        Dense::new(vec![z, o, z, z], vec![2, 2])
    }

    /// Identity operator = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 1]]: col0=[1, 0], col1=[0, 1]
        Dense::new(vec![o, z, z, o], vec![2, 2])
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
    pub fn x<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 1], [1, 0]]: col0=[0, 1], col1=[1, 0]
        Dense::new(vec![z, o, o, z], vec![2, 2])
    }

    /// Pauli Y = [[0, -i], [i, 0]]
    pub fn y<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let zi = T::Real::zero();
        let one = real_val::<T::Real>(1.0);
        let neg_i = T::from_real_imag(zi, real_val::<T::Real>(-1.0));
        let pos_i = T::from_real_imag(zi, one);
        // CM layout of [[0, -i], [i, 0]]: col0=[0, i], col1=[-i, 0]
        Dense::new(vec![z, pos_i, neg_i, z], vec![2, 2])
    }

    /// Pauli Z = [[1, 0], [0, -1]]
    pub fn z<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        let neg = real::<T>(-1.0);
        // CM layout of [[1, 0], [0, -1]]: col0=[1, 0], col1=[0, -1]
        Dense::new(vec![o, z, z, neg], vec![2, 2])
    }

    /// Hadamard = [[1, 1], [1, -1]] / sqrt(2)
    pub fn h<T: Scalar>(&self) -> Dense<T> {
        let inv_sqrt2 = real::<T>(std::f64::consts::FRAC_1_SQRT_2);
        let neg_inv_sqrt2 = real::<T>(-std::f64::consts::FRAC_1_SQRT_2);
        // CM layout of [[h, h], [h, -h]]: col0=[h, h], col1=[h, -h]
        Dense::new(
            vec![inv_sqrt2, inv_sqrt2, inv_sqrt2, neg_inv_sqrt2],
            vec![2, 2],
        )
    }

    /// S gate = [[1, 0], [0, i]]
    pub fn s<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        let i = T::from_real_imag(T::Real::zero(), real_val::<T::Real>(1.0));
        // CM layout of [[1, 0], [0, i]]: col0=[1, 0], col1=[0, i]
        Dense::new(vec![o, z, z, i], vec![2, 2])
    }

    /// T gate = [[1, 0], [0, exp(iπ/4)]]
    pub fn t<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        let angle = std::f64::consts::FRAC_PI_4;
        let t_val = T::from_real_imag(
            real_val::<T::Real>(angle.cos()),
            real_val::<T::Real>(angle.sin()),
        );
        // CM layout of [[1, 0], [0, t]]: col0=[1, 0], col1=[0, t]
        Dense::new(vec![o, z, z, t_val], vec![2, 2])
    }

    /// Identity = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 1]]: col0=[1, 0], col1=[0, 1]
        Dense::new(vec![o, z, z, o], vec![2, 2])
    }

    /// |0⟩⟨0| = [[1, 0], [0, 0]]
    pub fn proj0<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 0]]: col0=[1, 0], col1=[0, 0]
        Dense::new(vec![o, z, z, z], vec![2, 2])
    }

    /// |1⟩⟨1| = [[0, 0], [0, 1]]
    pub fn proj1<T: Scalar>(&self) -> Dense<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 0], [0, 1]]: col0=[0, 0], col1=[0, 1]
        Dense::new(vec![z, z, z, o], vec![2, 2])
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
