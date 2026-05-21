//! SiteOps trait and concrete operator dictionaries.
//!
//! Operator data is stored in the active backend's preferred memory
//! order ([`NativeBackend`]'s preferred is `ColumnMajor`). For a 2×2
//! matrix `[[a, b], [c, d]]`, the flat layout under column-major is
//! `[a, c, b, d]`.

use arnet::{DenseTensor, MemoryOrder, NativeBackend, Scalar};
use num_traits::{NumCast, Zero};

/// Trait for site-local operator dictionaries.
///
/// Provides the physical dimension of the local Hilbert space.
/// Concrete types (e.g., [`SpinHalf`]) supply type-safe methods
/// returning specific operators as dense tensors.
pub trait SiteOps {
    /// Physical dimension of the local Hilbert space.
    fn dim(&self) -> usize;
}

/// Build a 2×2 column-major DenseTensor pinned to NativeBackend.
///
/// `data` must be the four entries in column-major order: `[m(0,0),
/// m(1,0), m(0,1), m(1,1)]`. NativeBackend's preferred order is
/// ColumnMajor, so the resulting tensor satisfies the Tier 1
/// invariant immediately.
fn make_2x2_cm<T: Scalar>(data: Vec<T>) -> DenseTensor<T> {
    debug_assert_eq!(data.len(), 4, "make_2x2_cm: expected 4 entries");
    DenseTensor::from_raw_parts(
        data,
        vec![2, 2],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    )
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
        // CM layout of [[0.5, 0], [0, -0.5]]: col0=[0.5, 0], col1=[0, -0.5]
        make_2x2_cm(vec![half, z, z, neg_half])
    }

    /// S+ (raising operator) = [[0, 1], [0, 0]]
    pub fn sp<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 1], [0, 0]]: col0=[0, 0], col1=[1, 0]
        make_2x2_cm(vec![z, z, o, z])
    }

    /// S- (lowering operator) = [[0, 0], [1, 0]]
    pub fn sm<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 0], [1, 0]]: col0=[0, 1], col1=[0, 0]
        make_2x2_cm(vec![z, o, z, z])
    }

    /// Identity operator = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 1]]: col0=[1, 0], col1=[0, 1]
        make_2x2_cm(vec![o, z, z, o])
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
        // CM layout of [[0, 1], [1, 0]]: col0=[0, 1], col1=[1, 0]
        make_2x2_cm(vec![z, o, o, z])
    }

    /// Pauli Y = [[0, -i], [i, 0]]
    pub fn y<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let zi = T::Real::zero();
        let one = real_val::<T::Real>(1.0);
        let neg_i = T::from_real_imag(zi, real_val::<T::Real>(-1.0));
        let pos_i = T::from_real_imag(zi, one);
        // CM layout of [[0, -i], [i, 0]]: col0=[0, i], col1=[-i, 0]
        make_2x2_cm(vec![z, pos_i, neg_i, z])
    }

    /// Pauli Z = [[1, 0], [0, -1]]
    pub fn z<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        let neg = real::<T>(-1.0);
        // CM layout of [[1, 0], [0, -1]]: col0=[1, 0], col1=[0, -1]
        make_2x2_cm(vec![o, z, z, neg])
    }

    /// Hadamard = [[1, 1], [1, -1]] / sqrt(2)
    pub fn h<T: Scalar>(&self) -> DenseTensor<T> {
        let inv_sqrt2 = real::<T>(std::f64::consts::FRAC_1_SQRT_2);
        let neg_inv_sqrt2 = real::<T>(-std::f64::consts::FRAC_1_SQRT_2);
        // CM layout of [[h, h], [h, -h]]: col0=[h, h], col1=[h, -h]
        make_2x2_cm(vec![inv_sqrt2, inv_sqrt2, inv_sqrt2, neg_inv_sqrt2])
    }

    /// S gate = [[1, 0], [0, i]]
    pub fn s<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        let i = T::from_real_imag(T::Real::zero(), real_val::<T::Real>(1.0));
        // CM layout of [[1, 0], [0, i]]: col0=[1, 0], col1=[0, i]
        make_2x2_cm(vec![o, z, z, i])
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
        // CM layout of [[1, 0], [0, t]]: col0=[1, 0], col1=[0, t]
        make_2x2_cm(vec![o, z, z, t_val])
    }

    /// Identity = [[1, 0], [0, 1]]
    pub fn id<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 1]]: col0=[1, 0], col1=[0, 1]
        make_2x2_cm(vec![o, z, z, o])
    }

    /// |0⟩⟨0| = [[1, 0], [0, 0]]
    pub fn proj0<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[1, 0], [0, 0]]: col0=[1, 0], col1=[0, 0]
        make_2x2_cm(vec![o, z, z, z])
    }

    /// |1⟩⟨1| = [[0, 0], [0, 1]]
    pub fn proj1<T: Scalar>(&self) -> DenseTensor<T> {
        let z = T::zero();
        let o = T::one();
        // CM layout of [[0, 0], [0, 1]]: col0=[0, 0], col1=[0, 1]
        make_2x2_cm(vec![z, z, z, o])
    }
}

/// Convert an f64 value to a Scalar type via its Real component.
fn real<T: Scalar>(val: f64) -> T {
    let r: T::Real = NumCast::from(val).unwrap();
    T::from_real_imag(r, T::Real::zero())
}

/// Convert an f64 value to any type that implements [`NumCast`].
fn real_val<R: NumCast>(val: f64) -> R {
    NumCast::from(val).unwrap()
}
