//! (Anti-)Hermiticity check for the eigendecomposition-based exponentials.
//!
//! `eigh_dense` assumes a Hermitian matrix and does not check it, so the
//! Hermitian and anti-Hermitian exponentials validate their operand before
//! decomposing it. The check is split in two so that a block-sparse operator,
//! whose sector blocks are exponentiated one by one, can combine the
//! deviations of its sectors and judge the operator as a whole.

use std::cmp::Ordering;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_core::scale_safe_norm;
use ariadnetor_tensor::DenseTensorData;
use num_traits::{Float, ToPrimitive};

use crate::error::LinalgError;

/// Which adjoint relation an operand must satisfy.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Adjoint {
    /// `A† = A`.
    Hermitian,
    /// `A† = -A`.
    AntiHermitian,
}

/// `‖A − A†‖_F` for [`Adjoint::Hermitian`] or `‖A + A†‖_F` for
/// [`Adjoint::AntiHermitian`], of the row-major `n × n` matrix `matrix`.
///
/// # Panics
///
/// Panics if `matrix` is not row-major or does not hold `n * n` elements.
pub(crate) fn adjoint_deviation<T: Scalar>(
    matrix: &DenseTensorData<T>,
    n: usize,
    kind: Adjoint,
) -> T::Real {
    assert_eq!(
        matrix.order(),
        MemoryOrder::RowMajor,
        "adjoint_deviation: matrix must be row-major",
    );
    let a = matrix.data();
    assert_eq!(a.len(), n * n, "adjoint_deviation: matrix must be n x n");

    // Real and imaginary parts of `A[i, j] ∓ conj(A[j, i])`, summed as components.
    scale_safe_norm((0..n).flat_map(|i| {
        (0..n).flat_map(move |j| {
            let (x, y) = (a[i * n + j], a[j * n + i]);
            match kind {
                Adjoint::Hermitian => [x.re() - y.re(), x.im() + y.im()],
                Adjoint::AntiHermitian => [x.re() + y.re(), x.im() - y.im()],
            }
        })
    }))
}

/// Reject the operand unless its Frobenius norm `norm` is finite and
/// `deviation ≤ √ε · norm`, with `ε` the machine epsilon of `T::Real`. A NaN
/// or infinite entry, or a norm that overflows, makes `norm` non-finite. `op`
/// names the operation in the error message.
pub(crate) fn ensure_within_tolerance<T: Scalar>(
    deviation: T::Real,
    norm: T::Real,
    op: &str,
    kind: Adjoint,
) -> Result<(), LinalgError> {
    if !norm.is_finite() {
        return Err(LinalgError::InvalidArgument(format!(
            "{op} requires an operand with a finite Frobenius norm"
        )));
    }
    let bound = T::Real::epsilon().sqrt() * norm;
    // `partial_cmp` is `None` for a NaN deviation, which rejects.
    if !matches!(
        deviation.partial_cmp(&bound),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        let as_f64 = |x: T::Real| x.to_f64().unwrap_or(f64::NAN);
        let property = match kind {
            Adjoint::Hermitian => "a Hermitian",
            Adjoint::AntiHermitian => "an anti-Hermitian",
        };
        return Err(LinalgError::InvalidArgument(format!(
            "{op} requires {property} matrix: deviation {} exceeds the tolerance {} \
             (sqrt(epsilon) times the Frobenius norm)",
            as_f64(deviation),
            as_f64(bound),
        )));
    }
    Ok(())
}

/// [`ensure_within_tolerance`] for a single row-major `n × n` matrix.
pub(super) fn ensure_adjoint<T: Scalar>(
    matrix: &DenseTensorData<T>,
    n: usize,
    op: &str,
    kind: Adjoint,
) -> Result<(), LinalgError> {
    ensure_within_tolerance::<T>(
        adjoint_deviation(matrix, n, kind),
        matrix.norm_frobenius(),
        op,
        kind,
    )
}
