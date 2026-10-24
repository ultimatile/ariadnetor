//! Hermiticity check for the eigendecomposition-based exponentials.
//!
//! `eigh_dense` assumes a Hermitian matrix and does not check it, so the
//! Hermitian and anti-Hermitian exponentials validate their operand here
//! before decomposing it. The anti-Hermitian kernel passes `iA`, which is
//! Hermitian exactly when `A` is anti-Hermitian, with the same Frobenius norms.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_core::scale_safe_norm;
use ariadnetor_tensor::DenseTensorData;
use num_traits::{Float, ToPrimitive};

use crate::error::LinalgError;

/// Reject `matrix` unless `‖A‖_F` is finite and `‖A − A†‖_F ≤ √ε · ‖A‖_F`,
/// with `ε` the machine epsilon of `T::Real`.
///
/// `matrix` is the operand flattened to a row-major `n × n` matrix. A NaN or
/// infinite entry, or a norm that overflows, makes `‖A‖_F` non-finite.
/// `requirement` names the violated precondition in the error message.
///
/// # Panics
///
/// Panics if `matrix` is not row-major or does not hold `n * n` elements.
pub(super) fn ensure_hermitian<T: Scalar>(
    matrix: &DenseTensorData<T>,
    n: usize,
    requirement: &str,
) -> Result<(), LinalgError> {
    assert_eq!(
        matrix.order(),
        MemoryOrder::RowMajor,
        "ensure_hermitian: matrix must be row-major",
    );
    let a = matrix.data();
    assert_eq!(a.len(), n * n, "ensure_hermitian: matrix must be n x n");

    let norm = matrix.norm_frobenius();
    if !norm.is_finite() {
        return Err(LinalgError::InvalidArgument(format!(
            "{requirement}: the Frobenius norm is not finite"
        )));
    }
    // Real and imaginary parts of `A[i, j] - conj(A[j, i])`, summed as components.
    let residual = scale_safe_norm((0..n).flat_map(|i| {
        (0..n).flat_map(move |j| {
            let (x, y) = (a[i * n + j], a[j * n + i]);
            [x.re() - y.re(), x.im() + y.im()]
        })
    }));
    let bound = T::Real::epsilon().sqrt() * norm;
    if residual > bound {
        let as_f64 = |x: T::Real| x.to_f64().unwrap_or(f64::NAN);
        return Err(LinalgError::InvalidArgument(format!(
            "{requirement}: deviation {} exceeds the tolerance {} \
             (sqrt(epsilon) times the Frobenius norm)",
            as_f64(residual),
            as_f64(bound),
        )));
    }
    Ok(())
}
