//! Hermiticity check for the eigendecomposition-based exponentials.
//!
//! `eigh_dense` assumes a Hermitian matrix and does not check it, so the
//! Hermitian and anti-Hermitian exponentials validate their operand here
//! before decomposing it. The anti-Hermitian kernel passes `iA`, which is
//! Hermitian exactly when `A` is anti-Hermitian, with the same Frobenius norms.

use std::cmp::Ordering;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::MemoryOrder;
use ariadnetor_core::scale_safe_norm;
use ariadnetor_tensor::DenseTensorData;
use num_traits::{Float, ToPrimitive};

use crate::error::LinalgError;

/// Reject `matrix` unless `‖A − A†‖_F ≤ √ε · ‖A‖_F`, with `ε` the machine
/// epsilon of `T::Real`.
///
/// `matrix` is the operand flattened to a row-major `n × n` matrix. A NaN in
/// either norm rejects the input.
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

    // Real and imaginary parts of `A[i, j] - conj(A[j, i])`, summed as components.
    let residual = scale_safe_norm((0..n).flat_map(|i| {
        (0..n).flat_map(move |j| {
            let (x, y) = (a[i * n + j], a[j * n + i]);
            [x.re() - y.re(), x.im() + y.im()]
        })
    }));
    let bound = T::Real::epsilon().sqrt() * matrix.norm_frobenius();
    // `partial_cmp` is `None` when either side is NaN, which rejects.
    if !matches!(
        residual.partial_cmp(&bound),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        let (residual, bound) = (residual.to_f64(), bound.to_f64());
        return Err(LinalgError::InvalidArgument(format!(
            "{requirement}: deviation {residual:?} exceeds the tolerance {bound:?} \
             (sqrt(epsilon) times the Frobenius norm)"
        )));
    }
    Ok(())
}
