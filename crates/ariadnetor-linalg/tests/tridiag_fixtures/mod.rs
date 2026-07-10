//! Shared tridiagonal fixture helpers for the `tridiag_eigen`
//! integration test and the `tridiag_eigh` bench (which includes this
//! file via `#[path]`), so the bench measures exactly the matrix class
//! the contract tests verify — a drift in one target cannot silently
//! diverge from the other.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};
use num_traits::Float;

/// Deterministic non-trivial fixture: no zero subdiagonal entries, no
/// repeated diagonal values.
pub(crate) fn fixture<T: Float>(n: usize) -> (Vec<T>, Vec<T>) {
    let d = (0..n)
        .map(|i| T::from(2.0 + (i as f64 * 0.7).sin()).unwrap())
        .collect();
    let e = (0..n - 1)
        .map(|i| T::from(0.5 + 0.3 * (i as f64 * 1.3).cos()).unwrap())
        .collect();
    (d, e)
}

/// Assemble the dense column-major matrix defined by diagonal `d` and
/// subdiagonal `e`, as a `DenseTensor` input for the dense-eigh oracle.
pub(crate) fn assemble_dense<T: Scalar>(d: &[T], e: &[T]) -> DenseTensor<T> {
    let n = d.len();
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i + n * i] = d[i];
        if i + 1 < n {
            data[(i + 1) + n * i] = e[i];
            data[i + n * (i + 1)] = e[i];
        }
    }
    DenseTensor::from_data(DenseTensorData::from_raw_parts(
        data,
        vec![n, n],
        MemoryOrder::ColumnMajor,
    ))
}
