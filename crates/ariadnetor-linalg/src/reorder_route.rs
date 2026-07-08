//! Route a cross-order memory reorder through the backend transpose.
//!
//! The linalg row-major sandwich normalizes operands across memory order with
//! `ariadnetor_tensor::reorder_data`, a backend-less per-element loop that
//! never reaches HPTT. A cross-order reorder at fixed logical shape is a
//! physical axis-reversal permutation, so the backend transpose computes the
//! same bytes while using HPTT under `--features hptt`.

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{ComputeBackend, MemoryOrder, TransposeDescriptor};
use ariadnetor_tensor::DenseTensorData;

use crate::error::LinalgError;

/// Convert `tensor` to memory order `to` at fixed logical shape, routing the
/// conversion through `backend.transpose`.
///
/// A reorder from `from` to the opposite order keeps the logical index but
/// moves each element from its `from`-layout slot to its `to`-layout slot —
/// exactly `backend.transpose` with the reverse permutation and the SOURCE
/// order. That path uses HPTT for f64 / f32 / complex under `--features hptt`
/// (and the native naive kernel, which can parallelize, otherwise), where
/// `reorder_data` is always a sequential per-element loop. The transpose emits
/// the buffer for the reversed shape; it is re-wrapped under the original
/// logical shape and `to`, which is byte-identical to `reorder_data(tensor, to)`.
///
/// Rank ≤ 1 and `from == to` need no data movement and skip the transpose.
pub(crate) fn reorder_via_backend<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensorData<T>,
    to: MemoryOrder,
) -> Result<DenseTensorData<T>, LinalgError> {
    let from = tensor.order();
    if from == to {
        return Ok(tensor.clone());
    }
    let shape = tensor.shape().to_vec();
    // A rank-0 / rank-1 tensor has one byte layout shared by both orders, so
    // re-tagging the order is correct without moving data; this also keeps a
    // degenerate perm off the HPTT path.
    if shape.len() <= 1 {
        return Ok(DenseTensorData::from_raw_parts(
            tensor.data().to_vec(),
            shape,
            to,
        ));
    }
    let total = tensor.len();
    if total == 0 {
        return Ok(DenseTensorData::from_raw_parts(Vec::new(), shape, to));
    }
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    let policy = backend.par_for_transpose(&shape);
    let mut output = vec![T::zero(); total];
    backend.transpose(TransposeDescriptor {
        input: tensor.data(),
        output: &mut output,
        shape: &shape,
        perm: &perm,
        order: from,
        conj: false,
        policy,
    })?;
    Ok(DenseTensorData::from_raw_parts(output, shape, to))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariadnetor_core::Complex;
    use ariadnetor_native::NativeBackend;
    use ariadnetor_tensor::reorder_data;

    /// The routed reorder must be byte-identical to the naive `reorder_data`
    /// for every rank, order pair, and scalar. Distinct per-slot values make a
    /// wrong permutation (rather than a mere layout re-tag) observable. Run
    /// under both the default build (native naive kernel) and `--features hptt`
    /// (HPTT) — the two are distinct compilations that must both agree.
    fn check<T>(mk: impl Fn(usize) -> T)
    where
        T: Scalar + PartialEq + std::fmt::Debug,
    {
        let backend = NativeBackend::new();
        // Ranks 1–4 with non-symmetric extents so an axis-mismapping perm is
        // not masked by equal dimensions. `[2, 0, 3]` is rank>1 with zero
        // total, exercising the empty-tensor branch (which the rank≤1 fast
        // path would otherwise mask for a `[0]` shape).
        let shapes: &[&[usize]] = &[&[5], &[3, 4], &[2, 3, 4], &[2, 3, 2, 4], &[2, 0, 3]];
        let orders = [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor];
        for &shape in shapes {
            let total: usize = shape.iter().product();
            for &src in &orders {
                for &dst in &orders {
                    let data: Vec<T> = (0..total).map(&mk).collect();
                    let t = DenseTensorData::from_raw_parts(data, shape.to_vec(), src);
                    let expected = reorder_data(&t, dst);
                    let got = reorder_via_backend(&backend, &t, dst).unwrap();
                    assert_eq!(
                        got.data(),
                        expected.data(),
                        "data mismatch: shape={shape:?} {src:?}->{dst:?}"
                    );
                    assert_eq!(got.shape(), expected.shape(), "shape tag {shape:?}");
                    assert_eq!(got.order(), expected.order(), "order tag {shape:?}");
                }
            }
        }
    }

    #[test]
    fn byte_identity_f64() {
        check(|i| i as f64);
    }

    #[test]
    fn byte_identity_f32() {
        check(|i| i as f32);
    }

    #[test]
    fn byte_identity_c64() {
        check(|i| Complex::new(i as f64, (2 * i + 1) as f64));
    }

    #[test]
    fn byte_identity_c32() {
        check(|i| Complex::new(i as f32, (2 * i + 1) as f32));
    }
}
