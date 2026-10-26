//! Truncation targets of the dense truncated SVD: the relative target's
//! reference norm and the validation of target values.

use ariadnetor_linalg::{DenseHostOps, LinalgError, TruncErrTarget, TruncSvdParams};
use ariadnetor_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Square diagonal matrix, whose singular values are its diagonal entries.
/// A diagonal entry sits at index `i * n + i` in either memory order, so
/// the buffer is laid out column-major directly.
fn diag(entries: &[f64]) -> DenseTensor<f64> {
    let n = entries.len();
    let mut data = vec![0.0; n * n];
    for (i, &e) in entries.iter().enumerate() {
        data[i * n + i] = e;
    }
    DenseTensor::from_data(DenseTensorData::from_raw_parts(
        data,
        vec![n, n],
        MemoryOrder::ColumnMajor,
    ))
}

fn target(target: TruncErrTarget) -> TruncSvdParams {
    TruncSvdParams {
        chi_max: None,
        target_trunc_err: Some(target),
    }
}

/// `diag(4, 2, 1)` has norm `sqrt(21) ≈ 4.583`. `Relative(0.25)` is a
/// budget of about 1.146: it discards the 1 and stops at `sqrt(5)`.
/// `Relative(0.2)` (about 0.917) and `Absolute(0.25)` keep all three.
#[test]
fn relative_target_is_measured_against_the_matrix_norm() {
    for scale in [1.0, 1e3] {
        let a = diag(&[4.0 * scale, 2.0 * scale, scale]);
        let (_, s, _, err) = a
            .trunc_svd(1, &target(TruncErrTarget::Relative(0.25)))
            .unwrap();
        assert_eq!(s.len(), 2, "scale {scale}");
        assert!((err - scale).abs() < 1e-12 * scale, "scale {scale}: {err}");

        for kept_all in [
            TruncErrTarget::Relative(0.2),
            TruncErrTarget::Absolute(0.25),
        ] {
            let (_, s, _, err) = a.trunc_svd(1, &target(kept_all)).unwrap();
            assert_eq!(s.len(), 3, "scale {scale}, {kept_all:?}");
            assert_eq!(err, 0.0);
        }
    }
}

/// The norm of this spectrum overflows `f64`; the relative budget, about
/// `2.1e298`, still keeps the `1e300`.
#[test]
fn relative_target_survives_an_overflowing_matrix_norm() {
    let a = diag(&[1.5e308, 1.5e308, 1e300]);
    let (_, s, _, err) = a
        .trunc_svd(1, &target(TruncErrTarget::Relative(1e-10)))
        .unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(err, 0.0);
}

#[test]
fn invalid_target_values_are_rejected() {
    let a = diag(&[4.0, 2.0, 1.0]);
    for value in [f64::NAN, -0.1, f64::INFINITY, f64::NEG_INFINITY] {
        for t in [
            TruncErrTarget::Absolute(value),
            TruncErrTarget::Relative(value),
        ] {
            let result = a.trunc_svd(1, &target(t));
            assert!(
                matches!(&result, Err(LinalgError::InvalidArgument(msg)) if msg.contains("target_trunc_err")),
                "{t:?}: {:?}",
                result.err()
            );
        }
    }
}

#[test]
fn zero_targets_are_accepted() {
    let a = diag(&[4.0, 2.0, 1.0]);
    for t in [
        TruncErrTarget::Absolute(0.0),
        TruncErrTarget::Absolute(-0.0),
        TruncErrTarget::Relative(0.0),
    ] {
        let (_, s, _, _) = a.trunc_svd(1, &target(t)).unwrap();
        assert_eq!(s.len(), 3, "{t:?}");
    }
}

/// A matrix with a zero dimension has no singular values, and a target
/// keeps none rather than flooring the count past the spectrum.
#[test]
fn target_on_an_empty_spectrum_keeps_nothing() {
    let a: DenseTensor<f64> = DenseTensor::from_data(DenseTensorData::from_raw_parts(
        Vec::new(),
        vec![0, 3],
        MemoryOrder::ColumnMajor,
    ));
    for t in [TruncErrTarget::Absolute(0.1), TruncErrTarget::Relative(0.1)] {
        let (_, s, _, err) = a.trunc_svd(1, &target(t)).unwrap();
        assert_eq!(s.len(), 0, "{t:?}");
        assert_eq!(err, 0.0);
    }
}

/// Parameters are checked before the decomposition, so an invalid target
/// is reported even where the decomposition itself would fail on `nrow`.
#[test]
fn invalid_parameters_are_reported_before_the_decomposition() {
    let a = diag(&[4.0, 2.0, 1.0]);
    let result = a.trunc_svd(0, &target(TruncErrTarget::Relative(f64::NAN)));
    assert!(
        matches!(&result, Err(LinalgError::InvalidArgument(msg)) if msg.contains("target_trunc_err")),
        "{:?}",
        result.err()
    );
}
