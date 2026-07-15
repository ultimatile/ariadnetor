use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{
    BackendError, ComputeBackend, DeviceType, GemmDescriptor, MemoryOrder, QrDescriptor,
    SolveDescriptor, TransposeDescriptor,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseStorage, DenseTensor, DenseTensorData, Host, OpsFor};
use num_traits::{Float, NumCast, One, Zero};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::{IncrementalQr, QrAppendOutcome};
use crate::{LinalgError, inverse_with_backend, qr, tensordot};

/// Deterministic seeded random matrix with entries in `[-1, 1)`. The
/// imaginary part is dropped for real scalars by `from_real_imag`, so the
/// same helper exercises complex entries with genuinely nonzero phases.
fn pseudo_random<T: Scalar>(nrows: usize, ncols: usize, seed: u64) -> DenseTensor<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut next = move || rng.random::<f64>() * 2.0 - 1.0;
    let mut m = DenseTensor::<T>::zeros(vec![nrows, ncols]);
    for i in 0..nrows {
        for j in 0..ncols {
            let re = <T::Real as NumCast>::from(next()).expect("f64 sample fits the real type");
            let im = <T::Real as NumCast>::from(next()).expect("f64 sample fits the real type");
            m.set([i, j], T::from_real_imag(re, im));
        }
    }
    m
}

/// Native backend that serves `solves_allowed` linear solves and then fails
/// every later one. The triangular inversion in `append`'s inverse
/// maintenance is the only solve on that path, so arming it lets a chosen
/// append fail after its orthogonalization has already succeeded — the
/// error path where a mutate-then-compute order would corrupt the state.
struct SolveFailingBackend {
    inner: NativeBackend,
    solves_allowed: usize,
    seen: std::sync::Mutex<usize>,
}

impl ComputeBackend for SolveFailingBackend {
    fn name(&self) -> &'static str {
        "solve-failing"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn preferred_order(&self) -> MemoryOrder {
        self.inner.preferred_order()
    }

    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.gemm(desc)
    }

    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.transpose(desc)
    }

    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.qr(desc)
    }

    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        let mut seen = self.seen.lock().expect("solve counter is never poisoned");
        if *seen >= self.solves_allowed {
            return Err(BackendError::ExecutionFailed(
                "injected solve failure".into(),
            ));
        }
        *seen += 1;
        self.inner.solve(desc)
    }
}

impl<T: Scalar> OpsFor<DenseStorage<T>> for SolveFailingBackend {}

/// Backend that reports the opposite memory order to the host's, standing
/// in for a second backend whose kernels emit a different layout.
struct FlippedOrderBackend {
    inner: NativeBackend,
}

impl ComputeBackend for FlippedOrderBackend {
    fn name(&self) -> &'static str {
        "flipped-order"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn preferred_order(&self) -> MemoryOrder {
        match self.inner.preferred_order() {
            MemoryOrder::RowMajor => MemoryOrder::ColumnMajor,
            MemoryOrder::ColumnMajor => MemoryOrder::RowMajor,
        }
    }

    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.gemm(desc)
    }

    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.transpose(desc)
    }
}

impl<T: Scalar> OpsFor<DenseStorage<T>> for FlippedOrderBackend {}

/// `a - b` for scalars without a `Sub` bound (`Scalar` carries none).
fn sub<T: Scalar>(a: T, b: T) -> T {
    a + b.scale_real(-T::Real::one())
}

fn tol<T: Scalar>() -> T::Real {
    let hundred = <T::Real as NumCast>::from(100.0).expect("100 fits the real type");
    T::Real::epsilon() * hundred
}

/// Frobenius distance between two equal-shape matrices, via the
/// order-aware accessor so mixed memory orders compare correctly.
fn frob_dist<T: Scalar>(a: &DenseTensor<T>, b: &DenseTensor<T>) -> T::Real {
    assert_eq!(a.shape(), b.shape());
    let mut acc = T::Real::zero();
    for i in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            let d = sub(a.get([i, j]), b.get([i, j])).abs();
            acc = acc + d * d;
        }
    }
    acc.sqrt()
}

fn assert_orthonormal<T: Scalar>(q: &DenseTensor<T>, msg: &str) {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let gram = tensordot(backend, &q.conj(), q, &[0], &[0]).expect("gram contraction");
    let k = q.shape()[1];
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { T::one() } else { T::zero() };
            assert!(
                sub(gram.get([i, j]), expected).abs() <= tol::<T>(),
                "{msg}: gram deviates from identity at [{i},{j}]"
            );
        }
    }
}

/// `Q (Q^H Y) = Y` — every appended column lies in the span of `Q`.
fn assert_spans<T: Scalar>(q: &DenseTensor<T>, y: &DenseTensor<T>, msg: &str) {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let coeff = tensordot(backend, &q.conj(), y, &[0], &[0]).expect("span projection");
    let proj = tensordot(backend, q, &coeff, &[1], &[0]).expect("span reconstruction");
    let scale = y.norm().max(T::Real::one());
    assert!(
        frob_dist(&proj, y) <= tol::<T>() * scale,
        "{msg}: projection residual exceeds tolerance"
    );
}

/// Multi-append equivalence against one full QR of the stacked blocks:
/// orthonormal Q spanning the stack, matching `|diag R|`, and inverse row
/// norms matching a direct inversion of the full triangular factor.
fn check_append_equals_full_qr<T: Scalar>() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let blocks = [
        pseudo_random::<T>(12, 3, 1),
        pseudo_random::<T>(12, 4, 2),
        pseudo_random::<T>(12, 2, 3),
    ];

    let mut inc = IncrementalQr::<T>::new(12, true);
    for b in &blocks {
        assert_eq!(
            inc.append(backend, b).expect("append"),
            QrAppendOutcome::FullRank
        );
    }
    assert_eq!(inc.ncols(), 9);

    let stacked = DenseTensor::from_data(DenseTensorData::concatenate(
        &[blocks[0].data(), blocks[1].data(), blocks[2].data()],
        1,
    ));
    let (_, r_full) = qr(backend, &stacked, 1).expect("full QR");

    // |diag R| is unique across QR factorizations of a full-rank matrix
    // (phases cancel in the modulus), so the incremental diagonal must
    // match the from-scratch one.
    for (i, d) in inc.r_diag.iter().enumerate() {
        assert!(
            Float::abs(*d - r_full.get([i, i]).abs()) <= tol::<T>(),
            "diag mismatch at {i}"
        );
    }

    // Row norms of R^-1 are phase-invariant too: the two factors differ
    // by a unit-modulus column scaling of the inverse.
    let g_full = inverse_with_backend(backend, &r_full, 1).expect("invert full R");
    let row_sq = inc.r_inverse_row_sq_norms().expect("tracking is on");
    for (i, got) in row_sq.iter().enumerate() {
        let mut want = T::Real::zero();
        for j in 0..9 {
            let x = g_full.get([i, j]).abs();
            want = want + x * x;
        }
        assert!(
            Float::abs(*got - want) <= tol::<T>() * want.max(T::Real::one()),
            "inverse row norm mismatch at row {i}"
        );
    }

    // Read the raw basis through the private field: the point is BCGS2's
    // own orthonormality quality, which the repair pass in
    // `into_orthonormal_q` would mask.
    let q_raw = inc.q.as_ref().expect("appended").clone();
    assert_eq!(q_raw.shape(), &[12, 9]);
    assert_orthonormal(&q_raw, "multi-append raw Q");
    assert_spans(&q_raw, &stacked, "multi-append raw span");
    // The terminal accessor repairs (here: re-factorizes) without changing
    // the span.
    let q = inc.into_orthonormal_q(backend).expect("terminal accessor");
    assert_orthonormal(&q, "multi-append Q");
    assert_spans(&q, &stacked, "multi-append span");
}

#[test]
fn append_equals_full_qr_f64() {
    check_append_equals_full_qr::<f64>();
}

#[test]
fn append_equals_full_qr_c64() {
    check_append_equals_full_qr::<num_complex::Complex<f64>>();
}

#[test]
fn first_append_matches_plain_qr() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let block = pseudo_random::<f64>(8, 3, 7);
    let mut inc = IncrementalQr::<f64>::new(8, true);
    inc.append(backend, &block).expect("append");
    let (q_plain, _) = qr(backend, &block, 1).expect("plain QR");
    let q = inc
        .into_orthonormal_q(backend)
        .expect("single append skips the repair pass");
    assert!(
        frob_dist(&q, &q_plain) <= tol::<f64>(),
        "first append must reduce to the plain thin QR"
    );
}

#[test]
fn later_append_rank_deficiency_detected_and_recoverable() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let a = pseudo_random::<f64>(8, 3, 11);
    // Columns of `b` are linear combinations of `a`'s, so the appended
    // block is entirely inside the existing span.
    let mix = pseudo_random::<f64>(3, 2, 12);
    let b = tensordot(backend, &a, &mix, &[1], &[0]).expect("dependent block");

    let mut inc = IncrementalQr::<f64>::new(8, true);
    assert_eq!(
        inc.append(backend, &a).expect("full-rank append"),
        QrAppendOutcome::FullRank
    );
    assert_eq!(
        inc.append(backend, &b).expect("deficient append"),
        QrAppendOutcome::RankDeficient
    );
    // The inverse state is stale by design after termination.
    assert!(inc.r_inverse_row_sq_norms().is_none());

    // The terminal accessor re-orthonormalizes the assembled basis, whose
    // span still contains every appended column.
    let q_fixed = inc
        .into_orthonormal_q(backend)
        .expect("terminal repair pass");
    assert_orthonormal(&q_fixed, "re-orthonormalized Q");
    assert_spans(&q_fixed, &a, "span keeps the full-rank block");
    assert_spans(&q_fixed, &b, "span keeps the deficient block");
}

#[test]
fn first_append_rank_deficiency_keeps_q_orthonormal() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    // Two identical columns: rank 1, deficient already on the first
    // append — which is a plain Householder QR, so Q stays orthonormal.
    let col = pseudo_random::<f64>(6, 1, 21);
    let block = DenseTensor::from_data(DenseTensorData::concatenate(&[col.data(), col.data()], 1));
    let mut inc = IncrementalQr::<f64>::new(6, true);
    assert_eq!(
        inc.append(backend, &block).expect("append"),
        QrAppendOutcome::RankDeficient
    );
    let q = inc
        .into_orthonormal_q(backend)
        .expect("single append skips the repair pass");
    assert_orthonormal(&q, "first-append deficient Q");
}

#[test]
#[should_panic(expected = "terminated IncrementalQr")]
fn append_after_termination_panics() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let col = pseudo_random::<f64>(6, 1, 22);
    let block = DenseTensor::from_data(DenseTensorData::concatenate(&[col.data(), col.data()], 1));
    let mut inc = IncrementalQr::<f64>::new(6, true);
    let _ = inc.append(backend, &block);
    let _ = inc.append(backend, &col);
}

#[test]
fn append_rejects_a_backend_with_a_different_order() {
    let host = Host::shared();
    let mut inc = IncrementalQr::<f64>::new(6, true);
    inc.append(host.as_ref(), &pseudo_random::<f64>(6, 2, 61))
        .expect("first append");
    let ncols = inc.ncols();

    // A second backend whose kernels emit the opposite order would have the
    // stored factors and the new block disagree on layout; the append must
    // be rejected up front rather than fail deep inside an assembly step.
    let other = FlippedOrderBackend {
        inner: NativeBackend::new(),
    };
    assert!(matches!(
        inc.append(&other, &pseudo_random::<f64>(6, 2, 62)),
        Err(LinalgError::InvalidArgument(_))
    ));
    assert_eq!(inc.ncols(), ncols, "a rejected append must not mutate");
    inc.append(host.as_ref(), &pseudo_random::<f64>(6, 2, 63))
        .expect("the original backend still works");
}

#[test]
fn tracking_off_reports_no_row_norms() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let mut inc = IncrementalQr::<f64>::new(6, false);
    assert!(inc.r_inverse_row_sq_norms().is_none());
    inc.append(backend, &pseudo_random::<f64>(6, 2, 31))
        .expect("append");
    assert!(inc.r_inverse_row_sq_norms().is_none());
    assert_eq!(inc.ncols(), 2);
}

#[test]
fn append_validates_block_shape() {
    let backend = Host::shared();
    let backend = backend.as_ref();
    let mut inc = IncrementalQr::<f64>::new(6, true);
    // Wrong row count.
    assert!(matches!(
        inc.append(backend, &pseudo_random::<f64>(5, 2, 41)),
        Err(LinalgError::InvalidArgument(_))
    ));
    // Not a matrix.
    let cube = DenseTensor::<f64>::zeros(vec![6, 1, 1]);
    assert!(matches!(
        inc.append(backend, &cube),
        Err(LinalgError::InvalidArgument(_))
    ));
    // No columns.
    let empty = DenseTensor::<f64>::zeros(vec![6, 0]);
    assert!(matches!(
        inc.append(backend, &empty),
        Err(LinalgError::InvalidArgument(_))
    ));
    // Column count overflowing the row count.
    assert!(matches!(
        inc.append(backend, &pseudo_random::<f64>(6, 7, 42)),
        Err(LinalgError::InvalidArgument(_))
    ));
    // A failed append must not corrupt the state.
    assert_eq!(inc.ncols(), 0);
    inc.append(backend, &pseudo_random::<f64>(6, 2, 43))
        .expect("valid append still works");
}

#[test]
fn backend_failure_mid_append_leaves_state_unchanged() {
    // One solve served, so the first append completes and the second fails
    // inverting its own triangular block.
    let backend = SolveFailingBackend {
        inner: NativeBackend::new(),
        solves_allowed: 1,
        seen: std::sync::Mutex::new(0),
    };
    let mut inc = IncrementalQr::<f64>::new(8, true);
    assert_eq!(
        inc.append(&backend, &pseudo_random::<f64>(8, 3, 51))
            .expect("the first append's solve is served"),
        QrAppendOutcome::FullRank
    );
    let ncols = inc.ncols();
    let row_sq = inc
        .r_inverse_row_sq_norms()
        .expect("tracking is on")
        .to_vec();
    let q_before = inc.q.as_ref().expect("appended").clone();
    let g_before = inc.g.as_ref().expect("tracking is on").clone();
    let diag_before = inc.r_diag.clone();
    let appends_before = inc.appends;

    // Orthogonalization succeeds, the inversion does not — the point where a
    // mutate-then-compute order would leave the factorization half-grown.
    assert!(matches!(
        inc.append(&backend, &pseudo_random::<f64>(8, 2, 52)),
        Err(LinalgError::Backend(_))
    ));

    assert_eq!(inc.ncols(), ncols, "column count must not grow on failure");
    assert_eq!(
        inc.r_inverse_row_sq_norms().expect("tracking still on"),
        row_sq.as_slice(),
        "inverse row norms must not change on failure"
    );
    assert_eq!(
        frob_dist(inc.q.as_ref().expect("appended"), &q_before),
        0.0,
        "the basis must not absorb a failed block"
    );
    // The private half of the state too: a mutation here would corrupt
    // every later estimate or repair decision without moving `ncols`.
    assert_eq!(
        frob_dist(inc.g.as_ref().expect("tracking is on"), &g_before),
        0.0,
        "the maintained inverse must not change on failure"
    );
    assert_eq!(inc.r_diag, diag_before, "the diagonal must not grow");
    assert_eq!(
        inc.appends, appends_before,
        "the append count must not grow"
    );
    assert!(!inc.terminated, "a backend failure must not terminate");
    // The factorization stays usable: a later append on a working backend
    // proceeds as if the failure never happened.
    let host = Host::shared();
    inc.append(host.as_ref(), &pseudo_random::<f64>(8, 2, 53))
        .expect("state is intact after the failed append");
    assert_eq!(inc.ncols(), ncols + 2);
}
