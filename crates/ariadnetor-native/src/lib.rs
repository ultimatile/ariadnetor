//! CPU compute backend for ariadnetor
//!
//! Provides [`NativeBackend`] implementing `ComputeBackend` via:
//! - **GEMM**: faer (f64, f32, `Complex<f64>`, `Complex<f32>`)
//! - **SVD/QR/LQ/EIGH**: faer (f64, f32, `Complex<f64>`, `Complex<f32>`)
//! - **Tridiagonal EIGH**: faer (f64, f32; the eigensystem of a real
//!   symmetric tridiagonal matrix is real, so complex scalars are rejected)
//! - **Transpose**: HPTT (f64, f32, Complex) when the `hptt` feature is on, a naive kernel otherwise

#![deny(missing_docs)]

mod eig;
mod eigh;
mod gemm;
mod lq;
mod performance;
mod qr;
mod solve;
mod svd;
mod transpose;
mod tridiag_eigh;

use std::sync::{Arc, OnceLock};

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{
    BackendError, ComputeBackend, DeviceType, EigDescriptor, EighDescriptor, ExecPolicy,
    GemmDescriptor, LqDescriptor, MemoryOrder, OpDesc, QrDescriptor, ScalarKernels,
    SolveDescriptor, SvdDescriptor, TransposeDescriptor, TridiagEighDescriptor,
};
use num_complex::Complex;

pub use performance::{PerformanceManager, ThresholdTable};

/// Map an [`ExecPolicy`] to faer's per-call parallelism selector.
///
/// `Parallel(0)` defers to faer's Rayon default (current thread pool size).
/// `Parallel(n)` for `n > 0` is an advisory thread-count hint passed to
/// `faer::Par::rayon(n)`; faer dispatches on the global Rayon pool, so
/// `n` influences work partitioning rather than guaranteeing exactly
/// `n` OS threads. The naive transpose kernel honors `n` with the same
/// semantics.
pub(crate) fn to_faer_par(policy: ExecPolicy) -> faer::Par {
    match policy {
        ExecPolicy::Sequential => faer::Par::Seq,
        ExecPolicy::Parallel(0) => faer::Par::rayon(0),
        ExecPolicy::Parallel(n) => faer::Par::rayon(n),
    }
}

/// Native backend using faer for GEMM and, with the `hptt` feature, HPTT for
/// transpose (a naive kernel otherwise).
///
/// This is the sole owner of faer and hptt-rs dependencies in the workspace.
/// Other crates access these capabilities through the `ComputeBackend` trait.
/// Holds a `PerformanceManager` that drives the `par_for_*` dispatch
/// decisions for each op based on a hardware-aware threshold table.
#[derive(Debug, Clone)]
pub struct NativeBackend {
    perf: PerformanceManager,
}

impl NativeBackend {
    /// Construct a `NativeBackend` with thresholds auto-detected from the
    /// current machine via `ThresholdTable::detect()`.
    pub fn new() -> Self {
        Self {
            perf: PerformanceManager::new(ThresholdTable::detect()),
        }
    }

    /// Construct a `NativeBackend` with a user-supplied `PerformanceManager`.
    ///
    /// Use this to override the auto-detected threshold table, e.g. to pin
    /// the laptop profile on a workstation for reproducible benchmarks.
    pub fn with_perf(perf: PerformanceManager) -> Self {
        Self { perf }
    }

    /// Borrow the `PerformanceManager` driving this backend's dispatch.
    pub fn perf(&self) -> &PerformanceManager {
        &self.perf
    }

    /// Get a shared singleton instance.
    ///
    /// All tensors using the default backend share this single Arc,
    /// avoiding per-tensor allocation.
    pub fn shared() -> Arc<NativeBackend> {
        static INSTANCE: OnceLock<Arc<NativeBackend>> = OnceLock::new();
        INSTANCE
            .get_or_init(|| Arc::new(NativeBackend::new()))
            .clone()
    }
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared rejection for complex `tridiag_eigh` instantiations: a general
/// complex symmetric tridiagonal matrix is not Hermitian, so no complex
/// kernel exists — reject instead of silently reinterpreting the data.
/// One definition keeps the c64/c32 dispatch arms' error text identical.
fn tridiag_eigh_complex_unsupported() -> Result<(), BackendError> {
    Err(BackendError::NotSupported(
        "tridiag_eigh: requires a real scalar type".into(),
    ))
}

/// faer-backed decomposition / solve kernels accept column-major slices only.
/// Reject any other order at the dispatcher boundary so per-type kernels
/// never see a layout they cannot interpret.
fn require_column_major(op: &str, order: MemoryOrder) -> Result<(), BackendError> {
    if order != MemoryOrder::ColumnMajor {
        return Err(BackendError::InvalidArgument(format!(
            "NativeBackend::{op} supports ColumnMajor only, got {order:?}"
        )));
    }
    Ok(())
}

impl ComputeBackend for NativeBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn preferred_order(&self) -> MemoryOrder {
        MemoryOrder::ColumnMajor
    }

    /// GEMM: C = alpha * A * B + beta * C
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        T::dispatch_op(&NativeKernels, OpDesc::Gemm(desc))
    }

    /// Transpose tensor axes according to permutation.
    ///
    /// Uses HPTT for f64/f32/Complex when the `hptt` feature is enabled,
    /// otherwise a naive output-driven kernel.
    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        T::dispatch_op(&NativeKernels, OpDesc::Transpose(desc))
    }

    /// Thin SVD via faer: A = U * diag(S) * Vt
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// For complex types, Vt stores V^H (conjugate transpose).
    /// faer's SVD is column-major only; descriptors with any other
    /// order are rejected with `BackendError::InvalidArgument`.
    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("svd", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Svd(desc))
    }

    /// Thin QR via faer: A = Q * R
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// faer's QR is column-major only; descriptors with any other
    /// order are rejected with `BackendError::InvalidArgument`.
    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("qr", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Qr(desc))
    }

    /// Thin LQ via faer: A = L * Q
    ///
    /// Internally computes QR of A^H (adjoint), then takes conjugate transposes.
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// faer's QR (and hence this LQ) is column-major only; descriptors
    /// with any other order are rejected with `BackendError::InvalidArgument`.
    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("lq", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Lq(desc))
    }

    /// Self-adjoint eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// faer's eigendecomposition is column-major only; descriptors with
    /// any other order are rejected with `BackendError::InvalidArgument`.
    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("eigh", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Eigh(desc))
    }

    /// Real symmetric tridiagonal eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32; complex scalars are rejected
    /// with `BackendError::NotSupported` (a general complex symmetric
    /// tridiagonal matrix is not Hermitian, so no complex kernel
    /// exists). The eigenvector output is column-major only;
    /// descriptors with any other order are rejected with
    /// `BackendError::InvalidArgument`.
    fn tridiag_eigh<T: Scalar>(
        &self,
        desc: TridiagEighDescriptor<'_, T>,
    ) -> Result<(), BackendError> {
        require_column_major("tridiag_eigh", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::TridiagEigh(desc))
    }

    /// General eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// faer's eigendecomposition is column-major only; descriptors with
    /// any other order are rejected with `BackendError::InvalidArgument`.
    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("eig", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Eig(desc))
    }

    /// Linear solve via faer LU decomposition with partial pivoting
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// faer's LU is column-major only; descriptors with any other
    /// order are rejected with `BackendError::InvalidArgument`.
    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        require_column_major("solve", desc.order)?;
        T::dispatch_op(&NativeKernels, OpDesc::Solve(desc))
    }

    fn par_for_svd(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = (m as f64 * n as f64 * m.min(n) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().svd, work_proxy)
    }

    fn par_for_qr(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = (m as f64 * n as f64 * m.min(n) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().qr, work_proxy)
    }

    fn par_for_lq(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = (m as f64 * n as f64 * m.min(n) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().lq, work_proxy)
    }

    fn par_for_eigh(&self, n: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().eigh, n)
    }

    fn par_for_eig(&self, n: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().eig, n)
    }

    fn par_for_gemm(&self, m: usize, n: usize, k: usize) -> ExecPolicy {
        let work_proxy = (m as f64 * n as f64 * k as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().gemm, work_proxy)
    }

    fn par_for_solve(&self, n: usize, _nrhs: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().solve, n)
    }

    fn par_for_transpose(&self, shape: &[usize]) -> ExecPolicy {
        // Saturate on overflow so very large shapes don't wrap below the
        // threshold and silently dispatch Sequential.
        let total: usize = shape.iter().copied().fold(1usize, usize::saturating_mul);
        PerformanceManager::policy_by_n(self.perf.thresholds().transpose, total)
    }
}

/// faer / HPTT kernel set the call-site dispatcher routes to.
///
/// `DispatchScalar::dispatch_op` resolves a generic `OpDesc<'_, T>` to one of
/// these four methods, where the scalar is concrete; each method then matches the op
/// and calls the corresponding monomorphic kernel directly. This is what lets
/// the generic `ComputeBackend` methods reach the per-type kernels without an
/// `unsafe` `Descriptor<T>` -> `Descriptor<concrete>` reinterpretation.
struct NativeKernels;

impl ScalarKernels for NativeKernels {
    fn run_f64(&self, op: OpDesc<'_, f64>) -> Result<(), BackendError> {
        match op {
            OpDesc::Gemm(d) => gemm::gemm_f64(d),
            OpDesc::Svd(d) => svd::svd_f64(d),
            OpDesc::Qr(d) => qr::qr_f64(d),
            OpDesc::Lq(d) => lq::lq_f64(d),
            OpDesc::Eigh(d) => eigh::eigh_f64(d),
            OpDesc::TridiagEigh(d) => tridiag_eigh::tridiag_eigh_f64(d),
            OpDesc::Eig(d) => eig::eig_f64(d),
            OpDesc::Solve(d) => solve::solve_f64(d),
            OpDesc::Transpose(d) => transpose::transpose_f64(d),
        }
    }

    fn run_f32(&self, op: OpDesc<'_, f32>) -> Result<(), BackendError> {
        match op {
            OpDesc::Gemm(d) => gemm::gemm_f32(d),
            OpDesc::Svd(d) => svd::svd_f32(d),
            OpDesc::Qr(d) => qr::qr_f32(d),
            OpDesc::Lq(d) => lq::lq_f32(d),
            OpDesc::Eigh(d) => eigh::eigh_f32(d),
            OpDesc::TridiagEigh(d) => tridiag_eigh::tridiag_eigh_f32(d),
            OpDesc::Eig(d) => eig::eig_f32(d),
            OpDesc::Solve(d) => solve::solve_f32(d),
            OpDesc::Transpose(d) => transpose::transpose_f32(d),
        }
    }

    fn run_c64(&self, op: OpDesc<'_, Complex<f64>>) -> Result<(), BackendError> {
        match op {
            OpDesc::Gemm(d) => gemm::gemm_c64(d),
            OpDesc::Svd(d) => svd::svd_c64(d),
            OpDesc::Qr(d) => qr::qr_c64(d),
            OpDesc::Lq(d) => lq::lq_c64(d),
            OpDesc::Eigh(d) => eigh::eigh_c64(d),
            OpDesc::TridiagEigh(_) => tridiag_eigh_complex_unsupported(),
            OpDesc::Eig(d) => eig::eig_c64(d),
            OpDesc::Solve(d) => solve::solve_c64(d),
            OpDesc::Transpose(d) => transpose::transpose_c64(d),
        }
    }

    fn run_c32(&self, op: OpDesc<'_, Complex<f32>>) -> Result<(), BackendError> {
        match op {
            OpDesc::Gemm(d) => gemm::gemm_c32(d),
            OpDesc::Svd(d) => svd::svd_c32(d),
            OpDesc::Qr(d) => qr::qr_c32(d),
            OpDesc::Lq(d) => lq::lq_c32(d),
            OpDesc::Eigh(d) => eigh::eigh_c32(d),
            OpDesc::TridiagEigh(_) => tridiag_eigh_complex_unsupported(),
            OpDesc::Eig(d) => eig::eig_c32(d),
            OpDesc::Solve(d) => solve::solve_c32(d),
            OpDesc::Transpose(d) => transpose::transpose_c32(d),
        }
    }
}
