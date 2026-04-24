//! CPU compute backend for Ariadnetor
//!
//! Provides [`NativeBackend`] implementing `ComputeBackend` via:
//! - **GEMM**: faer (f64, f32, `Complex<f64>`, `Complex<f32>`)
//! - **SVD/QR/LQ/EIGH**: faer (f64, f32, `Complex<f64>`, `Complex<f32>`)
//! - **Transpose**: HPTT when available (f64, f32, Complex), naive fallback

mod eig;
mod eigh;
mod gemm;
mod lq;
mod performance;
mod qr;
mod solve;
mod svd;
mod transpose;

use std::sync::{Arc, OnceLock};

use arnet_core::Scalar;
use arnet_core::backend::{
    BackendError, ComputeBackend, DeviceType, EigDescriptor, EighDescriptor, ExecPolicy,
    GemmDescriptor, LqDescriptor, MemoryOrder, QrDescriptor, SolveDescriptor, SvdDescriptor,
    TransposeDescriptor,
};
use num_complex::Complex;

pub use performance::{PerformanceManager, ThresholdTable};

/// Map an [`ExecPolicy`] to faer's per-call parallelism selector.
///
/// `Parallel(0)` defers to faer's Rayon default (current thread pool size).
/// `Parallel(n)` for `n > 0` requests exactly `n` threads.
pub(crate) fn to_faer_par(policy: ExecPolicy) -> faer::Par {
    match policy {
        ExecPolicy::Sequential => faer::Par::Seq,
        ExecPolicy::Parallel(0) => faer::Par::rayon(0),
        ExecPolicy::Parallel(n) => faer::Par::rayon(n),
    }
}

/// Native backend using faer for GEMM and HPTT for transpose.
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
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId. Reinterpret generic fields
            // to concrete f64 via pointer casts; layout is identical.
            let desc_f64 = unsafe { reinterpret_gemm_desc::<T, f64>(desc) };
            gemm::gemm_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_gemm_desc::<T, f32>(desc) };
            gemm::gemm_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_gemm_desc::<T, Complex<f64>>(desc) };
            gemm::gemm_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_gemm_desc::<T, Complex<f32>>(desc) };
            gemm::gemm_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "GEMM is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Transpose tensor axes according to permutation.
    ///
    /// Uses HPTT for f64/f32/Complex when the `hptt` feature is enabled,
    /// with a naive fallback for all types.
    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        transpose::dispatch(desc)
    }

    /// Thin SVD via faer: A = U * diag(S) * Vt
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    /// For complex types, Vt stores V^H (conjugate transpose).
    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_svd_desc::<T, f64>(desc) };
            svd::svd_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_svd_desc::<T, f32>(desc) };
            svd::svd_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_svd_desc::<T, Complex<f64>>(desc) };
            svd::svd_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_svd_desc::<T, Complex<f32>>(desc) };
            svd::svd_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "SVD is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin QR via faer: A = Q * R
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_qr_desc::<T, f64>(desc) };
            qr::qr_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_qr_desc::<T, f32>(desc) };
            qr::qr_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_qr_desc::<T, Complex<f64>>(desc) };
            qr::qr_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_qr_desc::<T, Complex<f32>>(desc) };
            qr::qr_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "QR is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin LQ via faer: A = L * Q
    ///
    /// Internally computes QR of A^H (adjoint), then takes conjugate transposes.
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_lq_desc::<T, f64>(desc) };
            lq::lq_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_lq_desc::<T, f32>(desc) };
            lq::lq_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_lq_desc::<T, Complex<f64>>(desc) };
            lq::lq_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_lq_desc::<T, Complex<f32>>(desc) };
            lq::lq_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "LQ is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Self-adjoint eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_eigh_desc::<T, f64>(desc) };
            eigh::eigh_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_eigh_desc::<T, f32>(desc) };
            eigh::eigh_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_eigh_desc::<T, Complex<f64>>(desc) };
            eigh::eigh_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_eigh_desc::<T, Complex<f32>>(desc) };
            eigh::eigh_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "eigh is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// General eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_eig_desc::<T, f64>(desc) };
            eig::eig_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_eig_desc::<T, f32>(desc) };
            eig::eig_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_eig_desc::<T, Complex<f64>>(desc) };
            eig::eig_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_eig_desc::<T, Complex<f32>>(desc) };
            eig::eig_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "eig is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Linear solve via faer LU decomposition with partial pivoting
    ///
    /// Dispatches to faer for f64/f32/`Complex<f64>`/`Complex<f32>`.
    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_solve_desc::<T, f64>(desc) };
            solve::solve_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_solve_desc::<T, f32>(desc) };
            solve::solve_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_solve_desc::<T, Complex<f64>>(desc) };
            solve::solve_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_solve_desc::<T, Complex<f32>>(desc) };
            solve::solve_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "solve is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    fn par_for_svd(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = ((m * n * m.min(n)) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().svd, work_proxy)
    }

    fn par_for_qr(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = ((m * n * m.min(n)) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().qr, work_proxy)
    }

    fn par_for_lq(&self, m: usize, n: usize) -> ExecPolicy {
        let work_proxy = ((m * n * m.min(n)) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().lq, work_proxy)
    }

    fn par_for_eigh(&self, n: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().eigh, n)
    }

    fn par_for_eig(&self, n: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().eig, n)
    }

    fn par_for_gemm(&self, m: usize, n: usize, k: usize) -> ExecPolicy {
        let work_proxy = ((m * n * k) as f64).cbrt() as usize;
        PerformanceManager::policy_by_n(self.perf.thresholds().gemm, work_proxy)
    }

    fn par_for_solve(&self, n: usize, _nrhs: usize) -> ExecPolicy {
        PerformanceManager::policy_by_n(self.perf.thresholds().solve, n)
    }

    fn par_for_transpose(&self, shape: &[usize]) -> ExecPolicy {
        let total: usize = shape.iter().product();
        PerformanceManager::policy_by_n(self.perf.thresholds().transpose, total)
    }
}

// ---------------------------------------------------------------------------
// Generic -> concrete type reinterpretation
// ---------------------------------------------------------------------------

/// Reinterpret `GemmDescriptor<T>` as `GemmDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_gemm_desc<'a, T, U>(desc: GemmDescriptor<'a, T>) -> GemmDescriptor<'a, U> {
    let GemmDescriptor {
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
        trans_a,
        trans_b,
        order,
        policy,
    } = desc;
    unsafe {
        GemmDescriptor {
            m,
            n,
            k,
            alpha: std::ptr::read(&alpha as *const T as *const U),
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            b: std::slice::from_raw_parts(b.as_ptr() as *const U, b.len()),
            beta: std::ptr::read(&beta as *const T as *const U),
            c: std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut U, c.len()),
            trans_a,
            trans_b,
            order,
            policy,
        }
    }
}

/// Reinterpret `SvdDescriptor<T>` as `SvdDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Real` and `U::Real` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_svd_desc<'a, T: Scalar, U: Scalar>(
    desc: SvdDescriptor<'a, T>,
) -> SvdDescriptor<'a, U> {
    let SvdDescriptor {
        m,
        n,
        a,
        u,
        s,
        vt,
        policy,
    } = desc;
    unsafe {
        SvdDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            u: std::slice::from_raw_parts_mut(u.as_mut_ptr() as *mut U, u.len()),
            s: std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U::Real, s.len()),
            vt: std::slice::from_raw_parts_mut(vt.as_mut_ptr() as *mut U, vt.len()),
            policy,
        }
    }
}

/// Reinterpret `QrDescriptor<T>` as `QrDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_qr_desc<'a, T, U>(desc: QrDescriptor<'a, T>) -> QrDescriptor<'a, U> {
    let QrDescriptor {
        m,
        n,
        a,
        q,
        r,
        policy,
    } = desc;
    unsafe {
        QrDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
            r: std::slice::from_raw_parts_mut(r.as_mut_ptr() as *mut U, r.len()),
            policy,
        }
    }
}

/// Reinterpret `LqDescriptor<T>` as `LqDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_lq_desc<'a, T, U>(desc: LqDescriptor<'a, T>) -> LqDescriptor<'a, U> {
    let LqDescriptor {
        m,
        n,
        a,
        l,
        q,
        policy,
    } = desc;
    unsafe {
        LqDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            l: std::slice::from_raw_parts_mut(l.as_mut_ptr() as *mut U, l.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
            policy,
        }
    }
}

/// Reinterpret `EighDescriptor<T>` as `EighDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Real` and `U::Real` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_eigh_desc<'a, T: Scalar, U: Scalar>(
    desc: EighDescriptor<'a, T>,
) -> EighDescriptor<'a, U> {
    let EighDescriptor { n, a, w, v, policy } = desc;
    unsafe {
        EighDescriptor {
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            w: std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut U::Real, w.len()),
            v: std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut U, v.len()),
            policy,
        }
    }
}

/// Reinterpret `EigDescriptor<T>` as `EigDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Complex` and `U::Complex` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_eig_desc<'a, T: Scalar, U: Scalar>(
    desc: EigDescriptor<'a, T>,
) -> EigDescriptor<'a, U> {
    let EigDescriptor { n, a, w, v, policy } = desc;
    unsafe {
        EigDescriptor {
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            w: std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut U::Complex, w.len()),
            v: std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut U::Complex, v.len()),
            policy,
        }
    }
}

/// Reinterpret `SolveDescriptor<T>` as `SolveDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_solve_desc<'a, T, U>(desc: SolveDescriptor<'a, T>) -> SolveDescriptor<'a, U> {
    let SolveDescriptor {
        n,
        nrhs,
        a,
        b,
        x,
        policy,
    } = desc;
    unsafe {
        SolveDescriptor {
            n,
            nrhs,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            b: std::slice::from_raw_parts(b.as_ptr() as *const U, b.len()),
            x: std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut U, x.len()),
            policy,
        }
    }
}
