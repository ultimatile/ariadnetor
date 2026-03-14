//! CPU compute backend for Ariadnetor
//!
//! Provides [`CpuBackend`] implementing `ComputeBackend` via:
//! - **GEMM**: faer (f64, f32, Complex<f64>, Complex<f32>)
//! - **SVD/QR/LQ**: faer (f64, f32, Complex<f64>, Complex<f32>)
//! - **Transpose**: HPTT when available (f64, f32, Complex), naive fallback

mod transpose;

use arnet_core::backend::{BackendError, ComputeBackend, DeviceType, GemmDescriptor, LqDescriptor, QrDescriptor, SvdDescriptor, TransposeDescriptor};
use arnet_core::scalar::Scalar;
use num_complex::Complex;

/// CPU backend using faer for GEMM and HPTT for transpose.
///
/// This is the sole owner of faer and hptt-rs dependencies in the workspace.
/// Other crates access these capabilities through the `ComputeBackend` trait.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    /// GEMM: C = alpha * A * B + beta * C
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId. Reinterpret generic fields
            // to concrete f64 via pointer casts; layout is identical.
            let desc_f64 = unsafe { reinterpret_gemm_desc::<T, f64>(desc) };
            gemm_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_gemm_desc::<T, f32>(desc) };
            gemm_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_gemm_desc::<T, Complex<f64>>(desc) };
            gemm_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_gemm_desc::<T, Complex<f32>>(desc) };
            gemm_c32(desc_c32)
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
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    /// For complex types, Vt stores V^H (conjugate transpose).
    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_svd_desc::<T, f64>(desc) };
            svd_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_svd_desc::<T, f32>(desc) };
            svd_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_svd_desc::<T, Complex<f64>>(desc) };
            svd_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_svd_desc::<T, Complex<f32>>(desc) };
            svd_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "SVD is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin QR via faer: A = Q * R
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_qr_desc::<T, f64>(desc) };
            qr_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_qr_desc::<T, f32>(desc) };
            qr_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_qr_desc::<T, Complex<f64>>(desc) };
            qr_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_qr_desc::<T, Complex<f32>>(desc) };
            qr_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "QR is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin LQ via faer: A = L * Q
    ///
    /// Internally computes QR of A^H (adjoint), then takes conjugate transposes.
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_lq_desc::<T, f64>(desc) };
            lq_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_lq_desc::<T, f32>(desc) };
            lq_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_lq_desc::<T, Complex<f64>>(desc) };
            lq_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_lq_desc::<T, Complex<f32>>(desc) };
            lq_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "LQ is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Generic → concrete type reinterpretation
// ---------------------------------------------------------------------------

/// Reinterpret `GemmDescriptor<T>` as `GemmDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_gemm_desc<'a, T, U>(
    desc: GemmDescriptor<'a, T>,
) -> GemmDescriptor<'a, U> {
    let GemmDescriptor { m, n, k, alpha, a, b, beta, c, trans_a, trans_b } = desc;
    unsafe {
        GemmDescriptor {
            m, n, k,
            alpha: std::ptr::read(&alpha as *const T as *const U),
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            b: std::slice::from_raw_parts(b.as_ptr() as *const U, b.len()),
            beta: std::ptr::read(&beta as *const T as *const U),
            c: std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut U, c.len()),
            trans_a, trans_b,
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
    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    unsafe {
        SvdDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            u: std::slice::from_raw_parts_mut(u.as_mut_ptr() as *mut U, u.len()),
            s: std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U::Real, s.len()),
            vt: std::slice::from_raw_parts_mut(vt.as_mut_ptr() as *mut U, vt.len()),
        }
    }
}

/// Reinterpret `QrDescriptor<T>` as `QrDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_qr_desc<'a, T, U>(
    desc: QrDescriptor<'a, T>,
) -> QrDescriptor<'a, U> {
    let QrDescriptor { m, n, a, q, r } = desc;
    unsafe {
        QrDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
            r: std::slice::from_raw_parts_mut(r.as_mut_ptr() as *mut U, r.len()),
        }
    }
}

/// Reinterpret `LqDescriptor<T>` as `LqDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_lq_desc<'a, T, U>(
    desc: LqDescriptor<'a, T>,
) -> LqDescriptor<'a, U> {
    let LqDescriptor { m, n, a, l, q } = desc;
    unsafe {
        LqDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            l: std::slice::from_raw_parts_mut(l.as_mut_ptr() as *mut U, l.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM implementations (faer)
// ---------------------------------------------------------------------------

/// GEMM for f64 via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_f64(desc: GemmDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

    // Construct faer MatRef views from row-major flat slices.
    // faer's from_row_major_slice expects (data, nrows, ncols).
    // For transposed operands, swap dimensions.
    let lhs: faer::Mat<f64> = if trans_a {
        let view = MatRef::from_row_major_slice(a, k, m);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(a, m, k).to_owned()
    };

    let rhs: faer::Mat<f64> = if trans_b {
        let view = MatRef::from_row_major_slice(b, n, k);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(b, k, n).to_owned()
    };

    let product = &lhs * &rhs;

    // C = alpha * product + beta * C
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = alpha * product[(i, j)] + beta * c[idx];
        }
    }

    Ok(())
}

/// GEMM for f32 via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_f32(desc: GemmDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

    let lhs: faer::Mat<f32> = if trans_a {
        let view = MatRef::from_row_major_slice(a, k, m);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(a, m, k).to_owned()
    };

    let rhs: faer::Mat<f32> = if trans_b {
        let view = MatRef::from_row_major_slice(b, n, k);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(b, k, n).to_owned()
    };

    let product = &lhs * &rhs;

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = alpha * product[(i, j)] + beta * c[idx];
        }
    }

    Ok(())
}

/// GEMM for Complex<f64> via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_c64(desc: GemmDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

    let lhs: faer::Mat<Complex<f64>> = if trans_a {
        let view = MatRef::from_row_major_slice(a, k, m);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(a, m, k).to_owned()
    };

    let rhs: faer::Mat<Complex<f64>> = if trans_b {
        let view = MatRef::from_row_major_slice(b, n, k);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(b, k, n).to_owned()
    };

    let product = &lhs * &rhs;

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = alpha * product[(i, j)] + beta * c[idx];
        }
    }

    Ok(())
}

/// GEMM for Complex<f32> via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_c32(desc: GemmDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

    let lhs: faer::Mat<Complex<f32>> = if trans_a {
        let view = MatRef::from_row_major_slice(a, k, m);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(a, m, k).to_owned()
    };

    let rhs: faer::Mat<Complex<f32>> = if trans_b {
        let view = MatRef::from_row_major_slice(b, n, k);
        view.transpose().to_owned()
    } else {
        MatRef::from_row_major_slice(b, k, n).to_owned()
    };

    let product = &lhs * &rhs;

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = alpha * product[(i, j)] + beta * c[idx];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// SVD implementations (faer)
// ---------------------------------------------------------------------------

/// Thin SVD for f64 via faer: A = U * diag(S) * Vt
fn svd_f64(desc: SvdDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    // U (m×k, row-major)
    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    // Singular values (k)
    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    // Vt = V^T (k×n, row-major)
    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)];
        }
    }

    Ok(())
}

/// Thin SVD for f32 via faer: A = U * diag(S) * Vt
fn svd_f32(desc: SvdDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)];
        }
    }

    Ok(())
}

/// Thin SVD for Complex<f64> via faer: A = U * diag(S) * V^H
fn svd_c64(desc: SvdDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    // Singular values are always real; faer stores them as Complex with im=0
    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i].re;
    }

    // Vt = V^H (conjugate transpose)
    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin SVD for Complex<f32> via faer: A = U * diag(S) * V^H
fn svd_c32(desc: SvdDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i].re;
    }

    // Vt = V^H (conjugate transpose)
    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)].conj();
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// QR implementations (faer)
// ---------------------------------------------------------------------------

/// Thin QR for f64 via faer: A = Q * R
fn qr_f64(desc: QrDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    // Q (m×k, row-major) — thin Q
    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    // R (k×n, row-major) — thin R
    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for f32 via faer: A = Q * R
fn qr_f32(desc: QrDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f64> via faer: A = Q * R
fn qr_c64(desc: QrDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f32> via faer: A = Q * R
fn qr_c32(desc: QrDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let QrDescriptor { m, n, a, q, r } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = r_mat[(i, j)];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// LQ implementations (via adjoint → QR → adjoint)
// ---------------------------------------------------------------------------

/// Thin LQ for f64: A = L * Q, computed via QR of A^T
fn lq_f64(desc: LqDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // Transpose A (m×n, row-major) → A^T (n×m)
    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    // QR of A^T: A^T = Q_t * R_t where Q_t is n×k, R_t is k×m
    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (Q_t * R_t)^T = R_t^T * Q_t^T = L * Q
    // L = R_t^T (k×m transposed → m×k, row-major)
    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)];
        }
    }

    // Q = Q_t^T (n×k transposed → k×n, row-major)
    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for f32: A = L * Q, computed via QR of A^T
fn lq_f32(desc: LqDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f64>: A = L * Q, computed via QR of A^H
fn lq_c64(desc: LqDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // A^H (n×m) via conjugate transpose
    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    // QR of A^H: A^H = Q_t * R_t where Q_t is n×k, R_t is k×m
    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (A^H)^H = (Q_t * R_t)^H = R_t^H * Q_t^H = L * Q
    // L = R_t^H (m×k)
    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)].conj();
        }
    }

    // Q = Q_t^H (k×n)
    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f32>: A = L * Q, computed via QR of A^H
fn lq_c32(desc: LqDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    use faer::MatRef;

    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_row_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[i * k + j] = r_t[(j, i)].conj();
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[i * n + j] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_core::backend::ComputeBackend;

    #[test]
    fn test_backend_metadata() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.device_type(), DeviceType::Cpu);
        assert!(backend.is_available());
    }

    // --- GEMM tests ---

    #[test]
    fn test_gemm_f64_identity() {
        let backend = CpuBackend::new();

        // A = [[1, 0], [0, 1]] (2x2 identity)
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_gemm_f64_basic() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_f64_alpha_beta() {
        let backend = CpuBackend::new();

        // C = 2.0 * A * B + 3.0 * C_init
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [1.0f64; 4]; // C_init = all ones

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 2.0, a: &a, b: &b,
            beta: 3.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        // C = 2 * [19, 22, 43, 50] + 3 * [1, 1, 1, 1] = [41, 47, 89, 103]
        assert_eq!(c, [41.0, 47.0, 89.0, 103.0]);
    }

    #[test]
    fn test_gemm_f64_rectangular() {
        let backend = CpuBackend::new();

        // A (2x3) * B (3x2) = C (2x2)
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 3,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64, 139, 154]
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    // --- Transpose tests ---

    #[test]
    fn test_transpose_f64_2d() {
        let backend = CpuBackend::new();

        // 2x3 matrix → 3x2
        let input = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0f64; 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        // [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]]
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_f64_3d() {
        let backend = CpuBackend::new();

        // Shape [2,3,4], perm [1,0,2] → shape [3,2,4]
        let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let mut output = vec![0.0f64; 24];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3, 4],
            perm: &[1, 0, 2],
        };
        backend.transpose(desc).unwrap();

        // Verify a few elements: input[i][j][k] should equal output[j][i][k]
        // input[0][1][2] = 0*12 + 1*4 + 2 = 6 → output[1][0][2] = 1*8 + 0*4 + 2 = 10
        assert_eq!(output[10], 6.0);
        // input[1][0][3] = 1*12 + 0*4 + 3 = 15 → output[0][1][3] = 0*8 + 1*4 + 3 = 7
        assert_eq!(output[7], 15.0);
    }

    #[test]
    fn test_transpose_f32_2d() {
        let backend = CpuBackend::new();

        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0f32; 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_complex_f64_2d() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        let input = [
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0), Complex::new(9.0, 10.0), Complex::new(11.0, 12.0),
        ];
        let mut output = [Complex::new(0.0, 0.0); 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        assert_eq!(output[0], Complex::new(1.0, 2.0));
        assert_eq!(output[1], Complex::new(7.0, 8.0));
        assert_eq!(output[2], Complex::new(3.0, 4.0));
        assert_eq!(output[3], Complex::new(9.0, 10.0));
    }

    // --- SVD tests ---

    #[test]
    fn test_svd_f64_square() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]] (2x2)
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let mut u = [0.0f64; 4]; // 2x2
        let mut s = [0.0f64; 2]; // 2
        let mut vt = [0.0f64; 4]; // 2x2

        let desc = SvdDescriptor {
            m: 2, n: 2, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        // Singular values should be positive and in descending order
        assert!(s[0] > s[1]);
        assert!(s[1] >= 0.0);

        // Reconstruct: A ≈ U * diag(S) * Vt
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!((val - a[i * 2 + j]).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * 2 + j]);
            }
        }
    }

    #[test]
    fn test_svd_f64_rectangular() {
        let backend = CpuBackend::new();

        // A = [[1, 2, 3], [4, 5, 6]] (2x3), k = min(2,3) = 2
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n, k) = (2, 3, 2);
        let mut u = vec![0.0f64; m * k];
        let mut s = vec![0.0f64; k];
        let mut vt = vec![0.0f64; k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        // Reconstruct A ≈ U * diag(S) * Vt
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u[i * k + l] * s[l] * vt[l * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_svd_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut u = [0.0f32; 4];
        let mut s = [0.0f32; 2];
        let mut vt = [0.0f32; 4];

        let desc = SvdDescriptor {
            m: 2, n: 2, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!((val - a[i * 2 + j]).abs() < 1e-4,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * 2 + j]);
            }
        }
    }

    // --- QR tests ---

    #[test]
    fn test_qr_f64_square() {
        let backend = CpuBackend::new();

        let a = [1.0f64, 2.0, 3.0, 4.0];
        let (m, n, k) = (2, 2, 2);
        let mut q = [0.0f64; 4];
        let mut r = [0.0f64; 4];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        // Reconstruct: A ≈ Q * R
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
            }
        }
    }

    #[test]
    fn test_qr_f64_rectangular() {
        let backend = CpuBackend::new();

        // A (3×2), k = min(3,2) = 2
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n, k) = (3, 2, 2);
        let mut q = vec![0.0f64; m * k];
        let mut r = vec![0.0f64; k * n];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_qr_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let (m, n, k) = (2, 2, 2);
        let mut q = [0.0f32; 4];
        let mut r = [0.0f32; 4];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0f32;
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-4,
                    "QR reconstruction mismatch at ({i},{j})");
            }
        }
    }

    // --- LQ tests ---

    #[test]
    fn test_lq_f64_square() {
        let backend = CpuBackend::new();

        let a = [1.0f64, 2.0, 3.0, 4.0];
        let (m, n, k) = (2, 2, 2);
        let mut l = [0.0f64; 4];
        let mut q = [0.0f64; 4];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        // Reconstruct: A ≈ L * Q
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
            }
        }
    }

    #[test]
    fn test_lq_f64_rectangular() {
        let backend = CpuBackend::new();

        // A (2×3), k = min(2,3) = 2
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n, k) = (2, 3, 2);
        let mut l = vec![0.0f64; m * k];
        let mut q = vec![0.0f64; k * n];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_lq_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let (m, n, k) = (2, 2, 2);
        let mut l = [0.0f32; 4];
        let mut q = [0.0f32; 4];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0f32;
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-4,
                    "LQ reconstruction mismatch at ({i},{j})");
            }
        }
    }

    // --- Complex GEMM tests ---

    #[test]
    fn test_gemm_c64_basic() {
        let backend = CpuBackend::new();

        // A = [[1+i, 2+i], [3+i, 4+i]], B = [[5+i, 6+i], [7+i, 8+i]]
        let a = [
            Complex::new(1.0, 1.0), Complex::new(2.0, 1.0),
            Complex::new(3.0, 1.0), Complex::new(4.0, 1.0),
        ];
        let b = [
            Complex::new(5.0, 1.0), Complex::new(6.0, 1.0),
            Complex::new(7.0, 1.0), Complex::new(8.0, 1.0),
        ];
        let mut c = [Complex::new(0.0, 0.0); 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: Complex::new(1.0, 0.0), a: &a, b: &b,
            beta: Complex::new(0.0, 0.0), c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();

        // C[0,0] = (1+i)(5+i) + (2+i)(7+i) = (4+6i) + (13+9i) = 17+15i
        // Manually: (1+i)(5+i) = 5+i+5i+i² = 5+6i-1 = 4+6i
        //           (2+i)(7+i) = 14+2i+7i+i² = 14+9i-1 = 13+9i
        //           sum = 17+15i
        assert!((c[0].re - 17.0).abs() < 1e-10);
        assert!((c[0].im - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_c64_alpha_beta() {
        let backend = CpuBackend::new();

        // C = alpha * A * B + beta * C_init with complex alpha, beta
        let a = [
            Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        ];
        let b = [
            Complex::new(3.0, 4.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(3.0, 4.0),
        ];
        let mut c = [
            Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, 1.0),
        ];

        // alpha = 2, beta = i → C = 2*I*B + i*C_init = 2*B + i*C_init
        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: Complex::new(2.0, 0.0), a: &a, b: &b,
            beta: Complex::new(0.0, 1.0), c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();

        // C[0,0] = 2*(3+4i) + i*(1+i) = (6+8i) + (i+i²) = (6+8i) + (-1+i) = 5+9i
        assert!((c[0].re - 5.0).abs() < 1e-10);
        assert!((c[0].im - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_c32_basic() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(1.0f32, 1.0), Complex::new(2.0, 1.0),
            Complex::new(3.0, 1.0), Complex::new(4.0, 1.0),
        ];
        let b = [
            Complex::new(5.0f32, 1.0), Complex::new(6.0, 1.0),
            Complex::new(7.0, 1.0), Complex::new(8.0, 1.0),
        ];
        let mut c = [Complex::new(0.0f32, 0.0); 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: Complex::new(1.0, 0.0), a: &a, b: &b,
            beta: Complex::new(0.0, 0.0), c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();

        assert!((c[0].re - 17.0).abs() < 1e-4);
        assert!((c[0].im - 15.0).abs() < 1e-4);
    }

    // --- Complex SVD tests ---

    #[test]
    fn test_svd_c64_hermitian() {
        let backend = CpuBackend::new();

        // Hermitian matrix: A = [[2, 1-i], [1+i, 3]]
        let a = [
            Complex::new(2.0, 0.0), Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0), Complex::new(3.0, 0.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut u = vec![Complex::new(0.0, 0.0); m * k];
        let mut s = vec![0.0f64; k];
        let mut vt = vec![Complex::new(0.0, 0.0); k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        // Singular values should be positive and descending
        assert!(s[0] > s[1]);
        assert!(s[1] >= 0.0);

        // Reconstruct: A ≈ U * diag(S) * Vt (where Vt = V^H)
        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..k {
                    val += u[i * k + l] * s[l] * vt[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "SVD reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
            }
        }
    }

    #[test]
    fn test_svd_c64_rectangular() {
        let backend = CpuBackend::new();

        // A (2×3) complex
        let a = [
            Complex::new(1.0, 2.0), Complex::new(3.0, 0.0), Complex::new(0.0, 1.0),
            Complex::new(4.0, -1.0), Complex::new(2.0, 3.0), Complex::new(1.0, 1.0),
        ];
        let (m, n, k) = (2, 3, 2);
        let mut u = vec![Complex::new(0.0, 0.0); m * k];
        let mut s = vec![0.0f64; k];
        let mut vt = vec![Complex::new(0.0, 0.0); k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..k {
                    val += u[i * k + l] * s[l] * vt[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "SVD reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_svd_c64_unitary_check() {
        let backend = CpuBackend::new();

        // Verify U^H * U = I for SVD result
        let a = [
            Complex::new(1.0, 2.0), Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut u = vec![Complex::new(0.0, 0.0); m * k];
        let mut s = vec![0.0f64; k];
        let mut vt = vec![Complex::new(0.0, 0.0); k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        // U^H * U should be identity (k×k)
        for i in 0..k {
            for j in 0..k {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..m {
                    val += u[l * k + i].conj() * u[l * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(val.norm() - expected < 1e-10,
                    "U^H * U not identity at ({i},{j}): {val}");
            }
        }
    }

    #[test]
    fn test_svd_c32_basic() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(2.0f32, 0.0), Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0), Complex::new(3.0, 0.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut u = vec![Complex::new(0.0f32, 0.0); m * k];
        let mut s = vec![0.0f32; k];
        let mut vt = vec![Complex::new(0.0f32, 0.0); k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0f32, 0.0);
                for l in 0..k {
                    val += u[i * k + l] * s[l] * vt[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-4,
                    "SVD reconstruction mismatch at ({i},{j})");
            }
        }
    }

    // --- Complex QR tests ---

    #[test]
    fn test_qr_c64_square() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(1.0, 2.0), Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut q = vec![Complex::new(0.0, 0.0); m * k];
        let mut r = vec![Complex::new(0.0, 0.0); k * n];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        // Reconstruct: A ≈ Q * R
        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "QR reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
            }
        }

        // Q should be unitary: Q^H * Q = I
        for i in 0..k {
            for j in 0..k {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..m {
                    val += q[l * k + i].conj() * q[l * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val.norm() - expected).abs() < 1e-10,
                    "Q^H * Q not identity at ({i},{j}): {val}");
            }
        }
    }

    #[test]
    fn test_qr_c64_rectangular() {
        let backend = CpuBackend::new();

        // A (3×2) complex
        let a = [
            Complex::new(1.0, 1.0), Complex::new(2.0, -1.0),
            Complex::new(3.0, 0.0), Complex::new(0.0, 4.0),
            Complex::new(-1.0, 2.0), Complex::new(5.0, 1.0),
        ];
        let (m, n, k) = (3, 2, 2);
        let mut q = vec![Complex::new(0.0, 0.0); m * k];
        let mut r = vec![Complex::new(0.0, 0.0); k * n];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "QR reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_qr_c32_basic() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(1.0f32, 2.0), Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut q = vec![Complex::new(0.0f32, 0.0); m * k];
        let mut r = vec![Complex::new(0.0f32, 0.0); k * n];

        let desc = QrDescriptor {
            m, n, a: &a,
            q: &mut q, r: &mut r,
        };
        backend.qr(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0f32, 0.0);
                for l in 0..k {
                    val += q[i * k + l] * r[l * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-4,
                    "QR reconstruction mismatch at ({i},{j})");
            }
        }
    }

    // --- Complex LQ tests ---

    #[test]
    fn test_lq_c64_square() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(1.0, 2.0), Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut l = vec![Complex::new(0.0, 0.0); m * k];
        let mut q = vec![Complex::new(0.0, 0.0); k * n];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        // Reconstruct: A ≈ L * Q
        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * n + j]);
            }
        }

        // Q should have orthonormal rows: Q * Q^H = I
        for i in 0..k {
            for j in 0..k {
                let mut val = Complex::new(0.0, 0.0);
                for l_idx in 0..n {
                    val += q[i * n + l_idx] * q[j * n + l_idx].conj();
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val.norm() - expected).abs() < 1e-10,
                    "Q * Q^H not identity at ({i},{j}): {val}");
            }
        }
    }

    #[test]
    fn test_lq_c64_rectangular() {
        let backend = CpuBackend::new();

        // A (2×3) complex
        let a = [
            Complex::new(1.0, 1.0), Complex::new(2.0, -1.0), Complex::new(0.0, 3.0),
            Complex::new(4.0, 0.0), Complex::new(-1.0, 2.0), Complex::new(3.0, 1.0),
        ];
        let (m, n, k) = (2, 3, 2);
        let mut l = vec![Complex::new(0.0, 0.0); m * k];
        let mut q = vec![Complex::new(0.0, 0.0); k * n];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0, 0.0);
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-10,
                    "LQ reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_lq_c32_basic() {
        let backend = CpuBackend::new();

        let a = [
            Complex::new(1.0f32, 2.0), Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0), Complex::new(2.0, 1.0),
        ];
        let (m, n, k) = (2, 2, 2);
        let mut l = vec![Complex::new(0.0f32, 0.0); m * k];
        let mut q = vec![Complex::new(0.0f32, 0.0); k * n];

        let desc = LqDescriptor {
            m, n, a: &a,
            l: &mut l, q: &mut q,
        };
        backend.lq(desc).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut val = Complex::new(0.0f32, 0.0);
                for kk in 0..k {
                    val += l[i * k + kk] * q[kk * n + j];
                }
                let diff = (val - a[i * n + j]).norm();
                assert!(diff < 1e-4,
                    "LQ reconstruction mismatch at ({i},{j})");
            }
        }
    }
}
