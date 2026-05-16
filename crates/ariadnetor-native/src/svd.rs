//! SVD implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, SvdDescriptor};
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{ComputeSvdVectors, SvdParams, svd, svd_scratch};
use faer::{Mat, MatRef, Spec};
use num_complex::Complex;

use crate::to_faer_par;

/// Thin SVD for f64 via faer: A = U * diag(S) * Vt
pub(crate) fn svd_f64(desc: SvdDescriptor<'_, f64>) -> Result<(), BackendError> {
    let SvdDescriptor {
        m,
        n,
        a,
        u,
        s,
        vt,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<SvdParams, f64> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut u_mat = Mat::<f64>::zeros(m, k);
    let mut s_diag = Diag::<f64>::zeros(k);
    let mut v_mat = Mat::<f64>::zeros(n, k);

    let mut buf = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    svd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        Some(v_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer svd failed: {e:?}")))?;

    // U (m×k, column-major)
    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    // Singular values (k)
    for i in 0..k {
        s[i] = s_diag[i];
    }

    // Vt = V^T (k×n, column-major)
    for i in 0..k {
        for j in 0..n {
            vt[j * k + i] = v_mat[(j, i)];
        }
    }

    Ok(())
}

/// Thin SVD for f32 via faer: A = U * diag(S) * Vt
pub(crate) fn svd_f32(desc: SvdDescriptor<'_, f32>) -> Result<(), BackendError> {
    let SvdDescriptor {
        m,
        n,
        a,
        u,
        s,
        vt,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<SvdParams, f32> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut u_mat = Mat::<f32>::zeros(m, k);
    let mut s_diag = Diag::<f32>::zeros(k);
    let mut v_mat = Mat::<f32>::zeros(n, k);

    let mut buf = MemBuffer::new(svd_scratch::<f32>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    svd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        Some(v_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer svd failed: {e:?}")))?;

    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    for i in 0..k {
        s[i] = s_diag[i];
    }

    for i in 0..k {
        for j in 0..n {
            vt[j * k + i] = v_mat[(j, i)];
        }
    }

    Ok(())
}

/// Thin SVD for Complex<f64> via faer: A = U * diag(S) * V^H
pub(crate) fn svd_c64(desc: SvdDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let SvdDescriptor {
        m,
        n,
        a,
        u,
        s,
        vt,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<SvdParams, Complex<f64>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut u_mat = Mat::<Complex<f64>>::zeros(m, k);
    let mut s_diag = Diag::<Complex<f64>>::zeros(k);
    let mut v_mat = Mat::<Complex<f64>>::zeros(n, k);

    let mut buf = MemBuffer::new(svd_scratch::<Complex<f64>>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    svd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        Some(v_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer svd failed: {e:?}")))?;

    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    // Singular values are real; faer stores them as Complex with im=0
    for i in 0..k {
        s[i] = s_diag[i].re;
    }

    // Vt = V^H (conjugate transpose, k×n, column-major)
    for i in 0..k {
        for j in 0..n {
            vt[j * k + i] = v_mat[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin SVD for Complex<f32> via faer: A = U * diag(S) * V^H
pub(crate) fn svd_c32(desc: SvdDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let SvdDescriptor {
        m,
        n,
        a,
        u,
        s,
        vt,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<SvdParams, Complex<f32>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut u_mat = Mat::<Complex<f32>>::zeros(m, k);
    let mut s_diag = Diag::<Complex<f32>>::zeros(k);
    let mut v_mat = Mat::<Complex<f32>>::zeros(n, k);

    let mut buf = MemBuffer::new(svd_scratch::<Complex<f32>>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    svd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        Some(v_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer svd failed: {e:?}")))?;

    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    for i in 0..k {
        s[i] = s_diag[i].re;
    }

    for i in 0..k {
        for j in 0..n {
            vt[j * k + i] = v_mat[(j, i)].conj();
        }
    }

    Ok(())
}
