//! Self-adjoint eigenvalue decomposition implementations via faer for all supported scalar types

use ariadnetor_core::backend::{BackendError, EighDescriptor};
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
    ComputeEigenvectors, SelfAdjointEvdParams, self_adjoint_evd, self_adjoint_evd_scratch,
};
use faer::{Mat, MatRef, Spec};
use num_complex::Complex;

use crate::to_faer_par;

/// Self-adjoint eigenvalue decomposition for f64 via faer
///
/// Only the lower triangle of `A` is read.
pub(crate) fn eigh_f64(desc: EighDescriptor<'_, f64>) -> Result<(), BackendError> {
    let EighDescriptor {
        n,
        a,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, f64> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<f64>::zeros(n);
    let mut u_mat = Mat::<f64>::zeros(n, n);

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    self_adjoint_evd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer self_adjoint_evd failed: {e:?}")))?;

    for i in 0..n {
        w[i] = s_diag[i];
    }

    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for f32 via faer
///
/// Only the lower triangle of `A` is read.
pub(crate) fn eigh_f32(desc: EighDescriptor<'_, f32>) -> Result<(), BackendError> {
    let EighDescriptor {
        n,
        a,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, f32> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<f32>::zeros(n);
    let mut u_mat = Mat::<f32>::zeros(n, n);

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<f32>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    self_adjoint_evd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer self_adjoint_evd failed: {e:?}")))?;

    for i in 0..n {
        w[i] = s_diag[i];
    }

    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for Complex<f64> via faer
///
/// Only the lower triangle of `A` is read.
pub(crate) fn eigh_c64(desc: EighDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let EighDescriptor {
        n,
        a,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, Complex<f64>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<Complex<f64>>::zeros(n);
    let mut u_mat = Mat::<Complex<f64>>::zeros(n, n);

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<Complex<f64>>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    self_adjoint_evd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer self_adjoint_evd failed: {e:?}")))?;

    // Eigenvalues are real for self-adjoint matrices; faer stores as Complex with im=0
    for i in 0..n {
        w[i] = s_diag[i].re;
    }

    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for Complex<f32> via faer
///
/// Only the lower triangle of `A` is read.
pub(crate) fn eigh_c32(desc: EighDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let EighDescriptor {
        n,
        a,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, Complex<f32>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<Complex<f32>>::zeros(n);
    let mut u_mat = Mat::<Complex<f32>>::zeros(n, n);

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<Complex<f32>>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    self_adjoint_evd(
        a_mat,
        s_diag.as_mut(),
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer self_adjoint_evd failed: {e:?}")))?;

    for i in 0..n {
        w[i] = s_diag[i].re;
    }

    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}
