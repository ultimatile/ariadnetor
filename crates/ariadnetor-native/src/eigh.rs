//! Self-adjoint eigenvalue decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, EighDescriptor};
use faer::{MatRef, Side};
use num_complex::Complex;

/// Self-adjoint eigenvalue decomposition for f64 via faer
pub(crate) fn eigh_f64(desc: EighDescriptor<'_, f64>) -> Result<(), BackendError> {
    let EighDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_column_major_slice(a, n, n).to_owned();
    let eig = mat.self_adjoint_eigen(Side::Lower).map_err(|e| {
        BackendError::ExecutionFailed(format!("faer self_adjoint_eigen failed: {e:?}"))
    })?;

    // Eigenvalues (n, ascending)
    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    // Eigenvectors (n×n, column-major)
    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for f32 via faer
pub(crate) fn eigh_f32(desc: EighDescriptor<'_, f32>) -> Result<(), BackendError> {
    let EighDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_column_major_slice(a, n, n).to_owned();
    let eig = mat.self_adjoint_eigen(Side::Lower).map_err(|e| {
        BackendError::ExecutionFailed(format!("faer self_adjoint_eigen failed: {e:?}"))
    })?;

    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for Complex<f64> via faer
pub(crate) fn eigh_c64(desc: EighDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let EighDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_column_major_slice(a, n, n).to_owned();
    let eig = mat.self_adjoint_eigen(Side::Lower).map_err(|e| {
        BackendError::ExecutionFailed(format!("faer self_adjoint_eigen failed: {e:?}"))
    })?;

    // Eigenvalues are real for self-adjoint matrices; faer stores as Complex
    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i].re;
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// Self-adjoint eigenvalue decomposition for Complex<f32> via faer
pub(crate) fn eigh_c32(desc: EighDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let EighDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_column_major_slice(a, n, n).to_owned();
    let eig = mat.self_adjoint_eigen(Side::Lower).map_err(|e| {
        BackendError::ExecutionFailed(format!("faer self_adjoint_eigen failed: {e:?}"))
    })?;

    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i].re;
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[j * n + i] = u_mat[(i, j)];
        }
    }

    Ok(())
}
