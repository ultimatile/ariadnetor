//! General eigenvalue decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, EigDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// General eigenvalue decomposition for f64 via faer (real -> complex output)
pub(crate) fn eig_f64(desc: EigDescriptor<'_, f64>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_row_major_slice(a, n, n).to_owned();
    let eig = mat.eigen().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer eigen failed: {e:?}"))
    })?;

    // Eigenvalues (complex)
    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    // Eigenvectors (complex, n*n, row-major)
    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// General eigenvalue decomposition for f32 via faer (real -> complex output)
pub(crate) fn eig_f32(desc: EigDescriptor<'_, f32>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_row_major_slice(a, n, n).to_owned();
    let eig = mat.eigen().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer eigen failed: {e:?}"))
    })?;

    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// General eigenvalue decomposition for Complex<f64> via faer
pub(crate) fn eig_c64(desc: EigDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_row_major_slice(a, n, n).to_owned();
    let eig = mat.eigen().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer eigen failed: {e:?}"))
    })?;

    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = u_mat[(i, j)];
        }
    }

    Ok(())
}

/// General eigenvalue decomposition for Complex<f32> via faer
pub(crate) fn eig_c32(desc: EigDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v } = desc;

    let mat = MatRef::from_row_major_slice(a, n, n).to_owned();
    let eig = mat.eigen().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer eigen failed: {e:?}"))
    })?;

    let s_diag = eig.S();
    for i in 0..n {
        w[i] = s_diag[i];
    }

    let u_mat = eig.U();
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = u_mat[(i, j)];
        }
    }

    Ok(())
}
