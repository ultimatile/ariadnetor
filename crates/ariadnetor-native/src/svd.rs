//! SVD implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, SvdDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// Thin SVD for f64 via faer: A = U * diag(S) * Vt
pub(crate) fn svd_f64(desc: SvdDescriptor<'_, f64>) -> Result<(), BackendError> {
    let SvdDescriptor {
        m, n, a, u, s, vt, ..
    } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let thin = mat
        .thin_svd()
        .map_err(|e| BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}")))?;

    // U (m×k, column-major)
    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    // Singular values (k)
    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    // Vt = V^T (k×n, column-major)
    let v_mat = thin.V();
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
        m, n, a, u, s, vt, ..
    } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let thin = mat
        .thin_svd()
        .map_err(|e| BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}")))?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    let v_mat = thin.V();
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
        m, n, a, u, s, vt, ..
    } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let thin = mat
        .thin_svd()
        .map_err(|e| BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}")))?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    // Singular values are always real; faer stores them as Complex with im=0
    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i].re;
    }

    // Vt = V^H (conjugate transpose, k×n, column-major)
    let v_mat = thin.V();
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
        m, n, a, u, s, vt, ..
    } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let thin = mat
        .thin_svd()
        .map_err(|e| BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}")))?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i].re;
    }

    // Vt = V^H (conjugate transpose, k×n, column-major)
    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[j * k + i] = v_mat[(j, i)].conj();
        }
    }

    Ok(())
}
