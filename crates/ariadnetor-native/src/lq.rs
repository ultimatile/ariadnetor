//! LQ decomposition implementations via adjoint -> QR -> adjoint for all supported scalar types

use arnet_core::backend::{BackendError, LqDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// Thin LQ for f64: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f64(desc: LqDescriptor<'_, f64>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // Transpose A (m×n, column-major) -> A^T (n×m)
    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    // QR of A^T: A^T = Q_t * R_t where Q_t is n×k, R_t is k×m
    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (Q_t * R_t)^T = R_t^T * Q_t^T = L * Q
    // L = R_t^T (m×k, column-major)
    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = r_t[(j, i)];
        }
    }

    // Q = Q_t^T (k×n, column-major)
    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for f32: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f32(desc: LqDescriptor<'_, f32>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let at = a_mat.transpose().to_owned();

    let qr = at.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = r_t[(j, i)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f64>: A = L * Q, computed via QR of A^H
pub(crate) fn lq_c64(desc: LqDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    // A^H (n×m) via conjugate transpose
    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    // QR of A^H: A^H = Q_t * R_t where Q_t is n×k, R_t is k×m
    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    // A = (A^H)^H = (Q_t * R_t)^H = R_t^H * Q_t^H = L * Q
    // L = R_t^H (m×k, column-major)
    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = r_t[(j, i)].conj();
        }
    }

    // Q = Q_t^H (k×n, column-major)
    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f32>: A = L * Q, computed via QR of A^H
pub(crate) fn lq_c32(desc: LqDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let LqDescriptor { m, n, a, l, q } = desc;
    let k = m.min(n);

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let ah = a_mat.adjoint().to_owned();

    let qr = ah.qr();
    let q_t = qr.compute_thin_Q();
    let r_t = qr.thin_R();

    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = r_t[(j, i)].conj();
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}
