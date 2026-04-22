//! QR decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, QrDescriptor};
use faer::MatRef;
use num_complex::Complex;

/// Thin QR for f64 via faer: A = Q * R
pub(crate) fn qr_f64(desc: QrDescriptor<'_, f64>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r, .. } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    // Q (m×k, column-major)
    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    // R (k×n, column-major)
    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for f32 via faer: A = Q * R
pub(crate) fn qr_f32(desc: QrDescriptor<'_, f32>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r, .. } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f64> via faer: A = Q * R
pub(crate) fn qr_c64(desc: QrDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r, .. } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = r_mat[(i, j)];
        }
    }

    Ok(())
}

/// Thin QR for Complex<f32> via faer: A = Q * R
pub(crate) fn qr_c32(desc: QrDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let QrDescriptor { m, n, a, q, r, .. } = desc;
    let k = m.min(n);

    let mat = MatRef::from_column_major_slice(a, m, n).to_owned();
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    let r_mat = qr.thin_R();
    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = r_mat[(i, j)];
        }
    }

    Ok(())
}
