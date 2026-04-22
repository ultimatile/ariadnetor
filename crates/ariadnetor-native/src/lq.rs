//! LQ decomposition implementations via adjoint -> QR -> adjoint for all supported scalar types

use arnet_core::backend::{BackendError, LqDescriptor};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::no_pivoting::factor::{
    QrParams, qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::{Conj, Mat, MatRef, Spec};
use num_complex::Complex;

use crate::to_faer_par;

/// Thin LQ for f64: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f64(desc: LqDescriptor<'_, f64>) -> Result<(), BackendError> {
    let LqDescriptor {
        m,
        n,
        a,
        l,
        q,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, f64> = Default::default();
    // QR is on A^T (n×m), so block size is tuned for those dims
    let block_size = recommended_block_size::<f64>(n, m);

    // Transpose A (m×n) into owned (n×m) for in-place QR
    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut at = a_mat.transpose().to_owned();
    let mut q_coeff = Mat::<f64>::zeros(block_size, k);

    let req = qr_in_place_scratch::<f64>(n, m, block_size, par, params)
        .or(apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(n, block_size, k));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(at.as_mut(), q_coeff.as_mut(), par, stack, params);

    // Thin Q_t (n×k) from identity + Householder apply
    let mut q_t = Mat::<f64>::identity(n, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        at.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_t.as_mut(),
        par,
        stack,
    );

    // A = (Q_t * R_t)^T = R_t^T * Q_t^T = L * Q
    // L (m×k, column-major) = R_t^T where R_t is upper-triangular of at (k×m)
    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = if j <= i { at[(j, i)] } else { 0.0 };
        }
    }

    // Q (k×n, column-major) = Q_t^T
    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)];
        }
    }

    Ok(())
}

/// Thin LQ for f32: A = L * Q, computed via QR of A^T
pub(crate) fn lq_f32(desc: LqDescriptor<'_, f32>) -> Result<(), BackendError> {
    let LqDescriptor {
        m,
        n,
        a,
        l,
        q,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, f32> = Default::default();
    let block_size = recommended_block_size::<f32>(n, m);

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut at = a_mat.transpose().to_owned();
    let mut q_coeff = Mat::<f32>::zeros(block_size, k);

    let req = qr_in_place_scratch::<f32>(n, m, block_size, par, params)
        .or(apply_block_householder_sequence_on_the_left_in_place_scratch::<f32>(n, block_size, k));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(at.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_t = Mat::<f32>::identity(n, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        at.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_t.as_mut(),
        par,
        stack,
    );

    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = if j <= i { at[(j, i)] } else { 0.0 };
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
    let LqDescriptor {
        m,
        n,
        a,
        l,
        q,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, Complex<f64>> = Default::default();
    let block_size = recommended_block_size::<Complex<f64>>(n, m);

    // A^H (n×m) via conjugate transpose into owned storage
    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut ah = a_mat.adjoint().to_owned();
    let mut q_coeff = Mat::<Complex<f64>>::zeros(block_size, k);

    let req = qr_in_place_scratch::<Complex<f64>>(n, m, block_size, par, params).or(
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex<f64>>(
            n, block_size, k,
        ),
    );
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(ah.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_t = Mat::<Complex<f64>>::identity(n, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        ah.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_t.as_mut(),
        par,
        stack,
    );

    // A = (A^H)^H = (Q_t * R_t)^H = R_t^H * Q_t^H = L * Q
    // L (m×k, column-major) = R_t^H where R_t is upper-triangular of ah
    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = if j <= i {
                ah[(j, i)].conj()
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    // Q (k×n, column-major) = Q_t^H
    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}

/// Thin LQ for Complex<f32>: A = L * Q, computed via QR of A^H
pub(crate) fn lq_c32(desc: LqDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let LqDescriptor {
        m,
        n,
        a,
        l,
        q,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, Complex<f32>> = Default::default();
    let block_size = recommended_block_size::<Complex<f32>>(n, m);

    let a_mat = MatRef::from_column_major_slice(a, m, n);
    let mut ah = a_mat.adjoint().to_owned();
    let mut q_coeff = Mat::<Complex<f32>>::zeros(block_size, k);

    let req = qr_in_place_scratch::<Complex<f32>>(n, m, block_size, par, params).or(
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex<f32>>(
            n, block_size, k,
        ),
    );
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(ah.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_t = Mat::<Complex<f32>>::identity(n, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        ah.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_t.as_mut(),
        par,
        stack,
    );

    for i in 0..m {
        for j in 0..k {
            l[j * m + i] = if j <= i {
                ah[(j, i)].conj()
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    for i in 0..k {
        for j in 0..n {
            q[j * k + i] = q_t[(j, i)].conj();
        }
    }

    Ok(())
}
