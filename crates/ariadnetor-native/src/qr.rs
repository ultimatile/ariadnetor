//! QR decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, QrDescriptor};
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

/// Thin QR for f64 via faer: A = Q * R
pub(crate) fn qr_f64(desc: QrDescriptor<'_, f64>) -> Result<(), BackendError> {
    let QrDescriptor {
        m,
        n,
        a,
        q,
        r,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, f64> = Default::default();
    let block_size = recommended_block_size::<f64>(m, n);

    let mut a_owned = MatRef::from_column_major_slice(a, m, n).to_owned();
    let mut q_coeff = Mat::<f64>::zeros(block_size, k);

    let req = qr_in_place_scratch::<f64>(m, n, block_size, par, params)
        .or(apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(m, block_size, k));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(a_owned.as_mut(), q_coeff.as_mut(), par, stack, params);

    // Build thin Q by applying Householder sequence to identity(m, k)
    let mut q_mat = Mat::<f64>::identity(m, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        a_owned.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_mat.as_mut(),
        par,
        stack,
    );

    // Q (m×k, column-major)
    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    // R (k×n, column-major) — upper triangle of a_owned; strict lower = 0
    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = if i <= j { a_owned[(i, j)] } else { 0.0 };
        }
    }

    Ok(())
}

/// Thin QR for f32 via faer: A = Q * R
pub(crate) fn qr_f32(desc: QrDescriptor<'_, f32>) -> Result<(), BackendError> {
    let QrDescriptor {
        m,
        n,
        a,
        q,
        r,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, f32> = Default::default();
    let block_size = recommended_block_size::<f32>(m, n);

    let mut a_owned = MatRef::from_column_major_slice(a, m, n).to_owned();
    let mut q_coeff = Mat::<f32>::zeros(block_size, k);

    let req = qr_in_place_scratch::<f32>(m, n, block_size, par, params)
        .or(apply_block_householder_sequence_on_the_left_in_place_scratch::<f32>(m, block_size, k));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(a_owned.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_mat = Mat::<f32>::identity(m, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        a_owned.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = if i <= j { a_owned[(i, j)] } else { 0.0 };
        }
    }

    Ok(())
}

/// Thin QR for Complex<f64> via faer: A = Q * R
pub(crate) fn qr_c64(desc: QrDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let QrDescriptor {
        m,
        n,
        a,
        q,
        r,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, Complex<f64>> = Default::default();
    let block_size = recommended_block_size::<Complex<f64>>(m, n);

    let mut a_owned = MatRef::from_column_major_slice(a, m, n).to_owned();
    let mut q_coeff = Mat::<Complex<f64>>::zeros(block_size, k);

    let req = qr_in_place_scratch::<Complex<f64>>(m, n, block_size, par, params).or(
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex<f64>>(
            m, block_size, k,
        ),
    );
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(a_owned.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_mat = Mat::<Complex<f64>>::identity(m, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        a_owned.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = if i <= j {
                a_owned[(i, j)]
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    Ok(())
}

/// Thin QR for Complex<f32> via faer: A = Q * R
pub(crate) fn qr_c32(desc: QrDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let QrDescriptor {
        m,
        n,
        a,
        q,
        r,
        order: _,
        policy,
    } = desc;
    let k = m.min(n);
    let par = to_faer_par(policy);
    let params: Spec<QrParams, Complex<f32>> = Default::default();
    let block_size = recommended_block_size::<Complex<f32>>(m, n);

    let mut a_owned = MatRef::from_column_major_slice(a, m, n).to_owned();
    let mut q_coeff = Mat::<Complex<f32>>::zeros(block_size, k);

    let req = qr_in_place_scratch::<Complex<f32>>(m, n, block_size, par, params).or(
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex<f32>>(
            m, block_size, k,
        ),
    );
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    qr_in_place(a_owned.as_mut(), q_coeff.as_mut(), par, stack, params);

    let mut q_mat = Mat::<Complex<f32>>::identity(m, k);
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        a_owned.as_ref(),
        q_coeff.as_ref(),
        Conj::No,
        q_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..m {
        for j in 0..k {
            q[j * m + i] = q_mat[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            r[j * k + i] = if i <= j {
                a_owned[(i, j)]
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    Ok(())
}
