//! General eigenvalue decomposition implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, EigDescriptor};
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{ComputeEigenvectors, EvdParams, evd_cplx, evd_real, evd_scratch};
use faer::{Mat, MatRef, Spec};
use num_complex::Complex;

use crate::to_faer_par;

/// General eigenvalue decomposition for f64 via faer (real -> complex output).
///
/// Calls `evd_real` to obtain real-Schur-form output (separate real/imag
/// eigenvalue arrays and a real eigenvector matrix with conjugate pairs in
/// adjacent columns), then expands to complex values using the same
/// `real_to_cplx` recipe as faer's `Eigen::new_from_real`.
pub(crate) fn eig_f64(desc: EigDescriptor<'_, f64>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v, policy } = desc;
    let par = to_faer_par(policy);
    let params: Spec<EvdParams, f64> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_re = Diag::<f64>::zeros(n);
    let mut s_im = Diag::<f64>::zeros(n);
    let mut u_real = Mat::<f64>::zeros(n, n);

    let mut buf = MemBuffer::new(evd_scratch::<f64>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    evd_real(
        a_mat,
        s_re.as_mut(),
        s_im.as_mut(),
        None,
        Some(u_real.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer evd_real failed: {e:?}")))?;

    real_to_cplx_f64(n, &s_re, &s_im, &u_real, w, v);

    Ok(())
}

/// General eigenvalue decomposition for f32 via faer (real -> complex output).
pub(crate) fn eig_f32(desc: EigDescriptor<'_, f32>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v, policy } = desc;
    let par = to_faer_par(policy);
    let params: Spec<EvdParams, f32> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_re = Diag::<f32>::zeros(n);
    let mut s_im = Diag::<f32>::zeros(n);
    let mut u_real = Mat::<f32>::zeros(n, n);

    let mut buf = MemBuffer::new(evd_scratch::<f32>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    evd_real(
        a_mat,
        s_re.as_mut(),
        s_im.as_mut(),
        None,
        Some(u_real.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer evd_real failed: {e:?}")))?;

    real_to_cplx_f32(n, &s_re, &s_im, &u_real, w, v);

    Ok(())
}

/// General eigenvalue decomposition for Complex<f64> via faer.
pub(crate) fn eig_c64(desc: EigDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v, policy } = desc;
    let par = to_faer_par(policy);
    let params: Spec<EvdParams, Complex<f64>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<Complex<f64>>::zeros(n);
    let mut u_mat = Mat::<Complex<f64>>::zeros(n, n);

    let mut buf = MemBuffer::new(evd_scratch::<Complex<f64>>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    evd_cplx(
        a_mat,
        s_diag.as_mut(),
        None,
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer evd_cplx failed: {e:?}")))?;

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

/// General eigenvalue decomposition for Complex<f32> via faer.
pub(crate) fn eig_c32(desc: EigDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let EigDescriptor { n, a, w, v, policy } = desc;
    let par = to_faer_par(policy);
    let params: Spec<EvdParams, Complex<f32>> = Default::default();

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let mut s_diag = Diag::<Complex<f32>>::zeros(n);
    let mut u_mat = Mat::<Complex<f32>>::zeros(n, n);

    let mut buf = MemBuffer::new(evd_scratch::<Complex<f32>>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    evd_cplx(
        a_mat,
        s_diag.as_mut(),
        None,
        Some(u_mat.as_mut()),
        par,
        stack,
        params,
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("faer evd_cplx failed: {e:?}")))?;

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

// ---------------------------------------------------------------------------
// real -> complex eigen expansion
// Mirrors faer's Eigen::new_from_real real_to_cplx helper (solvers.rs).
// Assumes conjugate pairs in the real-Schur output occupy adjacent columns:
// s_im[j] != 0 implies (j, j+1) is a pair with s_re[j] == s_re[j+1].
// ---------------------------------------------------------------------------

fn real_to_cplx_f64(
    n: usize,
    s_re: &Diag<f64>,
    s_im: &Diag<f64>,
    u_real: &Mat<f64>,
    w: &mut [Complex<f64>],
    v: &mut [Complex<f64>],
) {
    let mut j = 0;
    while j < n {
        if s_im[j] == 0.0 {
            w[j] = Complex::new(s_re[j], 0.0);
            for i in 0..n {
                v[j * n + i] = Complex::new(u_real[(i, j)], 0.0);
            }
            j += 1;
        } else {
            w[j] = Complex::new(s_re[j], s_im[j]);
            w[j + 1] = Complex::new(s_re[j], -s_im[j]);
            for i in 0..n {
                v[j * n + i] = Complex::new(u_real[(i, j)], u_real[(i, j + 1)]);
                v[(j + 1) * n + i] = Complex::new(u_real[(i, j)], -u_real[(i, j + 1)]);
            }
            j += 2;
        }
    }
}

fn real_to_cplx_f32(
    n: usize,
    s_re: &Diag<f32>,
    s_im: &Diag<f32>,
    u_real: &Mat<f32>,
    w: &mut [Complex<f32>],
    v: &mut [Complex<f32>],
) {
    let mut j = 0;
    while j < n {
        if s_im[j] == 0.0 {
            w[j] = Complex::new(s_re[j], 0.0);
            for i in 0..n {
                v[j * n + i] = Complex::new(u_real[(i, j)], 0.0);
            }
            j += 1;
        } else {
            w[j] = Complex::new(s_re[j], s_im[j]);
            w[j + 1] = Complex::new(s_re[j], -s_im[j]);
            for i in 0..n {
                v[j * n + i] = Complex::new(u_real[(i, j)], u_real[(i, j + 1)]);
                v[(j + 1) * n + i] = Complex::new(u_real[(i, j)], -u_real[(i, j + 1)]);
            }
            j += 2;
        }
    }
}
