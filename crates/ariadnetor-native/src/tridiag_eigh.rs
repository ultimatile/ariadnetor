//! Real symmetric tridiagonal eigenvalue decomposition via faer.
//!
//! Wraps `faer::linalg::evd::tridiagonal_self_adjoint_evd`, which skips
//! the O(n^3) Householder reduction that the dense self-adjoint path
//! performs and runs directly on the diagonal / subdiagonal input. Only
//! real kernels exist: a real symmetric tridiagonal matrix has a fully
//! real eigensystem, and complex instantiations are rejected at the
//! dispatch layer in `lib.rs`.
//!
//! faer exposes no tridiagonal-specific scratch function (the inner
//! `tridiag_evd` module is private), so the buffers are sized with
//! `self_adjoint_evd_scratch`, whose layout is a strict superset of the
//! tridiagonal path's needs: requested with `ComputeEigenvectors::Yes`,
//! it covers the two real work columns and the divide-and-conquer
//! scratch that `tridiagonal_self_adjoint_evd` consumes.

use ariadnetor_core::backend::{BackendError, TridiagEighDescriptor};
use faer::diag::{DiagMut, DiagRef};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
    ComputeEigenvectors, SelfAdjointEvdParams, self_adjoint_evd_scratch,
    tridiagonal_self_adjoint_evd,
};
use faer::{MatMut, Spec};

use crate::to_faer_par;

/// Validate descriptor slice lengths before any faer call.
///
/// faer surfaces dimension misuse as a panic (an out-of-bounds
/// assertion deep inside the kernel), so the wrapper front-loads the
/// checks and returns a recoverable [`BackendError::InvalidArgument`]
/// instead.
fn validate_lengths<T>(desc: &TridiagEighDescriptor<'_, T>) -> Result<(), BackendError>
where
    T: ariadnetor_core::Scalar,
{
    let TridiagEighDescriptor { n, d, e, w, v, .. } = desc;
    let n = *n;
    if n < 1 {
        return Err(BackendError::InvalidArgument(
            "tridiag_eigh: n must be >= 1".into(),
        ));
    }
    if d.len() != n || e.len() != n - 1 || w.len() != n || v.len() != n * n {
        return Err(BackendError::InvalidArgument(format!(
            "tridiag_eigh: expected d/e/w/v lengths {}/{}/{}/{}, got {}/{}/{}/{}",
            n,
            n - 1,
            n,
            n * n,
            d.len(),
            e.len(),
            w.len(),
            v.len(),
        )));
    }
    Ok(())
}

/// Real symmetric tridiagonal eigenvalue decomposition for f64 via faer.
pub(crate) fn tridiag_eigh_f64(desc: TridiagEighDescriptor<'_, f64>) -> Result<(), BackendError> {
    validate_lengths(&desc)?;
    let TridiagEighDescriptor {
        n,
        d,
        e,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, f64> = Default::default();

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    // faer writes every entry of `s` and `u` itself (the QR path fills
    // `u` with the identity before applying rotations), so the
    // descriptor's output buffers can be handed over directly — no
    // intermediate allocation or copy-back is needed.
    tridiagonal_self_adjoint_evd(
        DiagRef::from_slice(d),
        DiagRef::from_slice(e),
        DiagMut::from_slice_mut(w),
        Some(MatMut::from_column_major_slice_mut(v, n, n)),
        par,
        stack,
        params,
    )
    .map_err(|e| {
        BackendError::ExecutionFailed(format!("faer tridiagonal_self_adjoint_evd failed: {e:?}"))
    })
}

/// Real symmetric tridiagonal eigenvalue decomposition for f32 via faer.
pub(crate) fn tridiag_eigh_f32(desc: TridiagEighDescriptor<'_, f32>) -> Result<(), BackendError> {
    validate_lengths(&desc)?;
    let TridiagEighDescriptor {
        n,
        d,
        e,
        w,
        v,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<SelfAdjointEvdParams, f32> = Default::default();

    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<f32>(
        n,
        ComputeEigenvectors::Yes,
        par,
        params,
    ));
    let stack = MemStack::new(&mut buf);

    // Same direct-write rationale as the f64 kernel above.
    tridiagonal_self_adjoint_evd(
        DiagRef::from_slice(d),
        DiagRef::from_slice(e),
        DiagMut::from_slice_mut(w),
        Some(MatMut::from_column_major_slice_mut(v, n, n)),
        par,
        stack,
        params,
    )
    .map_err(|e| {
        BackendError::ExecutionFailed(format!("faer tridiagonal_self_adjoint_evd failed: {e:?}"))
    })
}
