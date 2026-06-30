//! Ergonomic Host-defaulting operation surface.
//!
//! [`DenseHostOps`] and [`BlockSparseHostOps`] give dense / block-sparse
//! tensors method forms of the explicit-backend operation paths: each method
//! is a one-line delegation to its call-site-backend twin, passing the shared
//! [`Host`] handle, so the common single-substrate call site can omit the
//! backend argument (`t.svd(nrow)` instead of `svd(&backend, &t, nrow)`).
//!
//! The handle is always `Host::shared()`, keeping the call-site-supply
//! discipline intact: the operation dispatches on — and the result is built
//! by — the shared `Host` substrate, spelled through the [`Host`] alias rather
//! than a concrete backend type.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{DenseTensor, Host};

use crate::contract_dispatch::contract;
use crate::decompose_dispatch::{lq, qr, svd, trunc_svd};
use crate::decomposition::{LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult};
use crate::eigen::{EigResult, EighResult};
use crate::error::LinalgError;
use crate::scale_dispatch::diagonal_scale;
use crate::with_backend::{
    diag_with_backend, eig_with_backend, eigh_with_backend, eigvals_with_backend,
    eigvalsh_with_backend, expm_antihermitian_with_backend, expm_hermitian_with_backend,
    expm_with_backend, inverse_with_backend, permute_with_backend, solve_with_backend,
    trace_with_backend,
};

mod block_sparse;

pub use block_sparse::BlockSparseHostOps;

#[cfg(test)]
mod tests;

/// Host-defaulting method forms of the dense explicit-backend operations.
///
/// `einsum` has no method form: it takes its operands as a slice with no
/// natural receiver, and a receiver-plus-rest shape would change the
/// slice's meaning from "all operands" to "remaining operands". Use
/// [`crate::einsum_with_backend`] instead.
pub trait DenseHostOps<T: Scalar> {
    /// Host-defaulting counterpart of [`crate::svd`].
    fn svd(&self, nrow: usize) -> Result<SvdResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::trunc_svd`].
    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<TruncSvdResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::qr`].
    fn qr(&self, nrow: usize) -> Result<QrResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::lq`].
    fn lq(&self, nrow: usize) -> Result<LqResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigh_with_backend`].
    fn eigh(&self, nrow: usize) -> Result<EighResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigvalsh_with_backend`].
    fn eigvalsh(&self, nrow: usize) -> Result<DenseTensor<T::Real>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eig_with_backend`].
    fn eig(&self, nrow: usize) -> Result<EigResult<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigvals_with_backend`].
    fn eigvals(&self, nrow: usize) -> Result<DenseTensor<T::Complex>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::contract`];
    /// the receiver is the left operand.
    fn contract(&self, rhs: &DenseTensor<T>, notation: &str)
    -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::permute_with_backend`].
    fn permute(&self, perm: &[usize]) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::trace_with_backend`].
    fn trace(&self, pairs: &[(usize, usize)]) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::diag_with_backend`].
    fn diag(&self) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::diagonal_scale`].
    fn diagonal_scale(
        &self,
        weights: &[T::Real],
        axis: usize,
    ) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::solve_with_backend`];
    /// the receiver is the coefficient matrix `A` in `AX = B`.
    fn solve(&self, b: &DenseTensor<T>, nrow_a: usize) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::inverse_with_backend`].
    fn inverse(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_with_backend`].
    fn expm(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_hermitian_with_backend`].
    fn expm_hermitian(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_antihermitian_with_backend`].
    fn expm_antihermitian(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError>;
}

impl<T: Scalar> DenseHostOps<T> for DenseTensor<T> {
    fn svd(&self, nrow: usize) -> Result<SvdResult<T>, LinalgError> {
        svd(Host::shared().as_ref(), self, nrow)
    }

    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<TruncSvdResult<T>, LinalgError> {
        trunc_svd(Host::shared().as_ref(), self, nrow, params)
    }

    fn qr(&self, nrow: usize) -> Result<QrResult<T>, LinalgError> {
        qr(Host::shared().as_ref(), self, nrow)
    }

    fn lq(&self, nrow: usize) -> Result<LqResult<T>, LinalgError> {
        lq(Host::shared().as_ref(), self, nrow)
    }

    fn eigh(&self, nrow: usize) -> Result<EighResult<T>, LinalgError> {
        eigh_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eigvalsh(&self, nrow: usize) -> Result<DenseTensor<T::Real>, LinalgError> {
        eigvalsh_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eig(&self, nrow: usize) -> Result<EigResult<T>, LinalgError> {
        eig_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn eigvals(&self, nrow: usize) -> Result<DenseTensor<T::Complex>, LinalgError> {
        eigvals_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn contract(
        &self,
        rhs: &DenseTensor<T>,
        notation: &str,
    ) -> Result<DenseTensor<T>, LinalgError> {
        contract(Host::shared().as_ref(), self, rhs, notation)
    }

    fn permute(&self, perm: &[usize]) -> Result<DenseTensor<T>, LinalgError> {
        permute_with_backend(Host::shared().as_ref(), self, perm)
    }

    fn trace(&self, pairs: &[(usize, usize)]) -> Result<DenseTensor<T>, LinalgError> {
        trace_with_backend(Host::shared().as_ref(), self, pairs)
    }

    fn diag(&self) -> Result<DenseTensor<T>, LinalgError> {
        diag_with_backend(Host::shared().as_ref(), self)
    }

    fn diagonal_scale(
        &self,
        weights: &[T::Real],
        axis: usize,
    ) -> Result<DenseTensor<T>, LinalgError> {
        diagonal_scale(Host::shared().as_ref(), self, weights, axis)
    }

    fn solve(&self, b: &DenseTensor<T>, nrow_a: usize) -> Result<DenseTensor<T>, LinalgError> {
        solve_with_backend(Host::shared().as_ref(), self, b, nrow_a)
    }

    fn inverse(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError> {
        inverse_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError> {
        expm_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm_hermitian(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError> {
        expm_hermitian_with_backend(Host::shared().as_ref(), self, nrow)
    }

    fn expm_antihermitian(&self, nrow: usize) -> Result<DenseTensor<T>, LinalgError> {
        expm_antihermitian_with_backend(Host::shared().as_ref(), self, nrow)
    }
}
