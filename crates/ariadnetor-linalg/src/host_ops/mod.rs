//! Ergonomic Host-defaulting operation surface.
//!
//! [`DenseHostOps`] and [`BlockSparseHostOps`] give tensors on the default
//! [`Host`] substrate method forms of the explicit-backend operation paths:
//! each method is a one-line delegation to its `*_with_backend` twin,
//! passing the shared `Host` handle, so the common single-substrate call
//! site can omit the backend argument (`t.svd(nrow)` instead of
//! `svd_with_backend(&backend, &t, nrow)`).
//!
//! The methods derive no authority from the receiver: the handle is always
//! `Host::shared()`, never the tensor's own backend, keeping the
//! call-site-supply discipline intact. A receiver built with a custom
//! backend instance therefore dispatches — and wraps its results — with
//! the shared singleton, not the instance it was built with.

use std::ops::Mul;

use arnet_core::Scalar;
use arnet_tensor::{DenseTensor, Host};

use crate::decomposition::{LqResult, QrResult, SvdResult, TruncSvdParams, TruncSvdResult};
use crate::eigen::{EigResult, EighResult};
use crate::error::LinalgError;
use crate::with_backend::{
    contract_with_backend, diag_with_backend, diagonal_scale_with_backend, eig_with_backend,
    eigh_with_backend, eigvals_with_backend, eigvalsh_with_backend,
    expm_antihermitian_with_backend, expm_hermitian_with_backend, expm_with_backend,
    inverse_with_backend, lq_with_backend, qr_with_backend, solve_with_backend, svd_with_backend,
    trace_with_backend, transpose_with_backend, trunc_svd_with_backend,
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
/// [`crate::einsum_with_backend`] (or [`crate::einsum`]) instead.
pub trait DenseHostOps<T: Scalar> {
    /// Host-defaulting counterpart of [`crate::svd_with_backend`].
    fn svd(&self, nrow: usize) -> Result<SvdResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::trunc_svd_with_backend`].
    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<TruncSvdResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::qr_with_backend`].
    fn qr(&self, nrow: usize) -> Result<QrResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::lq_with_backend`].
    fn lq(&self, nrow: usize) -> Result<LqResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigh_with_backend`].
    fn eigh(&self, nrow: usize) -> Result<EighResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigvalsh_with_backend`].
    fn eigvalsh(&self, nrow: usize) -> Result<DenseTensor<T::Real, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eig_with_backend`].
    fn eig(&self, nrow: usize) -> Result<EigResult<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::eigvals_with_backend`].
    fn eigvals(&self, nrow: usize) -> Result<DenseTensor<T::Complex, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::contract_with_backend`];
    /// the receiver is the left operand.
    fn contract(
        &self,
        rhs: &DenseTensor<T, Host>,
        notation: &str,
    ) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::transpose_with_backend`].
    fn transpose(&self, perm: &[usize]) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::trace_with_backend`].
    fn trace(&self, pairs: &[(usize, usize)]) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::diag_with_backend`].
    fn diag(&self) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::diagonal_scale_with_backend`].
    ///
    /// Narrower than the free fn: the trait's `T: Scalar` bound excludes
    /// the non-`Scalar` element types `diagonal_scale_with_backend`
    /// accepts (`T: Clone + Mul + 'static`); the free fn remains the path
    /// for those.
    fn diagonal_scale<S2>(
        &self,
        weights: &[S2],
        axis: usize,
    ) -> Result<DenseTensor<T, Host>, LinalgError>
    where
        T: Mul<S2, Output = T>,
        S2: Clone;

    /// Host-defaulting counterpart of [`crate::solve_with_backend`];
    /// the receiver is the coefficient matrix `A` in `AX = B`.
    fn solve(
        &self,
        b: &DenseTensor<T, Host>,
        nrow_a: usize,
    ) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::inverse_with_backend`].
    fn inverse(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_with_backend`].
    fn expm(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_hermitian_with_backend`].
    fn expm_hermitian(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError>;

    /// Host-defaulting counterpart of [`crate::expm_antihermitian_with_backend`].
    fn expm_antihermitian(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError>;
}

impl<T: Scalar> DenseHostOps<T> for DenseTensor<T, Host> {
    fn svd(&self, nrow: usize) -> Result<SvdResult<T, Host>, LinalgError> {
        svd_with_backend(&Host::shared(), self, nrow)
    }

    fn trunc_svd(
        &self,
        nrow: usize,
        params: &TruncSvdParams,
    ) -> Result<TruncSvdResult<T, Host>, LinalgError> {
        trunc_svd_with_backend(&Host::shared(), self, nrow, params)
    }

    fn qr(&self, nrow: usize) -> Result<QrResult<T, Host>, LinalgError> {
        qr_with_backend(&Host::shared(), self, nrow)
    }

    fn lq(&self, nrow: usize) -> Result<LqResult<T, Host>, LinalgError> {
        lq_with_backend(&Host::shared(), self, nrow)
    }

    fn eigh(&self, nrow: usize) -> Result<EighResult<T, Host>, LinalgError> {
        eigh_with_backend(&Host::shared(), self, nrow)
    }

    fn eigvalsh(&self, nrow: usize) -> Result<DenseTensor<T::Real, Host>, LinalgError> {
        eigvalsh_with_backend(&Host::shared(), self, nrow)
    }

    fn eig(&self, nrow: usize) -> Result<EigResult<T, Host>, LinalgError> {
        eig_with_backend(&Host::shared(), self, nrow)
    }

    fn eigvals(&self, nrow: usize) -> Result<DenseTensor<T::Complex, Host>, LinalgError> {
        eigvals_with_backend(&Host::shared(), self, nrow)
    }

    fn contract(
        &self,
        rhs: &DenseTensor<T, Host>,
        notation: &str,
    ) -> Result<DenseTensor<T, Host>, LinalgError> {
        contract_with_backend(&Host::shared(), self, rhs, notation)
    }

    fn transpose(&self, perm: &[usize]) -> Result<DenseTensor<T, Host>, LinalgError> {
        transpose_with_backend(&Host::shared(), self, perm)
    }

    fn trace(&self, pairs: &[(usize, usize)]) -> Result<DenseTensor<T, Host>, LinalgError> {
        trace_with_backend(&Host::shared(), self, pairs)
    }

    fn diag(&self) -> Result<DenseTensor<T, Host>, LinalgError> {
        diag_with_backend(&Host::shared(), self)
    }

    fn diagonal_scale<S2>(
        &self,
        weights: &[S2],
        axis: usize,
    ) -> Result<DenseTensor<T, Host>, LinalgError>
    where
        T: Mul<S2, Output = T>,
        S2: Clone,
    {
        diagonal_scale_with_backend(&Host::shared(), self, weights, axis)
    }

    fn solve(
        &self,
        b: &DenseTensor<T, Host>,
        nrow_a: usize,
    ) -> Result<DenseTensor<T, Host>, LinalgError> {
        solve_with_backend(&Host::shared(), self, b, nrow_a)
    }

    fn inverse(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError> {
        inverse_with_backend(&Host::shared(), self, nrow)
    }

    fn expm(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError> {
        expm_with_backend(&Host::shared(), self, nrow)
    }

    fn expm_hermitian(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError> {
        expm_hermitian_with_backend(&Host::shared(), self, nrow)
    }

    fn expm_antihermitian(&self, nrow: usize) -> Result<DenseTensor<T, Host>, LinalgError> {
        expm_antihermitian_with_backend(&Host::shared(), self, nrow)
    }
}
