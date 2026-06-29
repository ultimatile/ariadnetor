//! Re-exports of the `arnet_linalg` Tensor-typed free-fn surface.
//!
//! `arnet_linalg` accepts `&DenseTensor<T>` and returns `DenseTensor<T>`, so the
//! umbrella re-exports each call site directly without copy bridges. Every
//! operation takes its backend explicitly; the single-backend ergonomic call
//! site is the [`DenseHostOps`](arnet_linalg::DenseHostOps) /
//! [`BlockSparseHostOps`](arnet_linalg::BlockSparseHostOps) extension trait,
//! re-exported from the crate root.

// ============================================================================
// Result type aliases — re-exported from arnet_linalg
// ============================================================================

pub use arnet_linalg::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// ============================================================================
// Tensor-keyed dispatch — `svd` / `trunc_svd` / `qr` / `lq` (decomposition),
// `contract` / `tensordot`, and `diagonal_scale` serve both Dense and
// BlockSparse via `LinalgDecompose` / `LinalgContract` / `LinalgScale`. The
// backend is supplied at the call site; the policy-explicit forms live under
// `expert`.
// ============================================================================

pub use arnet_linalg::{
    LinalgContract, LinalgDecompose, LinalgScale, contract, diagonal_scale, lq, qr, svd, tensordot,
    trunc_svd,
};

// ============================================================================
// Explicit-backend free fns — backend supplied at the call site
// ============================================================================

pub use arnet_linalg::{
    diag_with_backend, eig_with_backend, eigh_with_backend, eigvals_with_backend,
    eigvalsh_with_backend, einsum_with_backend, expm_antihermitian_with_backend,
    expm_hermitian_with_backend, expm_with_backend, inverse_with_backend, permute_with_backend,
    solve_with_backend, trace_with_backend,
};
