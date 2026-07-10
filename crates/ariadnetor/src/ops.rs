//! Re-exports of the `ariadnetor_linalg` Tensor-typed free-fn surface.
//!
//! `ariadnetor_linalg` accepts `&DenseTensor<T>` and returns `DenseTensor<T>`, so the
//! umbrella re-exports each call site directly without copy bridges. Every
//! operation takes its backend explicitly; the single-backend ergonomic call
//! site is the [`DenseHostOps`](ariadnetor_linalg::DenseHostOps) /
//! [`BlockSparseHostOps`](ariadnetor_linalg::BlockSparseHostOps) extension trait,
//! re-exported from the crate root.

// ============================================================================
// Result type aliases — re-exported from ariadnetor_linalg
// ============================================================================

pub use ariadnetor_linalg::{
    EigResult, EighResult, LqResult, QrResult, SvdResult, TridiagEighResult, TruncSvdResult,
};

// ============================================================================
// Tensor-keyed dispatch — `svd` / `trunc_svd` / `qr` / `lq` (decomposition),
// `contract` / `tensordot`, and `diagonal_scale` serve both Dense and
// BlockSparse via `LinalgDecompose` / `LinalgContract` / `LinalgScale`. The
// backend is supplied at the call site; the policy-explicit forms live under
// `expert`.
// ============================================================================

pub use ariadnetor_linalg::{
    LinalgContract, LinalgDecompose, LinalgScale, contract, diagonal_scale, lq, qr, svd, tensordot,
    trunc_svd,
};

// ============================================================================
// Explicit-backend free fns — backend supplied at the call site
// ============================================================================

pub use ariadnetor_linalg::{
    diag_with_backend, eig_with_backend, eigh_with_backend, eigvals_with_backend,
    eigvalsh_with_backend, einsum_with_backend, expm_antihermitian_with_backend,
    expm_hermitian_with_backend, expm_with_backend, inverse_with_backend, permute_with_backend,
    solve_with_backend, trace_with_backend, tridiag_eigh_with_backend,
};
