//! Re-exports of the `arnet_linalg` Tensor-typed free-fn surface.
//!
//! `arnet_linalg` accepts `&DenseTensor<T, B>` and returns
//! `DenseTensor<T, B>`, so the umbrella re-exports each call site
//! directly without copy bridges.

// ============================================================================
// Result type aliases — re-exported from arnet_linalg
// ============================================================================

pub use arnet_linalg::{EigResult, EighResult, LqResult, QrResult, SvdResult, TruncSvdResult};

// ============================================================================
// Tensor-typed free fns — re-exported from arnet_linalg
// ============================================================================

pub use arnet_linalg::{
    contract, diag, eig, eigh, eigvals, eigvalsh, einsum, expm, expm_antihermitian, expm_hermitian,
    inverse, lq, qr, solve, svd, trace, transpose, trunc_svd,
};

// ============================================================================
// Explicit-backend free fns — backend supplied at the call site
// ============================================================================

pub use arnet_linalg::{
    contract_with_backend, diag_with_backend, diagonal_scale_with_backend, eig_with_backend,
    eigh_with_backend, eigvals_with_backend, eigvalsh_with_backend, einsum_with_backend,
    expm_antihermitian_with_backend, expm_hermitian_with_backend, expm_with_backend,
    inverse_with_backend, lq_with_backend, qr_with_backend, solve_with_backend, svd_with_backend,
    trace_with_backend, transpose_with_backend, trunc_svd_with_backend,
};
