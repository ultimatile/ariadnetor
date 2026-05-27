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
