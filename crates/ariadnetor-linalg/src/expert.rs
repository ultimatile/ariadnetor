//! Expert layer: per-call [`ExecPolicy`](arnet_core::backend::ExecPolicy) control.
//!
//! The default operation surface (the `*_with_backend` free functions and the
//! `DenseHostOps` / `BlockSparseHostOps` methods) auto-selects a parallelism
//! policy per call by consulting the backend's `par_for_*` hooks. This module
//! is the escape hatch for callers that need to pin the policy explicitly —
//! `Sequential` to dodge faer's small-matrix parallel slowdown, or
//! `Parallel(n)` to opt a large problem into threads the auto-heuristic would
//! leave sequential.
//!
//! The functions are published under bare names (`expert::transpose`, not
//! `expert::transpose_with_policy`): the `expert::` path and the explicit
//! `ExecPolicy` argument already mark the call as the policy-pinned form, so the
//! suffix would be redundant. The defining functions keep their `*_with_policy`
//! names, preserving the pairing with the internal `*_with_policy_dense`
//! kernels they wrap.
//!
//! Only the non-decomposition ops live here today. The decomposition ops
//! (`svd` / `trunc_svd` / `qr` / `lq`) join this surface once their
//! layout-keyed dispatch lands (see issue
//! <https://github.com/ultimatile/ariadnetor/issues/299>); until then their
//! `*_with_policy` forms stay at the crate root.

pub use crate::contract::contract_with_policy as contract;
pub use crate::eigen::{eig_with_policy as eig, eigh_with_policy as eigh};
pub use crate::solve::solve_with_policy as solve;
pub use crate::transpose::transpose_with_policy as transpose;
