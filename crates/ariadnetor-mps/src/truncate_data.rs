//! `TensorData`-typed truncate shims — delegate to the `*_repr`
//! bodies in [`super::truncate`] by bumping each chain site's
//! storage `Arc` into a temporary `*Repr` chain, running the
//! matching legacy body, and writing the truncated sites back.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{
    BlockSparse, BlockSparseLayout, BlockSparseStorage, Dense, DenseLayout, DenseStorage, Sector,
};

use super::chain::TensorChain;
use super::truncate::{truncate_bsp_repr, truncate_dense_repr};
use super::types::{MpsRepr, TruncResult, TruncateParams};

/// `TensorChain<DenseStorage<T>, DenseLayout, B>` truncate shim.
///
/// Bumps each site's `Arc` buffer into a temporary
/// [`MpsRepr<Dense<T>, B>`], runs [`truncate_dense_repr`], then
/// writes the truncated sites back via the same Arc bump.
pub(super) fn truncate_dense<T, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    B: ComputeBackend,
    C: TensorChain<DenseStorage<T>, DenseLayout, B>,
{
    let n = chain.len();
    let backend_arc = Arc::clone(chain.backend_arc());
    let prior_form = chain.canonical_form().clone();
    let dense_storages: Vec<Dense<T>> = (0..n)
        .map(|j| Dense::from_tensor_data(chain.site(j).clone()))
        .collect();
    let mut repr = MpsRepr::with_backend(dense_storages, backend_arc);
    repr.0.canonical_form = prior_form;
    let result = truncate_dense_repr(&mut repr, params);

    for (j, dense) in repr.0.storages.into_iter().enumerate() {
        *chain.site_mut(j) = dense.into_tensor_data();
    }
    chain.set_canonical_form(repr.0.canonical_form);
    result
}

/// `TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>`
/// truncate shim. Output sites are retagged at
/// `backend.preferred_order()`, matching the
/// `arnet_linalg::*_block_sparse_repr` decomposition convention.
pub(super) fn truncate_bsp<T, S, B, C>(chain: &mut C, params: &TruncateParams) -> TruncResult<T>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
    C: TensorChain<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
{
    let n = chain.len();
    let backend_arc = Arc::clone(chain.backend_arc());
    let prior_form = chain.canonical_form().clone();
    let bsp_storages: Vec<BlockSparse<T, S>> = (0..n)
        .map(|j| BlockSparse::from_tensor_data(chain.site(j).clone()))
        .collect();
    let mut repr = MpsRepr::with_backend(bsp_storages, backend_arc);
    repr.0.canonical_form = prior_form;
    let result = truncate_bsp_repr(&mut repr, params);

    let preferred = chain.backend().preferred_order();
    for (j, bsp) in repr.0.storages.into_iter().enumerate() {
        *chain.site_mut(j) = bsp.into_tensor_data(preferred);
    }
    chain.set_canonical_form(repr.0.canonical_form);
    result
}
