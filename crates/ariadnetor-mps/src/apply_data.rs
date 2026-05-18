//! `TensorData`-typed apply shims — delegate to the `*_repr` bodies
//! in [`super::apply`] by bumping each site's storage `Arc` into a
//! temporary `*Repr` chain, running the matching legacy body, and
//! wrapping the result back.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_tensor::{
    BlockSparse, BlockSparseLayout, BlockSparseStorage, Dense, DenseLayout, DenseStorage, Sector,
    TensorData,
};

use super::apply::{
    apply_bsp_repr, apply_dense_repr, apply_zipup_bsp_repr, apply_zipup_dense_repr,
};
use super::types::{Mpo, MpoRepr, Mps, MpsRepr, TruncateParams};

fn dense_sites_to_repr<T: Scalar>(
    sites: &[TensorData<DenseStorage<T>, DenseLayout>],
) -> Vec<Dense<T>> {
    sites
        .iter()
        .map(|td| Dense::from_tensor_data(td.clone()))
        .collect()
}

fn dense_repr_to_sites<T: Scalar>(
    storages: Vec<Dense<T>>,
) -> Vec<TensorData<DenseStorage<T>, DenseLayout>> {
    storages.into_iter().map(|d| d.into_tensor_data()).collect()
}

fn bsp_sites_to_repr<T: Scalar, S: Sector>(
    sites: &[TensorData<BlockSparseStorage<T>, BlockSparseLayout<S>>],
) -> Vec<BlockSparse<T, S>> {
    sites
        .iter()
        .map(|td| BlockSparse::from_tensor_data(td.clone()))
        .collect()
}

fn bsp_repr_to_sites<T: Scalar, S: Sector>(
    storages: Vec<BlockSparse<T, S>>,
    order: arnet_core::MemoryOrder,
) -> Vec<TensorData<BlockSparseStorage<T>, BlockSparseLayout<S>>> {
    storages
        .into_iter()
        .map(|bs| bs.into_tensor_data(order))
        .collect()
}

/// `Mps<DenseStorage<T>, DenseLayout, B>` naive-apply shim.
pub(super) fn apply_dense<T, B>(
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    params: Option<&TruncateParams>,
) -> Mps<DenseStorage<T>, DenseLayout, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(&psi.0.backend);
    let op_repr = MpoRepr::with_backend(dense_sites_to_repr(&op.0.sites), Arc::clone(&backend_arc));
    let psi_repr =
        MpsRepr::with_backend(dense_sites_to_repr(&psi.0.sites), Arc::clone(&backend_arc));
    let result_repr = apply_dense_repr(&op_repr, &psi_repr, params);
    let mut result = Mps::with_backend(dense_repr_to_sites(result_repr.0.storages), backend_arc);
    result.0.canonical_form = result_repr.0.canonical_form;
    result
}

/// `Mps<DenseStorage<T>, DenseLayout, B>` zip-up apply shim.
pub(super) fn apply_zipup_dense<T, B>(
    op: &Mpo<DenseStorage<T>, DenseLayout, B>,
    psi: &Mps<DenseStorage<T>, DenseLayout, B>,
    params: Option<&TruncateParams>,
) -> Mps<DenseStorage<T>, DenseLayout, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(&psi.0.backend);
    let op_repr = MpoRepr::with_backend(dense_sites_to_repr(&op.0.sites), Arc::clone(&backend_arc));
    let psi_repr =
        MpsRepr::with_backend(dense_sites_to_repr(&psi.0.sites), Arc::clone(&backend_arc));
    let result_repr = apply_zipup_dense_repr(&op_repr, &psi_repr, params);
    let mut result = Mps::with_backend(dense_repr_to_sites(result_repr.0.storages), backend_arc);
    result.0.canonical_form = result_repr.0.canonical_form;
    result
}

/// `Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>` naive-apply
/// shim. Output sites are retagged at `backend.preferred_order()`,
/// matching the `arnet_linalg::*_block_sparse_repr` decomposition
/// convention.
pub(super) fn apply_bsp<T, S, B>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(&psi.0.backend);
    let preferred = backend_arc.preferred_order();
    let op_repr = MpoRepr::with_backend(bsp_sites_to_repr(&op.0.sites), Arc::clone(&backend_arc));
    let psi_repr = MpsRepr::with_backend(bsp_sites_to_repr(&psi.0.sites), Arc::clone(&backend_arc));
    let result_repr = apply_bsp_repr(&op_repr, &psi_repr, params);
    let mut result = Mps::with_backend(
        bsp_repr_to_sites(result_repr.0.storages, preferred),
        backend_arc,
    );
    result.0.canonical_form = result_repr.0.canonical_form;
    result
}

/// `Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>` zip-up apply
/// shim. Output sites are retagged at `backend.preferred_order()`.
pub(super) fn apply_zipup_bsp<T, S, B>(
    op: &Mpo<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    psi: &Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>,
    params: Option<&TruncateParams>,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let backend_arc = Arc::clone(&psi.0.backend);
    let preferred = backend_arc.preferred_order();
    let op_repr = MpoRepr::with_backend(bsp_sites_to_repr(&op.0.sites), Arc::clone(&backend_arc));
    let psi_repr = MpsRepr::with_backend(bsp_sites_to_repr(&psi.0.sites), Arc::clone(&backend_arc));
    let result_repr = apply_zipup_bsp_repr(&op_repr, &psi_repr, params);
    let mut result = Mps::with_backend(
        bsp_repr_to_sites(result_repr.0.storages, preferred),
        backend_arc,
    );
    result.0.canonical_form = result_repr.0.canonical_form;
    result
}
