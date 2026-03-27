//! MPO-MPS application: apply an MPO to an MPS

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::contract;
use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

use super::chain::TensorChain;
use super::types::{Mpo, Mps, TruncateParams};

/// Apply an MPO to an MPS, producing a new MPS.
///
/// For each site, contracts the MPO tensor (rank-4) with the MPS tensor
/// (rank-3) over the physical index, then fuses the bond dimensions:
///
/// ```text
/// W[w_L, d_ket, d_bra, w_R] × A[χ_L, d_ket, χ_R]
///   → result[w_L*χ_L, d_bra, w_R*χ_R]
/// ```
///
/// If `params` is `Some`, the result is orthogonalized and truncated.
/// If `None`, the exact (lossless) result is returned with `Unknown`
/// canonical form.
///
/// # Panics
///
/// Panics if the MPO and MPS have different lengths or either is empty.
pub fn apply<T, B>(op: &Mpo<T, B>, psi: &Mps<T, B>, params: Option<&TruncateParams>) -> Mps<T, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();
    let mut storages = Vec::with_capacity(n);

    for j in 0..n {
        let w = as_dense(op.storage(j)); // (w_L, d_ket, d_bra, w_R)
        let a = as_dense(psi.storage(j)); // (χ_L, d, χ_R)

        // Contract over physical index (b = d_ket = d):
        // W(a,b,c,d) × A(e,b,f) → result(a,e,c,d,f)
        // = (w_L, χ_L, d_bra, w_R, χ_R)
        let result =
            contract(backend, w, a, "abcd,ebf->aecdf").expect("MPO-MPS contraction failed");

        let shape = result.shape();
        let w_l = shape[0];
        let chi_l = shape[1];
        let d_bra = shape[2];
        let w_r = shape[3];
        let chi_r = shape[4];

        // Fuse bond dimensions: (w_L*χ_L, d_bra, w_R*χ_R)
        // Convert to row-major so reshape uses standard axis merge order.
        let result = result.to_contiguous(MemoryOrder::RowMajor);
        let fused = result.reshape(vec![w_l * chi_l, d_bra, w_r * chi_r]);

        storages.push(TensorStorage::Dense(fused));
    }

    let mut result_mps = Mps::with_backend(storages, Arc::clone(psi.backend_arc()));

    if let Some(trunc_params) = params {
        super::orthogonalize(&mut result_mps, 0);
        super::truncate(&mut result_mps, trunc_params);
    }

    result_mps
}

fn as_dense<T>(storage: &TensorStorage<T>) -> &DenseTensor<T> {
    match storage {
        TensorStorage::Dense(d) => d,
    }
}
