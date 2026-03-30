//! Shared test helpers for MPS tests.

use arnet::mps::{Mpo, Mps, TensorChain};
use arnet_tensor::{Dense, MemoryOrder};

/// Build a random-ish 4-site MPS from deterministic data.
pub fn make_4site_mps() -> Mps<f64> {
    let storages = vec![
        Dense::from_data_with_order(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![1, 2, 4],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            (1..=32).map(|i| i as f64 * 0.1).collect(),
            vec![4, 2, 4],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            (1..=24).map(|i| i as f64 * 0.1).collect(),
            vec![4, 2, 3],
            MemoryOrder::RowMajor,
        ),
        Dense::from_data_with_order(
            (1..=6).map(|i| i as f64 * 0.1).collect(),
            vec![3, 2, 1],
            MemoryOrder::RowMajor,
        ),
    ];
    Mps::from_storages(storages)
}

/// Check that a site tensor is left-canonical: Q^H Q ≈ I.
/// Reshape to (m, k) where m = product(shape[..rank-1]), k = shape[rank-1].
pub fn is_left_canonical(storage: &Dense<f64>, tol: f64) -> bool {
    let dense = match storage {
        d => d,
    };
    let shape = dense.shape();
    let rank = shape.len();
    let k = shape[rank - 1];
    let m: usize = shape[..rank - 1].iter().product();
    let mat = dense.reshape(vec![m, k]);

    let backend = arnet_native::NativeBackend::new();
    let qtq = arnet_linalg::contract(&backend, &mat, &mat, "ab,ac->bc").unwrap();

    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qtq.get(&[i, j]) - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check that a site tensor is right-canonical: Q Q^H ≈ I.
/// Reshape to (k, n) where k = shape[0], n = product(shape[1..]).
pub fn is_right_canonical(storage: &Dense<f64>, tol: f64) -> bool {
    let dense = match storage {
        d => d,
    };
    let shape = dense.shape();
    let k = shape[0];
    let n: usize = shape[1..].iter().product();
    let mat = dense.reshape(vec![k, n]);

    let backend = arnet_native::NativeBackend::new();
    let qqt = arnet_linalg::contract(&backend, &mat, &mat, "ab,cb->ac").unwrap();

    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qqt.get(&[i, j]) - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute the full state vector from an MPS by contracting all sites.
pub fn mps_to_dense(mps: &Mps<f64>) -> Dense<f64> {
    let backend = arnet_native::NativeBackend::new();
    let n = mps.len();

    let first = match mps.storage(0) {
        d => d.clone(),
    };
    let mut result = first;

    for j in 1..n {
        let site = match mps.storage(j) {
            d => d,
        };
        let r_rank = result.rank();
        let r_last: usize = *result.shape().last().unwrap();
        let r_rest: usize = result.shape()[..r_rank - 1].iter().product();
        let result_2d = result.reshape(vec![r_rest, r_last]);

        let s_first = site.shape()[0];
        let s_rest: usize = site.shape()[1..].iter().product();
        let site_2d = site.reshape(vec![s_first, s_rest]);

        let contracted =
            arnet_linalg::contract(&backend, &result_2d, &site_2d, "ab,bc->ac").unwrap();

        let contracted = contracted.to_contiguous(MemoryOrder::RowMajor);
        let mut new_shape: Vec<usize> = result.shape()[..r_rank - 1].to_vec();
        new_shape.extend_from_slice(&site.shape()[1..]);
        result = contracted.reshape(new_shape);
    }

    result
}

/// Build an identity MPO for a given number of sites and physical dimension.
pub fn make_identity_mpo(n: usize, d: usize) -> Mpo<f64> {
    let storages = (0..n)
        .map(|_| {
            let mut data = vec![0.0; d * d];
            for i in 0..d {
                data[i * d + i] = 1.0;
            }
            Dense::from_data_with_order(data, vec![1, d, d, 1], MemoryOrder::RowMajor)
        })
        .collect();
    Mpo::from_storages(storages)
}
