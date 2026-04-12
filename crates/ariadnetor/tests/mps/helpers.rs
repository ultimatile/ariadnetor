//! Shared test helpers for MPS tests.

use arnet::mps::{Mpo, Mps, TensorChain};
use arnet_linalg::{BlockSparseContractResult, contract_block_sparse};
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::U1Sector;
use arnet_tensor::{Dense, MemoryOrder};

/// Build a random-ish 4-site MPS from deterministic data.
pub fn make_4site_mps() -> Mps<Dense<f64>> {
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
pub fn is_left_canonical(dense: &Dense<f64>, tol: f64) -> bool {
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
pub fn is_right_canonical(dense: &Dense<f64>, tol: f64) -> bool {
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
pub fn mps_to_dense(mps: &Mps<Dense<f64>>) -> Dense<f64> {
    let backend = arnet_native::NativeBackend::new();
    let n = mps.len();

    let mut result = mps.storage(0).clone();

    for j in 1..n {
        let site = mps.storage(j);
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
pub fn make_identity_mpo(n: usize, d: usize) -> Mpo<Dense<f64>> {
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

// ============================================================================
// BlockSparse MPS helpers
// ============================================================================

// Convention: all sites are flux 0. Left bond = Out, physical = Out, right bond = In,
// so adjacent site bonds dual to each other (right[In] vs left[Out] with equal raw
// sectors), and `contract_block_sparse` accepts them without flipping.
//
// Physical dimension 2 with charges {U1(0): 1, U1(1): 1} on every site. Interior
// bond dimensions mix multi-element sectors so each site's QR / LQ exercises at
// least one non-trivial factorization.

fn make_u1_site(
    left_sectors: Vec<(U1Sector, usize)>,
    phys_sectors: Vec<(U1Sector, usize)>,
    right_sectors: Vec<(U1Sector, usize)>,
    counter: &mut f64,
) -> BlockSparse<f64, U1Sector> {
    let left = QNIndex::new(left_sectors, Direction::Out);
    let phys = QNIndex::new(phys_sectors, Direction::Out);
    let right = QNIndex::new(right_sectors, Direction::In);
    let mut site = BlockSparse::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));

    // Populate each allowed block with a deterministic monotone sequence so the
    // resulting site has non-trivial content in every live sector (avoiding
    // zero-valued blocks that would hide per-sector bugs).
    let coords: Vec<BlockCoord> = site.block_metas().iter().map(|m| m.coord.clone()).collect();
    for coord in coords {
        let data = site.block_data_mut(&coord).expect("allowed block");
        for slot in data.iter_mut() {
            *slot = *counter;
            *counter += 0.1;
        }
    }
    site
}

/// Build a 4-site U(1)-symmetric MPS with `f64` storage and per-site flux 0.
///
/// Bond structure (raw sector labels):
/// - Site 0 right = site 1 left = `{0:2, 1:1}`
/// - Site 1 right = site 2 left = `{0:2, 1:2, 2:1}`
/// - Site 2 right = site 3 left = `{0:2, 1:1}`
/// - Outer boundaries = `{0:1}` (trivial)
///
/// Each site has at least one non-trivial QR / LQ factorization sector, which is
/// essential for mutant-testing coverage of the BlockSparse canonicalize sweeps.
pub fn make_4site_u1_mps() -> Mps<BlockSparse<f64, U1Sector>> {
    let mut counter: f64 = 0.1;

    let site0 = make_u1_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        &mut counter,
    );
    let site1 = make_u1_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)],
        &mut counter,
    );
    let site2 = make_u1_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        &mut counter,
    );
    let site3 = make_u1_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );

    Mps::from_storages(vec![site0, site1, site2, site3])
}

/// Check that a block-sparse site is left-canonical: `Q^H Q ≈ I` on the new
/// (rightmost) bond.
///
/// For each sector `b` of the rightmost index, stack every allowed block with
/// `coord.last() == b` into a single `(Σ_i m_i) × k_b` matrix and verify the
/// accumulated `M^H M` is the `k_b × k_b` identity. Different new-bond sectors
/// are decoupled by symmetry, so per-sector checks are sufficient.
pub fn is_left_canonical_bsp(site: &BlockSparse<f64, U1Sector>, tol: f64) -> bool {
    let rank = site.rank();
    let new_bond = &site.indices()[rank - 1];

    for b in 0..new_bond.num_blocks() {
        let k_b = new_bond.block_dim(b);
        let mut mhm = vec![0.0_f64; k_b * k_b];

        for meta in site.block_metas() {
            if *meta.coord.0.last().unwrap() != b {
                continue;
            }
            let shape = site.block_shape(&meta.coord).expect("valid block coord");
            let k_block = shape[rank - 1];
            assert_eq!(k_block, k_b, "block dim mismatch with index metadata");
            let m_block: usize = shape[..rank - 1].iter().product();
            let data = site.block_data(&meta.coord).expect("stored block");

            // Row-major layout: data[row * k_b + col] where row ∈ 0..m_block.
            for row in 0..m_block {
                for p in 0..k_b {
                    let a = data[row * k_b + p];
                    for q in 0..k_b {
                        mhm[p * k_b + q] += a * data[row * k_b + q];
                    }
                }
            }
        }

        for p in 0..k_b {
            for q in 0..k_b {
                let expected = if p == q { 1.0 } else { 0.0 };
                if (mhm[p * k_b + q] - expected).abs() > tol {
                    return false;
                }
            }
        }
    }
    true
}

/// Check that a block-sparse site is right-canonical: `Q Q^H ≈ I` on the leftmost
/// bond.
///
/// Dual of [`is_left_canonical_bsp`]: for each sector `b` of the leftmost index,
/// accumulate `M M^H` from all allowed blocks with `coord[0] == b`, treating each
/// block as a `k_b × n_rest` row-major matrix.
pub fn is_right_canonical_bsp(site: &BlockSparse<f64, U1Sector>, tol: f64) -> bool {
    let left = &site.indices()[0];

    for b in 0..left.num_blocks() {
        let k_b = left.block_dim(b);
        let mut mmh = vec![0.0_f64; k_b * k_b];

        for meta in site.block_metas() {
            if meta.coord.0[0] != b {
                continue;
            }
            let shape = site.block_shape(&meta.coord).expect("valid block coord");
            let k_block = shape[0];
            assert_eq!(k_block, k_b, "block dim mismatch with index metadata");
            let n_block: usize = shape[1..].iter().product();
            let data = site.block_data(&meta.coord).expect("stored block");

            // Row-major layout: data[row * n_block + col] where row ∈ 0..k_b.
            for i in 0..k_b {
                for j in 0..k_b {
                    let mut acc = 0.0;
                    for col in 0..n_block {
                        acc += data[i * n_block + col] * data[j * n_block + col];
                    }
                    mmh[i * k_b + j] += acc;
                }
            }
        }

        for p in 0..k_b {
            for q in 0..k_b {
                let expected = if p == q { 1.0 } else { 0.0 };
                if (mmh[p * k_b + q] - expected).abs() > tol {
                    return false;
                }
            }
        }
    }
    true
}

/// Contract an entire block-sparse MPS into a single tensor by sequentially
/// merging adjacent sites. Used as an external state-equivalence witness in
/// canonicalize tests.
pub fn bsp_mps_contract_full(mps: &Mps<BlockSparse<f64, U1Sector>>) -> BlockSparse<f64, U1Sector> {
    let n = mps.len();
    assert!(n > 0, "cannot contract an empty MPS");

    let mut acc = mps.storage(0).clone();
    for j in 1..n {
        let site = mps.storage(j);
        let last_axis = acc.rank() - 1;
        let result = contract_block_sparse(mps.backend(), &acc, site, &[last_axis], &[0])
            .expect("chain contraction failed in bsp_mps_contract_full");
        acc = match result {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("chain contraction leaves at least the physical legs free")
            }
        };
    }
    acc
}

/// Assert that two block-sparse tensors are element-wise close.
///
/// Requires matching rank, flux, logical shape, per-axis QN indices
/// (`Direction` + sector/dimension pairs), and block set. The axis-level
/// check is what makes `BlockCoord`-based block lookups well-defined across
/// both tensors: without it, two tensors with the same logical shape but
/// different sector labeling could share `BlockCoord` indices that point to
/// semantically unrelated blocks and spuriously compare equal.
pub fn assert_block_sparse_close(
    a: &BlockSparse<f64, U1Sector>,
    b: &BlockSparse<f64, U1Sector>,
    tol: f64,
) {
    assert_eq!(a.rank(), b.rank(), "rank mismatch");
    assert_eq!(a.flux(), b.flux(), "flux mismatch");
    assert_eq!(a.shape(), b.shape(), "logical shape mismatch");

    for (axis, (ai, bi)) in a.indices().iter().zip(b.indices().iter()).enumerate() {
        assert_eq!(
            ai.direction(),
            bi.direction(),
            "axis {axis} direction mismatch"
        );
        assert_eq!(
            ai.blocks(),
            bi.blocks(),
            "axis {axis} sector / dimension layout mismatch"
        );
    }

    assert_eq!(a.num_blocks(), b.num_blocks(), "block count mismatch");

    for meta in a.block_metas() {
        let a_data = a.block_data(&meta.coord).expect("stored block in a");
        let b_data = b
            .block_data(&meta.coord)
            .unwrap_or_else(|| panic!("block {:?} missing in b", meta.coord));
        assert_eq!(
            a_data.len(),
            b_data.len(),
            "block {:?} size mismatch",
            meta.coord
        );
        for (idx, (aa, bb)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = (aa - bb).abs();
            assert!(
                diff < tol,
                "block {:?} element {} mismatch: {} vs {} (diff {})",
                meta.coord,
                idx,
                aa,
                bb,
                diff
            );
        }
    }
}

/// Build a 2-site U(1)-symmetric MPS in the total-charge-1 sector.
///
/// Physical charges {0, 1}, boundary left={0:1}, right={1:1}.
/// The state spans two basis vectors: |01⟩ (coeff 3) and |10⟩ (coeff 8),
/// giving bond dim 2 with two non-zero singular values — genuine
/// entanglement that truncation can meaningfully discard.
pub fn make_2site_entangled_u1_mps() -> Mps<BlockSparse<f64, U1Sector>> {
    let left0 = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut site0 = BlockSparse::<f64, U1Sector>::zeros(vec![left0, phys0, right0], U1Sector(0));
    site0.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap()[0] = 1.0;
    site0.block_data_mut(&BlockCoord(vec![0, 1, 1])).unwrap()[0] = 2.0;

    let left1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let phys1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right1 = QNIndex::new(vec![(U1Sector(1), 1)], Direction::In);
    let mut site1 = BlockSparse::<f64, U1Sector>::zeros(vec![left1, phys1, right1], U1Sector(0));
    site1.block_data_mut(&BlockCoord(vec![0, 1, 0])).unwrap()[0] = 3.0;
    site1.block_data_mut(&BlockCoord(vec![1, 0, 0])).unwrap()[0] = 4.0;

    Mps::from_storages(vec![site0, site1])
}
