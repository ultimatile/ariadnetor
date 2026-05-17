//! Shared test helpers for MPS tests.

use arnet_linalg::{BlockSparseContractResult, contract_block_sparse};
use arnet_mps::{Mpo, Mps, TensorChain};
use arnet_tensor::U1Sector;
use arnet_tensor::reorder;
use arnet_tensor::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::{Dense, MemoryOrder};

/// Build a Dense from row-major data and convert to column-major (NativeBackend order).
fn rm_dense(data: Vec<f64>, shape: Vec<usize>) -> Dense<f64> {
    let rm = Dense::new(data, shape, MemoryOrder::RowMajor);
    reorder(&rm, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor)
}

/// Single-basis-state dense MPS site for `|phys_c⟩` with bond dim 1.
///
/// Used to build product states with a known total particle number for the
/// dense-side analytical anchors that mirror the BlockSparse `bsp_basis_site`
/// helper. Physical dim is fixed at 2 (charges 0 and 1).
pub fn dense_basis_site(phys_c: usize) -> Dense<f64> {
    assert!(phys_c <= 1, "physical dim is 2 (charges 0, 1)");
    let mut data = vec![0.0; 2];
    data[phys_c] = 1.0;
    // Shape (1, 2, 1) has only one non-trivial axis, so RowMajor and
    // ColumnMajor flatten to the same byte order — no rm_dense needed.
    Dense::new(data, vec![1, 2, 1], MemoryOrder::ColumnMajor)
}

/// Dense total-particle-number MPO `N = Σ_j n_j` over `n` sites.
///
/// Mirror of `make_total_n_u1_mpo` for the dense path. Standard rank-2
/// finite-state-machine MPO: bond basis `{I, n}` with transitions
/// `I → I = 1`, `I → n = n_j`, `n → n = 1`. Boundary bonds collapse to
/// dim 1 (`I` on the left, `n` on the right).
///
/// Compared to the bond-dim-1 fixtures used by the existing dense apply
/// tests, this exercises the non-trivial `w_L⊗χ_L` and `w_R⊗χ_R` bond
/// fusion; pairing it with [`dense_basis_site`] gives an analytical
/// expectation value (`⟨c1…cn|N|c1…cn⟩ = Σ c_i`) that pins `apply_dense`
/// against algebra rather than against another implementation.
pub fn make_total_n_dense_mpo(n: usize) -> Mpo<Dense<f64>> {
    assert!(n >= 1, "need at least one site");
    let mut storages = Vec::with_capacity(n);
    for j in 0..n {
        // Site data is written in conceptual RowMajor (index
        // `wL*Bdk*Cdb*DwR + dk*Cdb*DwR + db*DwR + wR`) for readability and
        // converted to the backend's preferred order via rm_dense.
        let storage = match (j == 0, j == n - 1) {
            (true, true) => {
                // n == 1: the single site reduces to n_phys = diag(0, 1).
                rm_dense(vec![0.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1])
            }
            (true, false) => {
                // Left boundary, shape (1, 2, 2, 2). Non-zero entries:
                //   (0, 0, 0, 0) = I_phys at (0,0) = 1
                //   (0, 1, 1, 0) = I_phys at (1,1) = 1
                //   (0, 1, 1, 1) = n_phys at (1,1) = 1
                rm_dense(
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    vec![1, 2, 2, 2],
                )
            }
            (false, true) => {
                // Right boundary, shape (2, 2, 2, 1). Non-zero entries:
                //   (0, 1, 1, 0) = n_phys at (1,1) = 1   (bL=I → apply n)
                //   (1, 0, 0, 0) = I_phys at (0,0) = 1   (bL=n → apply I)
                //   (1, 1, 1, 0) = I_phys at (1,1) = 1
                rm_dense(
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    vec![2, 2, 2, 1],
                )
            }
            (false, false) => {
                // Interior, shape (2, 2, 2, 2). Non-zero entries:
                //   (0, 0, 0, 0), (0, 1, 1, 0)        — I→I block (I_phys)
                //   (0, 1, 1, 1)                      — I→n block (n_phys)
                //   (1, 0, 0, 1), (1, 1, 1, 1)        — n→n block (I_phys)
                #[rustfmt::skip]
                let data = vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ];
                rm_dense(data, vec![2, 2, 2, 2])
            }
        };
        storages.push(storage);
    }
    Mpo::from_storages(storages)
}

/// Build a random-ish 4-site MPS from deterministic data.
pub fn make_4site_mps() -> Mps<Dense<f64>> {
    let storages = vec![
        rm_dense(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 2, 4]),
        rm_dense((1..=32).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 4]),
        rm_dense((1..=24).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 3]),
        rm_dense((1..=6).map(|i| i as f64 * 0.1).collect(), vec![3, 2, 1]),
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
    // Reorder to RM for correct axis-merge reshape, then back to CM for contract.
    let rm = reorder(dense, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    let rm2d = rm.reshape(vec![m, k]);
    let mat = reorder(&rm2d, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let backend = arnet_native::NativeBackend::new();
    let qtq = arnet_linalg::contract_dense(&backend, &mat, &mat, "ab,ac->bc").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = arnet_tensor::flat_index(&[i, j], qtq.shape(), order);
            if (qtq.data()[idx] - expected).abs() > tol {
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
    // Reorder to RM for correct axis-merge reshape, then back to CM for contract.
    let rm = reorder(dense, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    let rm2d = rm.reshape(vec![k, n]);
    let mat = reorder(&rm2d, MemoryOrder::RowMajor, MemoryOrder::ColumnMajor);

    let backend = arnet_native::NativeBackend::new();
    let qqt = arnet_linalg::contract_dense(&backend, &mat, &mat, "ab,cb->ac").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = arnet_tensor::flat_index(&[i, j], qqt.shape(), order);
            if (qqt.data()[idx] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute the full state vector from an MPS by contracting all sites.
/// Returns the result in the backend's preferred order (ColumnMajor).
pub fn mps_to_dense(mps: &Mps<Dense<f64>>) -> Dense<f64> {
    let backend = arnet_native::NativeBackend::new();
    let order = MemoryOrder::ColumnMajor;
    let rm = MemoryOrder::RowMajor;
    let n = mps.len();

    let mut result = mps.storage(0).clone();

    for j in 1..n {
        let site = mps.storage(j);
        let r_rank = result.rank();
        let r_last: usize = *result.shape().last().unwrap();
        let r_rest: usize = result.shape()[..r_rank - 1].iter().product();
        // Reorder to RM for axis-merge reshape, then back to CM for contract.
        let result_rm = reorder(&result, order, rm);
        let result_2d_rm = result_rm.reshape(vec![r_rest, r_last]);
        let result_2d = reorder(&result_2d_rm, rm, order);

        let s_first = site.shape()[0];
        let s_rest: usize = site.shape()[1..].iter().product();
        let site_rm = reorder(site, order, rm);
        let site_2d_rm = site_rm.reshape(vec![s_first, s_rest]);
        let site_2d = reorder(&site_2d_rm, rm, order);

        let contracted =
            arnet_linalg::contract_dense(&backend, &result_2d, &site_2d, "ab,bc->ac").unwrap();

        // Reorder to RM for axis-split reshape, then back to CM.
        let contracted_rm = reorder(&contracted, order, rm);
        let mut new_shape: Vec<usize> = result.shape()[..r_rank - 1].to_vec();
        new_shape.extend_from_slice(&site.shape()[1..]);
        let multi_rm = contracted_rm.reshape(new_shape);
        result = reorder(&multi_rm, rm, order);
    }

    result
}

/// Build an identity MPO for a given number of sites and physical dimension.
pub fn make_identity_mpo(n: usize, d: usize) -> Mpo<Dense<f64>> {
    let storages = (0..n)
        .map(|_| {
            // Identity operator in RM, then convert to CM
            let mut data = vec![0.0; d * d];
            for i in 0..d {
                data[i * d + i] = 1.0;
            }
            rm_dense(data, vec![1, d, d, 1])
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

/// Build a U(1) total-particle-number MPO `N = Σ_j n_j` over `n` sites.
///
/// Standard rank-2 finite-state-machine MPO: bond basis `{I, n}` with
/// transitions `I → I = 1`, `I → n = n_j`, `n → n = 1`. All bond charges are
/// `0`, giving MPO bond dim 2 (single charge-0 sector of dim 2 on interior
/// bonds). Boundary bonds have dim 1, projecting onto `I` on the left and
/// `n` on the right.
///
/// Compared to [`make_identity_u1_mpo`] (bond dim 1), this fixture exercises
/// the non-trivial `w_L⊗χ_L` and `w_R⊗χ_R` bond fusions that any apply
/// implementation has to perform, so bugs that survive the identity-only
/// path can still be caught here.
pub fn make_total_n_u1_mpo(n: usize) -> Mpo<BlockSparse<f64, U1Sector>> {
    assert!(n >= 1, "need at least one site");
    let mut storages = Vec::with_capacity(n);
    for j in 0..n {
        let left_dim = if j == 0 { 1 } else { 2 };
        let right_dim = if j == n - 1 { 1 } else { 2 };
        let left = QNIndex::new(vec![(U1Sector(0), left_dim)], Direction::Out);
        let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
        let right = QNIndex::new(vec![(U1Sector(0), right_dim)], Direction::In);
        let mut site =
            BlockSparse::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));

        // Block-data layout follows the backend's preferred order. NativeBackend
        // is ColumnMajor, so for an interior block of shape (a, b, c, d) the
        // flat index is `bL + dk*a + db*a*b + bR*a*b*c`. With dk=db=0 (or 1)
        // pinned, the (bL, bR) pair flattens as `bL + bR*a` — i.e. the (bR, bL)
        // pair iterates fastest on bL.
        //
        // Boundary cases (j == 0, j == n - 1) shrink the corresponding bond to
        // dim 1 and may overlap (when n == 1 the site is both ends at once),
        // so the four (left-edge, right-edge) combinations are split
        // explicitly.

        // Charge-0 physical block: action on |0⟩⟨0|. n_phys at (0,0) = 0,
        // so all "I → n" transitions vanish here and only the FSM-stay
        // transitions (I→I, n→n) and the right-boundary projection survive.
        let block_phys0 = site
            .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
            .expect("charge-0 phys block");
        match (j == 0, j == n - 1) {
            (true, true) => {
                // n == 1: shape (1, 1, 1, 1) — total N reduces to n_phys at (0, 0) = 0.
                block_phys0[0] = 0.0;
            }
            (true, false) => {
                // Left edge, shape (1, 1, 1, 2). Single non-trivial axis is bR,
                // so RowMajor and ColumnMajor agree. Row [I→I, I→n] = [1, 0].
                block_phys0[0] = 1.0;
                block_phys0[1] = 0.0;
            }
            (false, true) => {
                // Right edge, shape (2, 1, 1, 1). Single non-trivial axis is bL.
                // Column [bL=I: apply n_phys, bL=n: apply I_phys]^T = [0, 1]^T.
                block_phys0[0] = 0.0;
                block_phys0[1] = 1.0;
            }
            (false, false) => {
                // Interior, shape (2, 1, 1, 2). Logical matrix in (bL, bR):
                // [[I→I, I→n], [n→I, n→n]] = [[1, 0], [0, 1]] (charge 0).
                // ColumnMajor flat index `bL + bR*2`:
                block_phys0[0] = 1.0; // (bL=I, bR=I) → 1
                block_phys0[1] = 0.0; // (bL=n, bR=I) → 0
                block_phys0[2] = 0.0; // (bL=I, bR=n) → 0 (n_phys at charge 0)
                block_phys0[3] = 1.0; // (bL=n, bR=n) → 1
            }
        }

        // Charge-1 physical block: action on |1⟩⟨1|. n_phys at (1,1) = 1, so
        // both the I→n FSM transition and the right-boundary projection take
        // value 1 here.
        let block_phys1 = site
            .block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
            .expect("charge-1 phys block");
        match (j == 0, j == n - 1) {
            (true, true) => {
                // n == 1: shape (1, 1, 1, 1) — total N reduces to n_phys at (1, 1) = 1.
                block_phys1[0] = 1.0;
            }
            (true, false) => {
                // Left edge, shape (1, 1, 1, 2). Row [I→I=1, I→n=1].
                block_phys1[0] = 1.0;
                block_phys1[1] = 1.0;
            }
            (false, true) => {
                // Right edge, shape (2, 1, 1, 1). Column [bL=I: n_phys=1,
                // bL=n: I_phys=1]^T (the right boundary projects bR onto n,
                // so the bL=n branch contributes the FSM-stay n→n with value 1).
                block_phys1[0] = 1.0;
                block_phys1[1] = 1.0;
            }
            (false, false) => {
                // Interior, shape (2, 1, 1, 2). Logical matrix in (bL, bR):
                // [[I→I=1, I→n=1], [n→I=0, n→n=1]] (upper triangular at charge 1).
                // ColumnMajor flat index `bL + bR*2`:
                block_phys1[0] = 1.0; // (bL=I, bR=I) → 1
                block_phys1[1] = 0.0; // (bL=n, bR=I) → 0
                block_phys1[2] = 1.0; // (bL=I, bR=n) → 1 (the I → n transition)
                block_phys1[3] = 1.0; // (bL=n, bR=n) → 1
            }
        }

        storages.push(site);
    }
    Mpo::from_storages(storages)
}

/// Build a U(1) identity MPO for the given number of sites.
///
/// MPO convention: (Out, In, Out, In) = (w_L, d_ket, d_bra, w_R).
/// Physical charges {0, 1}. Bond dim = 1. Flux = 0 per site.
pub fn make_identity_u1_mpo(n: usize) -> Mpo<BlockSparse<f64, U1Sector>> {
    let storages = (0..n)
        .map(|_| {
            let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
            let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
            let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
            let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
            let mut site =
                BlockSparse::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
            site.block_data_mut(&BlockCoord(vec![0, 0, 0, 0])).unwrap()[0] = 1.0;
            site.block_data_mut(&BlockCoord(vec![0, 1, 1, 0])).unwrap()[0] = 1.0;
            site
        })
        .collect();
    Mpo::from_storages(storages)
}
