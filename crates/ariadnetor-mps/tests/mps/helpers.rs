//! Shared test helpers for MPS tests.

use arnet::{
    BlockCoord, BlockSparseContractResult, BlockSparseTensor, ComputeBackend, DenseLayout,
    DenseStorage, DenseTensor, Direction, MemoryOrder, NativeBackend, QNIndex, Tensor, U1Sector,
    contract, contract_block_sparse,
};
use arnet_mps::{Mpo, Mps, TensorChain};

/// Build a `DenseTensor<f64>` from data already laid out in the active
/// backend's preferred order (NativeBackend → ColumnMajor).
pub fn cm_dense_tensor<T: arnet::Scalar>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
    DenseTensor::from_raw_parts(
        data,
        shape,
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    )
}

/// Build a `DenseTensor<f64>` from row-major data and convert to
/// column-major (NativeBackend's preferred order).
pub fn rm_dense_tensor(data: Vec<f64>, shape: Vec<usize>) -> DenseTensor<f64> {
    let total = shape.iter().product::<usize>();
    assert_eq!(data.len(), total, "rm_dense_tensor: data length mismatch");
    let rm =
        DenseTensor::from_raw_parts(data, shape, MemoryOrder::RowMajor, NativeBackend::shared());
    rm.reordered(MemoryOrder::ColumnMajor)
}

/// Single-basis-state dense MPS site for |phys_c⟩ with bond dim 1.
pub fn dense_basis_site(phys_c: usize) -> DenseTensor<f64> {
    assert!(phys_c <= 1, "physical dim is 2 (charges 0, 1)");
    let mut data = vec![0.0; 2];
    data[phys_c] = 1.0;
    DenseTensor::from_raw_parts(
        data,
        vec![1, 2, 1],
        MemoryOrder::ColumnMajor,
        NativeBackend::shared(),
    )
}

/// Dense total-particle-number MPO `N = Σ_j n_j` over `n` sites.
pub fn make_total_n_dense_mpo(n: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 1, "need at least one site");
    let mut sites = Vec::with_capacity(n);
    for j in 0..n {
        let site = match (j == 0, j == n - 1) {
            (true, true) => rm_dense_tensor(vec![0.0, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
            (true, false) => rm_dense_tensor(
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                vec![1, 2, 2, 2],
            ),
            (false, true) => rm_dense_tensor(
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                vec![2, 2, 2, 1],
            ),
            (false, false) => {
                #[rustfmt::skip]
                let data = vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ];
                rm_dense_tensor(data, vec![2, 2, 2, 2])
            }
        };
        sites.push(site);
    }
    Mpo::from_sites(sites)
}

/// Build a random-ish 4-site MPS from deterministic data.
pub fn make_4site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites = vec![
        rm_dense_tensor(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 2, 4]),
        rm_dense_tensor((1..=32).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 4]),
        rm_dense_tensor((1..=24).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 3]),
        rm_dense_tensor((1..=6).map(|i| i as f64 * 0.1).collect(), vec![3, 2, 1]),
    ];
    Mps::from_sites(sites)
}

/// Check that a site tensor is left-canonical: Q^H Q ≈ I.
pub fn is_left_canonical(site: &DenseTensor<f64>, tol: f64) -> bool {
    let shape = site.shape();
    let rank = shape.len();
    let k = shape[rank - 1];
    let m: usize = shape[..rank - 1].iter().product();
    let rm = site.reordered(MemoryOrder::RowMajor);
    let rm2d = dense_reshape_tensor(&rm, vec![m, k]);
    let mat = rm2d.reordered(MemoryOrder::ColumnMajor);

    let qtq = contract(&mat, &mat, "ab,ac->bc").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = arnet::flat_index(&[i, j], qtq.shape(), order);
            if (qtq.data_slice()[idx] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check that a site tensor is right-canonical: Q Q^H ≈ I.
pub fn is_right_canonical(site: &DenseTensor<f64>, tol: f64) -> bool {
    let shape = site.shape();
    let k = shape[0];
    let n: usize = shape[1..].iter().product();
    let rm = site.reordered(MemoryOrder::RowMajor);
    let rm2d = dense_reshape_tensor(&rm, vec![k, n]);
    let mat = rm2d.reordered(MemoryOrder::ColumnMajor);

    let qqt = contract(&mat, &mat, "ab,cb->ac").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = arnet::flat_index(&[i, j], qqt.shape(), order);
            if (qqt.data_slice()[idx] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Reshape helper for tests (zero-copy via legacy Dense).
fn dense_reshape_tensor<T, B>(t: &DenseTensor<T, B>, new_shape: Vec<usize>) -> DenseTensor<T, B>
where
    T: arnet::Scalar,
    B: ComputeBackend,
{
    let backend_arc = t.backend_arc().clone();
    let legacy = t.data().as_dense();
    let reshaped = legacy.reshape(new_shape);
    Tensor::<DenseStorage<T>, DenseLayout, B>::with_backend(
        reshaped.into_tensor_data(),
        backend_arc,
    )
}

/// Compute the full state vector from an MPS by contracting all sites.
pub fn mps_to_dense(mps: &Mps<DenseStorage<f64>, DenseLayout>) -> DenseTensor<f64> {
    let order = MemoryOrder::ColumnMajor;
    let rm = MemoryOrder::RowMajor;
    let n = mps.len();

    let mut result = mps.site(0).clone();

    for j in 1..n {
        let site = mps.site(j);
        let r_rank = result.rank();
        let r_last: usize = *result.shape().last().unwrap();
        let r_rest: usize = result.shape()[..r_rank - 1].iter().product();
        let result_rm = result.reordered(rm);
        let result_2d_rm = dense_reshape_tensor(&result_rm, vec![r_rest, r_last]);
        let result_2d = result_2d_rm.reordered(order);

        let s_first = site.shape()[0];
        let s_rest: usize = site.shape()[1..].iter().product();
        let site_rm = site.reordered(rm);
        let site_2d_rm = dense_reshape_tensor(&site_rm, vec![s_first, s_rest]);
        let site_2d = site_2d_rm.reordered(order);

        let contracted = contract(&result_2d, &site_2d, "ab,bc->ac").unwrap();

        let contracted_rm = contracted.reordered(rm);
        let mut new_shape: Vec<usize> = result.shape()[..r_rank - 1].to_vec();
        new_shape.extend_from_slice(&site.shape()[1..]);
        let multi_rm = dense_reshape_tensor(&contracted_rm, new_shape);
        result = multi_rm.reordered(order);
    }

    result
}

/// Build an identity MPO for a given number of sites and physical dimension.
pub fn make_identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let sites = (0..n)
        .map(|_| {
            let mut data = vec![0.0; d * d];
            for i in 0..d {
                data[i * d + i] = 1.0;
            }
            rm_dense_tensor(data, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(sites)
}

// ============================================================================
// BlockSparse MPS helpers
// ============================================================================

fn make_u1_site(
    left_sectors: Vec<(U1Sector, usize)>,
    phys_sectors: Vec<(U1Sector, usize)>,
    right_sectors: Vec<(U1Sector, usize)>,
    counter: &mut f64,
) -> BlockSparseTensor<f64, U1Sector> {
    let left = QNIndex::new(left_sectors, Direction::Out);
    let phys = QNIndex::new(phys_sectors, Direction::Out);
    let right = QNIndex::new(right_sectors, Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));

    let coords: Vec<BlockCoord> = site
        .data()
        .layout()
        .block_metas()
        .iter()
        .map(|m| m.coord.clone())
        .collect();
    for coord in coords {
        let data = site
            .data_mut()
            .block_data_mut(&coord)
            .expect("allowed block");
        for slot in data.iter_mut() {
            *slot = *counter;
            *counter += 0.1;
        }
    }
    site
}

/// Build a 4-site U(1)-symmetric MPS with `f64` storage and per-site flux 0.
pub fn make_4site_u1_mps() -> Mps<arnet::BlockSparseStorage<f64>, arnet::BlockSparseLayout<U1Sector>>
{
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

    Mps::from_sites(vec![site0, site1, site2, site3])
}

/// Check that a block-sparse site is left-canonical.
pub fn is_left_canonical_bsp(site: &BlockSparseTensor<f64, U1Sector>, tol: f64) -> bool {
    let layout = site.data().layout();
    let rank = site.rank();
    let new_bond = &layout.indices()[rank - 1];

    for b in 0..new_bond.num_blocks() {
        let k_b = new_bond.block_dim(b);
        let mut mhm = vec![0.0_f64; k_b * k_b];

        for meta in layout.block_metas() {
            if *meta.coord.0.last().unwrap() != b {
                continue;
            }
            let shape = layout.block_shape(&meta.coord).expect("valid block coord");
            let k_block = shape[rank - 1];
            assert_eq!(k_block, k_b, "block dim mismatch with index metadata");
            let m_block: usize = shape[..rank - 1].iter().product();
            let data = site.data().block_data(&meta.coord).expect("stored block");

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

/// Check that a block-sparse site is right-canonical.
pub fn is_right_canonical_bsp(site: &BlockSparseTensor<f64, U1Sector>, tol: f64) -> bool {
    let layout = site.data().layout();
    let left = &layout.indices()[0];

    for b in 0..left.num_blocks() {
        let k_b = left.block_dim(b);
        let mut mmh = vec![0.0_f64; k_b * k_b];

        for meta in layout.block_metas() {
            if meta.coord.0[0] != b {
                continue;
            }
            let shape = layout.block_shape(&meta.coord).expect("valid block coord");
            let k_block = shape[0];
            assert_eq!(k_block, k_b, "block dim mismatch with index metadata");
            let n_block: usize = shape[1..].iter().product();
            let data = site.data().block_data(&meta.coord).expect("stored block");

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

/// Contract an entire block-sparse MPS into a single tensor.
pub fn bsp_mps_contract_full(
    mps: &Mps<arnet::BlockSparseStorage<f64>, arnet::BlockSparseLayout<U1Sector>>,
) -> BlockSparseTensor<f64, U1Sector> {
    let n = mps.len();
    assert!(n > 0, "cannot contract an empty MPS");

    let mut acc = mps.site(0).clone();
    for j in 1..n {
        let site = mps.site(j);
        let last_axis = acc.rank() - 1;
        let result = contract_block_sparse(&acc, site, &[last_axis], &[0])
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
pub fn assert_block_sparse_close(
    a: &BlockSparseTensor<f64, U1Sector>,
    b: &BlockSparseTensor<f64, U1Sector>,
    tol: f64,
) {
    assert_eq!(a.rank(), b.rank(), "rank mismatch");
    let a_layout = a.data().layout();
    let b_layout = b.data().layout();
    assert_eq!(a_layout.flux(), b_layout.flux(), "flux mismatch");
    assert_eq!(a.shape(), b.shape(), "logical shape mismatch");

    for (axis, (ai, bi)) in a_layout
        .indices()
        .iter()
        .zip(b_layout.indices().iter())
        .enumerate()
    {
        assert_eq!(
            ai.direction(),
            bi.direction(),
            "axis {axis} direction mismatch",
        );
        assert_eq!(
            ai.blocks(),
            bi.blocks(),
            "axis {axis} sector / dimension layout mismatch",
        );
    }

    assert_eq!(
        a_layout.num_blocks(),
        b_layout.num_blocks(),
        "block count mismatch",
    );

    for meta in a_layout.block_metas() {
        let a_data = a.data().block_data(&meta.coord).expect("stored block in a");
        let b_data = b
            .data()
            .block_data(&meta.coord)
            .unwrap_or_else(|| panic!("block {:?} missing in b", meta.coord));
        assert_eq!(
            a_data.len(),
            b_data.len(),
            "block {:?} size mismatch",
            meta.coord,
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
                diff,
            );
        }
    }
}

/// Build a 2-site U(1)-symmetric MPS in the total-charge-1 sector.
pub fn make_2site_entangled_u1_mps()
-> Mps<arnet::BlockSparseStorage<f64>, arnet::BlockSparseLayout<U1Sector>> {
    let left0 = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    let phys0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let mut site0 =
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![left0, phys0, right0], U1Sector(0));
    site0
        .data_mut()
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()[0] = 1.0;
    site0
        .data_mut()
        .block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .unwrap()[0] = 2.0;

    let left1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let phys1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let right1 = QNIndex::new(vec![(U1Sector(1), 1)], Direction::In);
    let mut site1 =
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![left1, phys1, right1], U1Sector(0));
    site1
        .data_mut()
        .block_data_mut(&BlockCoord(vec![0, 1, 0]))
        .unwrap()[0] = 3.0;
    site1
        .data_mut()
        .block_data_mut(&BlockCoord(vec![1, 0, 0]))
        .unwrap()[0] = 4.0;

    Mps::from_sites(vec![site0, site1])
}

/// Build a U(1) total-particle-number MPO `N = Σ_j n_j` over `n` sites.
pub fn make_total_n_u1_mpo(
    n: usize,
) -> Mpo<arnet::BlockSparseStorage<f64>, arnet::BlockSparseLayout<U1Sector>> {
    assert!(n >= 1, "need at least one site");
    let mut sites = Vec::with_capacity(n);
    for j in 0..n {
        let left_dim = if j == 0 { 1 } else { 2 };
        let right_dim = if j == n - 1 { 1 } else { 2 };
        let left = QNIndex::new(vec![(U1Sector(0), left_dim)], Direction::Out);
        let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
        let right = QNIndex::new(vec![(U1Sector(0), right_dim)], Direction::In);
        let mut site =
            BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));

        let block_phys0 = site
            .data_mut()
            .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
            .expect("charge-0 phys block");
        match (j == 0, j == n - 1) {
            (true, true) => {
                block_phys0[0] = 0.0;
            }
            (true, false) => {
                block_phys0[0] = 1.0;
                block_phys0[1] = 0.0;
            }
            (false, true) => {
                block_phys0[0] = 0.0;
                block_phys0[1] = 1.0;
            }
            (false, false) => {
                block_phys0[0] = 1.0;
                block_phys0[1] = 0.0;
                block_phys0[2] = 0.0;
                block_phys0[3] = 1.0;
            }
        }

        let block_phys1 = site
            .data_mut()
            .block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
            .expect("charge-1 phys block");
        match (j == 0, j == n - 1) {
            (true, true) => {
                block_phys1[0] = 1.0;
            }
            (true, false) => {
                block_phys1[0] = 1.0;
                block_phys1[1] = 1.0;
            }
            (false, true) => {
                block_phys1[0] = 1.0;
                block_phys1[1] = 1.0;
            }
            (false, false) => {
                block_phys1[0] = 1.0;
                block_phys1[1] = 0.0;
                block_phys1[2] = 1.0;
                block_phys1[3] = 1.0;
            }
        }

        sites.push(site);
    }
    Mpo::from_sites(sites)
}

/// Build a U(1) identity MPO for the given number of sites.
pub fn make_identity_u1_mpo(
    n: usize,
) -> Mpo<arnet::BlockSparseStorage<f64>, arnet::BlockSparseLayout<U1Sector>> {
    let sites = (0..n)
        .map(|_| {
            let left = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
            let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
            let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
            let right = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
            let mut site =
                BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
            site.data_mut()
                .block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
                .unwrap()[0] = 1.0;
            site.data_mut()
                .block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
                .unwrap()[0] = 1.0;
            site
        })
        .collect();
    Mpo::from_sites(sites)
}
