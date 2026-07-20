//! Shared test helpers for MPS tests.

use ariadnetor_linalg::{contract, permute_with_backend, tensordot};
use ariadnetor_mps::{
    ApplyMethod, Mpo, Mps, MpsOps, TensorChain, TruncateParams, apply_with_method,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::MemoryOrder;
use ariadnetor_tensor::test_fixtures::legs;
use ariadnetor_tensor::{
    BlockCoord, BlockSparseTensor, DenseLayout, DenseStorage, DenseTensor, Direction, OpsFor,
    Storage, StorageFor, TensorLayout, U1Sector,
};
use ariadnetor_tensor::{ComputeBackendTensorExt, Host};

/// Build a `DenseTensor<f64>` from data already laid out in the active
/// backend's preferred order (NativeBackend → ColumnMajor).
pub(crate) fn cm_dense_tensor<T: ariadnetor_core::Scalar>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> DenseTensor<T> {
    Host::shared().dense(data, shape)
}

/// `apply_with_method` unwrapped: shared by tests whose inputs are finite,
/// where an `Err` can only mean the apply contract itself broke. Tests that
/// exercise the error path call `apply_with_method` directly.
pub(crate) fn apply_ok<T, St, L, B>(
    backend: &B,
    op: &Mpo<St, L>,
    psi: &Mps<St, L>,
    params: Option<&TruncateParams>,
    method: ApplyMethod,
) -> Mps<St, L>
where
    T: ariadnetor_core::Scalar,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    Mps<St, L>: MpsOps<T, Storage = St, Layout = L>,
    B: OpsFor<St>,
{
    apply_with_method(backend, op, psi, params, method)
        .expect("apply must succeed on finite inputs")
}

/// Build a `DenseTensor<f64>` whose logical content matches `data` read
/// in row-major order, materialized in the backend's preferred
/// (column-major) order. Constructing the buffer under the reversed
/// shape and reversing the axes turns the column-major linearization
/// into the row-major one (`cm_flat(rev(i); rev(shape)) ==
/// rm_flat(i; shape)`), so the call sites keep their readable
/// row-major listings without an order-tagging constructor.
pub(crate) fn rm_dense_tensor(data: Vec<f64>, shape: Vec<usize>) -> DenseTensor<f64> {
    let total = shape.iter().product::<usize>();
    assert_eq!(data.len(), total, "rm_dense_tensor: data length mismatch");
    let reversed_shape: Vec<usize> = shape.iter().rev().copied().collect();
    let reversed = Host::shared().dense(data, reversed_shape);
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    permute_with_backend(&NativeBackend::new(), &reversed, &perm)
        .expect("rm_dense_tensor: reverse-axis permute")
}

/// Single-basis-state dense MPS site for |phys_c⟩ with bond dim 1.
pub(crate) fn dense_basis_site(phys_c: usize) -> DenseTensor<f64> {
    assert!(phys_c <= 1, "physical dim is 2 (charges 0, 1)");
    let mut data = vec![0.0; 2];
    data[phys_c] = 1.0;
    Host::shared().dense(data, vec![1, 2, 1])
}

/// Dense total-particle-number MPO `N = Σ_j n_j` over `n` sites.
pub(crate) fn make_total_n_dense_mpo(n: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
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
pub(crate) fn make_4site_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites = vec![
        rm_dense_tensor(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![1, 2, 4]),
        rm_dense_tensor((1..=32).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 4]),
        rm_dense_tensor((1..=24).map(|i| i as f64 * 0.1).collect(), vec![4, 2, 3]),
        rm_dense_tensor((1..=6).map(|i| i as f64 * 0.1).collect(), vec![3, 2, 1]),
    ];
    Mps::from_sites(sites)
}

/// 3-site MPS with bond dim 2 and physical dim 2. Deterministic content.
pub(crate) fn make_3site_test_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        cm_dense_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ])
}

/// 3-site MPO with bond dim 2 and physical dim 2.
pub(crate) fn make_3site_test_mpo() -> Mpo<DenseStorage<f64>, DenseLayout> {
    Mpo::from_sites(vec![
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 2, 2]),
        cm_dense_tensor(
            (1..=16).map(|i| i as f64 * 0.05).collect(),
            vec![2, 2, 2, 2],
        ),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2, 1]),
    ])
}

/// Assert two dense f64 tensors agree element-wise within `tol`.
pub(crate) fn assert_dense_close(a: &DenseTensor<f64>, b: &DenseTensor<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    // Zipping raw storage is element-correct only when both tensors share
    // one memory order; guard it so an order mismatch fails loudly instead
    // of comparing wrong element pairs.
    assert_eq!(a.order(), b.order(), "memory order mismatch");
    for (i, (x, y)) in a.data_slice().iter().zip(b.data_slice().iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < tol, "elem {i} mismatch: {x} vs {y} (diff {diff})");
    }
}

/// Check that a site tensor is left-canonical: Q^H Q ≈ I.
pub(crate) fn is_left_canonical(site: &DenseTensor<f64>, tol: f64) -> bool {
    let shape = site.shape();
    let rank = shape.len();
    let k = shape[rank - 1];
    let m: usize = shape[..rank - 1].iter().product();
    let rm = site.reordered(MemoryOrder::RowMajor);
    let rm2d = rm.reshape(vec![m, k]);
    let mat = rm2d.reordered(MemoryOrder::ColumnMajor);

    let qtq = contract(&NativeBackend::new(), &mat, &mat, "ab,ac->bc").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = ariadnetor_tensor::flat_index(&[i, j], qtq.shape(), order);
            if (qtq.data_slice()[idx] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check that a site tensor is right-canonical: Q Q^H ≈ I.
pub(crate) fn is_right_canonical(site: &DenseTensor<f64>, tol: f64) -> bool {
    let shape = site.shape();
    let k = shape[0];
    let n: usize = shape[1..].iter().product();
    let rm = site.reordered(MemoryOrder::RowMajor);
    let rm2d = rm.reshape(vec![k, n]);
    let mat = rm2d.reordered(MemoryOrder::ColumnMajor);

    let qqt = contract(&NativeBackend::new(), &mat, &mat, "ab,cb->ac").unwrap();

    let order = MemoryOrder::ColumnMajor;
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            let idx = ariadnetor_tensor::flat_index(&[i, j], qqt.shape(), order);
            if (qqt.data_slice()[idx] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Contract a dense chain into its full state tensor (any scalar type).
pub(crate) fn densify<T: ariadnetor_core::Scalar>(
    backend: &NativeBackend,
    mps: &Mps<DenseStorage<T>, DenseLayout>,
) -> DenseTensor<T> {
    let mut acc = mps.site(0).clone();
    for j in 1..mps.len() {
        let last = acc.rank() - 1;
        acc = tensordot(backend, &acc, mps.site(j), &[last], &[0]).expect("chain contraction");
    }
    acc
}

/// Compute the full state vector from an MPS by contracting all sites.
pub(crate) fn mps_to_dense(mps: &Mps<DenseStorage<f64>, DenseLayout>) -> DenseTensor<f64> {
    densify(&NativeBackend::new(), mps)
}

/// Relative Frobenius distance ‖a − e‖ / ‖e‖ over dense state tensors.
///
/// Elementwise over the densified states keeps full floating-point
/// resolution; the inner-product form `‖a‖² + ‖e‖² − 2 Re⟨e|a⟩` cancels
/// catastrophically and cannot resolve relative errors below
/// ~sqrt(machine epsilon).
pub(crate) fn relative_frobenius<T: ariadnetor_core::Scalar>(
    a: &DenseTensor<T>,
    e: &DenseTensor<T>,
) -> f64 {
    use num_traits::NumCast;
    assert_eq!(a.shape(), e.shape(), "state shapes must agree");
    // Both operands come from the same contraction pipeline, so their
    // physical orders match and zipping raw storage is element-correct
    // (the assert_dense_close precedent).
    assert_eq!(a.order(), e.order(), "storage orders must agree");
    let mut diff_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for (&av, &ev) in a.data_slice().iter().zip(e.data_slice().iter()) {
        // Scalar carries Add but not Sub; negate through scale_real.
        let neg_one = -<T::Real as num_traits::One>::one();
        let d: f64 = <f64 as NumCast>::from((av + ev.scale_real(neg_one)).abs()).expect("fits f64");
        let n: f64 = <f64 as NumCast>::from(ev.abs()).expect("fits f64");
        diff_sq += d * d;
        norm_sq += n * n;
    }
    diff_sq.sqrt() / norm_sq.sqrt()
}

/// The chain with site `j` scaled by `factor` and every other site cloned.
/// Scaling any one site scales the whole product state, so this moves a
/// state's amplitude (or applies a complex phase) without touching its
/// direction.
pub(crate) fn with_site_scaled<T: ariadnetor_core::Scalar>(
    mps: &Mps<DenseStorage<T>, DenseLayout>,
    j: usize,
    factor: T,
) -> Mps<DenseStorage<T>, DenseLayout> {
    Mps::from_sites(
        (0..mps.len())
            .map(|i| {
                let site = mps.site(i);
                if i == j {
                    site.scaled(factor)
                } else {
                    site.clone()
                }
            })
            .collect(),
    )
}

/// Build an identity MPO for a given number of sites and a uniform
/// physical dimension (thin wrapper over the library constructor).
pub(crate) fn make_identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    Mpo::identity(&vec![d; n])
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
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (left_sectors, Direction::Out),
            (phys_sectors, Direction::Out),
            (right_sectors, Direction::In),
        ]),
        U1Sector(0),
    );

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

/// Build a 3-site U(1)-symmetric MPS whose middle bond has a sector with
/// dim 3 fed by **two distinct (left, phys) pathways**. The multi-path
/// structure makes the forward `(left*phys, right)` and backward
/// `(left, phys*right)` unfoldings of the per-sector block matrix
/// genuinely different matrices, so the per-sector SVD truncation choice
/// is not gauge-equivalent between the forward and backward sweeps. Total
/// flux is anchored at `U1Sector(1)`.
///
/// This fixture is paired with the BSP `forward_cap` observable-difference
/// test: cap = 2 against a sector with dim 3 and non-trivial forward /
/// backward unfolding asymmetry produces a discernible Frobenius
/// difference at the contracted-output level. Fixtures where every
/// sector has a single incoming pathway and per-sector phys dim 1 (e.g.
/// `make_4site_u1_mps`) collapse forward and backward SVDs into the same
/// matrix, hiding the cap's effect.
pub(crate) fn make_3site_u1_mps_multipath_middle()
-> Mps<ariadnetor_tensor::BlockSparseStorage<f64>, ariadnetor_tensor::BlockSparseLayout<U1Sector>> {
    let mut counter: f64 = 0.1;

    let site0 = make_u1_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 2)],
        &mut counter,
    );
    // Site 1's right sector 1 (dim 3) is fed by both (left=0, phys=1) and
    // (left=1, phys=0): two pathways that combine into the same right
    // sector, giving the per-sector forward and backward unfoldings
    // distinct shapes.
    let site1 = make_u1_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 2)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 3), (U1Sector(2), 2)],
        &mut counter,
    );
    // Site 2 anchors total flux at 1; two pathways (left=0+phys=1 and
    // left=1+phys=0) feed the right=1 boundary.
    let site2 = make_u1_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 3), (U1Sector(2), 2)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(1), 1)],
        &mut counter,
    );

    Mps::from_sites(vec![site0, site1, site2])
}

/// Build a 4-site U(1)-symmetric MPS with `f64` storage and per-site flux 0.
pub(crate) fn make_4site_u1_mps()
-> Mps<ariadnetor_tensor::BlockSparseStorage<f64>, ariadnetor_tensor::BlockSparseLayout<U1Sector>> {
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
pub(crate) fn is_left_canonical_bsp(site: &BlockSparseTensor<f64, U1Sector>, tol: f64) -> bool {
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
pub(crate) fn is_right_canonical_bsp(site: &BlockSparseTensor<f64, U1Sector>, tol: f64) -> bool {
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
pub(crate) fn bsp_mps_contract_full(
    mps: &Mps<
        ariadnetor_tensor::BlockSparseStorage<f64>,
        ariadnetor_tensor::BlockSparseLayout<U1Sector>,
    >,
) -> BlockSparseTensor<f64, U1Sector> {
    let n = mps.len();
    assert!(n > 0, "cannot contract an empty MPS");
    let backend = NativeBackend::new();

    let mut acc = mps.site(0).clone();
    for j in 1..n {
        let site = mps.site(j);
        let last_axis = acc.rank() - 1;
        acc = tensordot(&backend, &acc, site, &[last_axis], &[0])
            .expect("chain contraction failed in bsp_mps_contract_full");
    }
    acc
}

/// Assert that two block-sparse tensors are element-wise close.
pub(crate) fn assert_block_sparse_close(
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
pub(crate) fn make_2site_entangled_u1_mps()
-> Mps<ariadnetor_tensor::BlockSparseStorage<f64>, ariadnetor_tensor::BlockSparseLayout<U1Sector>> {
    let mut site0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
    site0
        .data_mut()
        .block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()[0] = 1.0;
    site0
        .data_mut()
        .block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .unwrap()[0] = 2.0;

    let mut site1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(1), 1)], Direction::In),
        ]),
        U1Sector(0),
    );
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
pub(crate) fn make_total_n_u1_mpo(
    n: usize,
) -> Mpo<ariadnetor_tensor::BlockSparseStorage<f64>, ariadnetor_tensor::BlockSparseLayout<U1Sector>>
{
    assert!(n >= 1, "need at least one site");
    let mut sites = Vec::with_capacity(n);
    for j in 0..n {
        let left_dim = if j == 0 { 1 } else { 2 };
        let right_dim = if j == n - 1 { 1 } else { 2 };
        let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(
            legs([
                (vec![(U1Sector(0), left_dim)], Direction::Out),
                (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
                (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
                (vec![(U1Sector(0), right_dim)], Direction::In),
            ]),
            U1Sector(0),
        );

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
pub(crate) fn make_identity_u1_mpo(
    n: usize,
) -> Mpo<ariadnetor_tensor::BlockSparseStorage<f64>, ariadnetor_tensor::BlockSparseLayout<U1Sector>>
{
    let sites = (0..n)
        .map(|_| {
            let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(
                legs([
                    (vec![(U1Sector(0), 1)], Direction::Out),
                    (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In),
                    (vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out),
                    (vec![(U1Sector(0), 1)], Direction::In),
                ]),
                U1Sector(0),
            );
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
