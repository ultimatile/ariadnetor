//! BlockSparse DmrgEnvs integration tests.
//!
//! Covers value correctness via densify-and-compare against the Dense
//! oracle, structural metadata (QNIndex / direction / flux), advance
//! consistency vs the Dense oracle, the N=2 edge case, and seven error
//! paths (QN mismatch, direction mismatch, malformed left edge,
//! malformed right edge, flux-disallowed boundary block, length
//! mismatch, empty chain).

use arnet_algorithms::dmrg::{DmrgEnvError, DmrgEnvs};
use arnet_mps::{Mpo, Mps, TensorChain};
use std::sync::Arc;

use arnet::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout,
    DenseStorage, DenseTensor, Direction, MemoryOrder, NativeBackend, QNIndex, Sector, U1Sector,
};

/// Run `DmrgEnvs::build` and assert it returns an error. Equivalent to
/// `Result::expect_err`, but doesn't require `BlockSparse: Debug`
/// (which is not derived upstream).
fn expect_build_err(
    mps: &Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
    mpo: &Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
) -> DmrgEnvError {
    match DmrgEnvs::build(mps, mpo) {
        Ok(_) => panic!("expected DmrgEnvs::build to error"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------
// Densify helper — convert a BlockSparseTensor<f64, U1Sector> into a row-major
// DenseTensor<f64> by scattering each allowed block into its global offset.
// ---------------------------------------------------------------------------

fn densify_bsp(bsp: &BlockSparseTensor<f64, U1Sector>) -> DenseTensor<f64> {
    let global_dims: Vec<usize> = bsp.shape().to_vec();
    let total: usize = global_dims.iter().product();
    let mut out = vec![0.0_f64; total];

    let rank = global_dims.len();
    // Per-axis prefix offsets so block (b0,b1,..) lands at the right
    // sub-tensor inside the dense buffer.
    let prefix_offsets: Vec<Vec<usize>> = bsp
        .indices()
        .iter()
        .map(|idx| {
            let mut acc = 0usize;
            (0..idx.num_blocks())
                .map(|b| {
                    let cur = acc;
                    acc += idx.block_dim(b);
                    cur
                })
                .collect()
        })
        .collect();

    for meta in bsp.block_metas() {
        let coord = &meta.coord;
        let block_shape = bsp.block_shape(coord).expect("allowed block");
        let block_data = bsp.block_data(coord).expect("allowed block");
        let offsets: Vec<usize> = (0..rank)
            .map(|axis| prefix_offsets[axis][coord.0[axis]])
            .collect();

        // Per-block data is stored in CM (NativeBackend's preferred order),
        // so iterate logical block coordinates while reading block_data via
        // a CM flat index. The scatter target `out` is built in RM and
        // reordered to CM at the end.
        let block_total: usize = block_shape.iter().product();
        let mut local = vec![0_usize; rank];
        for _ in 0..block_total {
            let mut cm_flat = 0_usize;
            let mut stride = 1_usize;
            for axis in 0..rank {
                cm_flat += local[axis] * stride;
                stride *= block_shape[axis];
            }
            let mut g = 0_usize;
            for axis in 0..rank {
                g = g * global_dims[axis] + (offsets[axis] + local[axis]);
            }
            out[g] = block_data[cm_flat];
            for axis in (0..rank).rev() {
                local[axis] += 1;
                if local[axis] < block_shape[axis] {
                    break;
                }
                local[axis] = 0;
            }
        }
    }

    // The scatter loop above writes `out` in row-major order; declare
    // the storage's `order` to match so the subsequent `reorder`'s
    // `from` argument is consistent with the storage authority.
    let rm = DenseTensor::from_raw_parts(
        out,
        global_dims,
        MemoryOrder::RowMajor,
        NativeBackend::shared(),
    );
    // The NativeBackend's preferred order is ColumnMajor; all Dense
    // tensors flowing through `contract` must be CM. Reorder once
    // before handing back.
    rm.reordered(MemoryOrder::ColumnMajor)
}

// ---------------------------------------------------------------------------
// Synthetic 4-site U(1) MPS with deterministic per-block distinct values.
// Bond structure mirrors arnet-mps's private make_4site_u1_mps fixture but
// is inlined here (the helper is not exposed by arnet_mps's public API).
// ---------------------------------------------------------------------------

fn make_u1_mps_site(
    left: Vec<(U1Sector, usize)>,
    phys: Vec<(U1Sector, usize)>,
    right: Vec<(U1Sector, usize)>,
    counter: &mut f64,
) -> BlockSparseTensor<f64, U1Sector> {
    let l = QNIndex::new(left, Direction::Out);
    let p = QNIndex::new(phys, Direction::Out);
    let r = QNIndex::new(right, Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![l, p, r], U1Sector(0));
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

fn make_4site_u1_mps_local() -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let mut counter: f64 = 0.1;
    let site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        &mut counter,
    );
    let site1 = make_u1_mps_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)],
        &mut counter,
    );
    let site2 = make_u1_mps_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        &mut counter,
    );
    let site3 = make_u1_mps_site(
        vec![(U1Sector(0), 2), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    Mps::from_sites(vec![site0, site1, site2, site3])
}

// ---------------------------------------------------------------------------
// Synthetic 4-site U(1) MPO with at least one bulk W bond carrying ≥ 2
// sectors including a negative charge (per the test fixture contract).
// W convention: (W_left=Out, ket=In, bra=Out, W_right=In). Per-site flux 0.
// ---------------------------------------------------------------------------

fn make_u1_mpo_site(
    w_left: Vec<(U1Sector, usize)>,
    w_right: Vec<(U1Sector, usize)>,
    counter: &mut f64,
) -> BlockSparseTensor<f64, U1Sector> {
    let w_l = QNIndex::new(w_left, Direction::Out);
    let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let w_r = QNIndex::new(w_right, Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![w_l, ket, bra, w_r], U1Sector(0));
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

fn make_4site_u1_mpo_local() -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    // Bulk W bond `{(-1):1, 0:1, +1:1}` — ≥ 2 sectors with a negative charge.
    let mut counter: f64 = 0.7;
    let site0 = make_u1_mpo_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let site1 = make_u1_mpo_site(
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let site2 = make_u1_mpo_site(
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let site3 = make_u1_mpo_site(
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    Mpo::from_sites(vec![site0, site1, site2, site3])
}

// ---------------------------------------------------------------------------
// Densify a Mps<BlockSparse> / Mpo<BlockSparse> into Dense counterparts so
// the same DmrgEnvs::build call exercises the Dense oracle path.
// ---------------------------------------------------------------------------

fn densify_mps(
    mps: &Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    let storages: Vec<DenseTensor<f64>> =
        (0..mps.len()).map(|i| densify_bsp(mps.site(i))).collect();
    Mps::from_sites(storages)
}

fn densify_mpo(
    mpo: &Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let storages: Vec<DenseTensor<f64>> =
        (0..mpo.len()).map(|i| densify_bsp(mpo.site(i))).collect();
    Mpo::from_sites(storages)
}

fn assert_dense_close(a: &DenseTensor<f64>, b: &DenseTensor<f64>, tol: f64, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
    let av = a.data_slice();
    let bv = b.data_slice();
    for (k, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
        let diff = (*x - *y).abs();
        assert!(
            diff <= tol,
            "{label}: divergence at flat index {k}: {x} vs {y} (|diff|={diff} > {tol})"
        );
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn bsp_envs_match_dense_via_densify() {
    let mps_bsp = make_4site_u1_mps_local();
    let mpo_bsp = make_4site_u1_mpo_local();
    let mps_dense = densify_mps(&mps_bsp);
    let mpo_dense = densify_mpo(&mpo_bsp);

    let envs_bsp = DmrgEnvs::build(&mps_bsp, &mpo_bsp).expect("build BS");
    let envs_dense = DmrgEnvs::build(&mps_dense, &mpo_dense).expect("build Dense");

    let n = envs_bsp.n_sites();
    // After build: left[0] populated; right[1..=n] populated; right[0] None.
    let left_bs_0 = densify_bsp(envs_bsp.left(0).expect("left[0]"));
    assert_dense_close(
        &left_bs_0,
        envs_dense.left(0).expect("left[0] dense"),
        1e-10,
        "left[0]",
    );
    for j in 1..=n {
        let bs = envs_bsp.right(j).expect("right slot");
        let bs_dense = densify_bsp(bs);
        assert_dense_close(
            &bs_dense,
            envs_dense.right(j).expect("right slot dense"),
            1e-10,
            &format!("right[{j}]"),
        );
    }
}

#[test]
fn bsp_advance_left_consistency_vs_dense() {
    let mps_bsp = make_4site_u1_mps_local();
    let mpo_bsp = make_4site_u1_mpo_local();
    let mps_dense = densify_mps(&mps_bsp);
    let mpo_dense = densify_mpo(&mpo_bsp);

    let mut envs_bsp = DmrgEnvs::build(&mps_bsp, &mpo_bsp).expect("build BS");
    let mut envs_dense = DmrgEnvs::build(&mps_dense, &mpo_dense).expect("build Dense");

    let n = envs_bsp.n_sites();
    for i in 0..(n - 1) {
        envs_bsp
            .advance_left(&mps_bsp, &mpo_bsp, i)
            .expect("advance_left BS");
        envs_dense
            .advance_left(&mps_dense, &mpo_dense, i)
            .expect("advance_left Dense");
        let bs_dense = densify_bsp(envs_bsp.left(i + 1).expect("left[i+1] BS"));
        assert_dense_close(
            &bs_dense,
            envs_dense.left(i + 1).expect("left[i+1] Dense"),
            1e-10,
            &format!("left[{}]", i + 1),
        );
    }
}

#[test]
fn bsp_advance_right_consistency_vs_dense() {
    let mps_bsp = make_4site_u1_mps_local();
    let mpo_bsp = make_4site_u1_mpo_local();
    let mps_dense = densify_mps(&mps_bsp);
    let mpo_dense = densify_mpo(&mpo_bsp);

    let mut envs_bsp = DmrgEnvs::build(&mps_bsp, &mpo_bsp).expect("build BS");
    let mut envs_dense = DmrgEnvs::build(&mps_dense, &mpo_dense).expect("build Dense");

    let n = envs_bsp.n_sites();
    // Cover an interior site: advance_right at j = n-1 reproduces right[n-1].
    let j = n - 1;
    envs_bsp
        .advance_right(&mps_bsp, &mpo_bsp, j)
        .expect("advance_right BS");
    envs_dense
        .advance_right(&mps_dense, &mpo_dense, j)
        .expect("advance_right Dense");
    let bs_dense = densify_bsp(envs_bsp.right(j).expect("right[j] BS"));
    assert_dense_close(
        &bs_dense,
        envs_dense.right(j).expect("right[j] Dense"),
        1e-10,
        &format!("right[{j}] (advance)"),
    );
}

#[test]
fn bsp_envs_n2_edge_case() {
    // N=2 chain: left[0] and right[2] are trivial; right[1] is the
    // single interior absorption result.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_bsp: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    let mut counter = 0.7_f64;
    let mpo_site0 = make_u1_mpo_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let mpo_site1 = make_u1_mpo_site(
        vec![(U1Sector(-1), 1), (U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mpo_bsp: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(vec![mpo_site0, mpo_site1]);

    let mps_dense = densify_mps(&mps_bsp);
    let mpo_dense = densify_mpo(&mpo_bsp);

    let envs_bsp = DmrgEnvs::build(&mps_bsp, &mpo_bsp).expect("build BS N=2");
    let envs_dense = DmrgEnvs::build(&mps_dense, &mpo_dense).expect("build Dense N=2");

    assert_eq!(envs_bsp.n_sites(), 2);
    let left_0 = densify_bsp(envs_bsp.left(0).expect("left[0]"));
    assert_dense_close(
        &left_0,
        envs_dense.left(0).expect("dense left[0]"),
        1e-10,
        "N=2 left[0]",
    );
    let right_1 = densify_bsp(envs_bsp.right(1).expect("right[1]"));
    assert_dense_close(
        &right_1,
        envs_dense.right(1).expect("dense right[1]"),
        1e-10,
        "N=2 right[1] (absorbed site 1)",
    );
    let right_2 = densify_bsp(envs_bsp.right(2).expect("right[2]"));
    assert_dense_close(
        &right_2,
        envs_dense.right(2).expect("dense right[2]"),
        1e-10,
        "N=2 right[2] trivial",
    );
}

#[test]
fn bsp_envs_structural_metadata() {
    let mps_bsp = make_4site_u1_mps_local();
    let mpo_bsp = make_4site_u1_mpo_local();
    let envs = DmrgEnvs::build(&mps_bsp, &mpo_bsp).expect("build");

    // Boundary at left[0]: dim-1 single-sector all three legs, flux=0.
    let l0 = envs.left(0).expect("left[0]");
    assert_eq!(l0.shape(), &[1, 1, 1], "left[0] shape");
    assert_eq!(*l0.flux(), U1Sector::identity());
    // env at left boundary: leg0 = MPS edge dir (Out), leg1 = flipped
    // MPO edge dir = flip(Out) = In, leg2 = flipped MPS edge dir = In.
    assert_eq!(l0.indices()[0].direction(), Direction::Out, "l0 leg0 dir");
    assert_eq!(l0.indices()[1].direction(), Direction::In, "l0 leg1 dir");
    assert_eq!(l0.indices()[2].direction(), Direction::In, "l0 leg2 dir");

    // Interior env right[2] absorbs sites {2, 3}; multi-block expected
    // because the bulk MPS / MPO bonds are multi-sector.
    let r2 = envs.right(2).expect("right[2]");
    assert_eq!(*r2.flux(), U1Sector::identity());
    assert!(
        r2.num_blocks() > 1,
        "right[2] should populate multiple block coordinates, got {}",
        r2.num_blocks()
    );
}

#[test]
fn bsp_envs_error_paths_qn_mismatch() {
    // MPS phys carries sectors {0, 1}; mismatched MPO ket carries {0, 2}.
    // The mismatch surfaces via contract_block_sparse during build's
    // right-sweep absorption.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    // MPO ket sector-set differs from MPS phys.
    let make_mismatched_mpo_site =
        |ket_sectors: Vec<(U1Sector, usize)>| -> BlockSparseTensor<f64, U1Sector> {
            let w_l = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
            let ket = QNIndex::new(ket_sectors, Direction::In);
            let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(2), 1)], Direction::Out);
            let w_r = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
            BlockSparseTensor::<f64, U1Sector>::zeros(vec![w_l, ket, bra, w_r], U1Sector(0))
        };
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> = Mpo::from_sites(vec![
        make_mismatched_mpo_site(vec![(U1Sector(0), 1), (U1Sector(2), 1)]),
        make_mismatched_mpo_site(vec![(U1Sector(0), 1), (U1Sector(2), 1)]),
    ]);

    let err = expect_build_err(&mps, &mpo);
    assert!(
        matches!(err, DmrgEnvError::Contract(_)),
        "expected Contract error wrapping LinalgError, got {err:?}"
    );
}

#[test]
fn bsp_envs_error_paths_direction_mismatch() {
    // MPS phys leg deliberately Direction::In (instead of Out): inside
    // extend_right_step step 2 the contracted pair `(site.phys, MPO.ket)`
    // collapses to `(In, In)` and contract_block_sparse rejects it.
    let make_bad_mps_site = || -> BlockSparseTensor<f64, U1Sector> {
        let l = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
        let p = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let r = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
        let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![l, p, r], U1Sector(0));
        let coords: Vec<BlockCoord> = site.block_metas().iter().map(|m| m.coord.clone()).collect();
        for coord in coords {
            let data = site.block_data_mut(&coord).unwrap();
            for slot in data.iter_mut() {
                *slot = 1.0;
            }
        }
        site
    };
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![make_bad_mps_site(), make_bad_mps_site()]);

    let make_mpo_site = || -> BlockSparseTensor<f64, U1Sector> {
        let w_l = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
        let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
        let w_r = QNIndex::new(vec![(U1Sector(0), 1)], Direction::In);
        let mut site =
            BlockSparseTensor::<f64, U1Sector>::zeros(vec![w_l, ket, bra, w_r], U1Sector(0));
        let coords: Vec<BlockCoord> = site.block_metas().iter().map(|m| m.coord.clone()).collect();
        for coord in coords {
            let data = site.block_data_mut(&coord).unwrap();
            for slot in data.iter_mut() {
                *slot = 1.0;
            }
        }
        site
    };
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(vec![make_mpo_site(), make_mpo_site()]);

    let err = expect_build_err(&mps, &mpo);
    assert!(
        matches!(err, DmrgEnvError::Contract(_)),
        "expected Contract error wrapping LinalgError direction failure, got {err:?}"
    );
}

#[test]
fn bsp_envs_error_paths_malformed_left_edge() {
    // MPS left edge has dim-2 single-sector — violates the dim-1 contract.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 2)], // dim 2 — not single-element
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    let mut counter = 0.7_f64;
    let mpo_site0 = make_u1_mpo_site(vec![(U1Sector(0), 1)], vec![(U1Sector(0), 1)], &mut counter);
    let mpo_site1 = make_u1_mpo_site(vec![(U1Sector(0), 1)], vec![(U1Sector(0), 1)], &mut counter);
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(vec![mpo_site0, mpo_site1]);

    let err = expect_build_err(&mps, &mpo);
    match err {
        DmrgEnvError::MalformedEdgeBond { leg } => assert_eq!(leg, "mps_left"),
        other => panic!("expected MalformedEdgeBond {{ leg: \"mps_left\" }}, got {other:?}"),
    }
}

#[test]
fn bsp_envs_error_paths_malformed_right_edge() {
    // MPO right edge has 2 sectors — violates single-sector contract.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    let mut counter = 0.7_f64;
    let mpo_site0 = make_u1_mpo_site(vec![(U1Sector(0), 1)], vec![(U1Sector(0), 1)], &mut counter);
    // Right MPO edge: two sectors — violates the contract.
    let mpo_site1 = make_u1_mpo_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        &mut counter,
    );
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(vec![mpo_site0, mpo_site1]);

    let err = expect_build_err(&mps, &mpo);
    match err {
        DmrgEnvError::MalformedEdgeBond { leg } => assert_eq!(leg, "mpo_right"),
        other => panic!("expected MalformedEdgeBond {{ leg: \"mpo_right\" }}, got {other:?}"),
    }
}

#[test]
fn bsp_envs_error_paths_flux_disallowed_boundary() {
    // MPS edge contributions to env's (0,0,0) flux check cancel (env_leg0
    // and env_leg2 carry the same sector with opposite directions); a
    // flux-disallowed boundary therefore requires a charged MPO edge.
    // Here the MPO left edge carries U1Sector(1), so the only allowed
    // env block has fused charge -1 ≠ identity and (0,0,0) is unallocated.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    // MPO with charged left edge — single-sector but sector ≠ identity.
    let make_charged_left_mpo_site = || -> BlockSparseTensor<f64, U1Sector> {
        let w_l = QNIndex::new(vec![(U1Sector(1), 1)], Direction::Out);
        let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
        let w_r = QNIndex::new(vec![(U1Sector(1), 1)], Direction::In);
        BlockSparseTensor::<f64, U1Sector>::zeros(vec![w_l, ket, bra, w_r], U1Sector(0))
    };
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> = Mpo::from_sites(vec![
        make_charged_left_mpo_site(),
        make_charged_left_mpo_site(),
    ]);

    let err = expect_build_err(&mps, &mpo);
    match err {
        DmrgEnvError::MalformedEdgeBond { leg } => assert_eq!(leg, "mpo_left"),
        other => {
            panic!("expected MalformedEdgeBond on flux-disallowed boundary, got {other:?}")
        }
    }
}

#[test]
fn bsp_envs_error_paths_length() {
    // MPS length 2, MPO length 3 — LengthMismatch.
    let mut counter = 0.1_f64;
    let mps_site0 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps_site1 = make_u1_mps_site(
        vec![(U1Sector(0), 1)],
        vec![(U1Sector(0), 1), (U1Sector(1), 1)],
        vec![(U1Sector(0), 1)],
        &mut counter,
    );
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::from_sites(vec![mps_site0, mps_site1]);

    let mut counter = 0.7_f64;
    let mpo_storages: Vec<BlockSparseTensor<f64, U1Sector>> = (0..3)
        .map(|_| make_u1_mpo_site(vec![(U1Sector(0), 1)], vec![(U1Sector(0), 1)], &mut counter))
        .collect();
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::from_sites(mpo_storages);

    let err = expect_build_err(&mps, &mpo);
    match err {
        DmrgEnvError::LengthMismatch { mps: m, mpo: o } => {
            assert_eq!(m, 2);
            assert_eq!(o, 3);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn bsp_envs_error_paths_empty_chain() {
    // An empty BlockSparse MPS / MPO triggers EmptyChain.
    let backend = NativeBackend::shared();
    let mps: Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mps::empty(Arc::clone(&backend));
    let mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> =
        Mpo::empty(Arc::clone(&backend));
    let err = expect_build_err(&mps, &mpo);
    assert!(
        matches!(err, DmrgEnvError::EmptyChain),
        "expected EmptyChain, got {err:?}"
    );
}
