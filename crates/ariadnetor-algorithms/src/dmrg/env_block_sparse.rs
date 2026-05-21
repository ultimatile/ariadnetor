//! BlockSparse implementation of [`super::env::DmrgEnvOps`].
//!
//! Mirrors the boundary convention used by `arnet_mps::inner::braket_bsp`
//! (the canonical BlockSparse braket reference): the env tensor's
//! axis 0 (top-bra-bond) carries the same `Direction` as the MPS edge
//! bond, axis 1 (W-bond) is flipped relative to the MPO edge, axis 2
//! (bot-ket-bond) is flipped relative to the MPS edge, and the env's
//! flux is `S::identity()`. The bra is built via
//! [`BlockSparseTensor::dagger`] (which flips QN directions, conjugates
//! values, and duals the flux), so flux is preserved through one
//! extend step regardless of per-site MPS charge.
//!
//! Boundary helpers reject inputs whose edge bonds violate the
//! dim-1 / single-sector contract or whose chosen edge sectors fail
//! to fuse to identity flux.

use std::sync::Arc;

use arnet::{
    BlockCoord, BlockSparseContractResult, BlockSparseLayout, BlockSparseStorage,
    BlockSparseTensor, ComputeBackend, Direction, LinalgError, MemoryOrder, QNIndex, Scalar,
    Sector, Tensor, contract_block_sparse, flat_index,
};

use super::env::{DmrgEnvError, DmrgEnvOps};

fn flip(d: Direction) -> Direction {
    match d {
        Direction::Out => Direction::In,
        Direction::In => Direction::Out,
    }
}

/// Swap axes 0 and 2 of a rank-3 BlockSparseTensor. The Dense
/// `extend_right_step` kernel ends with an einsum reorder
/// (`"ecbd,ade->abc"`) that places the bra's free axis first;
/// `contract_block_sparse` emits axes in the natural
/// `lhs_free | rhs_free` order, so the BlockSparse counterpart
/// receives `(c, b, a)` and must swap to `(a, b, c)` to match the env
/// axis convention.
///
/// `order` is the per-block memory layout used by the contracting
/// backend (`backend.preferred_order()`). Both old and new buffers
/// share that layout, so the helper is correct for either RowMajor or
/// ColumnMajor without code changes.
fn swap_axes_0_and_2<T, S, B>(
    t: &BlockSparseTensor<T, S, B>,
    order: MemoryOrder,
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    debug_assert_eq!(t.rank(), 3, "swap_axes_0_and_2 expects rank-3");
    let new_indices = vec![
        t.indices()[2].clone(),
        t.indices()[1].clone(),
        t.indices()[0].clone(),
    ];
    let backend_arc = Arc::clone(t.backend_arc());
    let _ = order;
    let mut out = BlockSparseTensor::<T, S, B>::zeros_with_backend(
        new_indices,
        t.flux().clone(),
        Arc::clone(&backend_arc),
    );
    let out_order = out.backend().preferred_order();
    let metas: Vec<_> = t.block_metas().to_vec();
    for meta in &metas {
        let old_coord = &meta.coord;
        let new_coord = BlockCoord(vec![old_coord.0[2], old_coord.0[1], old_coord.0[0]]);
        let old_data = t.block_data(old_coord).expect("allocated block").to_vec();
        let old_shape = t.block_shape(old_coord).expect("block shape");
        let new_shape = vec![old_shape[2], old_shape[1], old_shape[0]];
        let new_block = out
            .block_data_mut(&new_coord)
            .expect("flux-allowed under axis swap");
        let (d0, d1, d2) = (old_shape[0], old_shape[1], old_shape[2]);
        for a in 0..d2 {
            for b in 0..d1 {
                for c in 0..d0 {
                    let old_idx = flat_index(&[c, b, a], &old_shape, out_order);
                    let new_idx = flat_index(&[a, b, c], &new_shape, out_order);
                    new_block[new_idx] = old_data[old_idx];
                }
            }
        }
    }
    out
}

/// Verify that `idx` is a dim-1 single-sector edge bond.
fn check_dim1_single_sector<S: Sector>(
    idx: &QNIndex<S>,
    leg: &'static str,
) -> Result<(), DmrgEnvError> {
    if idx.num_blocks() == 1 && idx.block_dim(0) == 1 {
        Ok(())
    } else {
        Err(DmrgEnvError::MalformedEdgeBond { leg })
    }
}

/// Build the rank-3 boundary env tensor used by both
/// `trivial_left_boundary` and `trivial_right_boundary`.
fn build_boundary<T, S, B>(
    mps_edge: &QNIndex<S>,
    mpo_edge: &QNIndex<S>,
    mps_leg_name: &'static str,
    mpo_leg_name: &'static str,
    backend: &Arc<B>,
) -> Result<BlockSparseTensor<T, S, B>, DmrgEnvError>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    check_dim1_single_sector(mps_edge, mps_leg_name)?;
    check_dim1_single_sector(mpo_edge, mpo_leg_name)?;

    let env_leg0 = QNIndex::new(mps_edge.blocks().to_vec(), mps_edge.direction());
    let env_leg1 = QNIndex::new(mpo_edge.blocks().to_vec(), flip(mpo_edge.direction()));
    let env_leg2 = QNIndex::new(mps_edge.blocks().to_vec(), flip(mps_edge.direction()));

    let mut env = BlockSparseTensor::<T, S, B>::zeros_with_backend(
        vec![env_leg0, env_leg1, env_leg2],
        S::identity(),
        Arc::clone(backend),
    );
    let coord = BlockCoord(vec![0, 0, 0]);
    match env.block_data_mut(&coord) {
        Some(slot) => {
            slot[0] = T::one();
            Ok(env)
        }
        // The MPS contributions cancel by construction; if the chosen
        // edge sectors do not fuse to identity it is attributable to
        // the MPO leg.
        None => Err(DmrgEnvError::MalformedEdgeBond { leg: mpo_leg_name }),
    }
}

impl<T, S> DmrgEnvOps<T> for BlockSparseLayout<S>
where
    T: Scalar,
    S: Sector,
{
    type Storage = BlockSparseStorage<T>;

    fn assert_chain_order<B: ComputeBackend>(
        chain_backend: &Arc<B>,
        sites: &[Tensor<Self::Storage, Self, B>],
        ctx: &str,
    ) {
        let expected = chain_backend.preferred_order();
        for (i, site) in sites.iter().enumerate() {
            let got = site.data().layout().order();
            assert_eq!(
                got, expected,
                "{ctx}: site {i} order ({got:?}) != backend.preferred_order() ({expected:?})",
            );
            let site_backend_order = site.backend().preferred_order();
            assert_eq!(
                site_backend_order, expected,
                "{ctx}: site {i} cached backend preferred_order ({site_backend_order:?}) != chain backend preferred_order ({expected:?})",
            );
        }
    }

    fn trivial_left_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        mps_left_edge: &BlockSparseTensor<T, S, B>,
        mpo_left_edge: &BlockSparseTensor<T, S, B>,
    ) -> Result<BlockSparseTensor<T, S, B>, DmrgEnvError> {
        build_boundary(
            &mps_left_edge.indices()[0],
            &mpo_left_edge.indices()[0],
            "mps_left",
            "mpo_left",
            backend,
        )
    }

    fn trivial_right_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        mps_right_edge: &BlockSparseTensor<T, S, B>,
        mpo_right_edge: &BlockSparseTensor<T, S, B>,
    ) -> Result<BlockSparseTensor<T, S, B>, DmrgEnvError> {
        build_boundary(
            &mps_right_edge.indices()[2],
            &mpo_right_edge.indices()[3],
            "mps_right",
            "mpo_right",
            backend,
        )
    }

    /// Per-site left extension. Mirror of the
    /// `arnet_mps::inner::braket_bsp` loop body.
    fn extend_left_step<B: ComputeBackend>(
        _backend: &Arc<B>,
        env: &BlockSparseTensor<T, S, B>,
        site: &BlockSparseTensor<T, S, B>,
        mpo_site: &BlockSparseTensor<T, S, B>,
    ) -> Result<BlockSparseTensor<T, S, B>, LinalgError> {
        let bra = site.dagger();

        // env(a,b,c) × bra(a,d,e) → t1(b,c,d,e)
        let t1 = match contract_block_sparse(env, &bra, &[0], &[0])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 3 + rank 3 over 1 axis) keeps rank 4")
            }
        };

        // t1(b,c,d,e) × W(b,f,d,g) → t2(c,e,f,g)
        let t2 = match contract_block_sparse(&t1, mpo_site, &[0, 2], &[0, 2])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 4 over 2 axes) keeps rank 4")
            }
        };

        // t2(c,e,f,g) × site(c,f,h) → env'(e,g,h)
        let env_new = match contract_block_sparse(&t2, site, &[0, 2], &[0, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 3 over 2 axes) keeps rank 3")
            }
        };

        Ok(env_new)
    }

    /// Per-site right extension.
    fn extend_right_step<B: ComputeBackend>(
        backend: &Arc<B>,
        env: &BlockSparseTensor<T, S, B>,
        site: &BlockSparseTensor<T, S, B>,
        mpo_site: &BlockSparseTensor<T, S, B>,
    ) -> Result<BlockSparseTensor<T, S, B>, LinalgError> {
        let bra = site.dagger();

        // env(e,g,h) × site(c,f,h) → t1(e,g,c,f)
        let t1 = match contract_block_sparse(env, site, &[2], &[2])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 3 + rank 3 over 1 axis) keeps rank 4")
            }
        };

        // t1(e,g,c,f) × W(b,f,d,g) → t2(e,c,b,d)
        let t2 = match contract_block_sparse(&t1, mpo_site, &[1, 3], &[3, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 4 over 2 axes) keeps rank 4")
            }
        };

        // t2(e,c,b,d) × bra(a,d,e) → env_raw(c,b,a) — natural
        // contract_block_sparse output is `lhs_free | rhs_free`, so the
        // bra's free axis lands LAST and the t2 free axes (c,b) lead.
        let env_raw = match contract_block_sparse(&t2, &bra, &[0, 3], &[2, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 3 over 2 axes) keeps rank 3")
            }
        };

        Ok(swap_axes_0_and_2(&env_raw, backend.preferred_order()))
    }
}
