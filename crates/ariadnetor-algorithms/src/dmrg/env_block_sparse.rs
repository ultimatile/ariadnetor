//! BlockSparse implementation of [`super::env::DmrgEnvOps`].
//!
//! Mirrors the boundary convention used by `arnet_mps::inner::braket_bsp`
//! (the canonical BlockSparse braket reference): the env tensor's
//! axis 0 (top-bra-bond) carries the same `Direction` as the MPS edge
//! bond, axis 1 (W-bond) is flipped relative to the MPO edge, axis 2
//! (bot-ket-bond) is flipped relative to the MPS edge, and the env's
//! flux is `S::identity()`. The bra is built via
//! [`BlockSparseTensorData::dagger`] (which flips QN directions,
//! conjugates values, and duals the flux), so flux is preserved
//! through one extend step regardless of per-site MPS charge.
//!
//! Boundary helpers reject inputs whose edge bonds violate the
//! dim-1 / single-sector contract or whose chosen edge sectors fail
//! to fuse to identity flux.
//!
//! [`BlockSparseTensorData::dagger`]: arnet_tensor::BlockSparseTensorData::dagger

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{BlockSparseContractResult, LinalgError, contract_block_sparse};
use arnet_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, Direction,
    MemoryOrder, QNIndex, Sector, TensorData, flat_index,
};

use super::env::{DmrgEnvError, DmrgEnvOps};

fn flip(d: Direction) -> Direction {
    match d {
        Direction::Out => Direction::In,
        Direction::In => Direction::Out,
    }
}

/// Swap axes 0 and 2 of a rank-3 BlockSparse tensor in place of an
/// explicit axis-permutation primitive. The dense `extend_right_step`
/// kernel ends with an einsum reorder (`"ecbd,ade->abc"`) that places
/// the bra's free axis first; `contract_block_sparse` emits axes in
/// the natural `lhs_free | rhs_free` order, so the BlockSparse
/// counterpart receives `(c, b, a)` and must swap to `(a, b, c)` to
/// match the env axis convention.
///
/// The per-block memory layout is read from the input tensor's own
/// layout (`t.layout().order()`), so both buffers share the order
/// used by the contracting backend without needing an extra
/// `backend.preferred_order()` argument.
fn swap_axes_0_and_2<T, S>(t: &BlockSparseTensorData<T, S>) -> BlockSparseTensorData<T, S>
where
    T: Scalar,
    S: Sector,
{
    debug_assert_eq!(t.layout().rank(), 3, "swap_axes_0_and_2 expects rank-3");
    let order = t.layout().order();
    let old_indices = t.layout().indices();
    let new_indices = vec![
        old_indices[2].clone(),
        old_indices[1].clone(),
        old_indices[0].clone(),
    ];
    let mut out =
        BlockSparseTensorData::<T, S>::zeros(new_indices, t.layout().flux().clone(), order);
    let metas: Vec<BlockCoord> = t
        .layout()
        .block_metas()
        .iter()
        .map(|m| m.coord.clone())
        .collect();
    for old_coord in metas {
        let new_coord = BlockCoord(vec![old_coord.0[2], old_coord.0[1], old_coord.0[0]]);
        let old_shape = t.layout().block_shape(&old_coord).expect("block shape");
        let new_shape = vec![old_shape[2], old_shape[1], old_shape[0]];
        let old_data: Vec<T> = t.block_data(&old_coord).expect("allocated block").to_vec();
        let new_block = out
            .block_data_mut(&new_coord)
            .expect("flux-allowed under axis swap");
        let (d0, d1, d2) = (old_shape[0], old_shape[1], old_shape[2]);
        for a in 0..d2 {
            for b in 0..d1 {
                for c in 0..d0 {
                    let old_idx = flat_index(&[c, b, a], &old_shape, order);
                    let new_idx = flat_index(&[a, b, c], &new_shape, order);
                    new_block[new_idx] = old_data[old_idx];
                }
            }
        }
    }
    out
}

/// Verify that `idx` is a dim-1 single-sector edge bond. Returns
/// `Err(MalformedEdgeBond { leg })` if the contract is violated.
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
/// `trivial_left_boundary` and `trivial_right_boundary`. Validates the
/// dim-1 / single-sector contract on `mps_edge` and `mpo_edge`, builds
/// the env with `flux = S::identity()`, and writes `T::one()` into the
/// `(0,0,0)` block. If the chosen edge sectors fuse to a non-identity
/// flux, the `(0,0,0)` block is not allocated and we return
/// `MalformedEdgeBond`.
fn build_boundary<T, S>(
    mps_edge: &QNIndex<S>,
    mpo_edge: &QNIndex<S>,
    mps_leg_name: &'static str,
    mpo_leg_name: &'static str,
    order: MemoryOrder,
) -> Result<BlockSparseTensorData<T, S>, DmrgEnvError>
where
    T: Scalar,
    S: Sector,
{
    check_dim1_single_sector(mps_edge, mps_leg_name)?;
    check_dim1_single_sector(mpo_edge, mpo_leg_name)?;

    let env_leg0 = QNIndex::new(mps_edge.blocks().to_vec(), mps_edge.direction());
    let env_leg1 = QNIndex::new(mpo_edge.blocks().to_vec(), flip(mpo_edge.direction()));
    let env_leg2 = QNIndex::new(mps_edge.blocks().to_vec(), flip(mps_edge.direction()));

    let mut env = BlockSparseTensorData::<T, S>::zeros(
        vec![env_leg0, env_leg1, env_leg2],
        S::identity(),
        order,
    );
    let coord = BlockCoord(vec![0, 0, 0]);
    match env.block_data_mut(&coord) {
        Some(slot) => {
            slot[0] = T::one();
            Ok(env)
        }
        // The MPS contributions to the env's flux check cancel by
        // construction (env_leg0 carries the MPS edge sector with the
        // chain's direction, env_leg2 carries the same sector with the
        // flipped direction). The chosen edge sectors fuse to identity
        // iff the MPO edge sector is itself identity, so attribute the
        // anomaly to the MPO leg.
        None => Err(DmrgEnvError::MalformedEdgeBond { leg: mpo_leg_name }),
    }
}

impl<T, S> DmrgEnvOps<BlockSparseLayout<S>> for BlockSparseStorage<T>
where
    T: Scalar,
    S: Sector,
{
    fn trivial_left_boundary<B: ComputeBackend>(
        _backend: &B,
        mps_left_edge: &TensorData<Self, BlockSparseLayout<S>>,
        mpo_left_edge: &TensorData<Self, BlockSparseLayout<S>>,
    ) -> Result<TensorData<Self, BlockSparseLayout<S>>, DmrgEnvError> {
        build_boundary(
            &mps_left_edge.layout().indices()[0],
            &mpo_left_edge.layout().indices()[0],
            "mps_left",
            "mpo_left",
            mps_left_edge.layout().order(),
        )
    }

    fn trivial_right_boundary<B: ComputeBackend>(
        _backend: &B,
        mps_right_edge: &TensorData<Self, BlockSparseLayout<S>>,
        mpo_right_edge: &TensorData<Self, BlockSparseLayout<S>>,
    ) -> Result<TensorData<Self, BlockSparseLayout<S>>, DmrgEnvError> {
        build_boundary(
            &mps_right_edge.layout().indices()[2],
            &mpo_right_edge.layout().indices()[3],
            "mps_right",
            "mpo_right",
            mps_right_edge.layout().order(),
        )
    }

    /// Per-site left extension. Mirror of the
    /// `arnet_mps::inner::braket_bsp` loop body, with `bra =
    /// site.dagger()` (flips QN directions and duals the flux so the
    /// resulting env's flux returns to `S::identity()` after step 3).
    fn extend_left_step<B: ComputeBackend>(
        backend: &B,
        env: &TensorData<Self, BlockSparseLayout<S>>,
        site: &TensorData<Self, BlockSparseLayout<S>>,
        mpo_site: &TensorData<Self, BlockSparseLayout<S>>,
    ) -> Result<TensorData<Self, BlockSparseLayout<S>>, LinalgError> {
        let bra = site.dagger();

        // env(a,b,c) × bra(a,d,e) → t1(b,c,d,e)
        let t1 = match contract_block_sparse(backend, env, &bra, &[0], &[0])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 3 + rank 3 over 1 axis) keeps rank 4")
            }
        };

        // t1(b,c,d,e) × W(b,f,d,g) → t2(c,e,f,g)
        let t2 = match contract_block_sparse(backend, &t1, mpo_site, &[0, 2], &[0, 2])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 4 over 2 axes) keeps rank 4")
            }
        };

        // t2(c,e,f,g) × site(c,f,h) → env'(e,g,h)
        let env_new = match contract_block_sparse(backend, &t2, site, &[0, 2], &[0, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 3 over 2 axes) keeps rank 3")
            }
        };

        Ok(env_new)
    }

    /// Per-site right extension. Mirror of [`Self::extend_left_step`]
    /// with the contraction order reversed.
    fn extend_right_step<B: ComputeBackend>(
        backend: &B,
        env: &TensorData<Self, BlockSparseLayout<S>>,
        site: &TensorData<Self, BlockSparseLayout<S>>,
        mpo_site: &TensorData<Self, BlockSparseLayout<S>>,
    ) -> Result<TensorData<Self, BlockSparseLayout<S>>, LinalgError> {
        let bra = site.dagger();

        // env(e,g,h) × site(c,f,h) → t1(e,g,c,f)
        let t1 = match contract_block_sparse(backend, env, site, &[2], &[2])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 3 + rank 3 over 1 axis) keeps rank 4")
            }
        };

        // t1(e,g,c,f) × W(b,f,d,g) → t2(e,c,b,d)
        let t2 = match contract_block_sparse(backend, &t1, mpo_site, &[1, 3], &[3, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 4 over 2 axes) keeps rank 4")
            }
        };

        // t2(e,c,b,d) × bra(a,d,e) → env_raw(c,b,a) — natural
        // contract_block_sparse output is `lhs_free | rhs_free`, so the
        // bra's free axis lands LAST and the t2 free axes (c,b) lead.
        // The dense einsum string "ecbd,ade->abc" reorders explicitly;
        // BlockSparse needs an axis swap to recover the env convention.
        let env_raw = match contract_block_sparse(backend, &t2, &bra, &[0, 3], &[2, 1])? {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("partial contraction (rank 4 + rank 3 over 2 axes) keeps rank 3")
            }
        };

        Ok(swap_axes_0_and_2(&env_raw))
    }
}
