//! BlockSparse implementation of [`super::env::BraketEnvOps`].
//!
//! Mirrors the boundary convention used by `ariadnetor_mps::inner::braket_bsp`
//! (the canonical BlockSparse braket reference): the env tensor's
//! axis 0 (top-bra-bond) carries the same `Direction` as the bra edge
//! bond, axis 1 (W-bond) is flipped relative to the MPO edge, axis 2
//! (bot-ket-bond) is flipped relative to the ket edge, and the env's
//! flux is `S::identity()`. The bra is built via
//! [`BlockSparseTensor::dagger`] (which flips QN directions, conjugates
//! values, and duals the flux), so flux is preserved through one
//! extend step regardless of per-site charge.
//!
//! Boundary helpers reject inputs whose edge bonds violate the
//! dim-1 / single-sector contract, and require the bra / MPO / ket edge
//! charges to fuse to identity flux — i.e. the identity-flux `(0, 0, 0)`
//! boundary block must exist. That fusion condition admits an
//! MPO-charge-connected cross-sector `bra` / `ket` pair (a legitimate
//! non-zero overlap), exactly as `braket` does. A pair whose edges do
//! NOT fuse — a genuinely orthogonal, zero overlap — is rejected here as
//! `MalformedEdgeBond` rather than producing a zero environment (which
//! is what `braket` returns); that zero-environment case is the current
//! limitation, tracked separately.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{LinalgError, permute_block_sparse_with_backend, tensordot};
use ariadnetor_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, Host, QNIndex,
    Sector,
};

use super::env::{BraketEnvError, BraketEnvOps, BraketEnvs};

fn flip(d: Direction) -> Direction {
    match d {
        Direction::Out => Direction::In,
        Direction::In => Direction::Out,
    }
}

/// Swap axes 0 and 2 of a rank-3 BlockSparseTensor. The Dense
/// `extend_right_step` kernel ends with an einsum reorder
/// (`"ecbd,ade->abc"`) that places the bra's free axis first;
/// `tensordot` emits axes in the natural
/// `lhs_free | rhs_free` order, so the BlockSparse counterpart
/// receives `(c, b, a)` and must swap to `(a, b, c)` to match the env
/// axis convention.
fn swap_axes_0_and_2<T, S>(t: &BlockSparseTensor<T, S>) -> BlockSparseTensor<T, S>
where
    T: Scalar,
    S: Sector,
{
    debug_assert_eq!(t.rank(), 3, "swap_axes_0_and_2 expects rank-3");
    // `[2, 1, 0]` reverses axes 0 and 2 while keeping the W-bond axis 1;
    // `permute_block_sparse_with_backend` reorders indices, block
    // coordinates, and transposes each block's data. The unwrap covers
    // both failure modes: the permutation is fixed and valid for any
    // rank-3 input, and the layout-order check cannot fire because the
    // input is produced by a same-handle contraction, which emits
    // intermediates in the host substrate's preferred order.
    permute_block_sparse_with_backend(Host::shared().as_ref(), t, &[2, 1, 0])
        .expect("valid rank-3 permutation on a same-handle intermediate")
}

/// Verify that `idx` is a dim-1 single-sector edge bond.
fn check_dim1_single_sector<S: Sector>(
    idx: &QNIndex<S>,
    leg: &'static str,
) -> Result<(), BraketEnvError> {
    if idx.num_blocks() == 1 && idx.block_dim(0) == 1 {
        Ok(())
    } else {
        Err(BraketEnvError::MalformedEdgeBond { leg })
    }
}

/// Build the rank-3 boundary env tensor used by both
/// `trivial_left_boundary` and `trivial_right_boundary`.
///
/// Axis 0 carries the bra edge sector, axis 2 the (flipped) ket edge
/// sector. The `(0, 0, 0)` identity-flux block exists exactly when the
/// three edge charges fuse to identity; the boundary is valid iff it
/// does. This does not require `bra` and `ket` to share a sector — an
/// MPO edge carrying the compensating charge fuses a distinct-sector
/// pair, matching `braket`. An absent block (charges do not fuse) is a
/// zero-overlap boundary; it is reported as `MalformedEdgeBond` on the
/// MPO leg (a hint, exact for the common identity-edge case) rather than
/// producing a zero environment.
fn build_boundary<T, S>(
    bra_edge: &QNIndex<S>,
    mpo_edge: &QNIndex<S>,
    ket_edge: &QNIndex<S>,
    bra_leg_name: &'static str,
    mpo_leg_name: &'static str,
    ket_leg_name: &'static str,
) -> Result<BlockSparseTensor<T, S>, BraketEnvError>
where
    T: Scalar,
    S: Sector,
{
    check_dim1_single_sector(bra_edge, bra_leg_name)?;
    check_dim1_single_sector(mpo_edge, mpo_leg_name)?;
    check_dim1_single_sector(ket_edge, ket_leg_name)?;

    let env_leg0 = QNIndex::new(bra_edge.blocks().to_vec(), bra_edge.direction());
    let env_leg1 = QNIndex::new(mpo_edge.blocks().to_vec(), flip(mpo_edge.direction()));
    let env_leg2 = QNIndex::new(ket_edge.blocks().to_vec(), flip(ket_edge.direction()));

    let mut env =
        BlockSparseTensor::<T, S>::zeros(vec![env_leg0, env_leg1, env_leg2], S::identity());
    let coord = BlockCoord(vec![0, 0, 0]);
    match env.block_data_mut(&coord) {
        Some(slot) => {
            slot[0] = T::one();
            Ok(env)
        }
        // Absent block: the bra / MPO / ket edge charges do not fuse to
        // identity (a zero-overlap boundary). Attributed to the MPO leg
        // as a hint — exact for the common case where the MPS edges are
        // the identity sector, so the MPO edge carries the fusion.
        None => Err(BraketEnvError::MalformedEdgeBond { leg: mpo_leg_name }),
    }
}

impl<T, S> BraketEnvOps<T> for BraketEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>>
where
    T: Scalar,
    S: Sector,
{
    type Layout = BlockSparseLayout<S>;
    type Storage = BlockSparseStorage<T>;

    fn trivial_left_boundary(
        bra_left_edge: &BlockSparseTensor<T, S>,
        mpo_left_edge: &BlockSparseTensor<T, S>,
        ket_left_edge: &BlockSparseTensor<T, S>,
    ) -> Result<BlockSparseTensor<T, S>, BraketEnvError> {
        build_boundary(
            &bra_left_edge.indices()[0],
            &mpo_left_edge.indices()[0],
            &ket_left_edge.indices()[0],
            "bra_left",
            "mpo_left",
            "ket_left",
        )
    }

    fn trivial_right_boundary(
        bra_right_edge: &BlockSparseTensor<T, S>,
        mpo_right_edge: &BlockSparseTensor<T, S>,
        ket_right_edge: &BlockSparseTensor<T, S>,
    ) -> Result<BlockSparseTensor<T, S>, BraketEnvError> {
        build_boundary(
            &bra_right_edge.indices()[2],
            &mpo_right_edge.indices()[3],
            &ket_right_edge.indices()[2],
            "bra_right",
            "mpo_right",
            "ket_right",
        )
    }

    /// Per-site left extension. Mirror of the
    /// `ariadnetor_mps::inner::braket_bsp` loop body: bra =
    /// `bra_site.dagger()`, ket leg contracted last.
    fn extend_left_step(
        env: &BlockSparseTensor<T, S>,
        bra_site: &BlockSparseTensor<T, S>,
        mpo_site: &BlockSparseTensor<T, S>,
        ket_site: &BlockSparseTensor<T, S>,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        let backend = Host::shared();
        let bra = bra_site.dagger();

        // env(a,b,c) × bra(a,d,e) → t1(b,c,d,e)
        let t1 = tensordot(backend.as_ref(), env, &bra, &[0], &[0])?;

        // t1(b,c,d,e) × W(b,f,d,g) → t2(c,e,f,g)
        let t2 = tensordot(backend.as_ref(), &t1, mpo_site, &[0, 2], &[0, 2])?;

        // t2(c,e,f,g) × ket(c,f,h) → env'(e,g,h)
        let env_new = tensordot(backend.as_ref(), &t2, ket_site, &[0, 2], &[0, 1])?;

        Ok(env_new)
    }

    /// Per-site right extension.
    fn extend_right_step(
        env: &BlockSparseTensor<T, S>,
        bra_site: &BlockSparseTensor<T, S>,
        mpo_site: &BlockSparseTensor<T, S>,
        ket_site: &BlockSparseTensor<T, S>,
    ) -> Result<BlockSparseTensor<T, S>, LinalgError> {
        let backend = Host::shared();
        let bra = bra_site.dagger();

        // env(e,g,h) × ket(c,f,h) → t1(e,g,c,f)
        let t1 = tensordot(backend.as_ref(), env, ket_site, &[2], &[2])?;

        // t1(e,g,c,f) × W(b,f,d,g) → t2(e,c,b,d)
        let t2 = tensordot(backend.as_ref(), &t1, mpo_site, &[1, 3], &[3, 1])?;

        // t2(e,c,b,d) × bra(a,d,e) → env_raw(c,b,a) — natural
        // contract output is `lhs_free | rhs_free`, so the bra's free
        // axis lands LAST and the t2 free axes (c,b) lead.
        let env_raw = tensordot(backend.as_ref(), &t2, &bra, &[0, 3], &[2, 1])?;

        Ok(swap_axes_0_and_2(&env_raw))
    }
}
