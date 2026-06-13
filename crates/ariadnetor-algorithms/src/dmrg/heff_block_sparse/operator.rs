//! `EffectiveHamiltonian2SiteBlockSparse` and its
//! `LinearOp<T, NativeBackend>` implementation.
//!
//! The flat-buffer matvec lives here: scatter the flat input into a
//! pre-allocated psi `BlockSparseTensor` template, run four
//! `contract_block_sparse_with_backend` calls against the operator's
//! chain backend handle (the natural `lhs_free | rhs_free` order ends
//! in `[chi_l, d_i, d_{i+1}, chi_r]`, matching the input shape with
//! no axis swap), gather the rank-4 result back into a flat vector.
//! The template is owned by the operator so per-matvec allocation is
//! bounded to a single `BlockSparseTensor::clone` plus the transient
//! contract intermediates.

use std::sync::Arc;

use arnet_core::{ComputeBackend, Scalar};
use arnet_linalg::{BlockSparseContractResult, contract_block_sparse_with_backend};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockCoord, BlockSparseTensor, DenseTensor, Sector};

use super::super::heff_error::DmrgHeffError;
use crate::krylov::LinearOp;

/// Effective Hamiltonian operator for the BlockSparse 2-site DMRG
/// block at sites `(i, i+1)`.
pub struct EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B = NativeBackend>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    pub(super) left: &'a BlockSparseTensor<T, S, B>,
    pub(super) w_i: &'a BlockSparseTensor<T, S, B>,
    pub(super) w_ip1: &'a BlockSparseTensor<T, S, B>,
    pub(super) right: &'a BlockSparseTensor<T, S, B>,
    /// Zero-filled rank-4 BlockSparseTensor with the psi indices /
    /// flux. Cloned on every `apply` to receive the scattered flat
    /// input; also reused on the gather side so the output indexing
    /// matches.
    pub(super) psi_template: BlockSparseTensor<T, S, B>,
    /// Cumulative flat offsets for each block in
    /// `psi_template.block_metas()` order.
    pub(super) block_offsets: Vec<usize>,
    /// Cached `BlockCoord` per template block in
    /// `psi_template.block_metas()` order. Cached so the matvec hot
    /// path (`scatter_flat_to_template`) does not clone the coord
    /// `Vec` on every Lanczos / ARPACK iteration.
    pub(super) block_coords: Vec<BlockCoord>,
    /// Chain backend handle; the matvec body's contractions dispatch
    /// through it rather than deriving authority from the operands.
    backend: Arc<B>,
    psi_flux: S,
    dim: usize,
}

impl<'a, T, S, B> EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    /// Construct from env / MPO references plus the surrounding MPS
    /// sites (used only to derive the psi template's indices and
    /// flux).
    ///
    /// Asserts that the four contracted operands (`left`, `w_i`,
    /// `w_ip1`, `right`) have a layout `MemoryOrder` matching the
    /// `backend` argument's `preferred_order()`. The matvec body's
    /// `contract_block_sparse_with_backend` calls put intermediates
    /// into the backend's preferred order; an operand whose layout was
    /// built with a different order would fail the release-active
    /// layout-order check at the next contract entry, but failing here
    /// gives the caller a per-operand diagnostic before any matvec
    /// runs. The MPS sites (`mps_i`, `mps_ip1`) are used only to
    /// derive the psi template; their order is checked at the public
    /// step entry's Tier 2 site-scan, not here.
    pub fn new(
        left: &'a BlockSparseTensor<T, S, B>,
        w_i: &'a BlockSparseTensor<T, S, B>,
        w_ip1: &'a BlockSparseTensor<T, S, B>,
        right: &'a BlockSparseTensor<T, S, B>,
        mps_i: &BlockSparseTensor<T, S, B>,
        mps_ip1: &BlockSparseTensor<T, S, B>,
        backend: Arc<B>,
    ) -> Result<Self, DmrgHeffError> {
        debug_assert_eq!(left.rank(), 3, "left.rank == 3");
        debug_assert_eq!(right.rank(), 3, "right.rank == 3");
        debug_assert_eq!(w_i.rank(), 4, "W[i].rank == 4");
        debug_assert_eq!(w_ip1.rank(), 4, "W[i+1].rank == 4");
        debug_assert_eq!(mps_i.rank(), 3, "MPS[i].rank == 3");
        debug_assert_eq!(mps_ip1.rank(), 3, "MPS[i+1].rank == 3");

        let expected = backend.preferred_order();
        for (operand, tensor) in [
            ("left_env", left),
            ("w_i", w_i),
            ("w_ip1", w_ip1),
            ("right_env", right),
        ] {
            let actual = tensor.data().layout().order();
            if actual != expected {
                return Err(DmrgHeffError::OrderMismatch {
                    operand,
                    detail: format!(
                        "layout order {actual:?}, expected {expected:?} (chain backend preferred_order)"
                    ),
                });
            }
        }

        let psi_indices = vec![
            mps_i.indices()[0].clone(),
            mps_i.indices()[1].clone(),
            mps_ip1.indices()[1].clone(),
            mps_ip1.indices()[2].clone(),
        ];
        let psi_flux = mps_i.flux().fuse(mps_ip1.flux());
        let psi_template = BlockSparseTensor::<T, S, B>::zeros_with_backend(
            psi_indices,
            psi_flux.clone(),
            Arc::clone(&backend),
        );

        let mut block_offsets = Vec::with_capacity(psi_template.num_blocks() + 1);
        let mut block_coords = Vec::with_capacity(psi_template.num_blocks());
        let mut running = 0_usize;
        for meta in psi_template.block_metas() {
            block_offsets.push(running);
            block_coords.push(meta.coord.clone());
            running += meta.size;
        }
        block_offsets.push(running);
        let dim = running;

        Ok(Self {
            left,
            w_i,
            w_ip1,
            right,
            psi_template,
            block_offsets,
            block_coords,
            backend,
            psi_flux,
            dim,
        })
    }

    /// Length of the flat vector the matvec consumes / produces.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 2-site flux derived from `flux(MPS[i]) ⊕ flux(MPS[i+1])`.
    pub fn psi_flux(&self) -> &S {
        &self.psi_flux
    }
}

impl<'a, T, S, B> LinearOp<T, NativeBackend> for EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    fn apply(&self, v: &DenseTensor<T, NativeBackend>) -> DenseTensor<T, NativeBackend> {
        assert_eq!(
            v.shape(),
            &[self.dim],
            "BlockSparse heff matvec input must have shape [dim]"
        );
        let psi = scatter_flat_to_template(
            v.data_slice(),
            &self.psi_template,
            &self.block_offsets,
            &self.block_coords,
        );

        // env(a,b,c) × psi(c,i,j,f) → tmp1(a,b,i,j,f)
        let tmp1 =
            match contract_block_sparse_with_backend(&self.backend, self.left, &psi, &[2], &[0])
                .expect("BlockSparse heff step 1: validated by entry point")
            {
                BlockSparseContractResult::Tensor(t) => t,
                BlockSparseContractResult::Scalar(_) => {
                    unreachable!("rank 3 + rank 4 over 1 axis keeps rank 5")
                }
            };

        // tmp1(a,b,i,j,f) × W[i](b,i,s,m) → tmp2(a,j,f,s,m)
        let tmp2 = match contract_block_sparse_with_backend(
            &self.backend,
            &tmp1,
            self.w_i,
            &[1, 2],
            &[0, 1],
        )
        .expect("BlockSparse heff step 2: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 4 over 2 axes keeps rank 5")
            }
        };

        // tmp2(a,j,f,s,m) × W[i+1](m,j,t,g) → tmp3(a,f,s,t,g)
        let tmp3 = match contract_block_sparse_with_backend(
            &self.backend,
            &tmp2,
            self.w_ip1,
            &[1, 4],
            &[1, 0],
        )
        .expect("BlockSparse heff step 3: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 4 over 2 axes keeps rank 5")
            }
        };

        // tmp3(a,f,s,t,g) × right(h,g,f) → out(a,s,t,h)
        let out = match contract_block_sparse_with_backend(
            &self.backend,
            &tmp3,
            self.right,
            &[1, 4],
            &[2, 1],
        )
        .expect("BlockSparse heff step 4: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 3 over 2 axes keeps rank 4")
            }
        };

        assert_eq!(
            out.flux(),
            &self.psi_flux,
            "BlockSparse heff matvec output flux must equal psi_flux"
        );
        assert_eq!(
            out.indices().len(),
            self.psi_template.indices().len(),
            "BlockSparse heff matvec output rank must match template"
        );
        for (axis, (out_idx, tmpl_idx)) in out
            .indices()
            .iter()
            .zip(self.psi_template.indices().iter())
            .enumerate()
        {
            assert_eq!(
                out_idx.direction(),
                tmpl_idx.direction(),
                "BlockSparse heff matvec output axis {axis} direction must match template"
            );
            assert_eq!(
                out_idx.blocks(),
                tmpl_idx.blocks(),
                "BlockSparse heff matvec output axis {axis} sector list must match template"
            );
        }

        let flat = gather_template_aware(&out, &self.psi_template, &self.block_offsets, self.dim);
        // 1D output is layout-invariant. Bind to `NativeBackend` (the
        // Krylov family keeps its 1-D scratch space there); the input
        // is read-only here, so it imposes no order constraint.
        let native = NativeBackend::shared();
        DenseTensor::from_raw_parts(flat, vec![self.dim], native)
    }
}

/// Scatter a flat slice into a clone of the psi template.
pub(super) fn scatter_flat_to_template<T, S, B>(
    flat: &[T],
    template: &BlockSparseTensor<T, S, B>,
    block_offsets: &[usize],
    block_coords: &[BlockCoord],
) -> BlockSparseTensor<T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let mut out = template.clone();
    for (k, coord) in block_coords.iter().enumerate() {
        let lo = block_offsets[k];
        let hi = block_offsets[k + 1];
        let dst = out
            .block_data_mut(coord)
            .expect("template's allocated block must be writable");
        debug_assert_eq!(
            dst.len(),
            hi - lo,
            "scatter: block size mismatch with offsets"
        );
        dst.copy_from_slice(&flat[lo..hi]);
    }
    out
}

/// Gather a rank-4 BlockSparse output back into a flat `Vec<T>` of
/// length `dim`.
fn gather_template_aware<T, S, B>(
    out_4d: &BlockSparseTensor<T, S, B>,
    template: &BlockSparseTensor<T, S, B>,
    block_offsets: &[usize],
    dim: usize,
) -> Vec<T>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    let mut flat = vec![T::zero(); dim];
    for (k, meta) in template.block_metas().iter().enumerate() {
        if let Some(src) = out_4d.block_data(&meta.coord) {
            let lo = block_offsets[k];
            let hi = block_offsets[k + 1];
            debug_assert_eq!(src.len(), hi - lo, "gather: block size mismatch");
            flat[lo..hi].copy_from_slice(src);
        }
    }
    flat
}
