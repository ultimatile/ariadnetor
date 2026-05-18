//! `EffectiveHamiltonian2SiteBlockSparse` and its `LinearOp<T>`
//! implementation.
//!
//! The flat-buffer matvec lives here: scatter the flat input into a
//! pre-allocated psi `BlockSparseTensorData` template, run four
//! `contract_block_sparse` calls (the natural `lhs_free | rhs_free`
//! order ends in `[chi_l, d_i, d_{i+1}, chi_r]`, matching the input
//! shape with no axis swap), gather the rank-4 result back into a
//! flat vector. The template is owned by the operator so per-matvec
//! allocation is bounded to a single template clone plus the
//! transient contract intermediates.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::{ComputeBackend, MemoryOrder};
use arnet_linalg::{BlockSparseContractResult, contract_block_sparse};
use arnet_native::NativeBackend;
use arnet_tensor::{BlockSparseTensorData, DenseTensorData, Sector};

use crate::krylov::LinearOp;

/// Effective Hamiltonian operator for the BlockSparse 2-site
/// DMRG block at sites `(i, i+1)`. Built once per local update;
/// the psi template is allocated at construction so each
/// [`LinearOp::apply`] call avoids re-enumerating flux-allowed
/// blocks.
pub struct EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B = NativeBackend>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    pub(super) left: &'a BlockSparseTensorData<T, S>,
    pub(super) w_i: &'a BlockSparseTensorData<T, S>,
    pub(super) w_ip1: &'a BlockSparseTensorData<T, S>,
    pub(super) right: &'a BlockSparseTensorData<T, S>,
    /// Zero-filled rank-4 BlockSparse with the psi indices / flux.
    /// Cloned on every `apply` to receive the scattered flat input;
    /// also reused on the gather side so the output indexing
    /// matches.
    pub(super) psi_template: BlockSparseTensorData<T, S>,
    /// Cumulative flat offsets for each block in the template's
    /// `block_metas()` order. `block_offsets[k]` is the starting
    /// flat index of the k-th block; `block_offsets` has length
    /// `num_blocks + 1` with the final entry equal to `dim`.
    pub(super) block_offsets: Vec<usize>,
    psi_flux: S,
    dim: usize,
    pub(super) backend: Arc<B>,
}

impl<'a, T, S, B> EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    /// Construct from env / MPO references plus the surrounding
    /// MPS sites (used only to derive the psi template's indices
    /// and flux).
    ///
    /// Validates ranks via `debug_assert!`; the public entry point
    /// [`super::dmrg_2site_step_block_sparse`] performs full QN /
    /// Direction / dim / flux validation up front so this
    /// constructor's `.expect`-style invariants cannot fire on
    /// well-formed input.
    pub fn new(
        left: &'a BlockSparseTensorData<T, S>,
        w_i: &'a BlockSparseTensorData<T, S>,
        w_ip1: &'a BlockSparseTensorData<T, S>,
        right: &'a BlockSparseTensorData<T, S>,
        mps_i: &BlockSparseTensorData<T, S>,
        mps_ip1: &BlockSparseTensorData<T, S>,
        backend: Arc<B>,
    ) -> Self {
        debug_assert_eq!(left.layout().rank(), 3, "left.rank == 3");
        debug_assert_eq!(right.layout().rank(), 3, "right.rank == 3");
        debug_assert_eq!(w_i.layout().rank(), 4, "W[i].rank == 4");
        debug_assert_eq!(w_ip1.layout().rank(), 4, "W[i+1].rank == 4");
        debug_assert_eq!(mps_i.layout().rank(), 3, "MPS[i].rank == 3");
        debug_assert_eq!(mps_ip1.layout().rank(), 3, "MPS[i+1].rank == 3");

        let psi_indices = vec![
            mps_i.layout().indices()[0].clone(),
            mps_i.layout().indices()[1].clone(),
            mps_ip1.layout().indices()[1].clone(),
            mps_ip1.layout().indices()[2].clone(),
        ];
        let psi_flux = mps_i.layout().flux().fuse(mps_ip1.layout().flux());
        // The psi template's per-block memory order must match the
        // env / MPO operands' order so `contract_block_sparse`
        // accepts the pair. All env / MPO tensors are produced by
        // the same active backend and therefore share a single
        // order; read it off `left` rather than re-querying the
        // backend.
        let order = left.layout().order();
        let psi_template =
            BlockSparseTensorData::<T, S>::zeros(psi_indices, psi_flux.clone(), order);

        let mut block_offsets = Vec::with_capacity(psi_template.layout().num_blocks() + 1);
        let mut running = 0_usize;
        for meta in psi_template.layout().block_metas() {
            block_offsets.push(running);
            running += meta.size;
        }
        block_offsets.push(running);
        let dim = running;

        Self {
            left,
            w_i,
            w_ip1,
            right,
            psi_template,
            block_offsets,
            psi_flux,
            dim,
            backend,
        }
    }

    /// Length of the flat vector the matvec consumes / produces.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 2-site flux derived from `flux(MPS[i]) ⊕ flux(MPS[i+1])`.
    /// Equals `Vt.flux()` after the SVD split.
    pub fn psi_flux(&self) -> &S {
        &self.psi_flux
    }
}

impl<'a, T, S, B> LinearOp<T> for EffectiveHamiltonian2SiteBlockSparse<'a, T, S, B>
where
    T: Scalar,
    S: Sector,
    B: ComputeBackend,
{
    fn apply(&self, v: &DenseTensorData<T>) -> DenseTensorData<T> {
        assert_eq!(
            v.shape(),
            &[self.dim],
            "BlockSparse heff matvec input must have shape [dim]"
        );
        let psi = scatter_flat_to_template(v.data(), &self.psi_template, &self.block_offsets);
        let backend = &*self.backend;

        // env(a,b,c) × psi(c,i,j,f) → tmp1(a,b,i,j,f)
        let tmp1 = match contract_block_sparse(backend, self.left, &psi, &[2], &[0])
            .expect("BlockSparse heff step 1: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 3 + rank 4 over 1 axis keeps rank 5")
            }
        };

        // tmp1(a,b,i,j,f) × W[i](b,i,s,m) → tmp2(a,j,f,s,m)
        let tmp2 = match contract_block_sparse(backend, &tmp1, self.w_i, &[1, 2], &[0, 1])
            .expect("BlockSparse heff step 2: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 4 over 2 axes keeps rank 5")
            }
        };

        // tmp2(a,j,f,s,m) × W[i+1](m,j,t,g) → tmp3(a,f,s,t,g)
        let tmp3 = match contract_block_sparse(backend, &tmp2, self.w_ip1, &[1, 4], &[1, 0])
            .expect("BlockSparse heff step 3: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 4 over 2 axes keeps rank 5")
            }
        };

        // tmp3(a,f,s,t,g) × right(h,g,f) → out(a,s,t,h)
        let out = match contract_block_sparse(backend, &tmp3, self.right, &[1, 4], &[2, 1])
            .expect("BlockSparse heff step 4: validated by entry point")
        {
            BlockSparseContractResult::Tensor(t) => t,
            BlockSparseContractResult::Scalar(_) => {
                unreachable!("rank 5 + rank 3 over 2 axes keeps rank 4")
            }
        };

        // Structural invariants: the gather walks the psi
        // template's `block_metas()` and looks up each coord in
        // `out`. If `out`'s indices / flux disagree with the
        // template (only reachable when a caller constructs
        // `EffectiveHamiltonian2SiteBlockSparse::new` directly with
        // unvalidated inputs — `dmrg_2site_step_block_sparse`'s
        // entry-point validation rules this out), the gather
        // would silently return wrong (mostly-zero) results. Hard
        // assertions here turn the silent failure into a loud
        // panic regardless of build profile. Cost is O(rank ×
        // num_sectors), small relative to the four BlockSparse
        // contractions above.
        assert_eq!(
            out.layout().flux(),
            &self.psi_flux,
            "BlockSparse heff matvec output flux must equal psi_flux"
        );
        assert_eq!(
            out.layout().indices().len(),
            self.psi_template.layout().indices().len(),
            "BlockSparse heff matvec output rank must match template"
        );
        for (axis, (out_idx, tmpl_idx)) in out
            .layout()
            .indices()
            .iter()
            .zip(self.psi_template.layout().indices().iter())
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
        // 1D output is layout-invariant; the Lanczos basis uses
        // `MemoryOrder::ColumnMajor`, so tag the output likewise to
        // match without forcing a downstream reorder.
        DenseTensorData::from_raw_parts(flat, vec![self.dim], MemoryOrder::ColumnMajor)
    }
}

/// Scatter a flat slice into a clone of the psi template. Each
/// block's per-block buffer is filled by direct memcpy from the
/// flat slice's `[block_offsets[k]..block_offsets[k+1]]` slab —
/// the per-block memory order (set by the template's layout) is
/// preserved bit-for-bit.
pub(super) fn scatter_flat_to_template<T, S>(
    flat: &[T],
    template: &BlockSparseTensorData<T, S>,
    block_offsets: &[usize],
) -> BlockSparseTensorData<T, S>
where
    T: Scalar,
    S: Sector,
{
    let mut out = template.clone();
    // `template` and `out` are separate `BlockSparseTensorData`
    // instances after `clone`, so iterating `template.layout().block_metas()`
    // while mutating `out.block_data_mut(...)` is a clean two-borrow
    // pattern — no per-call `Vec<BlockCoord>` cache needed.
    let coords: Vec<_> = template
        .layout()
        .block_metas()
        .iter()
        .map(|m| (m.coord.clone(), m.size))
        .collect();
    for (k, (coord, size)) in coords.iter().enumerate() {
        let lo = block_offsets[k];
        let hi = block_offsets[k + 1];
        debug_assert_eq!(*size, hi - lo, "scatter: block size mismatch with offsets");
        let dst = out
            .block_data_mut(coord)
            .expect("template's allocated block must be writable");
        dst.copy_from_slice(&flat[lo..hi]);
    }
    out
}

/// Gather a rank-4 BlockSparse output back into a flat `Vec<T>` of
/// length `dim`. Walks the psi template's block enumeration; for
/// each coord, looks up `out_4d.block_data(coord)`. A `None`
/// lookup leaves the corresponding flat slab at zero (which is
/// safe because `contract_block_sparse` allocates every
/// flux-allowed coord — `None` would only occur if the output's
/// indices / flux do not match the template, which the entry
/// point pre-validation prevents).
fn gather_template_aware<T, S>(
    out_4d: &BlockSparseTensorData<T, S>,
    template: &BlockSparseTensorData<T, S>,
    block_offsets: &[usize],
    dim: usize,
) -> Vec<T>
where
    T: Scalar,
    S: Sector,
{
    let mut flat = vec![T::zero(); dim];
    for (k, meta) in template.layout().block_metas().iter().enumerate() {
        if let Some(src) = out_4d.block_data(&meta.coord) {
            let lo = block_offsets[k];
            let hi = block_offsets[k + 1];
            debug_assert_eq!(src.len(), hi - lo, "gather: block size mismatch");
            flat[lo..hi].copy_from_slice(src);
        }
    }
    flat
}
