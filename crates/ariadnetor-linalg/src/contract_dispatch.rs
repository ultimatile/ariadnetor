//! Layout-keyed dispatch for two-operand tensor contraction.
//!
//! [`LinalgContract`] is implemented on the concrete layout types
//! ([`DenseLayout`] and [`BlockSparseLayout<S>`]) and pairs a storage type via
//! [`LinalgContract::Storage`], routing to its storage-specific kernel. Callers
//! parameterized over `L: LinalgContract<T>` issue one generic
//! `contract(backend, &lhs, &rhs, notation)` call that serves both flavors,
//! mirroring the [`LinalgDecompose`](crate::LinalgDecompose) pattern for the
//! four decompositions.
//!
//! Both layouts take the same `&str` einsum notation (two operands, free output
//! ordering), parsed and validated by [`crate::contract_spec`]. The dense kernel
//! consumes the notation directly; the block-sparse kernel runs the natural-order
//! tensordot and then, when the notation requests a different leg order, reorders
//! the output via [`permute_block_sparse_dense`] — a composition justified by the
//! abelian-`Sector` invariance of the block sparsity pattern under leg
//! permutation. A full contraction returns a rank-0 tensor on both layouts.
//!
//! # Auto-policy vs explicit policy
//!
//! As with [`LinalgDecompose`], the auto-policy *choice* is layout-specific: the
//! dense path consults the backend's GEMM hook over the reshaped `(m, n, k)`,
//! while the block-sparse path pins [`ExecPolicy::Sequential`]. Both forms live
//! on the trait; the public auto entry point is the [`contract`] free fn below,
//! and the policy-explicit form is published under a bare name through
//! [`crate::expert`].

use arnet_core::Scalar;
use arnet_core::backend::ExecPolicy;
use arnet_core::compute_permutation;
use arnet_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, OpsFor, Sector, Storage,
    StorageFor, Tensor, TensorLayout,
};

use crate::block_sparse_contract::contract_block_sparse_with_policy_dense;
use crate::block_sparse_permute::permute_block_sparse_dense;
use crate::contract::{contract_axes_dense, contract_dense, contract_with_policy_dense};
use crate::contract_spec::ContractSpec;
use crate::error::LinalgError;
use crate::tensor_bridge::check_bsp_data_layout_order_matches;

/// Layout-keyed dispatch trait for two-operand tensor contraction.
///
/// Implemented for [`DenseLayout`] and [`BlockSparseLayout<S>`], each pairing a
/// storage type via [`Storage`](Self::Storage) and routing to its
/// storage-specific kernel. The return is a uniform `Tensor<Self::Storage, Self>`
/// — a full contraction yields a rank-0 tensor — so, unlike
/// [`LinalgDecompose`](crate::LinalgDecompose), no associated output type is
/// needed.
pub trait LinalgContract<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout.
    type Storage: Storage + StorageFor<Self>;

    /// Contract `lhs` and `rhs` over the einsum `notation`, with the layout's
    /// auto-selected execution policy.
    fn contract<B: OpsFor<Self::Storage>>(
        backend: &B,
        lhs: &Tensor<Self::Storage, Self>,
        rhs: &Tensor<Self::Storage, Self>,
        notation: &str,
    ) -> Result<Tensor<Self::Storage, Self>, LinalgError>;

    /// Contract `lhs` and `rhs` over the einsum `notation`, with a
    /// caller-specified execution policy.
    fn contract_with_policy<B: OpsFor<Self::Storage>>(
        backend: &B,
        lhs: &Tensor<Self::Storage, Self>,
        rhs: &Tensor<Self::Storage, Self>,
        notation: &str,
        policy: ExecPolicy,
    ) -> Result<Tensor<Self::Storage, Self>, LinalgError>;

    /// Tensordot `lhs` and `rhs` over the given axis pairs, emitting the output
    /// legs in their natural order (free left legs then free right legs, each in
    /// input axis order), with the layout's auto-selected execution policy.
    ///
    /// The axis-pair face of [`contract`](Self::contract). Both kernels are
    /// natively axis-based, so each implementation dispatches to its kernel
    /// directly over the axis pairs. The output order is natural — no reorder
    /// pass runs.
    fn tensordot<B: OpsFor<Self::Storage>>(
        backend: &B,
        lhs: &Tensor<Self::Storage, Self>,
        rhs: &Tensor<Self::Storage, Self>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<Tensor<Self::Storage, Self>, LinalgError>;
}

// ---------------------------------------------------------------------------
// Dense implementation
// ---------------------------------------------------------------------------

impl<T: Scalar> LinalgContract<T> for DenseLayout {
    type Storage = DenseStorage<T>;

    fn contract<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<DenseStorage<T>, DenseLayout>,
        rhs: &Tensor<DenseStorage<T>, DenseLayout>,
        notation: &str,
    ) -> Result<Tensor<DenseStorage<T>, DenseLayout>, LinalgError> {
        let result = contract_dense(backend, lhs.data(), rhs.data(), notation)?;
        Ok(Tensor::from_data(result))
    }

    fn contract_with_policy<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<DenseStorage<T>, DenseLayout>,
        rhs: &Tensor<DenseStorage<T>, DenseLayout>,
        notation: &str,
        policy: ExecPolicy,
    ) -> Result<Tensor<DenseStorage<T>, DenseLayout>, LinalgError> {
        let result = contract_with_policy_dense(backend, lhs.data(), rhs.data(), notation, policy)?;
        Ok(Tensor::from_data(result))
    }

    fn tensordot<B: OpsFor<DenseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<DenseStorage<T>, DenseLayout>,
        rhs: &Tensor<DenseStorage<T>, DenseLayout>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<Tensor<DenseStorage<T>, DenseLayout>, LinalgError> {
        // The dense kernel is axis-native, so call it directly over the axis
        // pairs — no notation synthesis, and no leg-count cap from labels.
        let result = contract_axes_dense(backend, lhs.data(), rhs.data(), axes_lhs, axes_rhs)?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// BlockSparse implementation
// ---------------------------------------------------------------------------

impl<T: Scalar, S: Sector> LinalgContract<T> for BlockSparseLayout<S> {
    type Storage = BlockSparseStorage<T>;

    fn contract<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        rhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        notation: &str,
    ) -> Result<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>, LinalgError> {
        Self::contract_with_policy(backend, lhs, rhs, notation, ExecPolicy::Sequential)
    }

    fn contract_with_policy<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        rhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        notation: &str,
        policy: ExecPolicy,
    ) -> Result<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>, LinalgError> {
        check_bsp_data_layout_order_matches(lhs.data(), backend, "contract_block_sparse: lhs")?;
        check_bsp_data_layout_order_matches(rhs.data(), backend, "contract_block_sparse: rhs")?;

        let spec = ContractSpec::from_notation(notation)?;
        // Reject a notation whose operand arity does not match the tensor rank,
        // mirroring the dense kernel's rank check. Without this, a notation
        // naming fewer axes than the operand has would silently treat the
        // undeclared axes as free and return an outer product.
        if lhs.rank() != spec.lhs_arity {
            return Err(LinalgError::InvalidArgument(format!(
                "LHS tensor rank {} doesn't match notation arity {}",
                lhs.rank(),
                spec.lhs_arity
            )));
        }
        if rhs.rank() != spec.rhs_arity {
            return Err(LinalgError::InvalidArgument(format!(
                "RHS tensor rank {} doesn't match notation arity {}",
                rhs.rank(),
                spec.rhs_arity
            )));
        }
        let natural = contract_block_sparse_with_policy_dense(
            backend,
            lhs.data(),
            rhs.data(),
            &spec.axes_lhs,
            &spec.axes_rhs,
            policy,
        )?;

        // Reorder the natural-order output legs into the requested order. The
        // permutation maps `new_leg[i] = natural_leg[perm[i]]`; an identity
        // permutation (including the rank-0 full-contraction case) skips the pass.
        let result = match compute_permutation(&spec.natural_labels, &spec.out_labels) {
            Some(perm) => permute_block_sparse_dense(backend, &natural, &perm)?,
            None => natural,
        };
        Ok(Tensor::from_data(result))
    }

    fn tensordot<B: OpsFor<BlockSparseStorage<T>>>(
        backend: &B,
        lhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        rhs: &Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>,
        axes_lhs: &[usize],
        axes_rhs: &[usize],
    ) -> Result<Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>>, LinalgError> {
        // The block-sparse kernel is natively axis-based and emits natural order,
        // so call it directly — no notation round-trip, and the kernel's own
        // `validate_contraction_axes` covers axis range / duplication / QN
        // compatibility.
        check_bsp_data_layout_order_matches(lhs.data(), backend, "tensordot_block_sparse: lhs")?;
        check_bsp_data_layout_order_matches(rhs.data(), backend, "tensordot_block_sparse: rhs")?;
        let result = contract_block_sparse_with_policy_dense(
            backend,
            lhs.data(),
            rhs.data(),
            axes_lhs,
            axes_rhs,
            ExecPolicy::Sequential,
        )?;
        Ok(Tensor::from_data(result))
    }
}

// ---------------------------------------------------------------------------
// Unified free fns — type-erase the layout into `L: LinalgContract<T>` so
// callers write `contract(backend, &lhs, &rhs, notation)` without naming the
// storage. `L` resolves from the `Tensor` arguments and `T` through
// `L::Storage`, the same inference the unified decomposition fns rely on.
// ---------------------------------------------------------------------------

/// Two-operand contraction over the einsum `notation`, using the supplied
/// backend and the layout's auto-selected execution policy.
pub fn contract<T, L, B>(
    backend: &B,
    lhs: &Tensor<L::Storage, L>,
    rhs: &Tensor<L::Storage, L>,
    notation: &str,
) -> Result<Tensor<L::Storage, L>, LinalgError>
where
    T: Scalar,
    L: LinalgContract<T>,
    B: OpsFor<L::Storage>,
{
    L::contract(backend, lhs, rhs, notation)
}

/// Tensordot: contract `lhs` and `rhs` over the given axis pairs, emitting the
/// output legs in their natural order (free left legs then free right legs, each
/// in input axis order). The axis-pair face of [`contract`], dispatched through
/// the same layout-keyed trait, so it serves both Dense and BlockSparse and
/// returns a uniform tensor (rank-0 for a full contraction).
///
/// Use this for plain tensordot, where the operand ranks may be generic;
/// reach for [`contract`] when the output legs need a non-natural order.
///
/// # Errors
///
/// `InvalidArgument` if `axes_lhs` and `axes_rhs` differ in length, an axis is
/// out of range, or an axis repeats within either list.
pub fn tensordot<T, L, B>(
    backend: &B,
    lhs: &Tensor<L::Storage, L>,
    rhs: &Tensor<L::Storage, L>,
    axes_lhs: &[usize],
    axes_rhs: &[usize],
) -> Result<Tensor<L::Storage, L>, LinalgError>
where
    T: Scalar,
    L: LinalgContract<T>,
    B: OpsFor<L::Storage>,
{
    L::tensordot(backend, lhs, rhs, axes_lhs, axes_rhs)
}

#[cfg(test)]
mod tests;
