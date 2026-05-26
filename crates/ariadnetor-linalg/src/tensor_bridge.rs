//! Cross-crate kernel-entry invariant helpers.

use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockSparseTensor, Sector};

/// Debug-time invariant: a block-sparse tensor reaching a linalg kernel
/// has its layout order equal to the backend's preferred order.
///
/// The block-sparse kernels operate at `backend.preferred_order()` and
/// the on-disk per-sector packed buffer is laid out in that order. A
/// mismatched layout tag would silently misinterpret the data. The
/// per-chain-construction / cross-chain-op-entry design pins this
/// equality at every chain-construction boundary; the assert below
/// catches misuse from direct `TensorData::new` constructions in
/// debug builds and is free in release.
#[track_caller]
pub(crate) fn assert_bsp_layout_order_matches_backend<T, S, B>(
    tensor: &BlockSparseTensor<T, S, B>,
    label: &'static str,
) where
    S: Sector,
    B: ComputeBackend,
{
    debug_assert_eq!(
        tensor.data().layout().order(),
        tensor.backend().preferred_order(),
        "{label}: layout order {:?} doesn't match backend preferred order {:?}; \
         construct the tensor via the chain-pinned constructors or align it \
         to the backend's preferred order before the linalg call",
        tensor.data().layout().order(),
        tensor.backend().preferred_order(),
    );
}
