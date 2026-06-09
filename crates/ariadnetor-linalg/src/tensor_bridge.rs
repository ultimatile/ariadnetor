//! Cross-crate kernel-entry invariant helpers.

use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockSparseTensor, BlockSparseTensorData, Sector};

use crate::error::LinalgError;

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

/// Release-active invariant for the explicit-backend block-sparse entry
/// points: the joined data's layout order must equal the supplied backend's
/// preferred order.
///
/// Unlike [`assert_bsp_layout_order_matches_backend`], the backend here is
/// supplied at the call site and is not pinned to the tensor by construction,
/// so a mismatch can occur at run time and would silently reinterpret the
/// per-sector packed buffer. The check is therefore release-active and returns
/// a [`LinalgError`] rather than a debug assertion. This is the same invariant
/// relocation the backend-unbundling Stage B applies to every block-sparse
/// operation entry point.
pub(crate) fn check_bsp_data_layout_order_matches<T, S>(
    data: &BlockSparseTensorData<T, S>,
    backend: &impl ComputeBackend,
    label: &'static str,
) -> Result<(), LinalgError>
where
    S: Sector,
{
    let data_order = data.layout().order();
    let backend_order = backend.preferred_order();
    if data_order != backend_order {
        return Err(LinalgError::InvalidArgument(format!(
            "{label}: layout order {data_order:?} doesn't match the supplied \
             backend's preferred order {backend_order:?}; align the tensor to \
             the backend's preferred order before the linalg call"
        )));
    }
    Ok(())
}
