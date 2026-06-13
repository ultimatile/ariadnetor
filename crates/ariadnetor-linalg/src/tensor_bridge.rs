//! Cross-crate kernel-entry invariant helpers.

use arnet_core::backend::ComputeBackend;
use arnet_tensor::{BlockSparseTensorData, Sector};

use crate::error::LinalgError;

/// Release-active invariant for the explicit-backend block-sparse entry
/// points: the joined data's layout order must equal the supplied backend's
/// preferred order.
///
/// The backend here is supplied at the call site and is not pinned to the
/// tensor by construction, so a mismatch can occur at run time and would
/// silently reinterpret the
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
