//! The self-describing CBOR manifest: chain-level type-identity tags plus a
//! per-site descriptor table.
//!
//! Memory order is mirrored into a local [`OrderTag`] so `serde` stays off the
//! foreign `MemoryOrder` type (which lives in `ariadnetor-core`).

use ariadnetor_core::MemoryOrder;
use ariadnetor_tensor::{BodyMeta, ScalarTag, SectorTag, StorageTag};
use serde::{Deserialize, Serialize};

use crate::CanonicalForm;

/// Serializable mirror of [`MemoryOrder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderTag {
    /// Row-major (C order).
    RowMajor,
    /// Column-major (Fortran order).
    ColumnMajor,
}

impl From<MemoryOrder> for OrderTag {
    fn from(order: MemoryOrder) -> Self {
        match order {
            MemoryOrder::RowMajor => OrderTag::RowMajor,
            MemoryOrder::ColumnMajor => OrderTag::ColumnMajor,
        }
    }
}

impl From<OrderTag> for MemoryOrder {
    fn from(tag: OrderTag) -> Self {
        match tag {
            OrderTag::RowMajor => MemoryOrder::RowMajor,
            OrderTag::ColumnMajor => MemoryOrder::ColumnMajor,
        }
    }
}

/// Per-site descriptor: memory order, the tensor-body metadata, and the byte
/// length of this site's slice of the data section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiteMeta {
    /// Memory order the site's numeric bytes are laid out in.
    pub order: OrderTag,
    /// Tensor-body descriptor (shape / block structure).
    pub body: BodyMeta,
    /// Length in bytes of this site's numeric data in the data section.
    pub data_len: u64,
}

/// The chain-level manifest, CBOR-encoded ahead of the numeric data section.
///
/// The type-identity tags let a load reject a file whose scalar / storage /
/// sector type differs from the requested one before any numeric byte is
/// decoded. `format_version` enables clean rejection of unknown versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MpsManifest {
    /// On-disk format version.
    pub format_version: u32,
    /// Stored scalar element type.
    pub scalar_type: ScalarTag,
    /// Stored tensor storage kind.
    pub storage_type: StorageTag,
    /// Stored sector type, or `None` for dense (sectorless) chains.
    pub sector_type: Option<SectorTag>,
    /// Canonical form, round-tripped verbatim.
    pub canonical_form: CanonicalForm,
    /// Per-site descriptors, in chain order.
    pub sites: Vec<SiteMeta>,
}
