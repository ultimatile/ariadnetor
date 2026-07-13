//! Self-describing metadata DTOs for the per-tensor body.
//!
//! These are the CBOR-serialized descriptors a load reconstructs a tensor
//! from. Sector values inside them travel as raw byte payloads (see
//! [`SerializableSector`](super::SerializableSector)); everything else is
//! plain serde. Memory order lives one level up in the MPS manifest, so these
//! descriptors carry no order field.

use serde::{Deserialize, Serialize};

/// Type-identity tag for the tensor storage kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTag {
    /// Dense storage.
    Dense,
    /// Block-sparse storage.
    BlockSparse,
}

/// Serializable mirror of [`Direction`](crate::Direction).
///
/// A local mirror keeps `serde` off the public `Direction` type while still
/// letting the leg direction round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectionTag {
    /// Outgoing leg.
    Out,
    /// Incoming leg.
    In,
}

/// Per-tensor body descriptor: everything a load needs besides the raw
/// numeric bytes and the memory order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyMeta {
    /// Dense tensor: reconstructed from `shape` + order + raw data.
    Dense {
        /// Logical shape.
        shape: Vec<usize>,
    },
    /// Block-sparse tensor: reconstructed by re-enumerating allowed blocks
    /// from `indices` + `flux`, then filling the packed buffer.
    BlockSparse {
        /// Raw little-endian bytes of the conserved flux sector.
        flux: Vec<u8>,
        /// Per-leg quantum-number indices.
        indices: Vec<QnIndexDto>,
    },
}

/// Serializable form of a per-leg quantum-number index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QnIndexDto {
    /// Leg direction.
    pub direction: DirectionTag,
    /// Sector–dimension blocks, in the stored (ascending-sector) order.
    pub blocks: Vec<QnBlockDto>,
}

/// One `(sector, block_dim)` entry of a [`QnIndexDto`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QnBlockDto {
    /// Raw little-endian bytes of the sector value.
    pub sector: Vec<u8>,
    /// Block dimension (always `> 0` in a well-formed file).
    pub dim: u64,
}
