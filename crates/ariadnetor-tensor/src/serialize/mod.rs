//! Per-tensor serialization primitives (Mid-layer).
//!
//! This module owns the tensor-level half of MPS serialization: the scalar
//! and sector codecs, the metadata DTOs, and the encode / decode of a single
//! Dense or BlockSparse tensor. The chain-level framing, container, and
//! public `save_mps` / `load_mps` entry points live in `ariadnetor-mps`.
//!
//! The split follows the format's two layers: self-describing CBOR metadata
//! (the DTOs here) plus an explicit little-endian numeric body (the scalar
//! codec here). Decode never panics on crafted input — it validates and does
//! checked arithmetic before calling the reconstruction constructors.

mod codec;
mod meta;
mod scalar;
mod sector;

#[cfg(test)]
mod tests;

pub use codec::{
    TensorCodecError, decode_block_sparse, decode_dense, encode_block_sparse, encode_dense,
};
pub use meta::{BodyMeta, DirectionTag, QnBlockDto, QnIndexDto, StorageTag};
pub use scalar::{ScalarCodec, ScalarDecodeError, ScalarTag};
pub use sector::{SectorDecodeError, SectorTag, SerializableSector};
