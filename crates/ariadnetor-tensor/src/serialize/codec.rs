//! Per-tensor encode / decode against the metadata DTOs.
//!
//! Encoding splits a tensor into a [`BodyMeta`] descriptor (CBOR-bound, one
//! level up) plus a flat little-endian numeric body. Decoding validates the
//! descriptor and does checked extent arithmetic *before* invoking the
//! panicking reconstruction constructors, so crafted input yields a typed
//! [`TensorCodecError`] rather than a panic. Memory order is supplied by the
//! caller (it lives in the MPS manifest, not the per-tensor descriptor).

use ariadnetor_core::backend::MemoryOrder;
use thiserror::Error;

use super::meta::{BodyMeta, DirectionTag, QnBlockDto, QnIndexDto};
use super::scalar::{ScalarCodec, ScalarDecodeError};
use super::sector::{SectorDecodeError, SerializableSector};
use crate::block_sparse::BlockLayoutError;
use crate::{
    BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, DenseTensorData, Direction,
    QNIndex, TensorData, TensorLayout,
};

/// Failure decoding a tensor from its descriptor and numeric body.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorCodecError {
    /// A derived byte / element count overflowed `usize`.
    #[error("integer overflow computing the {0}")]
    Overflow(&'static str),
    /// The numeric body length did not match the descriptor's implied extent.
    #[error("extent mismatch: expected {expected} bytes of numeric data, found {found}")]
    ExtentMismatch {
        /// Bytes the descriptor implies.
        expected: usize,
        /// Bytes actually present.
        found: usize,
    },
    /// A sector payload had bytes left over after decoding one value.
    #[error("sector payload has trailing bytes")]
    TrailingSectorBytes,
    /// A quantum-number index violated a decode-time precondition.
    #[error("malformed quantum-number index: {0}")]
    MalformedIndex(&'static str),
    /// A scalar could not be read from the numeric body.
    #[error(transparent)]
    Scalar(#[from] ScalarDecodeError),
    /// A sector value could not be decoded.
    #[error(transparent)]
    Sector(#[from] SectorDecodeError),
    /// Block enumeration overflowed while rebuilding a block-sparse layout.
    #[error(transparent)]
    Layout(#[from] BlockLayoutError),
}

/// Product of `shape`, or `None` on `usize` overflow.
fn checked_product(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

fn dir_to_tag(direction: Direction) -> DirectionTag {
    match direction {
        Direction::Out => DirectionTag::Out,
        Direction::In => DirectionTag::In,
    }
}

fn tag_to_dir(tag: DirectionTag) -> Direction {
    match tag {
        DirectionTag::Out => Direction::Out,
        DirectionTag::In => Direction::In,
    }
}

/// Serialize the flat scalar buffer into `buf` as raw little-endian bytes.
fn write_body<T: ScalarCodec>(flat: &[T], buf: &mut Vec<u8>) {
    buf.reserve(flat.len() * T::BYTE_LEN);
    for &x in flat {
        x.write_le(buf);
    }
}

/// Read exactly `extent` scalars from `body`, requiring it to be exactly the
/// right length first (so a crafted descriptor cannot force a giant
/// allocation, and truncation / trailing bytes are rejected up front).
fn read_body<T: ScalarCodec>(extent: usize, mut body: &[u8]) -> Result<Vec<T>, TensorCodecError> {
    let needed = extent
        .checked_mul(T::BYTE_LEN)
        .ok_or(TensorCodecError::Overflow("numeric body length"))?;
    if body.len() != needed {
        return Err(TensorCodecError::ExtentMismatch {
            expected: needed,
            found: body.len(),
        });
    }
    let mut data = Vec::with_capacity(extent);
    for _ in 0..extent {
        data.push(T::read_le(&mut body)?);
    }
    Ok(data)
}

/// Encode a dense tensor into its descriptor and numeric body.
pub fn encode_dense<T: ScalarCodec>(tensor: &DenseTensorData<T>) -> (BodyMeta, Vec<u8>) {
    let meta = BodyMeta::Dense {
        shape: tensor.shape().to_vec(),
    };
    let mut body = Vec::new();
    write_body(tensor.data(), &mut body);
    (meta, body)
}

/// Decode a dense tensor from its shape, memory order, and numeric body.
pub fn decode_dense<T: ScalarCodec>(
    shape: &[usize],
    order: MemoryOrder,
    body: &[u8],
) -> Result<DenseTensorData<T>, TensorCodecError> {
    let extent = checked_product(shape).ok_or(TensorCodecError::Overflow("dense extent"))?;
    let data = read_body::<T>(extent, body)?;
    // `read_body` guarantees `data.len() == extent == product(shape)`, so the
    // length assertion inside `from_raw_parts` cannot fire.
    Ok(DenseTensorData::from_raw_parts(data, shape.to_vec(), order))
}

/// Encode a block-sparse tensor into its descriptor and numeric body.
///
/// The packed flat buffer is stored verbatim; reconstruction re-derives
/// identical block offsets because enumeration is deterministic in
/// `(indices, flux, order)`.
pub fn encode_block_sparse<T: ScalarCodec, S: SerializableSector>(
    tensor: &BlockSparseTensorData<T, S>,
) -> (BodyMeta, Vec<u8>) {
    let layout = tensor.layout();

    let mut flux = Vec::new();
    layout.flux().encode_value(&mut flux);

    let indices = layout
        .indices()
        .iter()
        .map(|idx| {
            let blocks = idx
                .blocks()
                .iter()
                .map(|(sector, dim)| {
                    let mut bytes = Vec::new();
                    sector.encode_value(&mut bytes);
                    QnBlockDto {
                        sector: bytes,
                        dim: *dim as u64,
                    }
                })
                .collect();
            QnIndexDto {
                direction: dir_to_tag(idx.direction()),
                blocks,
            }
        })
        .collect();

    let mut body = Vec::new();
    write_body(tensor.storage().data(), &mut body);
    (BodyMeta::BlockSparse { flux, indices }, body)
}

/// Decode a single stored sector value, requiring the payload to be fully
/// consumed.
fn decode_full_sector<S: SerializableSector>(mut payload: &[u8]) -> Result<S, TensorCodecError> {
    let sector = S::decode_value(&mut payload)?;
    if !payload.is_empty() {
        return Err(TensorCodecError::TrailingSectorBytes);
    }
    Ok(sector)
}

/// Rebuild a validated [`QNIndex`] from its DTO.
///
/// Pre-checks dims are `> 0` and sectors are strictly ascending (unique and
/// sorted) so the panicking [`QNIndex::new`] cannot fire.
fn decode_qn_index<S: SerializableSector>(
    dto: &QnIndexDto,
) -> Result<QNIndex<S>, TensorCodecError> {
    let mut blocks: Vec<(S, usize)> = Vec::with_capacity(dto.blocks.len());
    for block in &dto.blocks {
        let sector = decode_full_sector::<S>(&block.sector)?;
        let dim = usize::try_from(block.dim)
            .map_err(|_| TensorCodecError::Overflow("block dimension"))?;
        if dim == 0 {
            return Err(TensorCodecError::MalformedIndex("zero block dimension"));
        }
        if let Some((prev, _)) = blocks.last()
            && *prev >= sector
        {
            return Err(TensorCodecError::MalformedIndex(
                "sectors must be unique and ascending",
            ));
        }
        blocks.push((sector, dim));
    }
    Ok(QNIndex::new(blocks, tag_to_dir(dto.direction)))
}

/// Decode a block-sparse tensor from its flux payload, indices, memory order,
/// and numeric body.
pub fn decode_block_sparse<T: ScalarCodec, S: SerializableSector>(
    flux: &[u8],
    indices: &[QnIndexDto],
    order: MemoryOrder,
    body: &[u8],
) -> Result<BlockSparseTensorData<T, S>, TensorCodecError> {
    let flux = decode_full_sector::<S>(flux)?;
    let mut qn_indices = Vec::with_capacity(indices.len());
    for dto in indices {
        qn_indices.push(decode_qn_index::<S>(dto)?);
    }

    // The numeric body caps how many stored elements the tensor can have, so
    // pass it as the enumeration budget: this bounds the block table's memory
    // against a compact descriptor before `read_body` runs its exact check.
    let max_extent = body.len() / T::BYTE_LEN;
    let layout = BlockSparseLayout::try_new(qn_indices, flux, order, max_extent)?;
    let data = read_body::<T>(layout.storage_extent(), body)?;
    // `read_body` guarantees `data.len() == storage_extent`, so the
    // storage/layout length assertion inside `TensorData::new` cannot fire.
    Ok(TensorData::new(BlockSparseStorage::new(data), layout))
}
