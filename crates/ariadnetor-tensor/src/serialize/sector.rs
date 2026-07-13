//! Sealed serialization capability for the built-in sector types.
//!
//! [`SerializableSector`] adds four things the codec needs on top of the
//! plain [`Sector`](crate::Sector) algebra: a type-identity tag, panic-free
//! `checked_fuse` / `checked_dual` for decode-side block enumeration, and a
//! raw value codec. It is sealed via a crate-private supertrait, so it stays
//! strictly narrower than `Sector` — the generic
//! [`BlockSparseLayout::new`](crate::BlockSparseLayout::new) is untouched, and
//! a downstream `Sector` reaches neither the codec nor
//! [`try_new`](crate::BlockSparseLayout::try_new).
//!
//! Sector values are **not** `serde`-derived: a derived public `Deserialize`
//! would let a caller construct an out-of-range `Z2Sector`, bypassing the
//! `0..=1` invariant its constructor enforces. Instead values travel as raw
//! payloads and [`decode_value`](SerializableSector::decode_value) range-checks
//! before constructing.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Sector, U1Sector, Z2Sector};

pub(crate) mod sealed {
    /// Crate-private supertrait that seals [`SerializableSector`](super::SerializableSector).
    pub trait Sealed {}
}

/// Recursive type-identity tag for a sector type.
///
/// Distinguishes the structurally-compatible newtypes: a `U1Sector` value `0`
/// or `1` is byte-compatible with a `Z2Sector`, so the tag is what lets a load
/// reject a U(1) file opened as Z₂.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectorTag {
    /// `U1Sector`.
    U1,
    /// `Z2Sector`.
    Z2,
    /// Direct-product `(A, B)`.
    Pair(Box<SectorTag>, Box<SectorTag>),
}

/// Failure decoding a raw sector value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SectorDecodeError {
    /// The payload ended before a full sector value could be read.
    #[error("unexpected end of data decoding a sector value")]
    UnexpectedEof,
    /// A Z₂ payload held a value outside `{0, 1}`.
    #[error("invalid Z2 sector value {0}: must be 0 or 1")]
    InvalidZ2(u8),
}

/// Sealed serialization capability for the built-in sectors.
///
/// See the module-level documentation for why this is narrower than `Sector`
/// and why sector values are not `serde`-derived.
pub trait SerializableSector: Sector + sealed::Sealed + Sized {
    /// Type-identity tag written into the manifest.
    fn type_tag() -> SectorTag;

    /// Fuse two sectors, returning `None` instead of panicking on overflow.
    fn checked_fuse(&self, other: &Self) -> Option<Self>;

    /// Dual sector, returning `None` instead of panicking on overflow.
    fn checked_dual(&self) -> Option<Self>;

    /// Append this sector's raw little-endian value bytes to `buf`.
    fn encode_value(&self, buf: &mut Vec<u8>);

    /// Read one sector value from the front of `bytes`, advancing the slice.
    fn decode_value(bytes: &mut &[u8]) -> Result<Self, SectorDecodeError>;
}

// --- U1 --------------------------------------------------------------------

impl sealed::Sealed for U1Sector {}

impl SerializableSector for U1Sector {
    fn type_tag() -> SectorTag {
        SectorTag::U1
    }

    fn checked_fuse(&self, other: &Self) -> Option<Self> {
        self.0.checked_add(other.0).map(U1Sector)
    }

    fn checked_dual(&self) -> Option<Self> {
        self.0.checked_neg().map(U1Sector)
    }

    fn encode_value(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.0.to_le_bytes());
    }

    fn decode_value(bytes: &mut &[u8]) -> Result<Self, SectorDecodeError> {
        if bytes.len() < 4 {
            return Err(SectorDecodeError::UnexpectedEof);
        }
        let (head, tail) = bytes.split_at(4);
        *bytes = tail;
        Ok(U1Sector(i32::from_le_bytes(
            head.try_into().expect("length checked"),
        )))
    }
}

// --- Z2 --------------------------------------------------------------------

impl sealed::Sealed for Z2Sector {}

impl SerializableSector for Z2Sector {
    fn type_tag() -> SectorTag {
        SectorTag::Z2
    }

    fn checked_fuse(&self, other: &Self) -> Option<Self> {
        // XOR of two in-range values stays in `{0, 1}`, so `new` cannot panic.
        Some(Z2Sector::new(self.value() ^ other.value()))
    }

    fn checked_dual(&self) -> Option<Self> {
        Some(*self)
    }

    fn encode_value(&self, buf: &mut Vec<u8>) {
        buf.push(self.value());
    }

    fn decode_value(bytes: &mut &[u8]) -> Result<Self, SectorDecodeError> {
        let (&first, tail) = bytes
            .split_first()
            .ok_or(SectorDecodeError::UnexpectedEof)?;
        *bytes = tail;
        if first > 1 {
            return Err(SectorDecodeError::InvalidZ2(first));
        }
        Ok(Z2Sector::new(first))
    }
}

// --- Direct product --------------------------------------------------------

impl<A: SerializableSector, B: SerializableSector> sealed::Sealed for (A, B) {}

impl<A: SerializableSector, B: SerializableSector> SerializableSector for (A, B) {
    fn type_tag() -> SectorTag {
        SectorTag::Pair(Box::new(A::type_tag()), Box::new(B::type_tag()))
    }

    fn checked_fuse(&self, other: &Self) -> Option<Self> {
        Some((
            self.0.checked_fuse(&other.0)?,
            self.1.checked_fuse(&other.1)?,
        ))
    }

    fn checked_dual(&self) -> Option<Self> {
        Some((self.0.checked_dual()?, self.1.checked_dual()?))
    }

    fn encode_value(&self, buf: &mut Vec<u8>) {
        self.0.encode_value(buf);
        self.1.encode_value(buf);
    }

    fn decode_value(bytes: &mut &[u8]) -> Result<Self, SectorDecodeError> {
        let a = A::decode_value(bytes)?;
        let b = B::decode_value(bytes)?;
        Ok((a, b))
    }
}
