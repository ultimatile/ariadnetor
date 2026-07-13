//! Scalar type tag and explicit little-endian byte codec.
//!
//! Numeric tensor bodies are written as raw little-endian scalar bytes
//! rather than pushed through a generic serialization data model: that keeps
//! the complex representation independent of any derive, avoids inflating
//! temporary buffers, and makes the on-disk memory layout explicit. Encoding
//! is bit-exact, so signed zero, infinities, and distinct NaN payloads all
//! round-trip.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Complex, Scalar};

/// Type-identity tag for the stored scalar element.
///
/// Written into the manifest so a load rejects a file whose scalar type
/// differs from the requested `T` before decoding any numeric bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarTag {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `Complex<f32>`.
    C32,
    /// `Complex<f64>`.
    C64,
}

/// Failure decoding a scalar from the numeric body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScalarDecodeError {
    /// The body ended before a full scalar could be read.
    #[error("unexpected end of numeric body: needed {needed} bytes, {available} available")]
    UnexpectedEof {
        /// Bytes required for the scalar.
        needed: usize,
        /// Bytes remaining in the body.
        available: usize,
    },
}

/// Bit-exact little-endian codec for the sealed set of tensor scalars.
///
/// Bounding on the sealed [`Scalar`] trait means only `f32`, `f64`,
/// `Complex<f32>`, and `Complex<f64>` can implement this — the codec is
/// closed to the same four types the storage layer supports.
pub trait ScalarCodec: Scalar {
    /// Type-identity tag for this scalar.
    const TAG: ScalarTag;

    /// Number of little-endian bytes one scalar occupies.
    const BYTE_LEN: usize;

    /// Append this scalar's little-endian bytes to `buf`.
    fn write_le(self, buf: &mut Vec<u8>);

    /// Read one scalar from the front of `bytes`, advancing the slice.
    fn read_le(bytes: &mut &[u8]) -> Result<Self, ScalarDecodeError>;
}

/// Split `n` bytes off the front of `bytes`, advancing it, or report EOF.
fn take_bytes<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8], ScalarDecodeError> {
    if bytes.len() < n {
        return Err(ScalarDecodeError::UnexpectedEof {
            needed: n,
            available: bytes.len(),
        });
    }
    let (head, tail) = bytes.split_at(n);
    *bytes = tail;
    Ok(head)
}

impl ScalarCodec for f32 {
    const TAG: ScalarTag = ScalarTag::F32;
    const BYTE_LEN: usize = 4;

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &mut &[u8]) -> Result<Self, ScalarDecodeError> {
        let head = take_bytes(bytes, 4)?;
        Ok(f32::from_le_bytes(head.try_into().expect("length checked")))
    }
}

impl ScalarCodec for f64 {
    const TAG: ScalarTag = ScalarTag::F64;
    const BYTE_LEN: usize = 8;

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &mut &[u8]) -> Result<Self, ScalarDecodeError> {
        let head = take_bytes(bytes, 8)?;
        Ok(f64::from_le_bytes(head.try_into().expect("length checked")))
    }
}

impl ScalarCodec for Complex<f32> {
    const TAG: ScalarTag = ScalarTag::C32;
    const BYTE_LEN: usize = 8;

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.re.to_le_bytes());
        buf.extend_from_slice(&self.im.to_le_bytes());
    }

    fn read_le(bytes: &mut &[u8]) -> Result<Self, ScalarDecodeError> {
        let re = f32::read_le(bytes)?;
        let im = f32::read_le(bytes)?;
        Ok(Complex::new(re, im))
    }
}

impl ScalarCodec for Complex<f64> {
    const TAG: ScalarTag = ScalarTag::C64;
    const BYTE_LEN: usize = 16;

    fn write_le(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.re.to_le_bytes());
        buf.extend_from_slice(&self.im.to_le_bytes());
    }

    fn read_le(bytes: &mut &[u8]) -> Result<Self, ScalarDecodeError> {
        let re = f64::read_le(bytes)?;
        let im = f64::read_le(bytes)?;
        Ok(Complex::new(re, im))
    }
}
