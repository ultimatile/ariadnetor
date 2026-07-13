//! Typed errors for MPS serialization. Decode never panics: every malformed
//! input maps to one of these.

use ariadnetor_tensor::{ScalarTag, SectorTag, StorageTag, TensorCodecError};
use thiserror::Error;

/// Failure saving or loading an MPS.
#[derive(Debug, Error)]
pub enum MpsIoError {
    /// Underlying reader / writer error.
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    /// The stream did not begin with the expected magic bytes.
    #[error("bad magic: not an ariadnetor MPS stream")]
    BadMagic,
    /// The file's format version is outside the range this build supports
    /// (newer than the maximum, or below the minimum).
    #[error("unsupported format version {found} (max supported {max})")]
    UnsupportedVersion {
        /// Version found in the file.
        found: u32,
        /// Highest version this build can read.
        max: u32,
    },
    /// The requested scalar type differs from the stored one.
    #[error("scalar type mismatch: file has {found:?}, requested {expected:?}")]
    ScalarTagMismatch {
        /// Scalar type the caller asked to load as.
        expected: ScalarTag,
        /// Scalar type recorded in the file.
        found: ScalarTag,
    },
    /// The requested storage kind differs from the stored one.
    #[error("storage type mismatch: file has {found:?}, requested {expected:?}")]
    StorageTagMismatch {
        /// Storage kind the caller asked to load as.
        expected: StorageTag,
        /// Storage kind recorded in the file.
        found: StorageTag,
    },
    /// The requested sector type differs from the stored one.
    #[error("sector type mismatch: file has {found:?}, requested {expected:?}")]
    SectorTagMismatch {
        /// Sector type the caller asked to load as (`None` for dense).
        expected: Option<SectorTag>,
        /// Sector type recorded in the file.
        found: Option<SectorTag>,
    },
    /// The stream ended before a required section was fully read.
    #[error("unexpected end of MPS stream")]
    UnexpectedEof,
    /// A structural inconsistency was detected while decoding.
    #[error("corrupt MPS stream: {detail}")]
    Corrupt {
        /// Human-readable description of the inconsistency.
        detail: String,
    },
    /// A site's numeric data length did not match its descriptor's extent.
    #[error("extent mismatch: expected {expected} bytes, found {found}")]
    ExtentMismatch {
        /// Bytes the descriptor implies.
        expected: usize,
        /// Bytes actually available.
        found: usize,
    },
}

impl From<TensorCodecError> for MpsIoError {
    fn from(err: TensorCodecError) -> Self {
        match err {
            TensorCodecError::ExtentMismatch { expected, found } => {
                MpsIoError::ExtentMismatch { expected, found }
            }
            other => MpsIoError::Corrupt {
                detail: other.to_string(),
            },
        }
    }
}
