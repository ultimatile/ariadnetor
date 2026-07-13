//! Single-stream container framing and the public save / load entry points.
//!
//! Layout: `[8-byte magic] [u64 LE manifest length] [CBOR manifest]
//! [numeric data section]`. The magic and version give a clean rejection of
//! foreign or future-version streams before any tensor is touched.

use std::fs;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ariadnetor_tensor::{ScalarCodec, Storage, StorageFor, TensorLayout};

use super::codec::MpsCodec;
use super::error::MpsIoError;
use super::manifest::MpsManifest;
use crate::{Mps, TensorChain};

/// Stream magic identifying an ariadnetor MPS container.
const MAGIC: &[u8; 8] = b"ARIADMPS";
/// Per-process counter disambiguating concurrent temp-file names in
/// [`save_mps_to_path`], so two in-flight saves to one destination never share
/// a temp path.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Format version this build writes.
const FORMAT_VERSION: u32 = 1;
/// Highest format version this build can read.
const MAX_VERSION: u32 = 1;

/// Map a short-read `io::Error` to the typed EOF variant.
fn map_eof(err: io::Error) -> MpsIoError {
    if err.kind() == io::ErrorKind::UnexpectedEof {
        MpsIoError::UnexpectedEof
    } else {
        MpsIoError::Io(err)
    }
}

/// Serialize an MPS to a single stream, losslessly and deterministically.
///
/// The state captures everything a warm continuation needs, so a restart is
/// just *load, then feed back in* — this saves the state, not a run-replay
/// envelope (energy baseline, sweep budget, and environments are recomputed or
/// reset on re-entry; they are not persisted). The stream is
/// `[magic] [manifest length] [CBOR manifest] [numeric data]`; identical input
/// always produces identical bytes.
pub fn save_mps<St, L>(mps: &Mps<St, L>, mut writer: impl Write) -> Result<(), MpsIoError>
where
    Mps<St, L>: MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    let (sites, data) = mps.encode_sites();
    let manifest = MpsManifest {
        format_version: FORMAT_VERSION,
        scalar_type: <<Mps<St, L> as MpsCodec>::Scalar as ScalarCodec>::TAG,
        storage_type: <Mps<St, L> as MpsCodec>::STORAGE_TAG,
        sector_type: <Mps<St, L> as MpsCodec>::sector_tag(),
        canonical_form: mps.canonical_form().clone(),
        sites,
    };

    let mut manifest_bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut manifest_bytes).map_err(|e| MpsIoError::Corrupt {
        detail: format!("manifest serialization failed: {e}"),
    })?;

    writer.write_all(MAGIC)?;
    writer.write_all(&(manifest_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&manifest_bytes)?;
    writer.write_all(&data)?;
    Ok(())
}

/// Deserialize an MPS from a stream written by [`save_mps`].
///
/// The manifest's scalar / storage / sector type-identity tags are checked
/// against the requested `T` / `St` / `S` before any value is decoded, so
/// loading a file as the wrong type fails cleanly (e.g. a U(1) file loaded as
/// Z₂) instead of silently misreading. Malformed input never panics — it maps
/// to a typed [`MpsIoError`].
pub fn load_mps<St, L>(mut reader: impl Read) -> Result<Mps<St, L>, MpsIoError>
where
    Mps<St, L>: MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(map_eof)?;
    if &magic != MAGIC {
        return Err(MpsIoError::BadMagic);
    }

    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes).map_err(map_eof)?;
    let manifest_len =
        usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| MpsIoError::Corrupt {
            detail: "manifest length exceeds usize".to_string(),
        })?;

    // `take` bounds the read so a crafted length cannot force a giant
    // allocation before the (possibly truncated) bytes actually arrive.
    let mut manifest_bytes = Vec::new();
    let read = (&mut reader)
        .take(manifest_len as u64)
        .read_to_end(&mut manifest_bytes)?;
    if read != manifest_len {
        return Err(MpsIoError::UnexpectedEof);
    }

    // Decode through a cursor so the declared manifest length can be
    // cross-checked against the bytes CBOR actually consumed — `from_reader`
    // silently ignores trailing bytes, which would otherwise let an inflated
    // manifest length swallow data-section bytes undetected.
    let mut cursor = Cursor::new(manifest_bytes.as_slice());
    let manifest: MpsManifest =
        ciborium::from_reader(&mut cursor).map_err(|e| MpsIoError::Corrupt {
            detail: format!("manifest decode failed: {e}"),
        })?;
    if cursor.position() != manifest_len as u64 {
        return Err(MpsIoError::Corrupt {
            detail: "manifest length does not match its encoded size".to_string(),
        });
    }

    if manifest.format_version > MAX_VERSION {
        return Err(MpsIoError::UnsupportedVersion {
            found: manifest.format_version,
            max: MAX_VERSION,
        });
    }

    let expected_scalar = <<Mps<St, L> as MpsCodec>::Scalar as ScalarCodec>::TAG;
    if manifest.scalar_type != expected_scalar {
        return Err(MpsIoError::ScalarTagMismatch {
            expected: expected_scalar,
            found: manifest.scalar_type,
        });
    }
    let expected_storage = <Mps<St, L> as MpsCodec>::STORAGE_TAG;
    if manifest.storage_type != expected_storage {
        return Err(MpsIoError::StorageTagMismatch {
            expected: expected_storage,
            found: manifest.storage_type,
        });
    }
    let expected_sector = <Mps<St, L> as MpsCodec>::sector_tag();
    if manifest.sector_type != expected_sector {
        return Err(MpsIoError::SectorTagMismatch {
            expected: expected_sector,
            found: manifest.sector_type,
        });
    }

    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    <Mps<St, L> as MpsCodec>::decode_sites(&manifest.sites, &data, manifest.canonical_form)
}

/// Serialize an MPS to `path` with atomic visibility.
///
/// Writes a temp file in the destination directory (same filesystem, so the
/// `rename` is atomic), flushes and `fsync`s it, then atomically renames into
/// place. A concurrent reader sees either the old file or the complete new
/// one, never a partial write. Full crash durability (parent-directory
/// `fsync`) is an optional hardening beyond this base guarantee.
pub fn save_mps_to_path<St, L>(mps: &Mps<St, L>, path: impl AsRef<Path>) -> Result<(), MpsIoError>
where
    Mps<St, L>: MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    let path = path.as_ref();
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let file_name = path.file_name().ok_or_else(|| {
        MpsIoError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "destination path has no file name",
        ))
    })?;
    let tmp = dir.join(format!(
        "{}.tmp.{}.{}",
        file_name.to_string_lossy(),
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));

    let write = || -> Result<(), MpsIoError> {
        let file = fs::File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        save_mps(mps, &mut writer)?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|e| MpsIoError::Io(e.into_error()))?;
        file.sync_all()?;
        Ok(())
    };

    if let Err(err) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(MpsIoError::Io(err));
    }
    Ok(())
}

/// Load an MPS previously written by [`save_mps_to_path`] (or [`save_mps`]).
pub fn load_mps_from_path<St, L>(path: impl AsRef<Path>) -> Result<Mps<St, L>, MpsIoError>
where
    Mps<St, L>: MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    let file = fs::File::open(path.as_ref())?;
    load_mps(BufReader::new(file))
}
