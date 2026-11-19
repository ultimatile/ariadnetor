//! Single-stream container framing and the public save / load entry points.
//!
//! Layout: `[8-byte magic] [u64 LE manifest length] [CBOR manifest]
//! [numeric data section]`. The magic and version give a clean rejection of
//! foreign or unsupported-version streams before any tensor is touched.

use std::fs;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;
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
/// Lowest format version this build can read.
const MIN_VERSION: u32 = 1;
/// Highest format version this build can read.
const MAX_VERSION: u32 = 1;
/// Bounded attempts to claim an unused temp-file name in [`save_mps_to_path`].
const MAX_TEMP_ATTEMPTS: u32 = 16;

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

    // Reject both future versions and any below the supported floor: with a
    // single defined version and no migration logic, an out-of-range version
    // would otherwise be decoded against the wrong schema.
    if manifest.format_version < MIN_VERSION || manifest.format_version > MAX_VERSION {
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
    let file_name = path.file_name().ok_or_else(|| {
        MpsIoError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "destination path has no file name",
        ))
    })?;

    // Claim a fresh temp file with `create_new` (O_EXCL): it refuses to open an
    // existing path — including a pre-planted symlink — so an attacker who can
    // write the destination directory cannot redirect the write through a
    // guessed temp name onto a victim file. Retry a bounded number of times if
    // the name is already taken.
    let (tmp, file) = {
        let mut attempt = 0;
        loop {
            let candidate = path.with_file_name(format!(
                "{}.tmp.{}.{}",
                file_name.to_string_lossy(),
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => break (candidate, file),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    attempt += 1;
                    if attempt >= MAX_TEMP_ATTEMPTS {
                        return Err(MpsIoError::Io(e));
                    }
                }
                Err(e) => return Err(MpsIoError::Io(e)),
            }
        }
    };

    let write = || -> Result<(), MpsIoError> {
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

/// Temp-name retry tests. They live beside the private counter because
/// provoking a collision means planting the names the next saves will try.
#[cfg(test)]
pub(super) mod tests {
    use std::fs;
    use std::io;
    use std::ops::Range;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensorData, MemoryOrder, Tensor};

    use super::{
        MAX_TEMP_ATTEMPTS, MpsIoError, TEMP_COUNTER, load_mps_from_path, save_mps_to_path,
    };
    use crate::{Mps, TensorChain};

    /// Held by every unit test that saves to a path, so no save consumes
    /// counter values a retry test has planted names for.
    static COUNTER_LOCK: Mutex<()> = Mutex::new(());

    /// Take [`COUNTER_LOCK`] for the rest of the calling test.
    pub(in crate::serialize) fn lock_temp_counter() -> MutexGuard<'static, ()> {
        COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const FILE_NAME: &str = "state.mps";

    fn tiny_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
        Mps::from_sites(vec![Tensor::from_data(DenseTensorData::from_raw_parts(
            vec![0.6, 0.8],
            vec![1, 2, 1],
            MemoryOrder::ColumnMajor,
        ))])
    }

    /// A fresh, empty directory unique to this process and `label`.
    fn scratch_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ariadnetor_mps_{label}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// Occupy the temp names `save_mps_to_path` derives from counter values
    /// in `counters`.
    fn plant_temp_names(dir: &Path, counters: Range<u64>) {
        for k in counters {
            let name = format!("{FILE_NAME}.tmp.{}.{k}", std::process::id());
            fs::write(dir.join(name), b"").expect("plant temp name");
        }
    }

    #[test]
    fn save_to_path_retries_past_taken_temp_names() {
        let _guard = lock_temp_counter();
        let dir = scratch_dir("temp_name_collision");
        let next = TEMP_COUNTER.load(Ordering::Relaxed);
        plant_temp_names(&dir, next..next + 4);

        let path = dir.join(FILE_NAME);
        save_mps_to_path(&tiny_mps(), &path).expect("a free temp name follows the taken ones");

        let loaded: Mps<DenseStorage<f64>, DenseLayout> =
            load_mps_from_path(&path).expect("load the saved state");
        assert_eq!(loaded.site(0).data_slice(), tiny_mps().site(0).data_slice());
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn save_to_path_gives_up_after_bounded_temp_name_collisions() {
        let _guard = lock_temp_counter();
        let dir = scratch_dir("temp_name_exhaustion");
        let next = TEMP_COUNTER.load(Ordering::Relaxed);
        plant_temp_names(&dir, next..next + u64::from(MAX_TEMP_ATTEMPTS) + 32);

        let path = dir.join(FILE_NAME);
        let err = save_mps_to_path(&tiny_mps(), &path)
            .expect_err("every temp name in the destination directory is taken");
        assert!(
            matches!(&err, MpsIoError::Io(e) if e.kind() == io::ErrorKind::AlreadyExists),
            "got {err:?}"
        );
        assert!(
            !path.exists(),
            "a failed save must not create the destination"
        );
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }

    #[test]
    fn save_to_path_does_not_retry_other_open_errors() {
        let _guard = lock_temp_counter();
        let dir = scratch_dir("temp_name_missing_dir");
        let path = dir.join("missing").join(FILE_NAME);

        let before = TEMP_COUNTER.load(Ordering::Relaxed);
        let err = save_mps_to_path(&tiny_mps(), &path)
            .expect_err("the destination directory does not exist");
        let consumed = TEMP_COUNTER.load(Ordering::Relaxed) - before;

        assert!(
            matches!(&err, MpsIoError::Io(e) if e.kind() == io::ErrorKind::NotFound),
            "got {err:?}"
        );
        // The retry budget is for taken names only; a missing directory
        // fails on the first attempt. Every path save in this binary holds
        // the counter lock, so no other save advances the counter meanwhile.
        assert_eq!(consumed, 1, "temp names tried for a non-collision error");
        fs::remove_dir_all(&dir).expect("remove scratch dir");
    }
}
