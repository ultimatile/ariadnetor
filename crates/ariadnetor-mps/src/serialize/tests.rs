//! Chain-level contract tests: roundtrip matrix, determinism, and typed
//! error paths. Per-tensor mechanics are covered in `ariadnetor-tensor`.

use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, BlockSparseTensorData, BodyMeta, Complex, DenseLayout,
    DenseStorage, DenseTensorData, Direction, MemoryOrder, QNIndex, QnBlockDto, QnIndexDto,
    ScalarCodec, ScalarTag, SectorTag, Storage, StorageFor, StorageTag, Tensor, TensorLayout,
    U1Sector, Z2Sector,
};

use super::{
    MpsIoError, MpsManifest, OrderTag, SiteMeta, load_mps, load_mps_from_path, save_mps,
    save_mps_to_path,
};
use crate::{CanonicalForm, Mps, TensorChain};

const ORDERS: [MemoryOrder; 2] = [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor];

// --- Value generators ------------------------------------------------------

fn f64_val(n: u32) -> f64 {
    f64::from(n) * 0.5
}
fn f32_val(n: u32) -> f32 {
    n as f32 * 0.25
}
fn c64_val(n: u32) -> Complex<f64> {
    Complex::new(f64::from(n), -f64::from(n))
}
fn c32_val(n: u32) -> Complex<f32> {
    Complex::new(n as f32, 1.0 - n as f32)
}

// --- Chain builders --------------------------------------------------------

fn dense_mps<T: ScalarCodec>(
    order: MemoryOrder,
    n_sites: usize,
    mut val: impl FnMut(u32) -> T,
) -> Mps<DenseStorage<T>, DenseLayout> {
    let mut counter = 0u32;
    let sites = (0..n_sites)
        .map(|_| {
            let data: Vec<T> = (0..8)
                .map(|_| {
                    counter += 1;
                    val(counter)
                })
                .collect();
            Tensor::from_data(DenseTensorData::from_raw_parts(data, vec![2, 2, 2], order))
        })
        .collect();
    Mps::from_sites(sites)
}

fn bsp_site<T: ScalarCodec, S: ariadnetor_tensor::SerializableSector>(
    indices: Vec<QNIndex<S>>,
    flux: S,
    order: MemoryOrder,
    counter: &mut u32,
    mut val: impl FnMut(u32) -> T,
) -> Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>> {
    let td = BlockSparseTensorData::from_block_fn(indices, flux, order, |_, shape| {
        let n: usize = shape.iter().product();
        (0..n)
            .map(|_| {
                *counter += 1;
                val(*counter)
            })
            .collect()
    });
    Tensor::from_data(td)
}

fn u1_indices() -> (Vec<QNIndex<U1Sector>>, U1Sector) {
    (
        vec![
            QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out),
            QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In),
        ],
        U1Sector(0),
    )
}

fn z2_indices() -> (Vec<QNIndex<Z2Sector>>, Z2Sector) {
    (
        vec![
            QNIndex::new(
                vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 2)],
                Direction::Out,
            ),
            QNIndex::new(
                vec![(Z2Sector::new(0), 1), (Z2Sector::new(1), 3)],
                Direction::In,
            ),
        ],
        Z2Sector::new(0),
    )
}

type Tuple = (U1Sector, Z2Sector);
fn tuple_indices() -> (Vec<QNIndex<Tuple>>, Tuple) {
    let leg = || {
        QNIndex::new(
            vec![
                ((U1Sector(0), Z2Sector::new(0)), 2),
                ((U1Sector(1), Z2Sector::new(1)), 1),
            ],
            Direction::Out,
        )
    };
    let leg_in = QNIndex::new(
        vec![
            ((U1Sector(0), Z2Sector::new(0)), 2),
            ((U1Sector(1), Z2Sector::new(1)), 1),
        ],
        Direction::In,
    );
    (vec![leg(), leg_in], (U1Sector(0), Z2Sector::new(0)))
}

fn bsp_mps<T: ScalarCodec, S: ariadnetor_tensor::SerializableSector>(
    make: impl Fn() -> (Vec<QNIndex<S>>, S),
    order: MemoryOrder,
    n_sites: usize,
    val: impl Fn(u32) -> T + Copy,
) -> Mps<BlockSparseStorage<T>, BlockSparseLayout<S>> {
    let mut counter = 0u32;
    let sites = (0..n_sites)
        .map(|_| {
            let (indices, flux) = make();
            bsp_site(indices, flux, order, &mut counter, val)
        })
        .collect();
    Mps::from_sites(sites)
}

// --- Roundtrip harness -----------------------------------------------------

/// save → load → save must be byte-identical, and the reloaded chain must
/// match length and canonical form. Byte equality also proves the numeric
/// data (raw scalar bytes) survived bit-exactly.
fn check_roundtrip<St, L>(mps: Mps<St, L>)
where
    Mps<St, L>: super::MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    let mut buf = Vec::new();
    save_mps(&mps, &mut buf).expect("save");
    let loaded: Mps<St, L> = load_mps(buf.as_slice()).expect("load");
    let mut buf2 = Vec::new();
    save_mps(&loaded, &mut buf2).expect("re-save");
    assert_eq!(buf, buf2, "save->load->save not byte-identical");
    assert_eq!(mps.len(), loaded.len(), "site count changed");
    assert_eq!(
        mps.canonical_form(),
        loaded.canonical_form(),
        "canonical form changed"
    );
}

#[test]
fn dense_roundtrip_matrix() {
    for order in ORDERS {
        for n in [1usize, 3] {
            check_roundtrip(dense_mps(order, n, f64_val));
            check_roundtrip(dense_mps(order, n, f32_val));
            check_roundtrip(dense_mps(order, n, c64_val));
            check_roundtrip(dense_mps(order, n, c32_val));
        }
    }
}

#[test]
fn bsp_roundtrip_matrix() {
    for order in ORDERS {
        for n in [1usize, 3] {
            check_roundtrip(bsp_mps(u1_indices, order, n, f64_val));
            check_roundtrip(bsp_mps(u1_indices, order, n, c64_val));
            check_roundtrip(bsp_mps(z2_indices, order, n, f64_val));
            check_roundtrip(bsp_mps(z2_indices, order, n, c64_val));
            check_roundtrip(bsp_mps(tuple_indices, order, n, f64_val));
            check_roundtrip(bsp_mps(tuple_indices, order, n, c64_val));
        }
    }
}

#[test]
fn canonical_form_variants_roundtrip() {
    let forms = [
        CanonicalForm::Unknown,
        CanonicalForm::Left,
        CanonicalForm::Right,
        CanonicalForm::Partial {
            left_end: 1,
            right_start: 3,
        },
        CanonicalForm::Mixed { center: 2 },
    ];
    for form in forms {
        let mut mps = dense_mps(MemoryOrder::RowMajor, 4, f64_val);
        mps.set_canonical_form(form.clone());
        check_roundtrip(mps);
    }
}

#[test]
fn byte_exact_scalar_edge_cases_roundtrip() {
    // Signed zero, ±inf, distinct NaN payloads survive a full chain roundtrip.
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0xfff8_0000_dead_beef);
    let data = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        nan_a,
        nan_b,
        1.0,
        2.0,
    ];
    let site = Tensor::from_data(DenseTensorData::from_raw_parts(
        data.clone(),
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    ));
    let mps = Mps::from_sites(vec![site]);
    let mut buf = Vec::new();
    save_mps(&mps, &mut buf).expect("save");
    let loaded: Mps<DenseStorage<f64>, DenseLayout> = load_mps(buf.as_slice()).expect("load");
    let got = loaded.site(0).data().data();
    for (expected, actual) in data.iter().zip(got) {
        assert_eq!(expected.to_bits(), actual.to_bits(), "bit pattern changed");
    }
}

// --- Error paths -----------------------------------------------------------

/// Load `bytes` as the given chain type and return the error, without
/// requiring the (Debug-less) `Mps` on the Ok side.
fn load_err<St, L>(bytes: &[u8]) -> MpsIoError
where
    Mps<St, L>: super::MpsCodec,
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    match load_mps::<St, L>(bytes) {
        Ok(_) => panic!("expected a load error"),
        Err(e) => e,
    }
}

fn dense_f64_bytes() -> Vec<u8> {
    let mut buf = Vec::new();
    save_mps(&dense_mps(MemoryOrder::RowMajor, 2, f64_val), &mut buf).expect("save");
    buf
}

/// Frame a hand-built manifest + data section the same way the container does,
/// for crafting streams the public API cannot produce directly.
fn frame(manifest: &MpsManifest, data: &[u8]) -> Vec<u8> {
    let mut mbytes = Vec::new();
    ciborium::into_writer(manifest, &mut mbytes).expect("manifest encode");
    let mut out = Vec::new();
    out.extend_from_slice(b"ARIADMPS");
    out.extend_from_slice(&(mbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&mbytes);
    out.extend_from_slice(data);
    out
}

#[test]
fn bad_magic() {
    let mut bytes = dense_f64_bytes();
    bytes[0] = b'X';
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes);
    assert!(matches!(err, MpsIoError::BadMagic), "got {err:?}");
}

#[test]
fn truncated_stream() {
    let bytes = dense_f64_bytes();
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes[..4]);
    assert!(matches!(err, MpsIoError::UnexpectedEof), "got {err:?}");
}

#[test]
fn trailing_bytes_in_data_section() {
    let mut bytes = dense_f64_bytes();
    bytes.extend_from_slice(&[0u8; 8]); // extra scalar-worth of garbage
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes);
    assert!(matches!(err, MpsIoError::Corrupt { .. }), "got {err:?}");
}

#[test]
fn unsupported_version() {
    let manifest = MpsManifest {
        format_version: 999,
        scalar_type: ScalarTag::F64,
        storage_type: StorageTag::Dense,
        sector_type: None,
        canonical_form: CanonicalForm::Unknown,
        sites: Vec::new(),
    };
    let bytes = frame(&manifest, &[]);
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes);
    assert!(
        matches!(err, MpsIoError::UnsupportedVersion { found: 999, max: 1 }),
        "got {err:?}"
    );
}

#[test]
fn unsupported_version_below_range() {
    // Version 0 is below the supported floor and must be rejected, not decoded
    // as version 1.
    let manifest = MpsManifest {
        format_version: 0,
        scalar_type: ScalarTag::F64,
        storage_type: StorageTag::Dense,
        sector_type: None,
        canonical_form: CanonicalForm::Unknown,
        sites: Vec::new(),
    };
    let bytes = frame(&manifest, &[]);
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes);
    assert!(
        matches!(err, MpsIoError::UnsupportedVersion { found: 0, max: 1 }),
        "got {err:?}"
    );
}

#[test]
fn manifest_length_mismatch() {
    // A declared manifest length longer than the actual CBOR encoding must be
    // rejected, not silently absorbed (CBOR decode ignores trailing bytes).
    let manifest = MpsManifest {
        format_version: 1,
        scalar_type: ScalarTag::F64,
        storage_type: StorageTag::Dense,
        sector_type: None,
        canonical_form: CanonicalForm::Unknown,
        sites: Vec::new(),
    };
    let mut mbytes = Vec::new();
    ciborium::into_writer(&manifest, &mut mbytes).expect("manifest encode");
    let inflated = (mbytes.len() + 4) as u64;
    let mut out = Vec::new();
    out.extend_from_slice(b"ARIADMPS");
    out.extend_from_slice(&inflated.to_le_bytes());
    out.extend_from_slice(&mbytes);
    out.extend_from_slice(&[0u8; 4]); // padding to reach the inflated length
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&out);
    assert!(matches!(err, MpsIoError::Corrupt { .. }), "got {err:?}");
}

#[test]
fn scalar_tag_mismatch() {
    // Saved as f64, loaded as f32.
    let bytes = dense_f64_bytes();
    let err = load_err::<DenseStorage<f32>, DenseLayout>(&bytes);
    assert!(
        matches!(
            err,
            MpsIoError::ScalarTagMismatch {
                expected: ScalarTag::F32,
                found: ScalarTag::F64
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn storage_tag_mismatch() {
    // Saved dense, loaded as block-sparse.
    let bytes = dense_f64_bytes();
    let err = load_err::<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>(&bytes);
    assert!(
        matches!(
            err,
            MpsIoError::StorageTagMismatch {
                expected: StorageTag::BlockSparse,
                found: StorageTag::Dense
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn sector_tag_mismatch_u1_as_z2() {
    // A U(1) file loaded as Z₂: byte-compatible values, distinct tags.
    let mut buf = Vec::new();
    save_mps(
        &bsp_mps(u1_indices, MemoryOrder::RowMajor, 1, f64_val),
        &mut buf,
    )
    .expect("save");
    let err = load_err::<BlockSparseStorage<f64>, BlockSparseLayout<Z2Sector>>(&buf);
    assert!(
        matches!(
            err,
            MpsIoError::SectorTagMismatch {
                expected: Some(SectorTag::Z2),
                found: Some(SectorTag::U1)
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn malformed_body_maps_to_corrupt() {
    // A block-sparse manifest with a zero-dim block: the tensor codec rejects
    // it, and the chain layer surfaces it as Corrupt (never a panic).
    let site = SiteMeta {
        order: OrderTag::RowMajor,
        body: BodyMeta::BlockSparse {
            flux: vec![0], // Z2 flux 0
            indices: vec![QnIndexDto {
                direction: ariadnetor_tensor::DirectionTag::Out,
                blocks: vec![QnBlockDto {
                    sector: vec![0],
                    dim: 0, // illegal
                }],
            }],
        },
        data_len: 0,
    };
    let manifest = MpsManifest {
        format_version: 1,
        scalar_type: ScalarTag::F64,
        storage_type: StorageTag::BlockSparse,
        sector_type: Some(SectorTag::Z2),
        canonical_form: CanonicalForm::Unknown,
        sites: vec![site],
    };
    let bytes = frame(&manifest, &[]);
    let err = load_err::<BlockSparseStorage<f64>, BlockSparseLayout<Z2Sector>>(&bytes);
    assert!(matches!(err, MpsIoError::Corrupt { .. }), "got {err:?}");
}

#[test]
fn path_roundtrip_atomic() {
    let path = std::env::temp_dir().join(format!(
        "ariadnetor_mps_path_roundtrip_{}.mps",
        std::process::id()
    ));
    let mps = dense_mps(MemoryOrder::ColumnMajor, 2, f64_val);
    save_mps_to_path(&mps, &path).expect("save to path");
    let loaded: Mps<DenseStorage<f64>, DenseLayout> =
        load_mps_from_path(&path).expect("load from path");
    assert_eq!(mps.len(), loaded.len());
    let mut a = Vec::new();
    let mut b = Vec::new();
    save_mps(&mps, &mut a).unwrap();
    save_mps(&loaded, &mut b).unwrap();
    assert_eq!(a, b, "path roundtrip changed the stream");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn oversized_site_data_len() {
    // A site claims more numeric bytes than the data section holds.
    let site = SiteMeta {
        order: OrderTag::RowMajor,
        body: BodyMeta::Dense { shape: vec![2] },
        data_len: 999,
    };
    let manifest = MpsManifest {
        format_version: 1,
        scalar_type: ScalarTag::F64,
        storage_type: StorageTag::Dense,
        sector_type: None,
        canonical_form: CanonicalForm::Unknown,
        sites: vec![site],
    };
    let bytes = frame(&manifest, &[0u8; 16]);
    let err = load_err::<DenseStorage<f64>, DenseLayout>(&bytes);
    assert!(matches!(err, MpsIoError::UnexpectedEof), "got {err:?}");
}
