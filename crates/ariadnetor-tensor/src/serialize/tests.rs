//! Tensor-level codec tests: encode/decode roundtrips and malformed-input
//! rejection. Chain-level (MPS) coverage lives in `ariadnetor-mps`.

use super::*;
use crate::{
    BlockLayoutError, BlockSparseTensorData, Complex, DenseTensorData, Direction, MemoryOrder,
    QNIndex, U1Sector, Z2Sector,
};

const ORDERS: [MemoryOrder; 2] = [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor];

/// Roundtrip a dense tensor and assert determinism + metadata/body/order/shape
/// preservation. Body comparison is bit-exact, so NaN payloads and signed
/// zeros are covered.
fn check_dense_roundtrip<T: ScalarCodec>(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) {
    let original = DenseTensorData::from_raw_parts(data, shape, order);
    let (meta, body) = encode_dense(&original);
    let decoded =
        decode_dense::<T>(original.shape(), order, &body).expect("dense roundtrip decode");
    let (meta2, body2) = encode_dense(&decoded);
    assert_eq!(meta, meta2, "dense metadata not preserved");
    assert_eq!(body, body2, "dense numeric body not preserved");
    assert_eq!(
        original.order(),
        decoded.order(),
        "dense order not preserved"
    );
    assert_eq!(
        original.shape(),
        decoded.shape(),
        "dense shape not preserved"
    );
}

fn expect_bsp(meta: &BodyMeta) -> (&[u8], &[QnIndexDto]) {
    match meta {
        BodyMeta::BlockSparse { flux, indices } => (flux, indices),
        BodyMeta::Dense { .. } => panic!("expected a block-sparse descriptor"),
    }
}

/// Roundtrip a block-sparse tensor and assert determinism + preservation.
fn check_bsp_roundtrip<T: ScalarCodec, S: SerializableSector>(
    original: BlockSparseTensorData<T, S>,
) {
    let order = original.layout().order();
    let (meta, body) = encode_block_sparse(&original);
    let (flux_bytes, idx_dtos) = expect_bsp(&meta);
    let decoded = decode_block_sparse::<T, S>(flux_bytes, idx_dtos, order, &body)
        .expect("block-sparse roundtrip decode");
    let (meta2, body2) = encode_block_sparse(&decoded);
    assert_eq!(meta, meta2, "block-sparse metadata not preserved");
    assert_eq!(body, body2, "block-sparse numeric body not preserved");
    assert_eq!(
        order,
        decoded.layout().order(),
        "block-sparse order not preserved"
    );
}

// --- Dense roundtrips ------------------------------------------------------

#[test]
fn dense_roundtrip_scalar_types() {
    for order in ORDERS {
        check_dense_roundtrip::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], order);
        check_dense_roundtrip::<f32>(vec![1.5, -2.5, 3.5, 0.0], vec![2, 2], order);
        check_dense_roundtrip::<Complex<f64>>(
            vec![Complex::new(1.0, -1.0), Complex::new(2.0, 3.0)],
            vec![2],
            order,
        );
        check_dense_roundtrip::<Complex<f32>>(
            vec![Complex::new(0.5, 0.5), Complex::new(-0.5, 1.5)],
            vec![1, 2],
            order,
        );
    }
}

#[test]
fn dense_roundtrip_byte_exact_edge_cases() {
    // Signed zero, ±inf, and distinct NaN payloads must survive bit-exactly.
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0x7ff8_0000_dead_beef);
    let data = vec![
        0.0_f64,
        -0.0_f64,
        f64::INFINITY,
        f64::NEG_INFINITY,
        nan_a,
        nan_b,
    ];
    let original = DenseTensorData::from_raw_parts(data.clone(), vec![6], MemoryOrder::RowMajor);
    let (_, body) = encode_dense(&original);
    let decoded =
        decode_dense::<f64>(&[6], MemoryOrder::RowMajor, &body).expect("edge-case decode");
    for (expected, got) in data.iter().zip(decoded.data()) {
        assert_eq!(
            expected.to_bits(),
            got.to_bits(),
            "bit pattern not preserved"
        );
    }
}

// --- Block-sparse roundtrips ----------------------------------------------

/// Fill every flux-allowed block with sequential values from `value`, so a
/// roundtrip has non-trivial data to preserve.
fn filled_bsp<T: ScalarCodec, S: SerializableSector>(
    indices: Vec<QNIndex<S>>,
    flux: S,
    order: MemoryOrder,
    mut value: impl FnMut(u32) -> T,
) -> BlockSparseTensorData<T, S> {
    let mut counter = 0u32;
    BlockSparseTensorData::from_block_fn(indices, flux, order, |_, shape| {
        let n: usize = shape.iter().product();
        (0..n)
            .map(|_| {
                counter += 1;
                value(counter)
            })
            .collect()
    })
}

/// A rank-2 U(1) tensor with an Out and an In leg; flux 0 selects the
/// charge-matched blocks.
fn u1_matrix<T: ScalarCodec>(
    order: MemoryOrder,
    value: impl FnMut(u32) -> T,
) -> BlockSparseTensorData<T, U1Sector> {
    let indices = vec![
        QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out),
        QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In),
    ];
    filled_bsp(indices, U1Sector(0), order, value)
}

fn z2_matrix<T: ScalarCodec>(
    order: MemoryOrder,
    value: impl FnMut(u32) -> T,
) -> BlockSparseTensorData<T, Z2Sector> {
    let indices = vec![
        QNIndex::new(
            vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 2)],
            Direction::Out,
        ),
        QNIndex::new(
            vec![(Z2Sector::new(0), 1), (Z2Sector::new(1), 3)],
            Direction::In,
        ),
    ];
    filled_bsp(indices, Z2Sector::new(0), order, value)
}

type TuplePair = (U1Sector, Z2Sector);

fn tuple_matrix<T: ScalarCodec>(
    order: MemoryOrder,
    value: impl FnMut(u32) -> T,
) -> BlockSparseTensorData<T, TuplePair> {
    let indices = vec![
        QNIndex::new(
            vec![
                ((U1Sector(0), Z2Sector::new(0)), 2),
                ((U1Sector(1), Z2Sector::new(1)), 1),
            ],
            Direction::Out,
        ),
        QNIndex::new(
            vec![
                ((U1Sector(0), Z2Sector::new(0)), 2),
                ((U1Sector(1), Z2Sector::new(1)), 1),
            ],
            Direction::In,
        ),
    ];
    filled_bsp(indices, (U1Sector(0), Z2Sector::new(0)), order, value)
}

fn f64_val(n: u32) -> f64 {
    f64::from(n) * 0.25
}

fn c64_val(n: u32) -> Complex<f64> {
    Complex::new(f64::from(n), -f64::from(n))
}

#[test]
fn bsp_roundtrip_matrix() {
    for order in ORDERS {
        check_bsp_roundtrip(u1_matrix(order, f64_val));
        check_bsp_roundtrip(u1_matrix(order, c64_val));
        check_bsp_roundtrip(z2_matrix(order, f64_val));
        check_bsp_roundtrip(z2_matrix(order, c64_val));
        check_bsp_roundtrip(tuple_matrix(order, f64_val));
        check_bsp_roundtrip(tuple_matrix(order, c64_val));
    }
}

// --- Error paths (typed, never panic) -------------------------------------

#[test]
fn dense_decode_extent_mismatch() {
    // shape 2x2 f64 needs 32 bytes; supply 8.
    let err = decode_dense::<f64>(&[2, 2], MemoryOrder::RowMajor, &[0u8; 8])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::ExtentMismatch {
            expected: 32,
            found: 8
        }
    );
}

#[test]
fn dense_decode_extent_overflow() {
    let err = decode_dense::<f64>(&[usize::MAX, 2], MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(err, TensorCodecError::Overflow("dense extent"));
}

/// Build a single-leg block-sparse descriptor for error-path probing.
fn one_leg(direction: DirectionTag, blocks: Vec<QnBlockDto>) -> Vec<QnIndexDto> {
    vec![QnIndexDto { direction, blocks }]
}

#[test]
fn bsp_decode_malformed_z2_value() {
    let indices = one_leg(
        DirectionTag::Out,
        vec![QnBlockDto {
            sector: vec![2], // out of {0, 1}
            dim: 1,
        }],
    );
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::Sector(SectorDecodeError::InvalidZ2(2))
    );
}

#[test]
fn bsp_decode_zero_dim_block() {
    let indices = one_leg(
        DirectionTag::Out,
        vec![QnBlockDto {
            sector: vec![0],
            dim: 0,
        }],
    );
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::MalformedIndex("zero block dimension")
    );
}

#[test]
fn bsp_decode_duplicate_sectors() {
    let indices = one_leg(
        DirectionTag::Out,
        vec![
            QnBlockDto {
                sector: vec![0],
                dim: 1,
            },
            QnBlockDto {
                sector: vec![0],
                dim: 1,
            },
        ],
    );
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::MalformedIndex("sectors must be unique and ascending")
    );
}

#[test]
fn bsp_decode_trailing_sector_bytes() {
    // flux payload for Z2 is one byte; a second byte is trailing garbage.
    let indices = one_leg(
        DirectionTag::Out,
        vec![QnBlockDto {
            sector: vec![0],
            dim: 1,
        }],
    );
    let err = decode_block_sparse::<f64, Z2Sector>(&[0, 99], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(err, TensorCodecError::TrailingSectorBytes);
}

#[test]
fn bsp_decode_fusion_overflow() {
    // Two Out legs each carrying U1 charge i32::MAX; fusing them overflows.
    let sector = i32::MAX.to_le_bytes().to_vec();
    let leg = QnIndexDto {
        direction: DirectionTag::Out,
        blocks: vec![QnBlockDto {
            sector: sector.clone(),
            dim: 1,
        }],
    };
    let indices = vec![leg.clone(), leg];
    // flux is a valid U1 value (0); the overflow happens during enumeration.
    let err = decode_block_sparse::<f64, U1Sector>(
        &0_i32.to_le_bytes(),
        &indices,
        MemoryOrder::RowMajor,
        &[],
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        TensorCodecError::Layout(BlockLayoutError::FusionOverflow)
    );
}

#[test]
fn bsp_decode_extent_overflow() {
    // A single leg whose block dimensions sum past usize::MAX overflows the
    // logical shape (computed before block enumeration).
    let indices = one_leg(
        DirectionTag::Out,
        vec![
            QnBlockDto {
                sector: vec![0],
                dim: usize::MAX as u64,
            },
            QnBlockDto {
                sector: vec![1],
                dim: 2,
            },
        ],
    );
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::Layout(BlockLayoutError::ExtentOverflow)
    );
}

#[test]
fn bsp_decode_size_overflow() {
    // A flux-allowed block whose element count (usize::MAX ^ 2) overflows.
    let big = (usize::MAX as u64).to_le_bytes(); // dim = usize::MAX
    let leg = QnIndexDto {
        direction: DirectionTag::Out,
        blocks: vec![QnBlockDto {
            sector: vec![0],
            dim: u64::from_le_bytes(big),
        }],
    };
    let indices = vec![leg.clone(), leg];
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &indices, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::Layout(BlockLayoutError::SizeOverflow)
    );
}

#[test]
fn bsp_decode_truncated_body() {
    // Valid single 1x1 Z2 block (flux 0) expects one f64 (8 bytes); supply none.
    let indices = one_leg(
        DirectionTag::Out,
        vec![QnBlockDto {
            sector: vec![0],
            dim: 1,
        }],
    );
    // Second leg In so block (0,0) fuses to flux 0.
    let leg_in = QnIndexDto {
        direction: DirectionTag::In,
        blocks: vec![QnBlockDto {
            sector: vec![0],
            dim: 1,
        }],
    };
    let mut idx = indices;
    idx.push(leg_in);
    let err = decode_block_sparse::<f64, Z2Sector>(&[0], &idx, MemoryOrder::RowMajor, &[])
        .err()
        .unwrap();
    assert_eq!(
        err,
        TensorCodecError::ExtentMismatch {
            expected: 8,
            found: 0
        }
    );
}
