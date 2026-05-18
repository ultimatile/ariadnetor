//! Foundation acceptance tests for the BlockSparse storage / layout
//! split: constructor surface visibility, logical-vs-storage extent
//! distinction, and scalar-only storage operations placement.

use arnet_core::backend::MemoryOrder;

use std::collections::HashMap;

use crate::block_sparse::{
    BlockCoord, BlockMeta, BlockSparseLayout, BlockSparseTensorData, Direction, QNIndex,
};
use crate::sector::U1Sector;
use crate::{Storage, TensorLayout};

/// Build a U(1)-symmetric BlockSparseTensorData with allowed blocks
/// (0,0) and (1,1) on a 5×5 logical shape.
fn sample_u1_rank2_data() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    )
}

#[test]
fn zeros_constructor_exposes_order_via_layout_only() {
    // `BlockSparseLayout`'s `order`, `flux`, `indices`, `blocks`, and
    // `shape` fields are private; the only public path to the memory
    // order is the `layout().order()` accessor. This test pins that
    // path and the constructor's order parameter wiring.
    let td = sample_u1_rank2_data();

    assert_eq!(td.layout().order(), MemoryOrder::RowMajor);

    // Construct again with ColumnMajor to confirm the parameter is
    // honored and surfaces through the accessor.
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let td_cm = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    assert_eq!(td_cm.layout().order(), MemoryOrder::ColumnMajor);
}

#[test]
fn block_data_accessor_joins_storage_and_layout() {
    // Joint storage+layout access: `block_data(coord)` consults the
    // layout's block_index, looks up the meta, and indexes into the
    // storage buffer. Verify it returns the right slice length and
    // the absence of a forbidden coord.
    let mut td = sample_u1_rank2_data();

    // Allowed block (0,0): 2x2 = 4 elements
    let d00 = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d00.len(), 4);
    assert!(d00.iter().all(|&v| v == 0.0));

    // Allowed block (1,1): 3x3 = 9 elements
    let d11 = td.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert_eq!(d11.len(), 9);

    // Forbidden by U(1) symmetry: (0,1) and (1,0)
    assert!(td.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(td.block_data(&BlockCoord(vec![1, 0])).is_none());

    // block_data_mut writes propagate through to block_data reads.
    {
        let slot = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        slot[0] = 7.0;
    }
    assert_eq!(td.block_data(&BlockCoord(vec![0, 0])).unwrap()[0], 7.0);
}

#[test]
fn logical_extent_and_storage_extent_diverge_when_symmetry_forbids_all_blocks() {
    // Construct an indices+flux combination where no block satisfies
    // the conservation law: row and col carry only U1(0)/Out and
    // U1(0)/In, but flux is U1(1). The logical shape is non-trivial
    // (2x3 = 6 cells), but the stored buffer is empty.
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);
    let layout: BlockSparseLayout<U1Sector> =
        BlockSparseLayout::new(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);

    let logical_extent: usize = TensorLayout::shape(&layout).iter().product();
    assert_eq!(logical_extent, 6, "logical extent = product(shape) = 2*3");
    assert_eq!(
        TensorLayout::storage_extent(&layout),
        0,
        "storage_extent = sum of allowed-block sizes = 0 under forbidden flux",
    );
    assert!(logical_extent != TensorLayout::storage_extent(&layout));
}

#[test]
fn norm_lives_on_storage_half_and_matches_legacy_block_sparse() {
    // Scalar-only data ops are placed on `BlockSparseStorage<T>` so
    // they need no layout, sector, or backend. This test pins the
    // placement (`.storage().norm()`) and verifies the numeric value
    // against the legacy `BlockSparse<T, S>::norm` definition
    // (Frobenius norm over all stored elements).
    let mut td = sample_u1_rank2_data();

    // Norm of a freshly-zeroed tensor is exactly zero.
    assert_eq!(td.storage().norm(), 0.0);

    // Fill block (0,0) with [3, 4, 0, 0] (Frobenius contribution 5)
    // and block (1,1) with all zeros (no contribution).
    {
        let slot = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        slot[0] = 3.0;
        slot[1] = 4.0;
    }
    let norm = td.storage().norm();
    assert!(
        (norm - 5.0).abs() < 1e-12,
        "expected Frobenius norm 5.0, got {norm}",
    );

    // `norm_frobenius` is an explicit alias for the same value.
    let nf = td.storage().norm_frobenius();
    assert!((nf - 5.0).abs() < 1e-12);

    // The storage-level `flat_len` (length of the packed buffer) is
    // the sum of allowed-block sizes; here 4 + 9 = 13.
    assert_eq!(td.storage().flat_len(), 13);
    assert_eq!(td.storage().stored_len(), 13);
}

#[test]
fn dagger_flips_directions_duals_flux_and_conjugates_data() {
    use num_complex::Complex64;

    // Build a rank-2 complex tensor with flux U1(1): allowed blocks
    // are coords where Out_i + In_j.dual() = 1, i.e. row charge 1.
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut td: BlockSparseTensorData<Complex64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);

    {
        let slot = td.block_data_mut(&BlockCoord(vec![1, 0])).unwrap();
        slot[0] = Complex64::new(1.0, 2.0);
        slot[1] = Complex64::new(-3.0, 4.0);
    }

    let dag = td.dagger();

    // Flux dualed.
    assert_eq!(*dag.layout().flux(), U1Sector(-1));

    // Directions flipped on every leg.
    let dirs: Vec<Direction> = dag
        .layout()
        .indices()
        .iter()
        .map(|i| i.direction())
        .collect();
    assert_eq!(dirs, vec![Direction::In, Direction::Out]);

    // Data element-wise conjugated.
    let d = dag.block_data(&BlockCoord(vec![1, 0])).unwrap();
    assert_eq!(d[0], Complex64::new(1.0, -2.0));
    assert_eq!(d[1], Complex64::new(-3.0, -4.0));

    // Involution.
    let back = dag.dagger();
    assert_eq!(*back.layout().flux(), U1Sector(1));
    let dirs_back: Vec<Direction> = back
        .layout()
        .indices()
        .iter()
        .map(|i| i.direction())
        .collect();
    assert_eq!(dirs_back, vec![Direction::Out, Direction::In]);
    let db = back.block_data(&BlockCoord(vec![1, 0])).unwrap();
    assert_eq!(db[0], Complex64::new(1.0, 2.0));
    assert_eq!(db[1], Complex64::new(-3.0, 4.0));
}

#[test]
fn conj_preserves_layout_and_conjugates_data() {
    use num_complex::Complex64;

    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut td: BlockSparseTensorData<Complex64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(0), MemoryOrder::RowMajor);

    {
        let slot = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        slot[0] = Complex64::new(1.0, 2.0);
    }

    let c = td.conj();

    // Flux preserved, directions preserved.
    assert_eq!(*c.layout().flux(), U1Sector(0));
    let dirs: Vec<Direction> = c.layout().indices().iter().map(|i| i.direction()).collect();
    assert_eq!(dirs, vec![Direction::Out, Direction::In]);

    // Data element-wise conjugated.
    let d = c.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d[0], Complex64::new(1.0, -2.0));
}

#[test]
fn from_block_fn_populates_each_allowed_block_via_closure() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let td: BlockSparseTensorData<f64, U1Sector> = BlockSparseTensorData::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
        |coord, block_shape| {
            // Tag the block by its coord index sum, scaled by the
            // block-shape product so each block carries a distinct
            // pattern.
            let tag = coord.0.iter().sum::<usize>() as f64;
            let len: usize = block_shape.iter().product();
            (0..len).map(|i| tag + (i as f64) * 0.1).collect()
        },
    );

    // Block (0,0): 2x2 = 4 elements, tag = 0.
    let d00 = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d00.len(), 4);
    for (i, &v) in d00.iter().enumerate() {
        let expected = (i as f64) * 0.1;
        assert!(
            (v - expected).abs() < 1e-12,
            "block (0,0)[{i}] = {v}, expected ~{expected}",
        );
    }

    // Block (1,1): 3x3 = 9 elements, tag = 2.
    let d11 = td.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert_eq!(d11.len(), 9);
    assert!((d11[0] - 2.0).abs() < 1e-12);
    assert!((d11[8] - 2.8).abs() < 1e-12);

    // Forbidden coords remain unstored (closure never called for them).
    assert!(td.block_data(&BlockCoord(vec![0, 1])).is_none());
}

#[test]
fn from_raw_parts_test_only_constructor_round_trips() {
    // Pre-validated raw-parts round-trip: caller supplies block
    // metadata + flat data, the test-only constructor wraps them
    // through `BlockSparseLayout::from_parts` and `TensorData::new`
    // (whose assertion still applies). Pins
    // `BlockSparseTensorData::from_raw_parts` so direct callers
    // observe `layout().order()`, `storage().flat_len()`, and
    // `block_data` behaving the same as via the enumerating
    // constructors.
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);

    let blocks = vec![
        BlockMeta {
            coord: BlockCoord(vec![0, 0]),
            offset: 0,
            size: 4,
        },
        BlockMeta {
            coord: BlockCoord(vec![1, 1]),
            offset: 4,
            size: 9,
        },
    ];
    let mut block_index = HashMap::new();
    block_index.insert(BlockCoord(vec![0, 0]), 0);
    block_index.insert(BlockCoord(vec![1, 1]), 1);

    let data: Vec<f64> = (0..13).map(|i| i as f64).collect();
    let td = BlockSparseTensorData::<f64, U1Sector>::from_raw_parts(
        data,
        blocks,
        block_index,
        vec![row, col],
        U1Sector(0),
        vec![5, 5],
        MemoryOrder::RowMajor,
    );

    assert_eq!(td.layout().shape(), &[5, 5]);
    assert_eq!(td.storage().flat_len(), 13);
    assert_eq!(
        td.block_data(&BlockCoord(vec![0, 0])).unwrap(),
        &[0.0, 1.0, 2.0, 3.0],
    );
    assert_eq!(td.layout().order(), MemoryOrder::RowMajor);
}

#[test]
#[should_panic(expected = "from_block_fn: closure returned")]
fn from_block_fn_panics_on_wrong_block_length() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let _: BlockSparseTensorData<f64, U1Sector> = BlockSparseTensorData::from_block_fn(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
        |_, _| vec![0.0; 99],
    );
}

#[test]
fn norm_sums_squares_across_all_allowed_blocks() {
    // Build a flux-0 U(1) tensor where allowed blocks (0,0) and (1,1) are
    // populated with values whose squared magnitudes sum to a
    // Pythagorean total: block (0,0) holds (3, 0, 0, 0) and block (1,1)
    // holds (4, 0, ..., 0). Then ‖t‖₂ = √(9 + 16) = 5, exact in f64.
    let mut td = sample_u1_rank2_data();
    td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap()[0] = 3.0;
    td.block_data_mut(&BlockCoord(vec![1, 1])).unwrap()[0] = 4.0;

    assert_eq!(td.norm(), 5.0);
}

#[test]
fn is_allowed_block_matches_flux_for_u1_rank2() {
    // Flux 0 over Out(0:2, 1:3) × In(0:2, 1:3): a block (i, j) is
    // allowed iff the directed sectors fuse to 0, i.e. row charge
    // equals column charge. (0, 0) and (1, 1) are allowed; (0, 1)
    // and (1, 0) are forbidden.
    let td = sample_u1_rank2_data();

    assert!(td.is_allowed_block(&BlockCoord(vec![0, 0])));
    assert!(td.is_allowed_block(&BlockCoord(vec![1, 1])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![0, 1])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![1, 0])));
}
