use ariadnetor_core::backend::{ComputeBackend, MemoryOrder};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::{legs, square_legs};
use ariadnetor_tensor::{BlockCoord, BlockSparseTensorData, Direction, U1Sector, Z2Sector};

use super::permute_block_sparse_dense;

fn backend() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    backend().preferred_order()
}

// ---------------------------------------------------------------------------
// Helper: build a rank-3 U1 tensor with known data
// ---------------------------------------------------------------------------

/// Rank-3 U1, flux=0, indices: Out(0:2, 1:3), Out(0:2, 1:1), In(0:2, 1:3).
/// Blocks: (0,0,0) 2×2×2, (0,1,1) 2×1×3, (1,0,1) 3×2×3, (1,1,0) 3×1×2.
fn sample_u1_rank3() -> BlockSparseTensorData<f64, U1Sector> {
    let mut bs = BlockSparseTensorData::zeros(
        legs([
            (vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out),
            (vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In),
        ]),
        U1Sector(0),
        order(),
    );

    // Fill blocks with distinct values for traceability
    let mut val = 1.0;
    for meta in bs.block_metas().to_vec() {
        let data = bs.block_data_mut(&meta.coord).unwrap();
        for elem in data.iter_mut() {
            *elem = val;
            val += 1.0;
        }
    }
    bs
}

// ---------------------------------------------------------------------------
// Identity permutation
// ---------------------------------------------------------------------------

#[test]
fn identity_permutation_is_clone() {
    let bs = sample_u1_rank3();
    let result = permute_block_sparse_dense(&backend(), &bs, &[0, 1, 2]).unwrap();
    assert_eq!(result.shape(), bs.shape());
    for meta in bs.block_metas() {
        let orig = bs.block_data(&meta.coord).unwrap();
        let perm = result.block_data(&meta.coord).unwrap();
        assert_eq!(orig, perm);
    }
}

// ---------------------------------------------------------------------------
// Simple transposition (rank-2)
// ---------------------------------------------------------------------------

#[test]
fn transpose_rank2() {
    let mut bs = BlockSparseTensorData::<f64, U1Sector>::zeros(
        square_legs(vec![(U1Sector(0), 2), (U1Sector(1), 3)]),
        U1Sector(0),
        order(),
    );

    // Block (0,0): 2×2 matrix [[1,2],[3,4]]
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    // Block (1,1): 3×3 matrix [[5,6,7],[8,9,10],[11,12,13]]
    let d = bs.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d.copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

    let result = permute_block_sparse_dense(&backend(), &bs, &[1, 0]).unwrap();

    // Shape should be transposed
    assert_eq!(result.shape(), &[5, 5]); // same total dims, swapped
    assert_eq!(result.indices()[0].direction(), Direction::In);
    assert_eq!(result.indices()[1].direction(), Direction::Out);

    // Block (0,0) in result = transpose of original (0,0): [[1,3],[2,4]]
    // Verify via double transpose (roundtrip), since the exact layout
    // depends on the backend's preferred memory order.
    let double = permute_block_sparse_dense(&backend(), &result, &[1, 0]).unwrap();
    for meta in bs.block_metas() {
        let orig = bs.block_data(&meta.coord).unwrap();
        let roundtrip = double.block_data(&meta.coord).unwrap();
        assert_eq!(orig, roundtrip, "double transpose should be identity");
    }
}

// ---------------------------------------------------------------------------
// Rank-3 cyclic permutation
// ---------------------------------------------------------------------------

#[test]
fn cyclic_permutation_rank3() {
    let bs = sample_u1_rank3();
    // perm = [1, 2, 0]: new[0]=old[1], new[1]=old[2], new[2]=old[0]
    let p1 = permute_block_sparse_dense(&backend(), &bs, &[1, 2, 0]).unwrap();
    let p2 = permute_block_sparse_dense(&backend(), &p1, &[1, 2, 0]).unwrap();
    let p3 = permute_block_sparse_dense(&backend(), &p2, &[1, 2, 0]).unwrap();

    // Three cyclic permutations should return to original
    assert_eq!(p3.shape(), bs.shape());
    for meta in bs.block_metas() {
        let orig = bs.block_data(&meta.coord).unwrap();
        let roundtrip = p3.block_data(&meta.coord).unwrap();
        for (i, (&a, &b)) in orig.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-14,
                "block {:?}[{i}]: {a} vs {b}",
                meta.coord
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Flux preservation
// ---------------------------------------------------------------------------

#[test]
fn permute_preserves_flux() {
    let bs = sample_u1_rank3();
    let result = permute_block_sparse_dense(&backend(), &bs, &[2, 0, 1]).unwrap();
    assert_eq!(result.flux(), bs.flux());
    // All blocks should satisfy flux conservation
    for meta in result.block_metas() {
        assert!(
            result.is_allowed_block(&meta.coord),
            "block {:?} violates flux",
            meta.coord
        );
    }
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn perm_wrong_length() {
    let bs = sample_u1_rank3();
    let result = permute_block_sparse_dense(&backend(), &bs, &[0, 1]);
    assert!(result.is_err());
    assert!(format!("{}", result.err().unwrap()).contains("perm length"));
}

#[test]
fn perm_out_of_range() {
    let bs = sample_u1_rank3();
    let result = permute_block_sparse_dense(&backend(), &bs, &[0, 1, 5]);
    assert!(result.is_err());
    assert!(format!("{}", result.err().unwrap()).contains("out of range"));
}

#[test]
fn perm_duplicate() {
    let bs = sample_u1_rank3();
    let result = permute_block_sparse_dense(&backend(), &bs, &[0, 1, 1]);
    assert!(result.is_err());
    assert!(format!("{}", result.err().unwrap()).contains("duplicate"));
}

// ---------------------------------------------------------------------------
// Z2 sector
// ---------------------------------------------------------------------------

#[test]
fn permute_z2_rank2() {
    let mut bs = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        square_legs(vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)]),
        Z2Sector::new(0),
        order(),
    );
    let d = bs.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    for (i, v) in d.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }

    let transposed = permute_block_sparse_dense(&backend(), &bs, &[1, 0]).unwrap();
    let double = permute_block_sparse_dense(&backend(), &transposed, &[1, 0]).unwrap();

    for meta in bs.block_metas() {
        let orig = bs.block_data(&meta.coord).unwrap();
        let roundtrip = double.block_data(&meta.coord).unwrap();
        assert_eq!(orig, roundtrip);
    }
}
