use super::*;
use arnet_native::NativeBackend;
use arnet_tensor::block_sparse::{BlockCoord, BlockSparse, Direction, QNIndex};
use arnet_tensor::sector::{U1Sector, Z2Sector};

fn b() -> NativeBackend {
    NativeBackend
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

#[test]
fn error_on_length_mismatch() {
    let idx = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let a = BlockSparse::<f64, U1Sector>::zeros(vec![idx.clone(), idx.clone()], U1Sector(0));
    assert!(contract_block_sparse(&b(), &a, &a, &[0], &[0, 1]).is_err());
}

#[test]
fn error_on_out_of_range() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    assert!(contract_block_sparse(&b(), &a, &a, &[5], &[0]).is_err());
}

#[test]
fn error_on_duplicate_axis() {
    let out = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let in_ = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a =
        BlockSparse::<f64, U1Sector>::zeros(vec![out.clone(), out, in_.clone(), in_], U1Sector(0));
    assert!(contract_block_sparse(&b(), &a, &a, &[0, 0], &[2, 3]).is_err());
}

#[test]
fn error_on_same_direction() {
    let out = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let in_ = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a = BlockSparse::<f64, U1Sector>::zeros(vec![out.clone(), in_.clone()], U1Sector(0));
    let c = BlockSparse::<f64, U1Sector>::zeros(vec![out, in_], U1Sector(0));
    // axis 0 (Out) vs axis 0 (Out) → same direction
    assert!(contract_block_sparse(&b(), &a, &c, &[0], &[0]).is_err());
}

#[test]
fn error_on_sector_mismatch() {
    let a_col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let b_row = QNIndex::new(vec![(U1Sector(1), 2)], Direction::Out);
    let a = BlockSparse::<f64, U1Sector>::zeros(
        vec![QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out), a_col],
        U1Sector(0),
    );
    let c = BlockSparse::<f64, U1Sector>::zeros(
        vec![b_row, QNIndex::new(vec![(U1Sector(1), 2)], Direction::In)],
        U1Sector(0),
    );
    assert!(contract_block_sparse(&b(), &a, &c, &[1], &[0]).is_err());
}

#[test]
fn error_on_dim_mismatch() {
    let a_col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let b_row = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let a = BlockSparse::<f64, U1Sector>::zeros(
        vec![QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out), a_col],
        U1Sector(0),
    );
    let c = BlockSparse::<f64, U1Sector>::zeros(
        vec![b_row, QNIndex::new(vec![(U1Sector(0), 3)], Direction::In)],
        U1Sector(0),
    );
    assert!(contract_block_sparse(&b(), &a, &c, &[1], &[0]).is_err());
}

// ---------------------------------------------------------------------------
// Partial contraction
// ---------------------------------------------------------------------------

#[test]
fn rank2_single_block_matmul() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![row.clone(), col.clone()], U1Sector(0));
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            let d = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
            assert!((d[0] - 19.0).abs() < 1e-10);
            assert!((d[1] - 22.0).abs() < 1e-10);
            assert!((d[2] - 43.0).abs() < 1e-10);
            assert!((d[3] - 50.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn rank2_multi_block_matmul() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![row.clone(), col.clone()], U1Sector(0));
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);

    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[6.0, 7.0, 8.0, 9.0]);
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[10.0]);

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            // Block (0,0): [[1,2],[3,4]]×[[6,7],[8,9]] = [[22,25],[50,57]]
            let d00 = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            assert!((d00[0] - 22.0).abs() < 1e-10);
            assert!((d00[1] - 25.0).abs() < 1e-10);
            assert!((d00[2] - 50.0).abs() < 1e-10);
            assert!((d00[3] - 57.0).abs() < 1e-10);
            // Block (1,1): [5]×[10] = [50]
            let d11 = out.block_data(&BlockCoord(vec![1, 1])).unwrap();
            assert!((d11[0] - 50.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn rank2_nonzero_flux() {
    let a_row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let shared = QNIndex::new(vec![(U1Sector(0), 4)], Direction::In);

    // A: flux=1, block (1,0) shape 3×4
    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![a_row, shared.clone()], U1Sector(1));
    a.block_data_mut(&BlockCoord(vec![1, 0]))
        .unwrap()
        .iter_mut()
        .for_each(|v| *v = 1.0);

    let b_col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let shared_out = QNIndex::new(vec![(U1Sector(0), 4)], Direction::Out);

    // B: flux=-1, block (0,1) shape 4×3
    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![shared_out, b_col], U1Sector(-1));
    c.block_data_mut(&BlockCoord(vec![0, 1]))
        .unwrap()
        .iter_mut()
        .for_each(|v| *v = 1.0);

    // Contract A axis 1 (In) with B axis 0 (Out)
    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            assert_eq!(out.flux(), &U1Sector(0)); // 1 + (-1) = 0
            // Output block (1,1): 3×3, each element = 4 (inner product of 4 ones)
            let d11 = out.block_data(&BlockCoord(vec![1, 1])).unwrap();
            assert_eq!(d11.len(), 9);
            assert!(d11.iter().all(|&v| (v - 4.0).abs() < 1e-10));
            // Block (0,0) stays zero (no contributing pairs)
            let d00 = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            assert!(d00.iter().all(|&v| v == 0.0));
        }
        _ => panic!("expected tensor"),
    }
}

// ---------------------------------------------------------------------------
// Full contraction → scalar
// ---------------------------------------------------------------------------

#[test]
fn full_contraction_scalar() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![row.clone(), col.clone()], U1Sector(0));
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![row, col], U1Sector(0));
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

    // axes_lhs=[0,1], axes_rhs=[1,0] so Out↔In pairing is correct
    // Σ_{i,j} A[i,j] * B[j,i] = 1*5+2*7+3*6+4*8 = 5+14+18+32 = 69
    match contract_block_sparse(&b(), &a, &c, &[0, 1], &[1, 0]).unwrap() {
        BlockSparseContractResult::Scalar(s) => assert!((s - 69.0).abs() < 1e-10),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn full_contraction_nonidentity_flux_gives_zero() {
    let shared = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let shared_in = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut a =
        BlockSparse::<f64, U1Sector>::zeros(vec![shared.clone(), shared_in.clone()], U1Sector(0));
    a.block_data_mut(&BlockCoord(vec![0, 0])).unwrap().fill(1.0);

    let c = BlockSparse::<f64, U1Sector>::zeros(vec![shared, shared_in], U1Sector(1));
    // flux_A=0 fuse flux_B=1 = 1 ≠ identity → zero
    match contract_block_sparse(&b(), &a, &c, &[0, 1], &[1, 0]).unwrap() {
        BlockSparseContractResult::Scalar(s) => assert_eq!(s, 0.0),
        _ => panic!("expected scalar"),
    }
}

// ---------------------------------------------------------------------------
// Accumulation: multiple block pairs → same output block
// ---------------------------------------------------------------------------

#[test]
fn accumulation_multiple_pairs_to_same_block() {
    // A: rank 3, flux=0
    let a0 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let a1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let a2 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);

    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![a0, a1, a2], U1Sector(0));
    // A blocks: (0,0,0) shape [2,1,1], (0,1,1) shape [2,1,1]
    a.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0]);
    a.block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0, 6.0]);

    // B: rank 3, flux=0
    let b0 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
    let b1 = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
    let b2 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![b0, b1, b2], U1Sector(0));
    // B blocks: (0,0,0) shape [1,1,2], (1,1,0) shape [1,1,2]
    c.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&[3.0, 4.0]);
    c.block_data_mut(&BlockCoord(vec![1, 1, 0]))
        .unwrap()
        .copy_from_slice(&[7.0, 8.0]);

    // Contract A[1,2] with B[0,1]: both pairs go to output (0,0)
    match contract_block_sparse(&b(), &a, &c, &[1, 2], &[0, 1]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            assert_eq!(out.num_blocks(), 1);
            let d = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            // C = [[1],[2]]×[[3,4]] + [[5],[6]]×[[7,8]]
            //   = [[3,4],[6,8]] + [[35,40],[42,48]] = [[38,44],[48,56]]
            assert!((d[0] - 38.0).abs() < 1e-10);
            assert!((d[1] - 44.0).abs() < 1e-10);
            assert!((d[2] - 48.0).abs() < 1e-10);
            assert!((d[3] - 56.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

// ---------------------------------------------------------------------------
// Contraction requiring intra-block transpose
// ---------------------------------------------------------------------------

#[test]
fn contraction_with_axis_transpose() {
    // A: rank 3 (i,j,k), contract axis 0 → lhs_perm = [1,2,0] (non-trivial)
    let a0 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let a1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let a2 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut a = BlockSparse::<f64, U1Sector>::zeros(vec![a0, a1, a2], U1Sector(0));
    // Block (0,0,0) shape [2,3,2] = 12 elements
    let ad = a.block_data_mut(&BlockCoord(vec![0, 0, 0])).unwrap();
    for (i, v) in ad.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }

    // B: rank 2, contract axis 1 → rhs_perm = [1,0] (non-trivial)
    let b0 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let b1 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut c = BlockSparse::<f64, U1Sector>::zeros(vec![b0, b1], U1Sector(0));
    let bd = c.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    for (i, v) in bd.iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }

    // Contract A axis 0 (Out) with B axis 1 (In)
    match contract_block_sparse(&b(), &a, &c, &[0], &[1]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            assert_eq!(out.rank(), 3); // [a1, a2, b0]
            let d = out.block_data(&BlockCoord(vec![0, 0, 0])).unwrap();
            assert_eq!(d.len(), 18); // 3×2×3

            // C[j,k,l] = Σ_i A[i,j,k] * B[l,i]
            // A data row-major [2,3,2]: A[0,0,0]=1..A[1,2,1]=12
            // B data row-major [3,2]: B[0,0]=1,B[0,1]=2,B[1,0]=3,...B[2,1]=6
            // C[0,0,0] = A[0,0,0]*B[0,0] + A[1,0,0]*B[0,1] = 1*1 + 7*2 = 15
            assert!((d[0] - 15.0).abs() < 1e-10);
            // C[2,1,2] = A[0,2,1]*B[2,0] + A[1,2,1]*B[2,1] = 6*5 + 12*6 = 102
            assert!((d[17] - 102.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

// ---------------------------------------------------------------------------
// Z2 symmetry
// ---------------------------------------------------------------------------

#[test]
fn z2_rank2_matmul() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 1)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 1)],
        Direction::In,
    );

    let mut a =
        BlockSparse::<f64, Z2Sector>::zeros(vec![row.clone(), col.clone()], Z2Sector::new(0));
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);

    let mut c = BlockSparse::<f64, Z2Sector>::zeros(vec![row, col], Z2Sector::new(0));
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&[2.0, 0.0, 0.0, 2.0]); // 2*I
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[3.0]);

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            // (0,0): [[1,2],[3,4]]×2I = [[2,4],[6,8]]
            let d00 = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            assert!((d00[0] - 2.0).abs() < 1e-10);
            assert!((d00[3] - 8.0).abs() < 1e-10);
            // (1,1): [5]×[3] = [15]
            let d11 = out.block_data(&BlockCoord(vec![1, 1])).unwrap();
            assert!((d11[0] - 15.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}
