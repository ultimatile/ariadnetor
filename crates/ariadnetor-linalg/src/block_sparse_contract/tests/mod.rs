use super::*;
use super::{BlockSparseContractResult, contract_block_sparse};
use arnet_core::backend::ComputeBackend;
use arnet_native::NativeBackend;
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Direction, MemoryOrder, QNIndex};
use arnet_tensor::{U1Sector, Z2Sector};

mod predicates;
mod transpose_flag;

fn b() -> NativeBackend {
    NativeBackend::new()
}

fn order() -> MemoryOrder {
    b().preferred_order()
}

/// Compute flat index from multi-index in the backend's preferred order.
fn flat_idx(multi: &[usize], shape: &[usize]) -> usize {
    let strides = compute_strides(shape, order());
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

/// Convert data from conceptual RowMajor layout to the backend's preferred order.
/// This preserves the mathematical meaning of the data regardless of backend convention.
fn to_order(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let ord = order();
    if matches!(ord, MemoryOrder::RowMajor) || shape.len() <= 1 {
        return data.to_vec();
    }
    let rm_strides = compute_strides(shape, MemoryOrder::RowMajor);
    let cm_strides = compute_strides(shape, MemoryOrder::ColumnMajor);
    let mut result = vec![0.0; data.len()];
    for (flat_rm, &val) in data.iter().enumerate() {
        let mut multi = vec![0; shape.len()];
        let mut rem = flat_rm;
        for i in 0..shape.len() {
            multi[i] = rem / rm_strides[i];
            rem %= rm_strides[i];
        }
        let flat_cm: usize = multi
            .iter()
            .zip(cm_strides.iter())
            .map(|(&m, &s)| m * s)
            .sum();
        result[flat_cm] = val;
    }
    result
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

#[test]
fn error_on_length_mismatch() {
    let idx = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![idx.clone(), idx.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    assert!(contract_block_sparse(&b(), &a, &a, &[0], &[0, 1]).is_err());
}

#[test]
fn error_on_out_of_range() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    assert!(contract_block_sparse(&b(), &a, &a, &[5], &[0]).is_err());
}

#[test]
fn error_on_duplicate_axis() {
    let out = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let in_ = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![out.clone(), out, in_.clone(), in_],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    assert!(contract_block_sparse(&b(), &a, &a, &[0, 0], &[2, 3]).is_err());
}

#[test]
fn error_on_same_direction() {
    let out = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let in_ = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![out.clone(), in_.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    let c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![out, in_],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    // axis 0 (Out) vs axis 0 (Out) → same direction
    assert!(contract_block_sparse(&b(), &a, &c, &[0], &[0]).is_err());
}

#[test]
fn error_on_sector_mismatch() {
    let a_col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let b_row = QNIndex::new(vec![(U1Sector(1), 2)], Direction::Out);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out), a_col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    let c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![b_row, QNIndex::new(vec![(U1Sector(1), 2)], Direction::In)],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    assert!(contract_block_sparse(&b(), &a, &c, &[1], &[0]).is_err());
}

#[test]
fn error_on_dim_mismatch() {
    let a_col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let b_row = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out), a_col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    let c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![b_row, QNIndex::new(vec![(U1Sector(0), 3)], Direction::In)],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
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

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            let d = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
            let expected = to_order(&[19.0, 22.0, 43.0, 50.0], &[2, 2]);
            assert!((d[0] - expected[0]).abs() < 1e-10);
            assert!((d[1] - expected[1]).abs() < 1e-10);
            assert!((d[2] - expected[2]).abs() < 1e-10);
            assert!((d[3] - expected[3]).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn rank2_multi_block_matmul() {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[6.0, 7.0, 8.0, 9.0], &[2, 2]));
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[10.0]);

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            // Block (0,0): [[1,2],[3,4]]×[[6,7],[8,9]] = [[22,25],[50,57]]
            let e00 = to_order(&[22.0, 25.0, 50.0, 57.0], &[2, 2]);
            let d00 = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            assert!((d00[0] - e00[0]).abs() < 1e-10);
            assert!((d00[1] - e00[1]).abs() < 1e-10);
            assert!((d00[2] - e00[2]).abs() < 1e-10);
            assert!((d00[3] - e00[3]).abs() < 1e-10);
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
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![a_row, shared.clone()],
        U1Sector(1),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![1, 0]))
        .unwrap()
        .iter_mut()
        .for_each(|v| *v = 1.0);

    let b_col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let shared_out = QNIndex::new(vec![(U1Sector(0), 4)], Direction::Out);

    // B: flux=-1, block (0,1) shape 4×3
    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![shared_out, b_col],
        U1Sector(-1),
        MemoryOrder::ColumnMajor,
    );
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

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

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
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![shared.clone(), shared_in.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0])).unwrap().fill(1.0);

    let c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![shared, shared_in],
        U1Sector(1),
        MemoryOrder::ColumnMajor,
    );
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

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![a0, a1, a2],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
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

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![b0, b1, b2],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
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
            let expected = to_order(&[38.0, 44.0, 48.0, 56.0], &[2, 2]);
            assert!((d[0] - expected[0]).abs() < 1e-10);
            assert!((d[1] - expected[1]).abs() < 1e-10);
            assert!((d[2] - expected[2]).abs() < 1e-10);
            assert!((d[3] - expected[3]).abs() < 1e-10);
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
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![a0, a1, a2],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    // Block (0,0,0) shape [2,3,2] = 12 elements
    // Conceptually RowMajor: A[i,j,k] = i*6+j*2+k+1
    let a_rm: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    a.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&a_rm, &[2, 3, 2]));

    // B: rank 2, contract axis 1 → rhs_perm = [1,0] (non-trivial)
    let b0 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let b1 = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![b0, b1],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    let b_rm: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&b_rm, &[3, 2]));

    // Contract A axis 0 (Out) with B axis 1 (In)
    match contract_block_sparse(&b(), &a, &c, &[0], &[1]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            assert_eq!(out.rank(), 3); // [a1, a2, b0]
            let d = out.block_data(&BlockCoord(vec![0, 0, 0])).unwrap();
            assert_eq!(d.len(), 18); // 3×2×3

            // C[j,k,l] = Σ_i A[i,j,k] * B[l,i]
            // A[0,0,0]=1, A[1,0,0]=7, B[0,0]=1, B[0,1]=2
            // C[0,0,0] = 1*1 + 7*2 = 15
            assert!((d[flat_idx(&[0, 0, 0], &[3, 2, 3])] - 15.0).abs() < 1e-10);
            // C[2,1,2] = A[0,2,1]*B[2,0] + A[1,2,1]*B[2,1] = 6*5 + 12*6 = 102
            assert!((d[flat_idx(&[2, 1, 2], &[3, 2, 3])] - 102.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

// ---------------------------------------------------------------------------
// Full contraction — no-transpose path
// ---------------------------------------------------------------------------

#[test]
fn full_contraction_identity_perm() {
    // [Out, In] × [In, Out] with axes=[0,1],[0,1] → rhs_perm is identity.
    // Exercises the no-transpose path in contract_to_scalar.
    let out_idx = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let in_idx = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![out_idx.clone(), in_idx.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![in_idx, out_idx],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

    // Σ A[i,j]*B[i,j] = 1*5 + 2*6 + 3*7 + 4*8 = 70
    match contract_block_sparse(&b(), &a, &c, &[0, 1], &[0, 1]).unwrap() {
        BlockSparseContractResult::Scalar(s) => {
            assert!((s - 70.0).abs() < 1e-10, "expected 70, got {s}");
        }
        _ => panic!("expected scalar"),
    }
}

#[test]
fn full_contraction_identity_perm_multi_block() {
    let out_idx = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let in_idx = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![out_idx.clone(), in_idx.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);

    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![in_idx, out_idx],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[2.0, 0.0, 0.0, 3.0], &[2, 2]));
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[4.0]);

    // Block (0,0): 1*2+2*0+3*0+4*3 = 14. Block (1,1): 5*4 = 20. Total = 34.
    match contract_block_sparse(&b(), &a, &c, &[0, 1], &[0, 1]).unwrap() {
        BlockSparseContractResult::Scalar(s) => {
            assert!((s - 34.0).abs() < 1e-10, "expected 34, got {s}");
        }
        _ => panic!("expected scalar"),
    }
}

// ---------------------------------------------------------------------------
// Rank mismatch contraction (output_rank formula)
// ---------------------------------------------------------------------------

#[test]
fn contraction_rank2_with_rank1() {
    // Matrix-vector: output_rank = 2+1-2 = 1 (tensor), not 2*1-2 = 0 (scalar).
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 3)], Direction::In);
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let v_out = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let mut v = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![v_out],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    v.block_data_mut(&BlockCoord(vec![0]))
        .unwrap()
        .copy_from_slice(&[1.0, 0.0, 1.0]); // rank-1: order doesn't matter

    match contract_block_sparse(&b(), &a, &v, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            assert_eq!(out.rank(), 1);
            let d = out.block_data(&BlockCoord(vec![0])).unwrap();
            // [[1,2,3],[4,5,6]] × [1,0,1] = [1+0+3, 4+0+6] = [4, 10]
            assert!((d[0] - 4.0).abs() < 1e-10);
            assert!((d[1] - 10.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor, got scalar"),
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

    let mut a = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        vec![row.clone(), col.clone()],
        Z2Sector::new(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);

    let mut c = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        vec![row, col],
        Z2Sector::new(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[2.0, 0.0, 0.0, 2.0], &[2, 2])); // 2*I
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[3.0]);

    match contract_block_sparse(&b(), &a, &c, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(out) => {
            // (0,0): [[1,2],[3,4]]×2I = [[2,4],[6,8]]
            let e00 = to_order(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
            let d00 = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
            assert!((d00[0] - e00[0]).abs() < 1e-10);
            assert!((d00[3] - e00[3]).abs() < 1e-10);
            // (1,1): [5]×[3] = [15]
            let d11 = out.block_data(&BlockCoord(vec![1, 1])).unwrap();
            assert!((d11[0] - 15.0).abs() < 1e-10);
        }
        _ => panic!("expected tensor"),
    }
}

// ---------------------------------------------------------------------------
// Permuted axis contraction (exercises trans_a/trans_b path)
// ---------------------------------------------------------------------------

/// Contract lhs axis 0 (Out) with rhs axis 1 (In): a_{ij} b_{ki} -> c_{jk}.
/// This triggers the GEMM trans_a path for rank-2 tensors with ascending prefix.
#[test]
fn contract_permuted_axes_rank2() {
    // A: 2×2 block (sector 0) + 1×1 block (sector 1), flux=0
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    // A block (0,0) = [[1,2],[3,4]], block (1,1) = [[5]]
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    let d = a.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
    d.copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let d = a.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
    d[0] = 5.0;

    // B = A (same tensor)
    let a2 = a.clone();

    // Standard: contract [1],[0] → A × A (matmul)
    let standard = match contract_block_sparse(&b(), &a, &a2, &[1], &[0]).unwrap() {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // Permuted: contract [0],[1] → A^T × A^T
    // a_{ij} b_{ki} -> c_{jk}: sum over i.
    // c_{jk} = sum_i a_{ij} b_{ki} = (A^T A^T)_{jk}
    let permuted = match contract_block_sparse(&b(), &a, &a2, &[0], &[1]).unwrap() {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // Verify: (A^T B^T)_{jk} = (B A)^T_{jk} = (A A)^T_{jk}
    // Standard block (0,0) = A×A block (0,0) = [[1,2],[3,4]]×[[1,2],[3,4]] = [[7,10],[15,22]]
    // Permuted block (0,0) = (A×A)^T block (0,0) = [[7,15],[10,22]]
    let s00 = standard.block_data(&BlockCoord(vec![0, 0])).unwrap();
    let p00 = permuted.block_data(&BlockCoord(vec![0, 0])).unwrap();

    // Standard in RM: [7, 10, 15, 22]. Permuted (transpose) in RM: [7, 15, 10, 22].
    let s00_rm = to_order(&[7.0, 10.0, 15.0, 22.0], &[2, 2]);
    let p00_rm = to_order(&[7.0, 15.0, 10.0, 22.0], &[2, 2]);
    for i in 0..4 {
        assert!(
            (s00[i] - s00_rm[i]).abs() < 1e-10,
            "standard[{i}]: {} vs {}",
            s00[i],
            s00_rm[i]
        );
        assert!(
            (p00[i] - p00_rm[i]).abs() < 1e-10,
            "permuted[{i}]: {} vs {}",
            p00[i],
            p00_rm[i]
        );
    }

    // Block (1,1): scalar 5*5=25 for both
    let s11 = standard.block_data(&BlockCoord(vec![1, 1])).unwrap();
    let p11 = permuted.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert!((s11[0] - 25.0).abs() < 1e-10);
    assert!((p11[0] - 25.0).abs() < 1e-10);
}

mod policy_forwarding;
