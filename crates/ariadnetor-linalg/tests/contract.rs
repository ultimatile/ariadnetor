use arnet_core::backend::ComputeBackend;
use arnet_linalg::contract;
use arnet_native::NativeBackend;
use arnet_tensor::{Dense, MemoryOrder};

#[test]
fn test_contract_matmul() {
    let backend = NativeBackend::new();
    let a =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b =
        Dense::from_data_with_order(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_contract_tensor_contraction() {
    let backend = NativeBackend::new();
    // C[i,l] = Σ_{j,k} A[i,j,k] × B[j,k,l]
    let a = Dense::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );

    let c = contract(&backend, &a, &b, "ijk,jkl->il").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_contract_f32() {
    let backend = NativeBackend::new();
    let a = Dense::from_data_with_order(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::from_data_with_order(
        vec![5.0f32, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0f32);
}

#[test]
fn test_contract_with_permutation() {
    let backend = NativeBackend::new();
    // A[i,k,j] × B[k,j] → C[i] requires permutation of LHS
    let a = Dense::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );
    let b =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);

    let c = contract(&backend, &a, &b, "ikj,kj->i").unwrap();

    assert_eq!(c.shape(), &[2]);
    assert_ne!(c.get(&[0]), 0.0);
}

#[test]
fn test_contract_rectangular() {
    let backend = NativeBackend::new();
    // A (2×2) × B (2×3) → C (2×3)
    let a =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b = Dense::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
}

#[test]
fn test_contract_invalid_notation() {
    let backend = NativeBackend::new();
    let a = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::<f64>::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );

    // Invalid: output index 'm' not in any input
    let result = contract(&backend, &a, &b, "ik,kj->im");
    assert!(result.is_err());
}

#[test]
fn test_contract_rank_mismatch() {
    let backend = NativeBackend::new();
    let a = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::<f64>::from_data_with_order(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        MemoryOrder::RowMajor,
    );

    // 3-index notation with rank-2 tensor
    let result = contract(&backend, &a, &b, "ijk,kl->ijl");
    assert!(result.is_err());
}

// ============================================================================
// Output index reordering tests
// ============================================================================

#[test]
fn test_contract_output_reorder_swap() {
    let backend = NativeBackend::new();
    let a = Dense::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        MemoryOrder::RowMajor,
    );
    let b = Dense::from_data_with_order(
        (1..=12).map(|x| x as f64).collect(),
        vec![3, 4],
        MemoryOrder::RowMajor,
    );

    let normal = contract(&backend, &a, &b, "ik,kj->ij").unwrap();
    let swapped = contract(&backend, &a, &b, "ik,kj->ji").unwrap();

    assert_eq!(normal.shape(), &[2, 4]);
    assert_eq!(swapped.shape(), &[4, 2]);

    for i in 0..2 {
        for j in 0..4 {
            assert_eq!(swapped.get(&[j, i]), normal.get(&[i, j]));
        }
    }
}

#[test]
fn test_contract_output_reorder_3d() {
    let backend = NativeBackend::new();
    // A[a,b,c] (2×3×2) × B[c,d] (2×4) → reorder to [d,b,a]
    let a = Dense::from_data_with_order(
        (1..=12).map(|x| x as f64).collect(),
        vec![2, 3, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::from_data_with_order(
        (1..=8).map(|x| x as f64).collect(),
        vec![2, 4],
        MemoryOrder::RowMajor,
    );

    let normal = contract(&backend, &a, &b, "abc,cd->abd").unwrap();
    let reordered = contract(&backend, &a, &b, "abc,cd->dba").unwrap();

    assert_eq!(normal.shape(), &[2, 3, 4]);
    assert_eq!(reordered.shape(), &[4, 3, 2]);

    for a_idx in 0..2 {
        for b_idx in 0..3 {
            for d in 0..4 {
                assert_eq!(
                    reordered.get(&[d, b_idx, a_idx]),
                    normal.get(&[a_idx, b_idx, d]),
                );
            }
        }
    }
}

#[test]
fn test_contract_rejects_batch_indices() {
    let backend = NativeBackend::new();
    let a = Dense::<f64>::from_data_with_order(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );
    let b = Dense::<f64>::from_data_with_order(
        vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        vec![2, 2, 2],
        MemoryOrder::RowMajor,
    );

    // Batch index 'b' appears in both inputs and output — contract() should reject
    let result = contract(&backend, &a, &b, "bik,bkj->bij");
    assert!(result.is_err());
}

#[test]
fn test_contract_output_memory_order() {
    let backend = NativeBackend::new();
    let a =
        Dense::from_data_with_order(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], MemoryOrder::RowMajor);
    let b =
        Dense::from_data_with_order(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], MemoryOrder::RowMajor);

    // No reorder case
    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();
    assert_eq!(c.memory_order(), backend.preferred_order());

    // Reorder case — must also be preferred_order
    let c_reordered = contract(&backend, &a, &b, "ik,kj->ji").unwrap();
    assert_eq!(c_reordered.memory_order(), backend.preferred_order());
}
