//! Contraction tests using arnet_linalg::contract with NativeBackend
//!
//! Migrated from ariadnetor-tensor integration tests after moving
//! contraction logic to the linalg crate.

use arnet_linalg::{contract, einsum, transpose};
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensor, DenseTensorData, MemoryOrder};

/// Build a `DenseTensor` from row-major data, reordered to column-major.
fn cm(data: Vec<f64>, shape: Vec<usize>) -> DenseTensor<f64, NativeBackend> {
    let rm = DenseTensorData::from_raw_parts(data, shape, MemoryOrder::RowMajor);
    let cm = arnet_tensor::reorder_data(&rm, MemoryOrder::ColumnMajor);
    DenseTensor::with_backend(cm, NativeBackend::shared())
}

/// Reorder a `DenseTensor` back to row-major for index-by-index assertions.
fn to_rm(tensor: &DenseTensor<f64, NativeBackend>) -> DenseTensor<f64, NativeBackend> {
    let rm = arnet_tensor::reorder_data(tensor.data(), MemoryOrder::RowMajor);
    DenseTensor::with_backend(rm, NativeBackend::shared())
}

// ============================================================================
// Basic contractions
// ============================================================================

#[test]
fn test_matrix_multiplication() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = to_rm(&contract(&a, &b, "ij,jk->ik").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_inner_product() {
    let a = cm(vec![1.0, 2.0, 3.0], vec![3]);
    let b = cm(vec![4.0, 5.0, 6.0], vec![3]);

    let c = contract(&a, &b, "i,i->").unwrap();

    // Scalar result → shape [1]
    assert_eq!(c.shape(), &[1]);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(c.data_slice()[0], 32.0);
}

#[test]
fn test_outer_product() {
    let a = cm(vec![2.0, 3.0], vec![2]);
    let b = cm(vec![4.0, 5.0, 6.0], vec![3]);

    let c = to_rm(&contract(&a, &b, "i,j->ij").unwrap());

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.get(&[0, 0]), 8.0);
    assert_eq!(c.get(&[0, 1]), 10.0);
    assert_eq!(c.get(&[1, 2]), 18.0);
}

#[test]
fn test_double_contraction() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = contract(&a, &b, "ij,ij->").unwrap();

    assert_eq!(c.shape(), &[1]);
    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert_eq!(c.data_slice()[0], 70.0);
}

// ============================================================================
// Complex contractions
// ============================================================================

#[test]
fn test_identity_multiplication() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let c = to_rm(&contract(&a, &b, "ij,jk->ik").unwrap());

    assert_eq!(c.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 1]), 2.0);
    assert_eq!(c.get(&[1, 0]), 3.0);
    assert_eq!(c.get(&[1, 1]), 4.0);
}

#[test]
fn test_hadamard_product_via_einsum() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    // Hadamard product routes through einsum_pair, not contract
    let c = to_rm(&einsum(&[&a, &b], "ij,ij->ij").unwrap());

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 2.0);
    assert_eq!(c.get(&[0, 1]), 6.0);
    assert_eq!(c.get(&[1, 0]), 12.0);
    assert_eq!(c.get(&[1, 1]), 20.0);
}

#[test]
fn test_hadamard_contract_rejects() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cm(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    // contract() should reject Hadamard (all batch, no contraction)
    let result = contract(&a, &b, "ij,ij->ij");
    assert!(result.is_err());
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_scalar_contraction() {
    let a = cm(vec![5.0], vec![]);
    let b = cm(vec![3.0], vec![]);

    let c = contract(&a, &b, ",->").unwrap();

    assert_eq!(c.shape(), &[1]);
    assert_eq!(c.data_slice()[0], 15.0);
}

#[test]
fn test_vector_matrix_contraction() {
    let v = cm(vec![1.0, 2.0, 3.0], vec![3]);
    let m = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let result = contract(&v, &m, "i,ij->j").unwrap();

    assert_eq!(result.shape(), &[2]);
    // [1 2 3] @ [[1 2], [3 4], [5 6]] = [22, 28]
    assert_eq!(result.data_slice()[0], 22.0);
    assert_eq!(result.data_slice()[1], 28.0);
}

// ============================================================================
// Contraction ordering tests
// ============================================================================

#[test]
fn test_actual_contraction_with_reordered_indices() {
    let a = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

    let c = contract(&a, &b, "ikj,jkl->il").unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_consistency_between_ijk_and_ikj_layouts() {
    let a_ijk = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = cm(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

    let result_ijk = contract(&a_ijk, &b, "ijk,jkl->il").unwrap();

    // Permute A from [i,j,k] to [i,k,j] layout
    let a_ikj = transpose(&a_ijk, &[0, 2, 1]).unwrap();
    let result_ikj = contract(&a_ikj, &b, "ikj,jkl->il").unwrap();

    assert_eq!(result_ijk.shape(), result_ikj.shape());
    assert_ne!(result_ijk.get(&[0, 0]), 0.0);
    assert_ne!(result_ikj.get(&[0, 0]), 0.0);
}
