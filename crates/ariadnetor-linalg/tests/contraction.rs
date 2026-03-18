//! Contraction tests using arnet_linalg::contract with NativeBackend
//!
//! Migrated from ariadnetor-tensor integration tests after moving
//! contraction logic to the linalg crate.

use arnet_native::NativeBackend;
use arnet_linalg::{contract, transpose};
use arnet_tensor::DenseTensor;

// ============================================================================
// Basic contractions
// ============================================================================

#[test]
fn test_matrix_multiplication() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ij,jk->ik").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_inner_product() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
    let b = DenseTensor::from_data(vec![4.0, 5.0, 6.0], vec![3]);

    let c = contract(&backend, &a, &b, "i,i->").unwrap();

    // Scalar result → shape [1]
    assert_eq!(c.shape(), &[1]);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(c.get(&[0]), 32.0);
}

#[test]
fn test_outer_product() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![2.0, 3.0], vec![2]);
    let b = DenseTensor::from_data(vec![4.0, 5.0, 6.0], vec![3]);

    let c = contract(&backend, &a, &b, "i,j->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.get(&[0, 0]), 8.0);
    assert_eq!(c.get(&[0, 1]), 10.0);
    assert_eq!(c.get(&[1, 2]), 18.0);
}

#[test]
fn test_double_contraction() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ij,ij->").unwrap();

    assert_eq!(c.shape(), &[1]);
    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert_eq!(c.get(&[0]), 70.0);
}

// ============================================================================
// Complex contractions
// ============================================================================

#[test]
fn test_identity_multiplication() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ij,jk->ik").unwrap();

    assert_eq!(c.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 1]), 2.0);
    assert_eq!(c.get(&[1, 0]), 3.0);
    assert_eq!(c.get(&[1, 1]), 4.0);
}

#[test]
fn test_hadamard_product() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ij,ij->ij").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 2.0);
    assert_eq!(c.get(&[0, 1]), 6.0);
    assert_eq!(c.get(&[1, 0]), 12.0);
    assert_eq!(c.get(&[1, 1]), 20.0);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_scalar_contraction() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![5.0], vec![]);
    let b = DenseTensor::from_data(vec![3.0], vec![]);

    let c = contract(&backend, &a, &b, ",->").unwrap();

    assert_eq!(c.shape(), &[1]);
    assert_eq!(c.get(&[0]), 15.0);
}

#[test]
fn test_vector_matrix_contraction() {
    let backend = NativeBackend::new();
    let v = DenseTensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
    let m = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let result = contract(&backend, &v, &m, "i,ij->j").unwrap();

    assert_eq!(result.shape(), &[2]);
    // [1 2 3] @ [[1 2], [3 4], [5 6]] = [22, 28]
    assert_eq!(result.get(&[0]), 22.0);
    assert_eq!(result.get(&[1]), 28.0);
}

// ============================================================================
// Contraction ordering tests (migrated from contraction_order_comprehensive)
// ============================================================================

#[test]
fn test_actual_contraction_with_reordered_indices() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );

    let c = contract(&backend, &a, &b, "ikj,jkl->il").unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_consistency_between_ijk_and_ikj_layouts() {
    let backend = NativeBackend::new();
    let a_ijk = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );

    let result_ijk = contract(&backend, &a_ijk, &b, "ijk,jkl->il").unwrap();

    // Permute A from [i,j,k] to [i,k,j] layout
    let a_ikj = transpose(&backend, &a_ijk, &[0, 2, 1]).unwrap();
    let result_ikj = contract(&backend, &a_ikj, &b, "ikj,jkl->il").unwrap();

    assert_eq!(result_ijk.shape(), result_ikj.shape());
    assert_ne!(result_ijk.get(&[0, 0]), 0.0);
    assert_ne!(result_ikj.get(&[0, 0]), 0.0);
}
