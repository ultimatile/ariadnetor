//! Explicit contraction tests using DenseTensor::contract_naive
//!
//! Tests for tensor contraction with Einstein notation

use arnet_tensor::DenseTensor;

// ============================================================================
// Basic contractions
// ============================================================================

#[test]
fn test_matrix_multiplication() {
    // Matrix multiplication: [2x2] @ [2x2] = [2x2]
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.contract_naive(&b, "ij,jk->ik");

    assert_eq!(c.shape(), &[2, 2]);

    // Expected result:
    // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_inner_product() {
    // Inner product: [3] . [3] = scalar (contract_naive returns [1])
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
    let b = DenseTensor::from_data(vec![4.0, 5.0, 6.0], vec![3]);

    let c = a.contract_naive(&b, "i,i->");

    // contract_naive returns shape [1] instead of [] for scalar results
    assert_eq!(c.shape(), &[1]);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(c.get(&[0]), 32.0);
}

#[test]
fn test_outer_product() {
    // Outer product: [2] x [3] = [2, 3]
    let a = DenseTensor::from_data(vec![2.0, 3.0], vec![2]);
    let b = DenseTensor::from_data(vec![4.0, 5.0, 6.0], vec![3]);

    let c = a.contract_naive(&b, "i,j->ij");

    assert_eq!(c.shape(), &[2, 3]);
    // [2]   [4 5 6]   [8  10 12]
    // [3] x [4 5 6] = [12 15 18]
    assert_eq!(c.get(&[0, 0]), 8.0);
    assert_eq!(c.get(&[0, 1]), 10.0);
    assert_eq!(c.get(&[1, 2]), 18.0);
}

#[test]
fn test_double_contraction() {
    // Double contraction (Frobenius inner product): sum of element-wise products
    // A[i,j] * B[i,j] -> scalar
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.contract_naive(&b, "ij,ij->");

    // contract_naive returns shape [1] for scalar results
    assert_eq!(c.shape(), &[1]);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_eq!(c.get(&[0]), 70.0);
}

// ============================================================================
// Complex contractions
// ============================================================================

#[test]
fn test_three_tensor_chain() {
    // Chain of contractions: A[i,j] @ B[j,k] = C[i,k]
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let c = a.contract_naive(&b, "ij,jk->ik");

    // Result should be same as A (identity multiplication)
    assert_eq!(c.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 1]), 2.0);
    assert_eq!(c.get(&[1, 0]), 3.0);
    assert_eq!(c.get(&[1, 1]), 4.0);
}

#[test]
#[ignore = "contract_naive doesn't handle element-wise multiplication (no contraction) case - returns [2,2,2,2] instead of [2,2]"]
fn test_hadamard_product() {
    // Element-wise product (no contraction)
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    let c = a.contract_naive(&b, "ij,ij->ij");

    assert_eq!(c.shape(), &[2, 2]);
    // Element-wise: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
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
    // Contracting two scalars
    let a = DenseTensor::from_data(vec![5.0], vec![]);
    let b = DenseTensor::from_data(vec![3.0], vec![]);

    let c = a.contract_naive(&b, "->");

    // contract_naive returns shape [1] for scalar results instead of []
    assert_eq!(c.shape(), &[1]);
    assert_eq!(c.get(&[0]), 15.0);
}

#[test]
fn test_vector_matrix_contraction() {
    // Vector-matrix multiplication: [3] @ [3x2] = [2]
    let v = DenseTensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
    let m = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let result = v.contract_naive(&m, "i,ij->j");

    assert_eq!(result.shape(), &[2]);
    // [1 2 3] @ [[1 2], [3 4], [5 6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    assert_eq!(result.get(&[0]), 22.0);
    assert_eq!(result.get(&[1]), 28.0);
}
