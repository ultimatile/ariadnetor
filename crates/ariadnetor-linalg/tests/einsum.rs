//! Tests for single-tensor einsum operations (trace, transpose, permutation)

use arnet_cpu::CpuBackend;
use arnet_linalg::einsum;
use arnet_tensor::DenseTensor;

// ============================================================================
// Transpose / permutation (no repeated indices)
// ============================================================================

#[test]
fn test_einsum_transpose_2d() {
    let backend = CpuBackend::new();
    // 2×3 matrix
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let b = einsum(&backend, &[&a], "ij->ji").unwrap();

    assert_eq!(b.shape(), &[3, 2]);
    // Row-major [1,2,3,4,5,6] transposed → [1,4,2,5,3,6]
    assert_eq!(b.get(&[0, 0]), 1.0);
    assert_eq!(b.get(&[0, 1]), 4.0);
    assert_eq!(b.get(&[1, 0]), 2.0);
    assert_eq!(b.get(&[1, 1]), 5.0);
    assert_eq!(b.get(&[2, 0]), 3.0);
    assert_eq!(b.get(&[2, 1]), 6.0);
}

#[test]
fn test_einsum_permutation_3d() {
    let backend = CpuBackend::new();
    // 2×3×4 tensor
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 4]);

    let b = einsum(&backend, &[&a], "ijk->kji").unwrap();

    assert_eq!(b.shape(), &[4, 3, 2]);
    // A[0,0,0] = 1 → B[0,0,0] = 1
    assert_eq!(b.get(&[0, 0, 0]), a.get(&[0, 0, 0]));
    // A[1,2,3] → B[3,2,1]
    assert_eq!(b.get(&[3, 2, 1]), a.get(&[1, 2, 3]));
    // A[0,1,2] → B[2,1,0]
    assert_eq!(b.get(&[2, 1, 0]), a.get(&[0, 1, 2]));
}

#[test]
fn test_einsum_identity_permutation() {
    let backend = CpuBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Identity permutation: no actual transpose needed
    let b = einsum(&backend, &[&a], "ij->ij").unwrap();

    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.data(), a.data());
}

// ============================================================================
// Trace (repeated indices not in output)
// ============================================================================

#[test]
fn test_einsum_full_trace() {
    let backend = CpuBackend::new();
    // 3×3 matrix
    let a = DenseTensor::from_data(
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        vec![3, 3],
    );

    let b = einsum(&backend, &[&a], "ii->").unwrap();

    // Trace = 1 + 2 + 3 = 6
    assert_eq!(b.shape(), &[1]);
    assert_eq!(b.get(&[0]), 6.0);
}

#[test]
fn test_einsum_partial_trace() {
    let backend = CpuBackend::new();
    // 2×2×3 tensor: A[i,i,j] → sum over diagonal of i, keep j
    let a = DenseTensor::from_data(
        vec![
            1.0, 2.0, 3.0, // [0,0,:]
            4.0, 5.0, 6.0, // [0,1,:]
            7.0, 8.0, 9.0, // [1,0,:]
            10.0, 11.0, 12.0, // [1,1,:]
        ],
        vec![2, 2, 3],
    );

    let b = einsum(&backend, &[&a], "iij->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,0,j] + A[1,1,j]
    // B[0] = 1 + 10 = 11
    // B[1] = 2 + 11 = 13
    // B[2] = 3 + 12 = 15
    assert_eq!(b.get(&[0]), 11.0);
    assert_eq!(b.get(&[1]), 13.0);
    assert_eq!(b.get(&[2]), 15.0);
}

// ============================================================================
// Trace + transpose
// ============================================================================

#[test]
fn test_einsum_trace_then_transpose() {
    let backend = CpuBackend::new();
    // 2×3×2 tensor: "iji->j" traces i (positions 0,2), keeps j
    // This is a valid trace+result case
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 2]);

    let b = einsum(&backend, &[&a], "iji->j").unwrap();

    assert_eq!(b.shape(), &[3]);
    // B[j] = A[0,j,0] + A[1,j,1]
    // A[0,0,0]=1, A[1,0,1]=8 → B[0]=9
    // A[0,1,0]=3, A[1,1,1]=10 → B[1]=13
    // A[0,2,0]=5, A[1,2,1]=12 → B[2]=17
    assert_eq!(b.get(&[0]), 9.0);
    assert_eq!(b.get(&[1]), 13.0);
    assert_eq!(b.get(&[2]), 17.0);
}

#[test]
fn test_einsum_trace_and_permute() {
    let backend = CpuBackend::new();
    // 2×3×4×2 tensor: "ijki->kj" traces i (positions 0,3), keeps j,k → permute to k,j
    let data: Vec<f64> = (1..=48).map(|x| x as f64).collect();
    let a = DenseTensor::from_data(data, vec![2, 3, 4, 2]);

    let b = einsum(&backend, &[&a], "ijki->kj").unwrap();

    assert_eq!(b.shape(), &[4, 3]);
    // After trace: C[j,k] = A[0,j,k,0] + A[1,j,k,1]
    // A is row-major [2,3,4,2]: A[i,j,k,l] = data[i*24 + j*8 + k*2 + l]
    // C[0,0] = A[0,0,0,0] + A[1,0,0,1] = 1 + 26 = 27
    // Then B[k,j] = C[j,k], so B[0,0] = C[0,0] = 27
    assert_eq!(b.get(&[0, 0]), 27.0);

    // C[1,2] = A[0,1,2,0] + A[1,1,2,1] = 13 + 38 = 51
    // B[2,1] = C[1,2] = 51
    assert_eq!(b.get(&[2, 1]), 51.0);
}

// ============================================================================
// 2-input delegation (verify einsum dispatches to contract)
// ============================================================================

#[test]
fn test_einsum_two_input_matmul() {
    let backend = CpuBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = einsum(&backend, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

// ============================================================================
// Error cases
// ============================================================================

#[test]
fn test_einsum_wrong_tensor_count() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Notation expects 2 inputs but only 1 given
    let result = einsum(&backend, &[&a], "ij,jk->ik");
    assert!(result.is_err());
}

#[test]
fn test_einsum_rank_mismatch() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // 3-index notation with rank-2 tensor
    let result = einsum(&backend, &[&a], "ijk->kji");
    assert!(result.is_err());
}

#[test]
fn test_einsum_diagonal_extraction_unsupported() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // "ii->i" is diagonal extraction, not yet supported
    let result = einsum(&backend, &[&a], "ii->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("diagonal extraction"));
}

#[test]
fn test_einsum_reduction_unsupported() {
    let backend = CpuBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // "ij->i" is a sum over j, not supported as single-tensor einsum
    let result = einsum(&backend, &[&a], "ij->i");
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("reduction"));
}
