use arnet_native::NativeBackend;
use arnet_linalg::contract;
use arnet_tensor::DenseTensor;

#[test]
fn test_contract_matmul() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

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
    let a = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );

    let c = contract(&backend, &a, &b, "ijk,jkl->il").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_ne!(c.get(&[0, 0]), 0.0);
}

#[test]
fn test_contract_f32() {
    let backend = NativeBackend::new();
    let a = DenseTensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]), 19.0f32);
}

#[test]
fn test_contract_with_permutation() {
    let backend = NativeBackend::new();
    // A[i,k,j] × B[k,j] → C[i] requires permutation of LHS
    let a = DenseTensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
    );
    let b = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let c = contract(&backend, &a, &b, "ikj,kj->i").unwrap();

    assert_eq!(c.shape(), &[2]);
    assert_ne!(c.get(&[0]), 0.0);
}

#[test]
fn test_contract_rectangular() {
    let backend = NativeBackend::new();
    // A (2×2) × B (2×3) → C (2×3)
    let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

    let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

    assert_eq!(c.shape(), &[2, 3]);
}

#[test]
fn test_contract_invalid_notation() {
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Invalid: output index 'm' not in any input
    let result = contract(&backend, &a, &b, "ik,kj->im");
    assert!(result.is_err());
}

#[test]
fn test_contract_rank_mismatch() {
    let backend = NativeBackend::new();
    let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // 3-index notation with rank-2 tensor
    let result = contract(&backend, &a, &b, "ijk,kl->ijl");
    assert!(result.is_err());
}
