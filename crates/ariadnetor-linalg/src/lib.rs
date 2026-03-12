//! Backend-agnostic linear algebra API for Ariadnetor
//!
//! Provides high-level tensor operations that delegate to a [`ComputeBackend`]
//! for the actual computation. This decouples tensor data from compute libraries
//! (faer, HPTT, etc.) so that `ariadnetor-tensor` carries no heavy dependencies.
//!
//! # Operations
//!
//! - [`transpose`]: Permute tensor axes via backend
//! - [`contract`]: Tensor contraction via Einstein summation (permute + GEMM)
//! - [`scale`]: Scalar multiplication (out-of-place)
//! - [`norm`]: Frobenius norm
//! - [`normalize`]: Normalize to unit norm (out-of-place)
//! - [`linear_combine`]: Linear combination of tensors

pub use arnet_core::backend::ComputeBackend;

use arnet_core::backend::{BackendError, GemmDescriptor, TransposeDescriptor};
use arnet_core::einsum::{ContractionPlan, EinsumExpr};
use arnet_core::scalar::{FloatCompute, Scalar};
use arnet_tensor::DenseTensor;
use num_traits::Zero;
use std::ops::{Add, Mul};

/// Transpose (permute axes) of a dense tensor using the provided backend.
///
/// # Arguments
///
/// * `backend` - Compute backend for the transpose operation
/// * `tensor` - Input tensor
/// * `perm` - Permutation of axes (e.g., `[1, 0]` transposes a 2D tensor)
///
/// # Errors
///
/// Returns `BackendError` if the backend fails to execute the transpose.
pub fn transpose<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    perm: &[usize],
) -> Result<DenseTensor<T>, BackendError> {
    let new_shape: Vec<usize> = perm.iter().map(|&i| tensor.shape()[i]).collect();
    let total = tensor.len();

    if total == 0 {
        return Ok(DenseTensor::from_data(vec![], new_shape));
    }

    let mut output = vec![T::zero(); total];

    let desc = TransposeDescriptor {
        input: tensor.data(),
        output: &mut output,
        shape: tensor.shape(),
        perm,
    };

    backend.transpose(desc)?;

    Ok(DenseTensor::from_data(output, new_shape))
}

/// Contract two tensors using Einstein summation notation with the provided backend.
///
/// Performs a single GEMM-based contraction:
/// 1. Parse Einstein notation
/// 2. Permute operands to align contracted indices (via backend transpose)
/// 3. Reshape to 2D matrices
/// 4. GEMM via backend
/// 5. Reshape result to output tensor shape
///
/// # Arguments
///
/// * `backend` - Compute backend for transpose and GEMM operations
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
/// * `notation` - Einstein summation notation (e.g., "ik,kj->ij")
///
/// # Errors
///
/// Returns `BackendError` if notation is invalid, dimensions mismatch,
/// or the backend fails to execute transpose/GEMM.
pub fn contract<T: Scalar>(
    backend: &impl ComputeBackend,
    lhs: &DenseTensor<T>,
    rhs: &DenseTensor<T>,
    notation: &str,
) -> Result<DenseTensor<T>, BackendError> {
    let expr = EinsumExpr::parse(notation)
        .map_err(|e| BackendError::ExecutionFailed(format!("Failed to parse einsum: {e}")))?;

    // Validate tensor ranks against notation
    if lhs.rank() != expr.lhs_indices.len() {
        return Err(BackendError::InvalidDimension(format!(
            "LHS tensor rank {} doesn't match notation {}",
            lhs.rank(),
            expr.lhs_indices.len()
        )));
    }
    if rhs.rank() != expr.rhs_indices.len() {
        return Err(BackendError::InvalidDimension(format!(
            "RHS tensor rank {} doesn't match notation {}",
            rhs.rank(),
            expr.rhs_indices.len()
        )));
    }

    let plan = ContractionPlan::from_expr(&expr);

    // Permute operands so contracted indices are adjacent for GEMM reshape
    let lhs_perm = plan.lhs_permutation(&expr.lhs_indices, &expr.rhs_indices);
    let rhs_perm = plan.rhs_permutation(&expr.rhs_indices);

    let lhs_permuted = if let Some(perm) = lhs_perm {
        transpose(backend, lhs, &perm)?
    } else {
        lhs.clone()
    };

    let rhs_permuted = if let Some(perm) = rhs_perm {
        transpose(backend, rhs, &perm)?
    } else {
        rhs.clone()
    };

    // Compute GEMM dimensions from the original shapes
    let m: usize = plan
        .free_lhs
        .iter()
        .map(|&idx| {
            let pos = expr
                .lhs_indices
                .iter()
                .position(|&x| x == idx)
                .expect("Free index not found in LHS");
            lhs.shape()[pos]
        })
        .product();

    let n: usize = plan
        .free_rhs
        .iter()
        .map(|&idx| {
            let pos = expr
                .rhs_indices
                .iter()
                .position(|&x| x == idx)
                .expect("Free index not found in RHS");
            rhs.shape()[pos]
        })
        .product();

    let k: usize = plan
        .contracted
        .iter()
        .map(|&idx| {
            let pos = expr
                .lhs_indices
                .iter()
                .position(|&x| x == idx)
                .expect("Contracted index not found in LHS");
            lhs.shape()[pos]
        })
        .product();

    // Handle scalar dimensions (empty product = 1)
    let m = m.max(1);
    let n = n.max(1);
    let k = k.max(1);

    // GEMM: C = 1 · A · B + 0 · C
    let mut c_data = vec![T::zero(); m * n];

    let desc = GemmDescriptor {
        m,
        n,
        k,
        alpha: T::one(),
        a: lhs_permuted.data(),
        b: rhs_permuted.data(),
        beta: T::zero(),
        c: &mut c_data,
        trans_a: false,
        trans_b: false,
    };

    backend.gemm(desc)?;

    // Compute output tensor shape from free indices in output order
    let output_shape = compute_output_shape(&plan, &expr, lhs.shape(), rhs.shape());

    Ok(DenseTensor::from_data(c_data, output_shape))
}

/// Derive output tensor shape from contraction plan and original input shapes.
fn compute_output_shape(
    plan: &ContractionPlan,
    expr: &EinsumExpr,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Vec<usize> {
    let mut output_shape = Vec::new();

    for &idx in &plan.free_lhs {
        let pos = expr
            .lhs_indices
            .iter()
            .position(|&x| x == idx)
            .expect("Free LHS index not found");
        output_shape.push(lhs_shape[pos]);
    }

    for &idx in &plan.free_rhs {
        let pos = expr
            .rhs_indices
            .iter()
            .position(|&x| x == idx)
            .expect("Free RHS index not found");
        output_shape.push(rhs_shape[pos]);
    }

    // Scalar result (no free indices) → shape [1]
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    output_shape
}

// ============================================================================
// Scalar operations (backend-free)
// ============================================================================

/// Scale tensor by a scalar factor (out-of-place).
///
/// Returns a new tensor with all elements multiplied by `factor`.
///
/// # Examples
///
/// ```
/// use arnet_linalg::scale;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 3]);
/// let scaled = scale(&tensor, 2.5);
/// assert_eq!(scaled.get(&[0, 0]), 2.5);
/// ```
pub fn scale<T>(tensor: &DenseTensor<T>, factor: T) -> DenseTensor<T>
where
    T: Clone + Mul<Output = T>,
{
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|x| x.clone() * factor.clone())
        .collect();
    DenseTensor::from_data(data, tensor.shape().to_vec())
}

/// Compute the Frobenius norm of a tensor.
///
/// Returns sqrt(sum |element|^2) as a real value.
///
/// # Examples
///
/// ```
/// use arnet_linalg::norm;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let n = norm(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// ```
pub fn norm<T: Scalar>(tensor: &DenseTensor<T>) -> T::Real {
    let sum_sq = tensor
        .data()
        .iter()
        .map(|&x| {
            let a = x.abs();
            a.mul(a)
        })
        .fold(T::Real::zero(), |acc, x| acc.add(x));
    sum_sq.sqrt()
}

/// Normalize a tensor to unit Frobenius norm (out-of-place).
///
/// Returns `(normalized_tensor, original_norm)`.
/// Panics if the tensor has zero norm.
///
/// # Examples
///
/// ```
/// use arnet_linalg::normalize;
/// use arnet_tensor::DenseTensor;
///
/// let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
/// let (normalized, n) = normalize(&tensor);
/// assert!((n - 2.0).abs() < 1e-10);
/// assert!((arnet_linalg::norm(&normalized) - 1.0).abs() < 1e-10);
/// ```
pub fn normalize<T: Scalar>(tensor: &DenseTensor<T>) -> (DenseTensor<T>, T::Real) {
    let n = norm(tensor);
    assert!(n != T::Real::zero(), "Cannot normalize zero tensor");
    let inv_norm = T::Real::one().div(n);
    let data: Vec<T> = tensor
        .data()
        .iter()
        .map(|&x| x.scale_real(inv_norm))
        .collect();
    (DenseTensor::from_data(data, tensor.shape().to_vec()), n)
}

/// Linear combination of tensors: sum_i coefs[i] * tensors[i].
///
/// All tensors must have the same shape.
///
/// # Errors
///
/// Returns an error if tensors have different shapes, the list is empty,
/// or tensors and coefficients have different lengths.
///
/// # Examples
///
/// ```
/// use arnet_linalg::linear_combine;
/// use arnet_tensor::DenseTensor;
///
/// let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
/// let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
///
/// // 2*a + 3*b = 2*1 + 3*2 = 8
/// let result = linear_combine(&[&a, &b], &[2.0, 3.0]).unwrap();
/// assert_eq!(result.get(&[0, 0]), 8.0);
/// ```
pub fn linear_combine<T>(
    tensors: &[&DenseTensor<T>],
    coefs: &[T],
) -> Result<DenseTensor<T>, String>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    if tensors.is_empty() {
        return Err("Cannot combine empty tensor list".to_string());
    }
    if tensors.len() != coefs.len() {
        return Err(format!(
            "Mismatched lengths: {} tensors vs {} coefficients",
            tensors.len(),
            coefs.len()
        ));
    }
    let shape = tensors[0].shape();
    for t in &tensors[1..] {
        if t.shape() != shape {
            return Err("All tensors must have the same shape".to_string());
        }
    }
    let len = tensors[0].len();
    let mut result = vec![T::zero(); len];
    for (tensor, coef) in tensors.iter().zip(coefs) {
        for (r, val) in result.iter_mut().zip(tensor.data()) {
            *r = r.clone() + coef.clone() * val.clone();
        }
    }
    Ok(DenseTensor::from_data(result, shape.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_cpu::CpuBackend;

    // --- Transpose tests ---

    #[test]
    fn test_transpose_f64_2d() {
        let backend = CpuBackend::new();
        let tensor =
            DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_f64_3d() {
        let backend = CpuBackend::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let result = transpose(&backend, &tensor, &[2, 0, 1]).unwrap();

        assert_eq!(result.shape(), &[4, 2, 3]);
        assert_eq!(result.len(), 24);
        // input[0][0][0] = 0 → output[0][0][0]
        assert_eq!(result.get(&[0, 0, 0]), 0.0);
        // input[0][0][1] = 1 → output[1][0][0]
        assert_eq!(result.get(&[1, 0, 0]), 1.0);
    }

    #[test]
    fn test_transpose_f32_2d() {
        let backend = CpuBackend::new();
        let tensor =
            DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_complex_f64_2d() {
        use num_complex::Complex;

        let backend = CpuBackend::new();
        let input = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
            Complex::new(9.0, 10.0),
            Complex::new(11.0, 12.0),
        ];
        let tensor = DenseTensor::from_data(input, vec![2, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.get(&[0, 0]), Complex::new(1.0, 2.0));
        assert_eq!(result.get(&[0, 1]), Complex::new(7.0, 8.0));
        assert_eq!(result.get(&[1, 0]), Complex::new(3.0, 4.0));
        assert_eq!(result.get(&[1, 1]), Complex::new(9.0, 10.0));
    }

    #[test]
    fn test_transpose_empty_tensor() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![], vec![0, 3]);

        let result = transpose(&backend, &tensor, &[1, 0]).unwrap();

        assert_eq!(result.shape(), &[3, 0]);
        assert_eq!(result.len(), 0);
    }

    // --- Contract tests ---

    #[test]
    fn test_contract_matmul() {
        let backend = CpuBackend::new();
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
        let backend = CpuBackend::new();
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
        let backend = CpuBackend::new();
        let a = DenseTensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

        let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]), 19.0f32);
    }

    #[test]
    fn test_contract_with_permutation() {
        let backend = CpuBackend::new();
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
        let backend = CpuBackend::new();
        // A (2×2) × B (2×3) → C (2×3)
        let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);

        let c = contract(&backend, &a, &b, "ik,kj->ij").unwrap();

        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_contract_invalid_notation() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let result = contract(&backend, &a, &b, "ik,kj");
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_rank_mismatch() {
        let backend = CpuBackend::new();
        let a = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = DenseTensor::<f64>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        // 3-index notation with rank-2 tensor
        let result = contract(&backend, &a, &b, "ijk,kl->ijl");
        assert!(result.is_err());
    }

    // --- Scale tests ---

    #[test]
    fn test_scale_f64() {
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let scaled = scale(&tensor, 2.5);
        assert_eq!(scaled.get(&[0, 0]), 2.5);
        assert_eq!(scaled.get(&[0, 1]), 5.0);
        assert_eq!(scaled.get(&[1, 0]), 7.5);
        assert_eq!(scaled.get(&[1, 1]), 10.0);
        // Original unchanged
        assert_eq!(tensor.get(&[0, 0]), 1.0);
    }

    #[test]
    fn test_scale_complex() {
        use num_complex::Complex;
        let tensor = DenseTensor::from_data(
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            vec![2],
        );
        let scaled = scale(&tensor, Complex::new(2.0, 3.0));
        // (1+0i)*(2+3i) = 2+3i
        assert_eq!(scaled.get(&[0]), Complex::new(2.0, 3.0));
        // (0+1i)*(2+3i) = -3+2i
        assert_eq!(scaled.get(&[1]), Complex::new(-3.0, 2.0));
    }

    // --- Norm tests ---

    #[test]
    fn test_norm_f64() {
        let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
        let n = norm(&tensor);
        assert!((n - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_complex() {
        use num_complex::Complex;
        // |3+4i| = 5, so norm of single element [3+4i] = 5
        let tensor = DenseTensor::from_data(vec![Complex::new(3.0, 4.0)], vec![1]);
        let n: f64 = norm(&tensor);
        assert!((n - 5.0).abs() < 1e-10);
    }

    // --- Normalize tests ---

    #[test]
    fn test_normalize_f64() {
        let tensor = DenseTensor::<f64>::ones(vec![2, 2]);
        let (normalized, n) = normalize(&tensor);
        assert!((n - 2.0).abs() < 1e-10);
        assert!((norm(&normalized) - 1.0).abs() < 1e-10);
        // Original unchanged
        assert_eq!(tensor.get(&[0, 0]), 1.0);
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero tensor")]
    fn test_normalize_zero_panics() {
        let tensor = DenseTensor::<f64>::zeros(vec![2, 2]);
        let _ = normalize(&tensor);
    }

    // --- Linear combine tests ---

    #[test]
    fn test_linear_combine_basic() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = DenseTensor::<f64>::constant(vec![2, 2], 2.0);
        let result = linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
        // 3*1 + 4*2 = 11
        assert_eq!(result.get(&[0, 0]), 11.0);
    }

    #[test]
    fn test_linear_combine_shape_mismatch() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        let b = DenseTensor::<f64>::constant(vec![3, 3], 2.0);
        assert!(linear_combine(&[&a, &b], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn test_linear_combine_empty() {
        let result = linear_combine::<f64>(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_combine_length_mismatch() {
        let a = DenseTensor::<f64>::constant(vec![2, 2], 1.0);
        assert!(linear_combine(&[&a], &[1.0, 2.0]).is_err());
    }
}
