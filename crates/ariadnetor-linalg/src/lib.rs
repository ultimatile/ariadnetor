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
//! - [`trace`]: Partial trace over bond index pairs
//! - [`svd`]: Thin SVD decomposition via backend
//! - [`trunc_svd`]: Truncated SVD with bond dimension control

pub use arnet_core::backend::ComputeBackend;

use arnet_core::backend::{BackendError, GemmDescriptor, SvdDescriptor, TransposeDescriptor};
use arnet_core::einsum::{ContractionPlan, EinsumExpr};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;
use num_traits::{Float, One, ToPrimitive, Zero};
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
            a * a
        })
        .fold(T::Real::zero(), |acc, x| acc + x);
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
    let inv_norm = T::Real::one() / n;
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

/// Partial trace over matched bond index pairs.
///
/// Each pair `(a, b)` traces over two bond indices by summing over
/// their shared diagonal. Paired bonds must have the same dimension.
/// The output tensor retains only the non-paired bonds in their original order.
///
/// # TCI-spec
///
/// Corresponds to `tci::trace` overload (2) (out-of-place).
///
/// # Errors
///
/// Returns an error if:
/// - A bond index is out of range
/// - A bond index appears in more than one pair
/// - Paired bonds have different dimensions
/// - `a == b` in any pair
///
/// # Examples
///
/// ```
/// use arnet_linalg::trace;
/// use arnet_tensor::DenseTensor;
///
/// // Matrix trace: tr([[1,2],[3,4]]) = 1 + 4 = 5
/// let mat = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let result = trace(&mat, &[(0, 1)]).unwrap();
/// assert_eq!(result.shape(), &[1]);
/// assert_eq!(result.get(&[0]), 5.0);
/// ```
pub fn trace<T: Scalar>(
    tensor: &DenseTensor<T>,
    pairs: &[(usize, usize)],
) -> Result<DenseTensor<T>, String> {
    let rank = tensor.rank();
    let shape = tensor.shape();

    // Empty pairs: return a clone
    if pairs.is_empty() {
        return Ok(tensor.clone());
    }

    // Validate pairs
    let mut used = vec![false; rank];
    let mut trace_dims = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if a >= rank || b >= rank {
            return Err(format!(
                "Bond index out of range: ({a}, {b}) for rank {rank}"
            ));
        }
        if a == b {
            return Err(format!("Self-pair not allowed: ({a}, {b})"));
        }
        if used[a] || used[b] {
            return Err(format!(
                "Bond index used in multiple pairs: ({a}, {b})"
            ));
        }
        if shape[a] != shape[b] {
            return Err(format!(
                "Dimension mismatch for pair ({a}, {b}): {} vs {}",
                shape[a], shape[b]
            ));
        }
        used[a] = true;
        used[b] = true;
        trace_dims.push(shape[a]);
    }

    // Free indices: those not in any pair, in original order
    let free_indices: Vec<usize> = (0..rank).filter(|i| !used[*i]).collect();
    let output_shape: Vec<usize> = if free_indices.is_empty() {
        vec![1] // Scalar result
    } else {
        free_indices.iter().map(|&i| shape[i]).collect()
    };
    let output_len: usize = output_shape.iter().product();

    // Precompute input strides (row-major)
    let input_strides = compute_strides(shape);

    // Precompute output strides for coordinate iteration
    let out_iter_shape: Vec<usize> = free_indices.iter().map(|&i| shape[i]).collect();

    // Total number of trace elements per output element
    let trace_total: usize = trace_dims.iter().product();

    let mut result = vec![T::zero(); output_len];

    // Iterate over each output element
    let mut out_coords = vec![0usize; free_indices.len()];
    let mut trace_coords = vec![0usize; pairs.len()];
    for (out_idx, res) in result.iter_mut().enumerate() {
        // Decode output flat index to coordinates
        if !free_indices.is_empty() {
            decode_coords(out_idx, &out_iter_shape, &mut out_coords);
        }

        let mut sum = T::zero();

        // Iterate over cartesian product of trace dimensions
        for trace_idx in 0..trace_total {
            // Decode trace flat index to coordinates
            decode_coords(trace_idx, &trace_dims, &mut trace_coords);

            // Build full input coordinate as flat index
            let mut flat = 0usize;
            for (i, &fi) in free_indices.iter().enumerate() {
                flat += out_coords[i] * input_strides[fi];
            }
            for (p, &(a, b)) in pairs.iter().enumerate() {
                let t = trace_coords[p];
                flat += t * input_strides[a] + t * input_strides[b];
            }

            sum = sum + tensor.data()[flat];
        }

        *res = sum;
    }

    Ok(DenseTensor::from_data(result, output_shape))
}

/// Compute row-major strides from shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Decode a flat index into coordinates given shape (row-major).
fn decode_coords(mut flat: usize, shape: &[usize], coords: &mut [usize]) {
    for i in (0..shape.len()).rev() {
        coords[i] = flat % shape[i];
        flat /= shape[i];
    }
}

// ============================================================================
// Decompositions (backend-dependent)
// ============================================================================

/// Result of a thin SVD decomposition: `(U, S, Vt)`.
///
/// - `U`: Left singular vectors
/// - `S`: Singular values (real-valued, descending)
/// - `Vt`: Right singular vectors transposed
pub type SvdResult<T> = (DenseTensor<T>, DenseTensor<<T as Scalar>::Real>, DenseTensor<T>);

/// Result of a truncated SVD decomposition: `(U, S, Vt, trunc_err)`.
///
/// - `U`: Left singular vectors (truncated)
/// - `S`: Singular values (real-valued, descending, truncated)
/// - `Vt`: Right singular vectors transposed (truncated)
/// - `trunc_err`: Truncation error — Frobenius norm of discarded singular values
pub type TruncSvdResult<T> = (
    DenseTensor<T>,
    DenseTensor<<T as Scalar>::Real>,
    DenseTensor<T>,
    <T as Scalar>::Real,
);

/// Parameters for truncated SVD.
///
/// Controls bond dimension via maximum rank (`chi_max`) and/or
/// target truncation error (`target_trunc_err`). When both are set,
/// the stricter (smaller) bound applies.
#[derive(Debug, Clone)]
pub struct TruncSvdParams {
    /// Maximum number of singular values to keep.
    pub chi_max: Option<usize>,
    /// Target truncation error threshold. Singular values are discarded from
    /// the smallest until the Frobenius norm of discarded values would exceed
    /// this threshold.
    pub target_trunc_err: Option<f64>,
}

/// Compute thin SVD of a tensor reshaped as a matrix.
///
/// The tensor is reshaped to a matrix with the first `nrow` axes
/// forming the row dimension and the remaining axes forming the column dimension.
/// Returns `(U, S, Vt)` where A ≈ U * diag(S) * Vt.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
///
/// # Returns
///
/// * `U` - Left singular vectors, shape `[m, k]` where `m = product(shape[..nrow])`, `k = min(m, n)`
/// * `S` - Singular values (real, descending), shape `[k]`
/// * `Vt` - Right singular vectors transposed, shape `[k, n]` where `n = product(shape[nrow..])`
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn svd<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
) -> Result<SvdResult<T>, BackendError> {
    let shape = tensor.shape();
    let rank = tensor.rank();

    if nrow == 0 || nrow >= rank {
        return Err(BackendError::InvalidDimension(format!(
            "nrow must be in 1..rank, got nrow={nrow} for rank={rank}"
        )));
    }

    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k = m.min(n);

    let mut u_data = vec![T::zero(); m * k];
    let mut s_data = vec![T::Real::zero(); k];
    let mut vt_data = vec![T::zero(); k * n];

    let desc = SvdDescriptor {
        m,
        n,
        a: tensor.data(),
        u: &mut u_data,
        s: &mut s_data,
        vt: &mut vt_data,
    };

    backend.svd(desc)?;

    let u_tensor = DenseTensor::from_data(u_data, vec![m, k]);
    let s_tensor = DenseTensor::from_data(s_data, vec![k]);
    let vt_tensor = DenseTensor::from_data(vt_data, vec![k, n]);

    Ok((u_tensor, s_tensor, vt_tensor))
}

/// Compute truncated SVD of a tensor reshaped as a matrix.
///
/// Performs a full thin SVD via [`svd`], then truncates singular values
/// according to `params`. The truncation keeps at most `chi_max` singular
/// values, and further discards the smallest values whose cumulative
/// Frobenius norm exceeds `target_trunc_err`.
///
/// # Arguments
///
/// * `backend` - Compute backend
/// * `tensor` - Input tensor
/// * `nrow` - Number of leading axes to group as rows (must be in `1..rank`)
/// * `params` - Truncation parameters (`chi_max` and/or `target_trunc_err`)
///
/// # Returns
///
/// * `U` - Left singular vectors, shape `[m, chi]`
/// * `S` - Singular values (real, descending), shape `[chi]`
/// * `Vt` - Right singular vectors transposed, shape `[chi, n]`
/// * `trunc_err` - Frobenius norm of discarded singular values
///
/// # Errors
///
/// Returns `BackendError` if `nrow` is out of range or the backend fails.
pub fn trunc_svd<T: Scalar>(
    backend: &impl ComputeBackend,
    tensor: &DenseTensor<T>,
    nrow: usize,
    params: &TruncSvdParams,
) -> Result<TruncSvdResult<T>, BackendError> {
    let (u_full, s_full, vt_full) = svd(backend, tensor, nrow)?;

    let shape = tensor.shape();
    let m: usize = shape[..nrow].iter().product();
    let n: usize = shape[nrow..].iter().product();
    let k_full = m.min(n);

    // Determine how many singular values to keep
    let mut chi = k_full;

    // Apply chi_max bound
    if let Some(chi_max) = params.chi_max {
        if chi_max == 0 {
            return Err(BackendError::InvalidDimension(
                "chi_max must be at least 1".into(),
            ));
        }
        chi = chi.min(chi_max);
    }

    // Apply target_trunc_err bound: keep the largest singular values
    // such that the norm of discarded values stays within the threshold
    if let Some(target_err) = params.target_trunc_err {
        // Accumulate discarded norm² from the smallest singular value upward.
        // Compare in f64 to avoid precision issues with the user-specified threshold.
        let target_sq = target_err * target_err;
        let s_data = s_full.data();
        let mut discarded_sq = 0.0_f64;
        let mut chi_err = k_full;
        for i in (0..k_full).rev() {
            let si = s_data[i];
            let si_sq: f64 = (si * si).to_f64().unwrap();
            let new_discarded_sq = discarded_sq + si_sq;
            if new_discarded_sq > target_sq {
                break;
            }
            discarded_sq = new_discarded_sq;
            chi_err = i;
        }
        // Ensure at least one singular value is kept even with aggressive error threshold
        chi = chi.min(chi_err).max(1);
    }

    if chi == k_full {
        // No truncation needed
        return Ok((u_full, s_full, vt_full, T::Real::zero()));
    }

    // Compute truncation error: Frobenius norm of discarded singular values
    let s_data = s_full.data();
    let mut err_sq = T::Real::zero();
    for &si in &s_data[chi..] {
        err_sq = err_sq + si * si;
    }
    let trunc_err = err_sq.sqrt();

    // Truncate U: [m, k_full] → [m, chi]
    let u_data = u_full.data();
    let mut u_trunc = vec![T::zero(); m * chi];
    for i in 0..m {
        u_trunc[i * chi..(i + 1) * chi].copy_from_slice(&u_data[i * k_full..i * k_full + chi]);
    }

    // Truncate S: [k_full] → [chi]
    let s_trunc: Vec<T::Real> = s_data[..chi].to_vec();

    // Truncate Vt: [k_full, n] → [chi, n]
    let vt_data = vt_full.data();
    let vt_trunc: Vec<T> = vt_data[..chi * n].to_vec();

    let u_tensor = DenseTensor::from_data(u_trunc, vec![m, chi]);
    let s_tensor = DenseTensor::from_data(s_trunc, vec![chi]);
    let vt_tensor = DenseTensor::from_data(vt_trunc, vec![chi, n]);

    Ok((u_tensor, s_tensor, vt_tensor, trunc_err))
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

    // --- Trace tests ---

    #[test]
    fn test_trace_matrix() {
        // tr([[1,2],[3,4]]) = 1 + 4 = 5
        let mat = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = trace(&mat, &[(0, 1)]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get(&[0]), 5.0);
    }

    #[test]
    fn test_trace_3x3_identity() {
        // tr(I_3) = 3
        let mut data = vec![0.0; 9];
        data[0] = 1.0;
        data[4] = 1.0;
        data[8] = 1.0;
        let mat = DenseTensor::<f64>::from_data(data, vec![3, 3]);
        let result = trace(&mat, &[(0, 1)]).unwrap();
        assert_eq!(result.get(&[0]), 3.0);
    }

    #[test]
    fn test_trace_partial_rank3() {
        // A[i,j,k] shape [2,3,3], trace over (1,2) → B[i] shape [2]
        // B[i] = Σ_j A[i,j,j]
        let mut data = vec![0.0; 18]; // 2*3*3
        // A[0,0,0]=1, A[0,1,1]=2, A[0,2,2]=3 → B[0] = 6
        data[0] = 1.0; // [0,0,0]
        data[4] = 2.0; // [0,1,1]
        data[8] = 3.0; // [0,2,2]
        // A[1,0,0]=4, A[1,1,1]=5, A[1,2,2]=6 → B[1] = 15
        data[9] = 4.0;  // [1,0,0]
        data[13] = 5.0; // [1,1,1]
        data[17] = 6.0; // [1,2,2]
        let tensor = DenseTensor::from_data(data, vec![2, 3, 3]);
        let result = trace(&tensor, &[(1, 2)]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get(&[0]), 6.0);
        assert_eq!(result.get(&[1]), 15.0);
    }

    #[test]
    fn test_trace_tci_example() {
        // TCI spec example: shape {3, 4, 2, 4, 2}, pairs {{1,3}, {2,4}} → shape {3}
        // result[i0] = Σ_{t1,t2} A[i0, t1, t2, t1, t2]
        let shape = vec![3, 4, 2, 4, 2];
        let total: usize = shape.iter().product();
        let mut data = vec![0.0f64; total];

        // Set specific elements to verify correctness
        // A[0, 0, 0, 0, 0] = 1.0
        // A[0, 1, 1, 1, 1] = 2.0
        // result[0] should be 3.0
        let strides = vec![4 * 2 * 4 * 2, 2 * 4 * 2, 4 * 2, 2, 1]; // [64, 16, 8, 2, 1]
        data[0 * strides[0] + 0 * strides[1] + 0 * strides[2] + 0 * strides[3] + 0 * strides[4]] =
            1.0;
        data[0 * strides[0] + 1 * strides[1] + 1 * strides[2] + 1 * strides[3] + 1 * strides[4]] =
            2.0;
        // A[1, 2, 0, 2, 0] = 5.0
        data[1 * strides[0] + 2 * strides[1] + 0 * strides[2] + 2 * strides[3] + 0 * strides[4]] =
            5.0;

        let tensor = DenseTensor::from_data(data, shape);
        let result = trace(&tensor, &[(1, 3), (2, 4)]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get(&[0]), 3.0);
        assert_eq!(result.get(&[1]), 5.0);
        assert_eq!(result.get(&[2]), 0.0);
    }

    #[test]
    fn test_trace_full_contraction() {
        // All bonds paired → scalar
        // A[i,j,i,j] shape [2,3,2,3], pairs [(0,2),(1,3)]
        // result = Σ_{i,j} A[i,j,i,j]
        let shape = vec![2, 3, 2, 3];
        let total: usize = shape.iter().product();
        let mut data = vec![0.0f64; total];
        let strides = vec![18, 6, 3, 1]; // [3*2*3, 2*3, 3, 1]
        // A[0,0,0,0]=1, A[1,1,1,1]=2, A[0,2,0,2]=3
        data[0] = 1.0;
        data[1 * strides[0] + 1 * strides[1] + 1 * strides[2] + 1 * strides[3]] = 2.0;
        data[0 * strides[0] + 2 * strides[1] + 0 * strides[2] + 2 * strides[3]] = 3.0;
        let tensor = DenseTensor::from_data(data, shape);
        let result = trace(&tensor, &[(0, 2), (1, 3)]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get(&[0]), 6.0);
    }

    #[test]
    fn test_trace_empty_pairs() {
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = trace(&tensor, &[]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), tensor.data());
    }

    #[test]
    fn test_trace_dimension_mismatch() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 6], vec![2, 3]);
        assert!(trace(&tensor, &[(0, 1)]).is_err());
    }

    #[test]
    fn test_trace_index_out_of_range() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(trace(&tensor, &[(0, 5)]).is_err());
    }

    #[test]
    fn test_trace_self_pair() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 4], vec![2, 2]);
        assert!(trace(&tensor, &[(1, 1)]).is_err());
    }

    #[test]
    fn test_trace_duplicate_index() {
        let tensor = DenseTensor::<f64>::from_data(vec![0.0; 8], vec![2, 2, 2]);
        assert!(trace(&tensor, &[(0, 1), (1, 2)]).is_err());
    }

    // --- SVD tests ---

    #[test]
    fn test_svd_f64_2d() {
        let backend = CpuBackend::new();
        // A = [[1, 2], [3, 4]] shape [2, 2], nrow=1 → 2×2 matrix
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);

        // Singular values should be positive and descending
        assert!(s.get(&[0]) > s.get(&[1]));
        assert!(s.get(&[1]) >= 0.0);

        // Reconstruct: A ≈ U * diag(S) * Vt
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_f64_rectangular() {
        let backend = CpuBackend::new();
        // shape [2, 3], nrow=1 → 2×3 matrix, k=2
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        let (m, n, k) = (2, 3, 2);
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(s.shape(), &[k]);
        assert_eq!(vt.shape(), &[k, n]);

        // Reconstruct
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_f64_higher_rank() {
        let backend = CpuBackend::new();
        // shape [2, 3, 4], nrow=2 → m=6, n=4, k=4
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let (u, s, vt) = svd(&backend, &tensor, 2).unwrap();

        let (m, n, k) = (6, 4, 4);
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(s.shape(), &[k]);
        assert_eq!(vt.shape(), &[k, n]);

        // Reconstruct and verify against original flat data
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                let orig = tensor.data()[i * n + j];
                assert!(
                    (val - orig).abs() < 1e-9,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {orig}"
                );
            }
        }
    }

    #[test]
    fn test_svd_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let (u, s, vt) = svd(&backend, &tensor, 1).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += u.get(&[i, k]) * s.get(&[k]) * vt.get(&[k, j]);
                }
                assert!(
                    (val - tensor.get(&[i, j])).abs() < 1e-4,
                    "Reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_svd_invalid_nrow() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // nrow=0 is invalid
        assert!(svd(&backend, &tensor, 0).is_err());
        // nrow=rank is invalid
        assert!(svd(&backend, &tensor, 2).is_err());
    }

    // --- Truncated SVD tests ---

    #[test]
    fn test_trunc_svd_chi_max() {
        let backend = CpuBackend::new();
        // 3×4 matrix with rank > 1
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 4],
        );

        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        // Truncated to chi=2
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 4]);

        // Singular values should be positive and descending
        assert!(s.get(&[0]) > s.get(&[1]));
        assert!(s.get(&[1]) > 0.0);

        // Truncation error should be positive (we discarded one singular value)
        assert!(trunc_err > 0.0);

        // Verify truncation error equals the discarded singular value
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();
        let expected_err = s_full.get(&[2]);
        assert!(
            (trunc_err - expected_err).abs() < 1e-10,
            "trunc_err={trunc_err} vs expected={expected_err}"
        );
    }

    #[test]
    fn test_trunc_svd_chi_max_zero_is_error() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let params = TruncSvdParams {
            chi_max: Some(0),
            target_trunc_err: None,
        };
        assert!(trunc_svd(&backend, &tensor, 1, &params).is_err());
    }

    #[test]
    fn test_trunc_svd_chi_max_no_truncation() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // chi_max >= k=2 means no truncation
        let params = TruncSvdParams {
            chi_max: Some(5),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);
        assert_eq!(trunc_err, 0.0);
    }

    #[test]
    fn test_trunc_svd_target_trunc_err() {
        let backend = CpuBackend::new();
        // 4×4 matrix
        let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![4, 4]);

        // Full SVD first to know the singular values
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();

        // Set threshold just above the smallest singular value
        let smallest_sv = s_full.get(&[s_full.len() - 1]);
        let params = TruncSvdParams {
            chi_max: None,
            target_trunc_err: Some(smallest_sv + 1e-10),
        };
        let (_u, s, _vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        // Should have discarded the smallest singular value
        assert!(s.len() < s_full.len());
        // Truncation error should be approximately equal to the discarded singular value
        assert!(trunc_err <= smallest_sv + 1e-10);
    }

    #[test]
    fn test_trunc_svd_both_params() {
        let backend = CpuBackend::new();
        // 4×4 matrix
        let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let tensor = DenseTensor::from_data(data, vec![4, 4]);

        // Full SVD to get singular values
        let (_, s_full, _, _) = trunc_svd(
            &backend,
            &tensor,
            1,
            &TruncSvdParams {
                chi_max: None,
                target_trunc_err: None,
            },
        )
        .unwrap();
        let k_full = s_full.len();

        // chi_max is the binding constraint: target_trunc_err=0 forces keeping all,
        // but chi_max limits to 2
        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: Some(0.0),
        };
        let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        assert_eq!(s.len(), 2);

        // target_trunc_err is the binding constraint: chi_max allows all,
        // but large target_trunc_err allows aggressive truncation
        let params = TruncSvdParams {
            chi_max: Some(k_full),
            target_trunc_err: Some(1e10),
        };
        let (_, s, _, _) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        // Large allowed error → aggressive truncation → minimum 1 value kept
        assert_eq!(s.len(), 1);

        // Neither constraint truncates: chi_max=k_full, target_trunc_err=0
        let params = TruncSvdParams {
            chi_max: Some(k_full),
            target_trunc_err: Some(0.0),
        };
        let (_, s, _, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();
        assert_eq!(s.len(), k_full);
        assert_eq!(trunc_err, 0.0);
    }

    #[test]
    fn test_trunc_svd_f32() {
        let backend = CpuBackend::new();
        let tensor = DenseTensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        let params = TruncSvdParams {
            chi_max: Some(1),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        assert_eq!(u.shape(), &[2, 1]);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 3]);
        assert!(trunc_err > 0.0);
    }

    #[test]
    fn test_trunc_svd_reconstruction() {
        let backend = CpuBackend::new();
        // Verify that truncated reconstruction is a valid low-rank approximation
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 4],
        );

        let params = TruncSvdParams {
            chi_max: Some(2),
            target_trunc_err: None,
        };
        let (u, s, vt, trunc_err) = trunc_svd(&backend, &tensor, 1, &params).unwrap();

        let (m, n, chi) = (3, 4, 2);

        // Reconstruct: A_approx = U * diag(S) * Vt
        let mut recon_err_sq = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..chi {
                    val += u.get(&[i, l]) * s.get(&[l]) * vt.get(&[l, j]);
                }
                let diff = val - tensor.data()[i * n + j];
                recon_err_sq += diff * diff;
            }
        }
        let recon_err = recon_err_sq.sqrt();

        // Reconstruction error should equal the truncation error
        // (Eckart-Young theorem: ||A - A_k||_F = sqrt(sum of discarded σ²))
        assert!(
            (recon_err - trunc_err).abs() < 1e-10,
            "recon_err={recon_err} vs trunc_err={trunc_err}"
        );
    }
}
