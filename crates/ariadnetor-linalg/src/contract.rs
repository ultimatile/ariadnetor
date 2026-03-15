use arnet_core::backend::{BackendError, ComputeBackend, GemmDescriptor};
use arnet_core::einsum::{ContractionPlan, EinsumExpr};
use arnet_core::scalar::Scalar;
use arnet_tensor::DenseTensor;

use crate::transpose::transpose;

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
