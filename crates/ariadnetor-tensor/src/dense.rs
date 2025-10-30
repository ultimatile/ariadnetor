//! Dense tensor storage
//!
//! Provides a dense tensor with row-major layout and Arc-based shared ownership.

use std::fmt;
use std::sync::Arc;
use num_traits::{Zero, One};

/// Dense tensor with shared ownership (Arc + Copy-on-Write)
///
/// # Type Parameters
///
/// * `T` - Element type (default: f64). Commonly used types:
///   - `f64`: Double precision floating point
///   - `f32`: Single precision floating point
///   - `Complex<f64>`: Double precision complex numbers
///   - `Complex<f32>`: Single precision complex numbers
///
/// # Memory Management
///
/// Uses `Arc<Vec<T>>` for efficient cloning:
/// - Cloning is O(1) (only increments reference count)
/// - Mutation triggers Copy-on-Write if reference count > 1
/// - Ideal for read-heavy workloads
///
/// # Layout
///
/// Data is stored in row-major order (C-contiguous).
/// For a 2D tensor `[2, 3]`:
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// → [a, b, c, d, e, f]
/// ```
#[derive(Clone)]
pub struct DenseTensor<T = f64> {
    /// Shared data buffer (row-major order)
    data: Arc<Vec<T>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Strides for indexing (row-major)
    strides: Vec<usize>,
}

impl<T> DenseTensor<T> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get shape as i64 slice for MLIR compatibility
    ///
    /// MLIR uses i64 for tensor dimensions, so we need conversion from usize.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&s| s as i64).collect()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Create a new tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use arnet_tensor::DenseTensor;
    ///
    /// let tensor = DenseTensor::zeros(vec![10, 20]);
    /// assert_eq!(tensor.shape(), &[10, 20]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Zero,
    {
        let total_elements: usize = shape.iter().product();
        let data = Arc::new(vec![T::zero(); total_elements]);
        let strides = Self::compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with ones
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::ones(vec![5, 5]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Zero,
    {
        let total_elements: usize = shape.iter().product();
        let data = Arc::new(vec![T::one(); total_elements]);
        let strides = Self::compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with a constant value
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::constant(vec![3, 3], 3.14);
    /// ```
    pub fn constant(shape: Vec<usize>, value: T) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = Arc::new(vec![value; total_elements]);
        let strides = Self::compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor from existing data
    ///
    /// # Arguments
    ///
    /// * `data` - Tensor data in row-major order
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the shape
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_elements,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            total_elements
        );

        let strides = Self::compute_strides(&shape);

        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data (triggers CoW if shared)
    ///
    /// # Copy-on-Write
    ///
    /// If the data is shared (Arc reference count > 1), this will clone the data
    /// before returning a mutable reference.
    pub fn data_mut(&mut self) -> &mut [T] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Get element at given indices
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn get(&self, indices: &[usize]) -> T {
        let flat_index = self.flat_index(indices);
        self.data[flat_index].clone()
    }

    /// Set element at given indices (triggers CoW if shared)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, indices: &[usize], value: T) {
        let flat_index = self.flat_index(indices);
        Arc::make_mut(&mut self.data)[flat_index] = value;
    }

    /// Fill tensor with a constant value (triggers CoW if shared)
    pub fn fill(&mut self, value: T) {
        Arc::make_mut(&mut self.data).fill(value);
    }

    /// Get pointer to the underlying data for FFI
    ///
    /// Returns a pointer that can be passed to JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable pointer to the underlying data for FFI (triggers CoW if shared)
    ///
    /// Returns a mutable pointer for writing results from JIT-compiled functions.
    /// The pointer remains valid as long as the DenseTensor is not moved or dropped.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        Arc::make_mut(&mut self.data).as_mut_ptr()
    }

    /// Permute tensor axes with automatic backend selection
    ///
    /// Automatically selects the best available implementation:
    /// - HPTT (if `hptt` feature enabled and type is f32/f64)
    /// - Naive fallback (always available, all types)
    ///
    /// # Arguments
    ///
    /// * `perm` - Permutation of axes (e.g., `[1, 0]` transposes a 2D tensor)
    ///
    /// # Panics
    ///
    /// Panics if the permutation is invalid (wrong length or duplicate indices)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let transposed = tensor.permute(&[1, 0]); // Uses HPTT if available
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self
    where
        T: Clone + Zero + 'static,
    {
        // Use HPTT for f64/f32/Complex when available
        #[cfg(feature = "hptt")]
        {
            use std::any::TypeId;

            let type_id = TypeId::of::<T>();

            if type_id == TypeId::of::<f64>() {
                // Safety: We just checked that T is f64
                let self_f64: &DenseTensor<f64> = unsafe { std::mem::transmute(self) };
                let result_f64 = self_f64.permute_hptt(perm);
                return unsafe { std::mem::transmute(result_f64) };
            }

            if type_id == TypeId::of::<f32>() {
                // Safety: We just checked that T is f32
                let self_f32: &DenseTensor<f32> = unsafe { std::mem::transmute(self) };
                let result_f32 = self_f32.permute_hptt(perm);
                return unsafe { std::mem::transmute(result_f32) };
            }

            if type_id == TypeId::of::<num_complex::Complex<f64>>() {
                // Safety: We just checked that T is Complex<f64>
                let self_c64: &DenseTensor<num_complex::Complex<f64>> =
                    unsafe { std::mem::transmute(self) };
                let result_c64 = self_c64.permute_hptt(perm);
                return unsafe { std::mem::transmute(result_c64) };
            }

            if type_id == TypeId::of::<num_complex::Complex<f32>>() {
                // Safety: We just checked that T is Complex<f32>
                let self_c32: &DenseTensor<num_complex::Complex<f32>> =
                    unsafe { std::mem::transmute(self) };
                let result_c32 = self_c32.permute_hptt(perm);
                return unsafe { std::mem::transmute(result_c32) };
            }
        }

        // Fallback: naive implementation (always available)
        self.permute_naive(perm)
    }

    /// Naive tensor permutation implementation
    ///
    /// This is the fallback implementation that works for all types and devices.
    /// For f32/f64, prefer using `permute()` which automatically selects HPTT.
    pub fn permute_naive(&self, perm: &[usize]) -> Self
    where
        T: Zero,
    {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides = Self::compute_strides(&new_shape);
        let total_elements = self.len();

        let mut result_data = vec![T::zero(); total_elements];

        // Iterate over all elements and reorder
        for old_idx in 0..total_elements {
            let old_coords = self.linear_to_coords(old_idx);
            let new_coords: Vec<usize> = perm.iter().map(|&i| old_coords[i]).collect();
            let new_idx = Self::coords_to_linear(&new_coords, &new_strides);

            result_data[new_idx] = self.data[old_idx].clone();
        }

        Self {
            data: Arc::new(result_data),
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Validate permutation
    fn validate_permutation(&self, perm: &[usize]) {
        assert_eq!(
            perm.len(),
            self.rank(),
            "Permutation length {} doesn't match tensor rank {}",
            perm.len(),
            self.rank()
        );

        let mut seen = vec![false; self.rank()];
        for &i in perm {
            assert!(
                i < self.rank(),
                "Permutation index {} out of range [0, {})",
                i,
                self.rank()
            );
            assert!(!seen[i], "Duplicate index {} in permutation", i);
            seen[i] = true;
        }
    }

    /// Convert linear index to multi-dimensional coordinates
    fn linear_to_coords(&self, idx: usize) -> Vec<usize> {
        let mut coords = vec![0; self.rank()];
        let mut remaining = idx;

        for i in 0..self.rank() {
            coords[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }

        coords
    }

    /// Convert multi-dimensional coordinates to linear index
    fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(&c, &s)| c * s)
            .sum()
    }

    /// Compute strides for row-major layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Convert multi-dimensional indices to flat index
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices {} doesn't match tensor rank {}",
            indices.len(),
            self.shape.len()
        );

        indices
            .iter()
            .zip(&self.shape)
            .for_each(|(&idx, &dim)| {
                assert!(
                    idx < dim,
                    "Index {} out of bounds for dimension {}",
                    idx,
                    dim
                )
            });

        indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Naive tensor contraction using Einstein summation notation
    ///
    /// This implementation performs the entire contraction in a single GEMM operation
    /// (non-pairwise strategy). For large contractions with many indices, this may not be
    /// optimal. Use pairwise contraction strategies (greedy path) for better performance
    /// in those cases.
    ///
    /// # Algorithm
    ///
    /// 1. Parse Einstein notation (e.g., "ijk,jkl->il")
    /// 2. Determine contraction plan (contracted vs free indices)
    /// 3. Permute LHS to [free_lhs..., contracted_in_rhs_order...]
    /// 4. Permute RHS to [contracted_in_rhs_order..., free_rhs...]
    /// 5. Reshape both tensors to 2D matrices
    /// 6. Perform GEMM: C[m,n] = A[m,k] × B[k,n]
    /// 7. Reshape result to final output shape
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side tensor
    /// * `notation` - Einstein summation notation (e.g., "ijk,jkl->il")
    ///
    /// # Returns
    ///
    /// Result tensor after contraction
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use arnet_tensor::DenseTensor;
    ///
    /// let a = DenseTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = DenseTensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    /// let c = a.contract_naive(&b, "ik,kj->ij"); // Matrix multiplication
    /// ```
    pub fn contract_naive(&self, rhs: &Self, notation: &str) -> Self
    where
        T: Clone + Zero + One + std::ops::AddAssign + std::ops::Mul<Output = T> + 'static,
    {
        use crate::einsum::{EinsumExpr, ContractionPlan};

        // Parse einsum notation
        let expr = EinsumExpr::parse(notation)
            .expect("Failed to parse einsum notation");

        // Validate shapes
        assert_eq!(
            self.rank(),
            expr.lhs_indices.len(),
            "LHS tensor rank {} doesn't match notation {}",
            self.rank(),
            expr.lhs_indices.len()
        );
        assert_eq!(
            rhs.rank(),
            expr.rhs_indices.len(),
            "RHS tensor rank {} doesn't match notation {}",
            rhs.rank(),
            expr.rhs_indices.len()
        );

        // Compute contraction plan
        let plan = ContractionPlan::from_expr(&expr);

        // Apply permutations if needed
        let lhs_perm = plan.lhs_permutation(&expr.lhs_indices, &expr.rhs_indices);
        let rhs_perm = plan.rhs_permutation(&expr.rhs_indices);

        let lhs_permuted = if let Some(perm) = lhs_perm {
            self.permute(&perm)
        } else {
            self.clone()
        };

        let rhs_permuted = if let Some(perm) = rhs_perm {
            rhs.permute(&perm)
        } else {
            rhs.clone()
        };

        // Compute dimensions for GEMM
        // LHS: [free_lhs..., contracted...]
        // RHS: [contracted..., free_rhs...]
        // Output: [free_lhs..., free_rhs...]

        let m: usize = plan.free_lhs.iter()
            .map(|&idx| {
                let pos = expr.lhs_indices.iter().position(|&x| x == idx)
                    .expect("Free index not found in LHS");
                self.shape[pos]
            })
            .product();

        let n: usize = plan.free_rhs.iter()
            .map(|&idx| {
                let pos = expr.rhs_indices.iter().position(|&x| x == idx)
                    .expect("Free index not found in RHS");
                rhs.shape[pos]
            })
            .product();

        let k: usize = plan.contracted.iter()
            .map(|&idx| {
                let pos = expr.lhs_indices.iter().position(|&x| x == idx)
                    .expect("Contracted index not found in LHS");
                self.shape[pos]
            })
            .product();

        // Handle edge cases
        let m = if m == 0 { 1 } else { m };
        let n = if n == 0 { 1 } else { n };
        let k = if k == 0 { 1 } else { k };

        // Call type-specific GEMM implementation
        self.gemm_dispatch(&lhs_permuted, &rhs_permuted, m, n, k, &plan, &expr, self.shape(), rhs.shape())
    }

    /// Dispatch GEMM to type-specific implementation
    fn gemm_dispatch(
        &self,
        lhs: &Self,
        rhs: &Self,
        m: usize,
        n: usize,
        k: usize,
        plan: &crate::einsum::ContractionPlan,
        expr: &crate::einsum::EinsumExpr,
        lhs_orig_shape: &[usize],
        rhs_orig_shape: &[usize],
    ) -> Self
    where
        T: Clone + Zero + One + std::ops::AddAssign + std::ops::Mul<Output = T> + 'static,
    {
        use std::any::TypeId;

        let type_id = TypeId::of::<T>();

        // Dispatch to faer for supported types (f64, f32)
        if type_id == TypeId::of::<f64>() {
            // Safety: We just checked that T is f64
            let lhs_f64: &DenseTensor<f64> = unsafe { std::mem::transmute(lhs) };
            let rhs_f64: &DenseTensor<f64> = unsafe { std::mem::transmute(rhs) };
            let result_f64 = lhs_f64.gemm_f64(rhs_f64, m, n, k);
            let reshaped = Self::reshape_output(&result_f64, lhs_orig_shape, rhs_orig_shape, plan, expr);
            return unsafe { std::mem::transmute(reshaped) };
        }

        if type_id == TypeId::of::<f32>() {
            // Safety: We just checked that T is f32
            let lhs_f32: &DenseTensor<f32> = unsafe { std::mem::transmute(lhs) };
            let rhs_f32: &DenseTensor<f32> = unsafe { std::mem::transmute(rhs) };
            let result_f32 = lhs_f32.gemm_f32(rhs_f32, m, n, k);
            let reshaped = Self::reshape_output(&result_f32, lhs_orig_shape, rhs_orig_shape, plan, expr);
            return unsafe { std::mem::transmute(reshaped) };
        }

        // Fallback: naive implementation (for Complex and other types)
        panic!("GEMM not yet implemented for this type");
    }

    /// Reshape output matrix back to tensor shape
    fn reshape_output<U>(
        matrix: &DenseTensor<U>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        plan: &crate::einsum::ContractionPlan,
        expr: &crate::einsum::EinsumExpr,
    ) -> DenseTensor<U>
    where
        U: Clone + Zero,
    {
        // Compute output shape: [free_lhs dimensions..., free_rhs dimensions...]
        let mut output_shape = Vec::new();

        // Add free_lhs dimensions in output order
        for &idx in &plan.free_lhs {
            let pos = expr.lhs_indices.iter().position(|&x| x == idx)
                .expect("Free LHS index not found");
            output_shape.push(lhs_shape[pos]);
        }

        // Add free_rhs dimensions in output order
        for &idx in &plan.free_rhs {
            let pos = expr.rhs_indices.iter().position(|&x| x == idx)
                .expect("Free RHS index not found");
            output_shape.push(rhs_shape[pos]);
        }

        // If no free indices, result is scalar (shape [1])
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        DenseTensor::from_data(matrix.data().to_vec(), output_shape)
    }
}

// HPTT-specific implementations for f64 and f32
#[cfg(feature = "hptt")]
impl DenseTensor<f64> {
    /// Permute tensor using HPTT (high-performance implementation for f64)
    ///
    /// This uses the HPTT library for optimized tensor transposition.
    /// HPTT provides 10-27× speedup over naive implementation through:
    /// - Explicit vectorization (AVX/ARM)
    /// - Cache-aware blocking
    /// - Loop reordering heuristics
    /// - Multi-threading (OpenMP)
    pub fn permute_hptt(&self, perm: &[usize]) -> Self {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let mut output = vec![0.0f64; self.len()];

        hptt::transpose_f64(
            perm,
            1.0,           // alpha
            self.data(),
            &self.shape,
            0.0,           // beta (overwrite)
            &mut output,
            1,             // num_threads (TODO: make configurable)
        )
        .expect("HPTT transpose_f64 failed");

        Self::from_data(output, new_shape)
    }

    /// GEMM operation for f64 tensors using faer
    ///
    /// Performs C = A × B where:
    /// - A is m × k
    /// - B is k × n
    /// - C is m × n
    pub(crate) fn gemm_f64(&self, rhs: &Self, m: usize, n: usize, k: usize) -> Self {
        use faer::{MatRef, Mat};

        // Create faer matrix views from row-major data
        let lhs_view = MatRef::from_row_major_slice(self.data(), m, k);
        let rhs_view = MatRef::from_row_major_slice(rhs.data(), k, n);

        // Perform GEMM using operator
        let output: Mat<f64> = &lhs_view * &rhs_view;

        // Convert back to row-major for DenseTensor
        let result_data: Vec<f64> = (0..m*n)
            .map(|i| {
                let row = i / n;
                let col = i % n;
                output[(row, col)]
            })
            .collect();

        Self::from_data(result_data, vec![m, n])
    }
}

#[cfg(feature = "hptt")]
impl DenseTensor<f32> {
    /// Permute tensor using HPTT (high-performance implementation for f32)
    pub fn permute_hptt(&self, perm: &[usize]) -> Self {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let mut output = vec![0.0f32; self.len()];

        hptt::transpose_f32(
            perm,
            1.0,
            self.data(),
            &self.shape,
            0.0,
            &mut output,
            1,
        )
        .expect("HPTT transpose_f32 failed");

        Self::from_data(output, new_shape)
    }

    /// GEMM operation for f32 tensors using faer
    pub(crate) fn gemm_f32(&self, rhs: &Self, m: usize, n: usize, k: usize) -> Self {
        use faer::{MatRef, Mat};

        let lhs_view = MatRef::from_row_major_slice(self.data(), m, k);
        let rhs_view = MatRef::from_row_major_slice(rhs.data(), k, n);

        let output: Mat<f32> = &lhs_view * &rhs_view;

        let result_data: Vec<f32> = (0..m*n)
            .map(|i| {
                let row = i / n;
                let col = i % n;
                output[(row, col)]
            })
            .collect();

        Self::from_data(result_data, vec![m, n])
    }
}

#[cfg(feature = "hptt")]
impl DenseTensor<num_complex::Complex<f64>> {
    /// Permute tensor using HPTT (high-performance implementation for Complex<f64>)
    pub fn permute_hptt(&self, perm: &[usize]) -> Self {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let mut output = vec![num_complex::Complex::new(0.0, 0.0); self.len()];

        hptt::transpose_c64(
            perm,
            num_complex::Complex::new(1.0, 0.0), // alpha
            self.data(),
            &self.shape,
            num_complex::Complex::new(0.0, 0.0), // beta (overwrite)
            &mut output,
            1,
        )
        .expect("HPTT transpose_c64 failed");

        Self::from_data(output, new_shape)
    }
}

#[cfg(feature = "hptt")]
impl DenseTensor<num_complex::Complex<f32>> {
    /// Permute tensor using HPTT (high-performance implementation for Complex<f32>)
    pub fn permute_hptt(&self, perm: &[usize]) -> Self {
        self.validate_permutation(perm);

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let mut output = vec![num_complex::Complex::new(0.0, 0.0); self.len()];

        hptt::transpose_c32(
            perm,
            num_complex::Complex::new(1.0, 0.0), // alpha
            self.data(),
            &self.shape,
            num_complex::Complex::new(0.0, 0.0), // beta (overwrite)
            &mut output,
            1,
        )
        .expect("HPTT transpose_c32 failed");

        Self::from_data(output, new_shape)
    }
}

impl<T> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseTensor(shape={:?}, elements={})",
            self.shape,
            self.len()
        )
    }
}

impl<T> fmt::Display for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor{:?}", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = DenseTensor::<f64>::zeros(vec![3, 4]);
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.len(), 12);
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DenseTensor::<f64>::from_data(data.clone(), vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &data[..]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut tensor = DenseTensor::<f64>::zeros(vec![3, 4]);
        tensor.set(&[1, 2], 42.0);
        assert_eq!(tensor.get(&[1, 2]), 42.0);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
        tensor.fill(3.14);
        for &val in tensor.data() {
            assert_eq!(val, 3.14);
        }
    }

    #[test]
    fn test_ones() {
        let tensor = DenseTensor::<f64>::ones(vec![2, 3]);
        for &val in tensor.data() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_copy_on_write() {
        let tensor1 = DenseTensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut tensor2 = tensor1.clone(); // Share data

        // Modification triggers CoW
        tensor2.set(&[0, 0], 999.0);

        // tensor1 should be unchanged
        assert_eq!(tensor1.get(&[0, 0]), 1.0);
        assert_eq!(tensor2.get(&[0, 0]), 999.0);
    }

    // Test different numeric types
    #[test]
    fn test_f32_tensor() {
        let tensor = DenseTensor::<f32>::zeros(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);

        let tensor = DenseTensor::<f32>::ones(vec![2, 2]);
        for &val in tensor.data() {
            assert_eq!(val, 1.0f32);
        }
    }

    #[test]
    fn test_complex_f64_tensor() {
        use num_complex::Complex;

        let tensor = DenseTensor::<Complex<f64>>::zeros(vec![2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        for &val in tensor.data() {
            assert_eq!(val, Complex::new(0.0, 0.0));
        }

        let tensor = DenseTensor::<Complex<f64>>::ones(vec![2, 2]);
        for &val in tensor.data() {
            assert_eq!(val, Complex::new(1.0, 0.0));
        }
    }

    #[test]
    fn test_complex_f32_tensor() {
        use num_complex::Complex;

        let mut tensor = DenseTensor::<Complex<f32>>::zeros(vec![2, 2]);
        let c = Complex::new(3.0f32, 4.0f32);
        tensor.set(&[0, 0], c);
        assert_eq!(tensor.get(&[0, 0]), c);
    }

    #[test]
    fn test_constant_with_complex() {
        use num_complex::Complex;

        let c = Complex::new(1.5, 2.5);
        let tensor = DenseTensor::constant(vec![3, 3], c);
        for &val in tensor.data() {
            assert_eq!(val, c);
        }
    }

    #[test]
    fn test_ffi_pointer_types() {
        // Test that as_ptr works for different types
        let tensor_f64 = DenseTensor::<f64>::zeros(vec![10]);
        let _ptr_f64: *const f64 = tensor_f64.as_ptr();

        let tensor_f32 = DenseTensor::<f32>::zeros(vec![10]);
        let _ptr_f32: *const f32 = tensor_f32.as_ptr();

        use num_complex::Complex;
        let tensor_c64 = DenseTensor::<Complex<f64>>::zeros(vec![10]);
        let _ptr_c64: *const Complex<f64> = tensor_c64.as_ptr();

        let tensor_c32 = DenseTensor::<Complex<f32>>::zeros(vec![10]);
        let _ptr_c32: *const Complex<f32> = tensor_c32.as_ptr();
    }

    // Permute tests
    #[test]
    fn test_permute_2d_transpose() {
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );

        // Transpose: [2, 3] -> [3, 2]
        let result = tensor.permute(&[1, 0]);

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.get(&[0, 0]), 1.0);
        assert_eq!(result.get(&[1, 0]), 2.0);
        assert_eq!(result.get(&[2, 0]), 3.0);
        assert_eq!(result.get(&[0, 1]), 4.0);
        assert_eq!(result.get(&[1, 1]), 5.0);
        assert_eq!(result.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_permute_3d() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::<f64>::from_data(data, vec![2, 3, 4]);

        // Permute: [2, 3, 4] -> [4, 2, 3]
        let result = tensor.permute(&[2, 0, 1]);

        assert_eq!(result.shape(), &[4, 2, 3]);
        assert_eq!(result.len(), 24);

        // Verify a few elements
        assert_eq!(result.get(&[0, 0, 0]), tensor.get(&[0, 0, 0]));
        assert_eq!(result.get(&[1, 0, 0]), tensor.get(&[0, 0, 1]));
        assert_eq!(result.get(&[2, 0, 0]), tensor.get(&[0, 0, 2]));
    }

    #[test]
    fn test_permute_identity() {
        let tensor = DenseTensor::<f64>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        // Identity permutation
        let result = tensor.permute(&[0, 1]);

        assert_eq!(result.shape(), tensor.shape());
        assert_eq!(result.data(), tensor.data());
    }

    #[test]
    fn test_permute_f32() {
        let tensor = DenseTensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let result = tensor.permute(&[1, 0]);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get(&[0, 0]), 1.0f32);
        assert_eq!(result.get(&[1, 0]), 2.0f32);
    }

    #[test]
    #[should_panic(expected = "Permutation length 3 doesn't match tensor rank 2")]
    fn test_permute_invalid_length() {
        let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
        tensor.permute(&[0, 1, 2]); // Wrong length
    }

    #[test]
    #[should_panic(expected = "Permutation index 2 out of range")]
    fn test_permute_invalid_index() {
        let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
        tensor.permute(&[0, 2]); // Index 2 out of range
    }

    #[test]
    #[should_panic(expected = "Duplicate index 1 in permutation")]
    fn test_permute_duplicate_index() {
        let tensor = DenseTensor::<f64>::zeros(vec![2, 3]);
        tensor.permute(&[1, 1]); // Duplicate index
    }

    // HPTT-specific tests
    #[cfg(feature = "hptt")]
    #[test]
    fn test_hptt_vs_naive_f64() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let tensor = DenseTensor::<f64>::from_data(data, vec![2, 3, 4]);

        let result_naive = tensor.permute_naive(&[2, 0, 1]);
        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);

        // Results should be identical
        assert_eq!(result_naive.shape(), result_hptt.shape());
        assert_eq!(result_naive.len(), result_hptt.len());

        for i in 0..result_naive.len() {
            assert_eq!(
                result_naive.data()[i],
                result_hptt.data()[i],
                "Mismatch at index {}",
                i
            );
        }
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_hptt_vs_naive_f32() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = DenseTensor::<f32>::from_data(data, vec![2, 3, 4]);

        let result_naive = tensor.permute_naive(&[2, 0, 1]);
        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);

        assert_eq!(result_naive.shape(), result_hptt.shape());
        for i in 0..result_naive.len() {
            assert_eq!(result_naive.data()[i], result_hptt.data()[i]);
        }
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_permute_auto_selects_hptt_f64() {
        let tensor = DenseTensor::<f64>::from_data(
            (0..24).map(|i| i as f64).collect(),
            vec![2, 3, 4],
        );

        // permute() should automatically use HPTT for f64
        let result = tensor.permute(&[2, 0, 1]);
        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);

        assert_eq!(result.shape(), result_hptt.shape());
        assert_eq!(result.data(), result_hptt.data());
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_permute_auto_selects_hptt_f32() {
        let tensor = DenseTensor::<f32>::from_data(
            (0..24).map(|i| i as f32).collect(),
            vec![2, 3, 4],
        );

        // permute() should automatically use HPTT for f32
        let result = tensor.permute(&[2, 0, 1]);
        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);

        assert_eq!(result.shape(), result_hptt.shape());
        assert_eq!(result.data(), result_hptt.data());
    }

    #[test]
    fn test_permute_complex_basic() {
        use num_complex::Complex;

        let data: Vec<Complex<f64>> = (0..4)
            .map(|i| Complex::new(i as f64, (i + 1) as f64))
            .collect();
        let tensor = DenseTensor::from_data(data, vec![2, 2]);

        // permute() should work for Complex (uses HPTT when available, naive otherwise)
        let result = tensor.permute(&[1, 0]);
        let result_naive = tensor.permute_naive(&[1, 0]);

        assert_eq!(result.shape(), result_naive.shape());
        assert_eq!(result.data(), result_naive.data());
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_hptt_large_tensor() {
        // Test with a larger tensor to verify HPTT works correctly
        let size = 10 * 20 * 30;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let tensor = DenseTensor::<f64>::from_data(data, vec![10, 20, 30]);

        let result = tensor.permute(&[2, 0, 1]);
        assert_eq!(result.shape(), &[30, 10, 20]);
        assert_eq!(result.len(), size);

        // Verify a few random elements
        assert_eq!(result.get(&[0, 0, 0]), tensor.get(&[0, 0, 0]));
        assert_eq!(result.get(&[5, 3, 7]), tensor.get(&[3, 7, 5]));
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_hptt_vs_naive_c64() {
        use num_complex::Complex;

        // Test Complex<f64> HPTT vs naive implementation
        let data: Vec<Complex<f64>> = (0..24)
            .map(|i| Complex::new(i as f64, (i + 1) as f64))
            .collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);
        let result_naive = tensor.permute_naive(&[2, 0, 1]);

        assert_eq!(result_hptt.shape(), result_naive.shape());
        assert_eq!(result_hptt.data(), result_naive.data());
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_hptt_vs_naive_c32() {
        use num_complex::Complex;

        // Test Complex<f32> HPTT vs naive implementation
        let data: Vec<Complex<f32>> = (0..24)
            .map(|i| Complex::new(i as f32, (i + 1) as f32))
            .collect();
        let tensor = DenseTensor::from_data(data, vec![2, 3, 4]);

        let result_hptt = tensor.permute_hptt(&[2, 0, 1]);
        let result_naive = tensor.permute_naive(&[2, 0, 1]);

        assert_eq!(result_hptt.shape(), result_naive.shape());
        assert_eq!(result_hptt.data(), result_naive.data());
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_permute_auto_selects_hptt_c64() {
        use num_complex::Complex;

        // Verify that permute() automatically selects HPTT for Complex<f64>
        let data: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let tensor = DenseTensor::from_data(data, vec![2, 2]);

        let result = tensor.permute(&[1, 0]);
        let result_naive = tensor.permute_naive(&[1, 0]);

        assert_eq!(result.shape(), result_naive.shape());
        assert_eq!(result.data(), result_naive.data());
    }

    #[cfg(feature = "hptt")]
    #[test]
    fn test_permute_auto_selects_hptt_c32() {
        use num_complex::Complex;

        // Verify that permute() automatically selects HPTT for Complex<f32>
        let data: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let tensor = DenseTensor::from_data(data, vec![2, 2]);

        let result = tensor.permute(&[1, 0]);
        let result_naive = tensor.permute_naive(&[1, 0]);

        assert_eq!(result.shape(), result_naive.shape());
        assert_eq!(result.data(), result_naive.data());
    }

    #[test]
    fn test_contract_matmul() {
        // Test matrix multiplication: C[i,j] = A[i,k] * B[k,j]
        let a = DenseTensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let b = DenseTensor::from_data(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
        );

        let c = a.contract_naive(&b, "ik,kj->ij");

        // Expected: [[1*5 + 2*7, 1*6 + 2*8],
        //            [3*5 + 4*7, 3*6 + 4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_contract_tensor_contraction() {
        // Test 3D tensor contraction: C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]
        let a = DenseTensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2], // i=2, j=2, k=2
        );
        let b = DenseTensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2], // j=2, k=2, l=2
        );

        let c = a.contract_naive(&b, "ijk,jkl->il");

        assert_eq!(c.shape(), &[2, 2]);
        // Verify it's non-zero (exact values depend on contraction)
        assert_ne!(c.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_contract_f32() {
        // Test that contract works with f32
        let a = DenseTensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let b = DenseTensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
        );

        let c = a.contract_naive(&b, "ik,kj->ij");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]), 19.0f32);
    }

    #[test]
    fn test_contract_with_permutation() {
        // Test case where permutation is needed: A[i,k,j] * B[k,j] -> C[i]
        let a = DenseTensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2], // i=2, k=2, j=2
        );
        let b = DenseTensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2], // k=2, j=2
        );

        let c = a.contract_naive(&b, "ikj,kj->i");

        assert_eq!(c.shape(), &[2]);
        // Result should be non-zero
        assert_ne!(c.get(&[0]), 0.0);
    }
}
