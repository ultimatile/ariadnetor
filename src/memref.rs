//! MemRef descriptor marshalling for JIT execution
//!
//! This module provides types and utilities for converting between Rust Tensor
//! data structures and MLIR's MemRef descriptors, which are needed for passing
//! tensor data to JIT-compiled functions.
//!
//! ## MemRef Descriptor Layout
//!
//! After conversion to LLVM dialect, a ranked MemRef has the following C-compatible layout:
//!
//! ```c
//! struct MemRefDescriptor<T, N> {
//!     T* allocated;        // Pointer to allocated memory buffer
//!     T* aligned;          // Pointer to aligned data (same as allocated for our use case)
//!     intptr_t offset;     // Distance to first element (0 for contiguous tensors)
//!     intptr_t sizes[N];   // Dimension sizes
//!     intptr_t strides[N]; // Dimension strides
//! }
//! ```
//!
//! ## Safety
//!
//! MemRef descriptors contain raw pointers and must be carefully managed:
//! - The lifetime of the descriptor must not exceed the lifetime of the source tensor
//! - Pointers must remain valid throughout JIT function execution
//! - Proper alignment must be maintained

use crate::tensor::Tensor;

/// Ranked MemRef descriptor for f64 tensors
///
/// This structure matches MLIR's MemRef descriptor layout after lowering to LLVM.
/// The generic parameter `N` represents the rank of the tensor.
///
/// # Memory Layout
///
/// The layout exactly matches MLIR's expectations:
/// 1. allocated pointer (8 bytes on 64-bit)
/// 2. aligned pointer (8 bytes on 64-bit)
/// 3. offset (8 bytes as i64)
/// 4. sizes array (N * 8 bytes)
/// 5. strides array (N * 8 bytes)
///
/// # Safety
///
/// This structure is `repr(C)` to ensure proper memory layout for FFI.
/// The pointers must remain valid for the lifetime of this descriptor.
#[repr(C)]
pub struct MemRefDescriptor<const N: usize> {
    /// Pointer to the allocated memory buffer
    pub allocated: *mut f64,
    /// Pointer to the aligned data (typically same as allocated)
    pub aligned: *mut f64,
    /// Offset to the first element in number of elements (0 for contiguous)
    pub offset: i64,
    /// Dimension sizes (one entry per dimension)
    pub sizes: [i64; N],
    /// Dimension strides (one entry per dimension)
    pub strides: [i64; N],
}

impl<const N: usize> MemRefDescriptor<N> {
    /// Create a new MemRef descriptor from a Tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Reference to the source tensor (must have rank N)
    ///
    /// # Returns
    ///
    /// A MemRef descriptor pointing to the tensor's data
    ///
    /// # Panics
    ///
    /// Panics if the tensor's rank doesn't match N
    ///
    /// # Safety
    ///
    /// The returned descriptor is only valid as long as the tensor remains alive
    /// and unmodified. The caller must ensure the tensor outlives the descriptor.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tn_mlir::{Tensor, memref::MemRefDescriptor};
    ///
    /// let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let descriptor = MemRefDescriptor::<2>::from_tensor_mut(&mut tensor);
    /// ```
    pub fn from_tensor_mut(tensor: &mut Tensor) -> Self {
        // Clone shape to avoid borrow checker issues
        let shape = tensor.shape().to_vec();
        assert_eq!(
            shape.len(),
            N,
            "Tensor rank {} doesn't match descriptor rank {}",
            shape.len(),
            N
        );

        // Get raw pointer to data
        let data_ptr = tensor.as_mut_ptr();

        // Create sizes array
        let mut sizes = [0i64; N];
        for (i, &size) in shape.iter().enumerate() {
            sizes[i] = size as i64;
        }

        // Calculate strides (row-major layout)
        // For a 2D tensor [2, 3]: strides = [3, 1]
        // For a 3D tensor [2, 3, 4]: strides = [12, 4, 1]
        let mut strides = [0i64; N];
        let mut stride = 1i64;
        for i in (0..N).rev() {
            strides[i] = stride;
            stride *= sizes[i];
        }

        Self {
            allocated: data_ptr,
            aligned: data_ptr,
            offset: 0,
            sizes,
            strides,
        }
    }

    /// Create a MemRef descriptor for an output tensor (allocated but uninitialized)
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor (must have length N)
    ///
    /// # Returns
    ///
    /// A tuple of (MemRefDescriptor, Tensor) where the descriptor points to
    /// the tensor's data
    ///
    /// # Panics
    ///
    /// Panics if the shape length doesn't match N
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (descriptor, tensor) = MemRefDescriptor::<2>::allocate_output(&[2, 2]);
    /// // descriptor can be passed to JIT function
    /// // tensor will receive the results
    /// ```
    pub fn allocate_output(shape: &[usize]) -> (Self, Tensor) {
        assert_eq!(
            shape.len(),
            N,
            "Shape length {} doesn't match descriptor rank {}",
            shape.len(),
            N
        );

        let mut tensor = Tensor::new(shape.to_vec());
        let descriptor = Self::from_tensor_mut(&mut tensor);
        (descriptor, tensor)
    }

    /// Get a raw pointer to this descriptor for FFI
    ///
    /// This is used to pass the descriptor to JIT-compiled functions via invoke_packed.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid as long as `self` remains alive and
    /// at the same memory location (don't move the descriptor after calling this).
    pub fn as_ptr(&self) -> *const Self {
        self as *const Self
    }

    /// Get a mutable raw pointer to this descriptor for FFI
    ///
    /// This is used when the JIT function needs to write to the descriptor
    /// (e.g., for output parameters).
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid as long as `self` remains alive and
    /// at the same memory location (don't move the descriptor after calling this).
    pub fn as_mut_ptr(&mut self) -> *mut Self {
        self as *mut Self
    }
}

// Unranked MemRef descriptor
//
// This is used when the rank is not known at compile time.
// Layout: struct { i64 rank; void* descriptor; }
//
// Note: Currently not implemented as we focus on statically-known ranks.
// This would be needed for truly dynamic tensor operations.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memref_descriptor_2d() {
        let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let descriptor = MemRefDescriptor::<2>::from_tensor_mut(&mut tensor);

        // Check sizes
        assert_eq!(descriptor.sizes, [2, 3]);

        // Check strides (row-major: [3, 1])
        assert_eq!(descriptor.strides, [3, 1]);

        // Check offset
        assert_eq!(descriptor.offset, 0);

        // Check pointers are not null
        assert!(!descriptor.allocated.is_null());
        assert!(!descriptor.aligned.is_null());
        assert_eq!(descriptor.allocated, descriptor.aligned);

        // Verify data is accessible through descriptor
        unsafe {
            assert_eq!(*descriptor.aligned.offset(0), 1.0);
            assert_eq!(*descriptor.aligned.offset(1), 2.0);
            assert_eq!(*descriptor.aligned.offset(5), 6.0);
        }
    }

    #[test]
    fn test_memref_descriptor_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut tensor = Tensor::from_data(data, vec![2, 2, 3]);
        let descriptor = MemRefDescriptor::<3>::from_tensor_mut(&mut tensor);

        // Check sizes
        assert_eq!(descriptor.sizes, [2, 2, 3]);

        // Check strides (row-major: [6, 3, 1])
        assert_eq!(descriptor.strides, [6, 3, 1]);

        // Check offset
        assert_eq!(descriptor.offset, 0);
    }

    #[test]
    fn test_allocate_output() {
        let (descriptor, tensor) = MemRefDescriptor::<2>::allocate_output(&[3, 4]);

        // Check descriptor
        assert_eq!(descriptor.sizes, [3, 4]);
        assert_eq!(descriptor.strides, [4, 1]);
        assert_eq!(descriptor.offset, 0);

        // Check tensor
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.data().len(), 12);
    }

    #[test]
    fn test_pointer_access() {
        let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut descriptor = MemRefDescriptor::<2>::from_tensor_mut(&mut tensor);

        // Get pointers
        let const_ptr = descriptor.as_ptr();
        let mut_ptr = descriptor.as_mut_ptr();

        assert!(!const_ptr.is_null());
        assert!(!mut_ptr.is_null());
    }

    #[test]
    #[should_panic(expected = "doesn't match descriptor rank")]
    fn test_mismatched_rank() {
        let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Try to create a 3D descriptor from a 2D tensor
        let _descriptor = MemRefDescriptor::<3>::from_tensor_mut(&mut tensor);
    }

    #[test]
    fn test_data_modification_through_descriptor() {
        let mut tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let descriptor = MemRefDescriptor::<2>::from_tensor_mut(&mut tensor);

        // Modify data through descriptor
        unsafe {
            *descriptor.aligned.offset(0) = 42.0;
        }

        // Check tensor was modified
        assert_eq!(tensor.get(&[0, 0]), 42.0);
    }
}
