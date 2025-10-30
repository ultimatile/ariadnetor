//! MemRef descriptor for MLIR FFI
//!
//! This module provides the MemRefDescriptor structure that matches
//! MLIR's MemRef layout for FFI calls.

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
    /// Create a new MemRef descriptor from raw parts
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `data_ptr` points to valid memory of sufficient size
    /// - `sizes` and `strides` correctly describe the memory layout
    /// - The pointer remains valid for the lifetime of this descriptor
    pub unsafe fn from_raw_parts(data_ptr: *mut f64, sizes: [i64; N], strides: [i64; N]) -> Self {
        Self {
            allocated: data_ptr,
            aligned: data_ptr,
            offset: 0,
            sizes,
            strides,
        }
    }
}
