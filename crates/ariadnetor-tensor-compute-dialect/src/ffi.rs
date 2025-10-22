//! FFI bindings to Tensor Compute Dialect C API

use std::ffi::c_void;

/// Opaque handle to a dialect
#[repr(C)]
pub struct MlirDialectHandle {
    ptr: *mut c_void,
}

// NOTE: Removed static linking to avoid global initialization conflicts with ExecutionEngine
// Linking is now handled by build.rs with dynamic linking
unsafe extern "C" {
    /// Returns a handle to the TC dialect for registration
    pub fn mlirGetDialectHandle__tc__() -> MlirDialectHandle;
}
