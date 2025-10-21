//! FFI bindings to TN-Compute Dialect C API

use std::ffi::c_void;

/// Opaque handle to a dialect
#[repr(C)]
pub struct MlirDialectHandle {
    ptr: *mut c_void,
}

// NOTE: Removed static linking to avoid global initialization conflicts with ExecutionEngine
// Linking is now handled by build.rs with dynamic linking
extern "C" {
    /// Returns a handle to the TN dialect for registration
    pub fn mlirGetDialectHandle__tn__() -> MlirDialectHandle;
}
