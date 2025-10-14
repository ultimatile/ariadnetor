//! FFI bindings to TN-Compute Dialect C API

use std::ffi::c_void;

/// Opaque handle to a dialect
#[repr(C)]
pub struct MlirDialectHandle {
    ptr: *mut c_void,
}

#[link(name = "MLIRTNCAPI", kind = "static")]
extern "C" {
    /// Returns a handle to the TN dialect for registration
    pub fn mlirGetDialectHandle__tn__() -> MlirDialectHandle;
}
