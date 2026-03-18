//! Runtime functions for tensor operations
//!
//! GEMM and transpose are now provided via `ComputeBackend` trait
//! (see `ariadnetor-native` for the native implementation using faer/HPTT).
//!
//! This module will host C ABI shims for JIT-compiled code in a future phase.
