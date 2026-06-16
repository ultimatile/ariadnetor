//! Pluggable compute backend trait.
//!
//! [`ComputeBackend`] unifies the numerical primitives the algorithm
//! layer needs (GEMM, transpose, SVD / QR / LQ / eigh / eig / solve)
//! behind a single trait so the algorithm layer never names a
//! concrete backend. Each backend declares its identity through the
//! `name` / `device_type` / `preferred_order` accessors and then
//! overrides only the operations it actually supports — the default
//! implementations return [`BackendError::NotSupported`], so a
//! partial backend still compiles. Per-call parallelism is selected
//! by the caller through [`ExecPolicy`] and shaped by the
//! per-operation `par_for_*` hooks; see those docstrings for how a
//! given backend interprets `Parallel(n)`.

use crate::scalar::Scalar;
use num_complex::Complex;

/// Device type for backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

/// Memory layout order for tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// Row-major (C order): last axis varies fastest.
    RowMajor,
    /// Column-major (Fortran order): first axis varies fastest.
    ColumnMajor,
}

/// Per-call execution policy for a compute backend operation.
///
/// `Parallel(0)` means "backend auto" — faer uses rayon's
/// `current_num_threads`, while HPTT resolves `0` via
/// `std::thread::available_parallelism()` before crossing the FFI
/// boundary (HPTT 0.4 rejects a literal `0`). `Parallel(n)` with
/// `n > 0` is a target worker count whose strictness depends on the
/// backend: HPTT spawns exactly `n` OpenMP threads, while faer and
/// the naive Rayon kernel treat `n` as a partitioning hint dispatched
/// on the global Rayon pool. `Sequential` forces single-threaded
/// execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecPolicy {
    Sequential,
    Parallel(usize),
}

/// GEMM operation descriptor
///
/// Data layout (A, B, C slices) is specified by the `order` field.
pub struct GemmDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: T,
    pub a: &'a [T],
    pub b: &'a [T],
    pub beta: T,
    pub c: &'a mut [T],
    pub trans_a: bool,
    pub trans_b: bool,
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// Transpose operation descriptor
pub struct TransposeDescriptor<'a, T> {
    pub input: &'a [T],
    pub output: &'a mut [T],
    pub shape: &'a [usize],
    pub perm: &'a [usize],
    pub order: MemoryOrder,
    /// Apply element-wise complex conjugation during transpose.
    /// No-op for real types.
    pub conj: bool,
    pub policy: ExecPolicy,
}

/// Thin SVD operation descriptor: A = U * diag(S) * Vt
///
/// Computes the thin SVD of an m×n matrix A.
/// Data layout (A, U, Vt slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Outputs: U (m×k), S (k singular values), Vt (k×n)
/// where k = min(m, n).
pub struct SvdDescriptor<'a, T: Scalar> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub u: &'a mut [T],
    pub s: &'a mut [T::Real],
    pub vt: &'a mut [T],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// Thin QR decomposition descriptor: A = Q * R
///
/// Computes the thin QR of an m×n matrix A.
/// Data layout (A, Q, R slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Outputs: Q (m×k), R (k×n)
/// where k = min(m, n).
pub struct QrDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub q: &'a mut [T],
    pub r: &'a mut [T],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// Thin LQ decomposition descriptor: A = L * Q
///
/// Computes the thin LQ of an m×n matrix A.
/// Data layout (A, L, Q slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Outputs: L (m×k), Q (k×n)
/// where k = min(m, n).
pub struct LqDescriptor<'a, T> {
    pub m: usize,
    pub n: usize,
    pub a: &'a [T],
    pub l: &'a mut [T],
    pub q: &'a mut [T],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// Self-adjoint eigenvalue decomposition descriptor: A = V * diag(W) * V^H
///
/// Computes eigenvalues and eigenvectors of an n×n self-adjoint matrix A.
/// Data layout (A, V slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Outputs: W (n real eigenvalues, ascending), V (n×n eigenvectors)
pub struct EighDescriptor<'a, T: Scalar> {
    pub n: usize,
    pub a: &'a [T],
    pub w: &'a mut [T::Real],
    pub v: &'a mut [T],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// General eigenvalue decomposition descriptor
///
/// Computes eigenvalues and right eigenvectors of an n×n matrix A.
/// Data layout (A, V slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Outputs are always complex: W (n complex eigenvalues), V (n×n eigenvectors)
pub struct EigDescriptor<'a, T: Scalar> {
    pub n: usize,
    pub a: &'a [T],
    pub w: &'a mut [T::Complex],
    pub v: &'a mut [T::Complex],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// Linear solve descriptor: AX = B via LU decomposition
///
/// Solves the system AX = B where A is an n×n matrix and B is n×nrhs.
/// Data layout (A, B, X slices) is specified by the `order` field;
/// a backend that does not support a given order returns
/// [`BackendError::InvalidArgument`].
/// Output X is written to `x` (n×nrhs).
pub struct SolveDescriptor<'a, T> {
    pub n: usize,
    pub nrhs: usize,
    pub a: &'a [T],
    pub b: &'a [T],
    pub x: &'a mut [T],
    pub order: MemoryOrder,
    pub policy: ExecPolicy,
}

/// One generic-descriptor backend operation, tagged by which op it is.
///
/// This is the unit that [`DispatchScalar::dispatch_op`] carries from a generic
/// `T: Scalar` context down to a concrete per-type kernel. Bundling every op
/// into one enum lets a backend expose a single typed entry point per scalar
/// (see [`ScalarKernels`]) instead of one per `(op, type)` pair, which is what
/// makes type-directed dispatch possible without reinterpreting a
/// `Descriptor<T>` into a `Descriptor<concrete>` through `unsafe`.
pub enum OpDesc<'a, T: Scalar> {
    Gemm(GemmDescriptor<'a, T>),
    Svd(SvdDescriptor<'a, T>),
    Qr(QrDescriptor<'a, T>),
    Lq(LqDescriptor<'a, T>),
    Eigh(EighDescriptor<'a, T>),
    Eig(EigDescriptor<'a, T>),
    Solve(SolveDescriptor<'a, T>),
    Transpose(TransposeDescriptor<'a, T>),
}

/// A backend's concrete per-scalar kernels, one entry point per supported type.
///
/// [`DispatchScalar::dispatch_op`] resolves a generic `OpDesc<'_, T>` to exactly
/// one of these methods, so inside each method the scalar is concrete and the op
/// match dispatches to a monomorphic kernel directly. A backend implements this
/// on a local kernel-set type; the four methods mirror the four sealed [`Scalar`]
/// types.
pub trait ScalarKernels {
    fn run_f64(&self, op: OpDesc<'_, f64>) -> Result<(), BackendError>;
    fn run_f32(&self, op: OpDesc<'_, f32>) -> Result<(), BackendError>;
    fn run_c64(&self, op: OpDesc<'_, Complex<f64>>) -> Result<(), BackendError>;
    fn run_c32(&self, op: OpDesc<'_, Complex<f32>>) -> Result<(), BackendError>;
}

/// Type-directed dispatch hook: reach a concrete per-type kernel from a generic
/// `T: Scalar`.
///
/// A backend method bounded only by `T: Scalar` cannot name a per-type kernel
/// directly. This supertrait of [`Scalar`] lets it call `T::dispatch_op(...)`,
/// which forwards to the matching [`ScalarKernels`] method where the scalar is
/// concrete — a type-level branch in place of an `unsafe`
/// `Descriptor<T>` -> `Descriptor<concrete>` reinterpretation. It is dispatch
/// plumbing between [`ComputeBackend`] and a backend's [`ScalarKernels`], not a
/// user entry point.
///
/// It is kept separate from [`Scalar`] so that `Scalar`'s own method list carries
/// no backend descriptor / error / kernel types; the supertrait bound still makes
/// every `Scalar` a `DispatchScalar`. The `where Self: Scalar` bound (rather than
/// `trait DispatchScalar: Scalar`) avoids a cycle with that supertrait while still
/// admitting `OpDesc<'_, Self>`, which requires `Self: Scalar`. Sealed: only the
/// four built-in scalar types implement it.
pub trait DispatchScalar: sealed::Sealed {
    fn dispatch_op<K: ScalarKernels>(kernels: &K, op: OpDesc<'_, Self>) -> Result<(), BackendError>
    where
        Self: Scalar;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

impl DispatchScalar for f64 {
    #[inline]
    fn dispatch_op<K: ScalarKernels>(
        kernels: &K,
        op: OpDesc<'_, Self>,
    ) -> Result<(), BackendError> {
        kernels.run_f64(op)
    }
}

impl DispatchScalar for f32 {
    #[inline]
    fn dispatch_op<K: ScalarKernels>(
        kernels: &K,
        op: OpDesc<'_, Self>,
    ) -> Result<(), BackendError> {
        kernels.run_f32(op)
    }
}

impl DispatchScalar for Complex<f64> {
    #[inline]
    fn dispatch_op<K: ScalarKernels>(
        kernels: &K,
        op: OpDesc<'_, Self>,
    ) -> Result<(), BackendError> {
        kernels.run_c64(op)
    }
}

impl DispatchScalar for Complex<f32> {
    #[inline]
    fn dispatch_op<K: ScalarKernels>(
        kernels: &K,
        op: OpDesc<'_, Self>,
    ) -> Result<(), BackendError> {
        kernels.run_c32(op)
    }
}

/// Pluggable compute backend trait
pub trait ComputeBackend: Send + Sync {
    /// Backend name
    fn name(&self) -> &'static str;

    /// Device type
    fn device_type(&self) -> DeviceType;

    /// Preferred memory order for this backend's data layout.
    ///
    /// Descriptor data (input/output slices) is expected in this order.
    /// The linalg layer converts tensors to this order before constructing descriptors.
    ///
    /// This is an **implementor-facing contract**, not a user entry point:
    /// backend implementors must report the layout their kernels assume so
    /// the linalg / algorithm layers can normalize to it. End users never
    /// call it — the public `Tensor` surface hides memory layout entirely.
    fn preferred_order(&self) -> MemoryOrder;

    /// Check if backend is available
    fn is_available(&self) -> bool {
        true
    }

    /// GEMM: C = alpha * A * B + beta * C
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError>;

    /// Transpose tensor
    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError>;

    /// Thin SVD: A = U * diag(S) * Vt
    fn svd<T: Scalar>(&self, _desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("svd".into()))
    }

    /// Thin QR: A = Q * R
    fn qr<T: Scalar>(&self, _desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("qr".into()))
    }

    /// Thin LQ: A = L * Q
    fn lq<T: Scalar>(&self, _desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("lq".into()))
    }

    /// Self-adjoint eigenvalue decomposition: A = V * diag(W) * V^H
    fn eigh<T: Scalar>(&self, _desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("eigh".into()))
    }

    /// General eigenvalue decomposition
    fn eig<T: Scalar>(&self, _desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("eig".into()))
    }

    /// Linear solve: AX = B via LU decomposition
    fn solve<T: Scalar>(&self, _desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        Err(BackendError::NotSupported("solve".into()))
    }

    /// Recommended execution policy for SVD at the given problem size.
    ///
    /// Default returns `Sequential`; performance-oriented backends (e.g. `NativeBackend`)
    /// override this with a hardware-aware threshold table.
    fn par_for_svd(&self, _m: usize, _n: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for QR at the given problem size.
    fn par_for_qr(&self, _m: usize, _n: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for LQ at the given problem size.
    fn par_for_lq(&self, _m: usize, _n: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for self-adjoint eigendecomposition.
    fn par_for_eigh(&self, _n: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for general eigendecomposition.
    fn par_for_eig(&self, _n: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for GEMM at the given problem size.
    fn par_for_gemm(&self, _m: usize, _n: usize, _k: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for linear solve.
    fn par_for_solve(&self, _n: usize, _nrhs: usize) -> ExecPolicy {
        ExecPolicy::Sequential
    }

    /// Recommended execution policy for tensor transpose.
    fn par_for_transpose(&self, _shape: &[usize]) -> ExecPolicy {
        ExecPolicy::Sequential
    }
}

/// Error originating from a compute backend.
///
/// All variants represent conditions detected by or attributed to the backend.
/// Linalg-layer validation (nrow range, square matrix checks, etc.) should use
/// a separate error mechanism, not `BackendError`.
///
/// Every variant carries its full context in its own `Display` message; none
/// wraps a structured inner error. `BackendError` is therefore a leaf in the
/// error chain — its `source()` is always `None`.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The backend does not support this operation.
    ///
    /// Returned when an operation is fundamentally unavailable on this backend
    /// (e.g., a GPU backend that lacks an eigenvalue solver). Upper layers
    /// should consider fallback strategies or alternative computation paths.
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// The descriptor passed to the backend violates its contract.
    ///
    /// This indicates a bug in the calling layer (typically linalg), not a user
    /// error. For example, buffer sizes inconsistent with declared dimensions.
    /// Callers should treat this as a panic-worthy condition in debug builds.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// The computation failed at runtime.
    ///
    /// The operation was supported and the arguments were valid, but execution
    /// failed due to numerical issues, resource exhaustion, or other runtime
    /// conditions (e.g., a matrix factorization that fails to converge).
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
}
