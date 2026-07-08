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
    /// Host CPU.
    Cpu,
    /// NVIDIA GPU via CUDA.
    Cuda,
    /// Apple GPU via Metal.
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
    /// Force single-threaded execution.
    Sequential,
    /// Run in parallel with the given target worker count; `0` means
    /// "backend auto" (see the type-level note for per-backend semantics).
    Parallel(usize),
}

/// GEMM operation descriptor
///
/// Data layout (A, B, C slices) is specified by the `order` field.
pub struct GemmDescriptor<'a, T> {
    /// Rows of `op(A)` and of `C`.
    pub m: usize,
    /// Columns of `op(B)` and of `C`.
    pub n: usize,
    /// Contracted dimension: columns of `op(A)` and rows of `op(B)`.
    pub k: usize,
    /// Scalar applied to the `op(A) * op(B)` product.
    pub alpha: T,
    /// Operand `A` (`m×k`, or `k×m` when `trans_a`).
    pub a: &'a [T],
    /// Operand `B` (`k×n`, or `n×k` when `trans_b`).
    pub b: &'a [T],
    /// Scalar applied to the existing `C` before accumulation.
    pub beta: T,
    /// Operand / output `C` (`m×n`), overwritten with the result.
    pub c: &'a mut [T],
    /// Whether `A` is transposed, i.e. `op(A) = Aᵀ`.
    pub trans_a: bool,
    /// Whether `B` is transposed, i.e. `op(B) = Bᵀ`.
    pub trans_b: bool,
    /// Memory layout of the `A` / `B` / `C` slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
    pub policy: ExecPolicy,
}

/// Transpose operation descriptor
pub struct TransposeDescriptor<'a, T> {
    /// Input tensor in `shape` order.
    pub input: &'a [T],
    /// Output buffer receiving the permuted tensor.
    pub output: &'a mut [T],
    /// Shape of the input tensor.
    pub shape: &'a [usize],
    /// Axis permutation: output axis `i` is input axis `perm[i]`.
    pub perm: &'a [usize],
    /// Memory layout of the `input` / `output` slices.
    pub order: MemoryOrder,
    /// Apply element-wise complex conjugation during transpose.
    /// No-op for real types.
    pub conj: bool,
    /// Per-call execution policy.
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
    /// Rows of `A`.
    pub m: usize,
    /// Columns of `A`.
    pub n: usize,
    /// Input matrix `A` (`m×n`).
    pub a: &'a [T],
    /// Output left singular vectors `U` (`m×k`, `k = min(m, n)`).
    pub u: &'a mut [T],
    /// Output singular values `S` (`k` real values).
    pub s: &'a mut [T::Real],
    /// Output right singular vectors `Vᴴ` (`k×n`).
    pub vt: &'a mut [T],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// Rows of `A`.
    pub m: usize,
    /// Columns of `A`.
    pub n: usize,
    /// Input matrix `A` (`m×n`).
    pub a: &'a [T],
    /// Output orthonormal factor `Q` (`m×k`, `k = min(m, n)`).
    pub q: &'a mut [T],
    /// Output upper-triangular factor `R` (`k×n`).
    pub r: &'a mut [T],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// Rows of `A`.
    pub m: usize,
    /// Columns of `A`.
    pub n: usize,
    /// Input matrix `A` (`m×n`).
    pub a: &'a [T],
    /// Output lower-triangular factor `L` (`m×k`, `k = min(m, n)`).
    pub l: &'a mut [T],
    /// Output orthonormal factor `Q` (`k×n`).
    pub q: &'a mut [T],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// Dimension of the square matrix `A`.
    pub n: usize,
    /// Input self-adjoint matrix `A` (`n×n`).
    pub a: &'a [T],
    /// Output eigenvalues `W` (`n` real values, ascending).
    pub w: &'a mut [T::Real],
    /// Output eigenvectors `V` (`n×n`).
    pub v: &'a mut [T],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// Dimension of the square matrix `A`.
    pub n: usize,
    /// Input matrix `A` (`n×n`).
    pub a: &'a [T],
    /// Output complex eigenvalues `W` (`n`).
    pub w: &'a mut [T::Complex],
    /// Output complex right eigenvectors `V` (`n×n`).
    pub v: &'a mut [T::Complex],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// Dimension of the square coefficient matrix `A`.
    pub n: usize,
    /// Number of right-hand-side columns.
    pub nrhs: usize,
    /// Coefficient matrix `A` (`n×n`).
    pub a: &'a [T],
    /// Right-hand side `B` (`n×nrhs`).
    pub b: &'a [T],
    /// Output solution `X` (`n×nrhs`).
    pub x: &'a mut [T],
    /// Memory layout of the matrix slices.
    pub order: MemoryOrder,
    /// Per-call execution policy.
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
    /// GEMM operation.
    Gemm(GemmDescriptor<'a, T>),
    /// Thin SVD operation.
    Svd(SvdDescriptor<'a, T>),
    /// Thin QR operation.
    Qr(QrDescriptor<'a, T>),
    /// Thin LQ operation.
    Lq(LqDescriptor<'a, T>),
    /// Self-adjoint eigendecomposition operation.
    Eigh(EighDescriptor<'a, T>),
    /// General eigendecomposition operation.
    Eig(EigDescriptor<'a, T>),
    /// Linear-solve operation.
    Solve(SolveDescriptor<'a, T>),
    /// Tensor-transpose operation.
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
    /// Run an operation with `f64` scalars.
    fn run_f64(&self, op: OpDesc<'_, f64>) -> Result<(), BackendError>;
    /// Run an operation with `f32` scalars.
    fn run_f32(&self, op: OpDesc<'_, f32>) -> Result<(), BackendError>;
    /// Run an operation with `Complex<f64>` scalars.
    fn run_c64(&self, op: OpDesc<'_, Complex<f64>>) -> Result<(), BackendError>;
    /// Run an operation with `Complex<f32>` scalars.
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
    /// Forward a generic `OpDesc<'_, Self>` to the concrete [`ScalarKernels`]
    /// method matching `Self`, turning a type parameter into a type-level branch.
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

    /// Transpose tensor.
    ///
    /// Unlike the compute kernels (GEMM / SVD / …), whose buffers must be in
    /// [`preferred_order`](Self::preferred_order), transpose is
    /// layout-parametric: `desc.order` is the memory layout of *this* call's
    /// input and output buffers, and implementors must honor it per call
    /// regardless of `preferred_order()`. This lets callers convert between
    /// memory orders (a fixed-shape axis reversal) by transposing under the
    /// source order.
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
