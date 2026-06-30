//! Test-only compute backend that records descriptor `ExecPolicy` values
//! while delegating the actual work to a `NativeBackend`.
//!
//! Used to verify policy forwarding: BSp default wrappers hardcode
//! `Sequential` and BSp `_with_policy` wrappers forward the caller's policy
//! into every per-sector dense or gemm call.

use std::sync::Mutex;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{
    BackendError, ComputeBackend, DeviceType, EigDescriptor, EighDescriptor, ExecPolicy,
    GemmDescriptor, LqDescriptor, MemoryOrder, QrDescriptor, SolveDescriptor, SvdDescriptor,
    TransposeDescriptor,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{BlockSparseStorage, DenseStorage, OpsFor};

/// Compute backend that records the `policy` field of every descriptor it
/// receives, then delegates to an inner `NativeBackend`.
///
/// One `Mutex<Vec<ExecPolicy>>` per op — tests poll whichever list matches
/// the op under inspection. Other ops' lists stay empty.
pub(crate) struct RecordingBackend {
    inner: NativeBackend,
    pub gemm_policies: Mutex<Vec<ExecPolicy>>,
    pub svd_policies: Mutex<Vec<ExecPolicy>>,
    pub qr_policies: Mutex<Vec<ExecPolicy>>,
    pub lq_policies: Mutex<Vec<ExecPolicy>>,
    pub eigh_policies: Mutex<Vec<ExecPolicy>>,
    pub eig_policies: Mutex<Vec<ExecPolicy>>,
    pub solve_policies: Mutex<Vec<ExecPolicy>>,
    pub transpose_policies: Mutex<Vec<ExecPolicy>>,
}

impl RecordingBackend {
    pub(crate) fn new() -> Self {
        Self {
            inner: NativeBackend::new(),
            gemm_policies: Mutex::new(Vec::new()),
            svd_policies: Mutex::new(Vec::new()),
            qr_policies: Mutex::new(Vec::new()),
            lq_policies: Mutex::new(Vec::new()),
            eigh_policies: Mutex::new(Vec::new()),
            eig_policies: Mutex::new(Vec::new()),
            solve_policies: Mutex::new(Vec::new()),
            transpose_policies: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn svd_recorded(&self) -> Vec<ExecPolicy> {
        self.svd_policies.lock().unwrap().clone()
    }

    pub(crate) fn qr_recorded(&self) -> Vec<ExecPolicy> {
        self.qr_policies.lock().unwrap().clone()
    }

    pub(crate) fn lq_recorded(&self) -> Vec<ExecPolicy> {
        self.lq_policies.lock().unwrap().clone()
    }

    pub(crate) fn gemm_recorded(&self) -> Vec<ExecPolicy> {
        self.gemm_policies.lock().unwrap().clone()
    }
}

impl ComputeBackend for RecordingBackend {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn preferred_order(&self) -> MemoryOrder {
        self.inner.preferred_order()
    }

    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        self.gemm_policies.lock().unwrap().push(desc.policy);
        self.inner.gemm(desc)
    }

    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        self.transpose_policies.lock().unwrap().push(desc.policy);
        self.inner.transpose(desc)
    }

    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        self.svd_policies.lock().unwrap().push(desc.policy);
        self.inner.svd(desc)
    }

    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        self.qr_policies.lock().unwrap().push(desc.policy);
        self.inner.qr(desc)
    }

    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        self.lq_policies.lock().unwrap().push(desc.policy);
        self.inner.lq(desc)
    }

    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        self.eigh_policies.lock().unwrap().push(desc.policy);
        self.inner.eigh(desc)
    }

    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        self.eig_policies.lock().unwrap().push(desc.policy);
        self.inner.eig(desc)
    }

    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        self.solve_policies.lock().unwrap().push(desc.policy);
        self.inner.solve(desc)
    }

    // par_for_* methods intentionally left at the trait defaults (all
    // Sequential). The tests that exercise `svd_block_sparse(..)` etc. call
    // the BSp default wrapper, which hardcodes Sequential without consulting
    // par_for_*, so the trait default matches the BSp wrapper's intent. Tests
    // for the expert wrapper pass policy explicitly.
}

/// Test-only backend that returns `MemoryOrder::RowMajor` from
/// `preferred_order()` so RM-only branches in layout-aware ops can be
/// exercised. The production `NativeBackend` is column-major, leaving the
/// RM branch otherwise unreachable from tests.
///
/// All kernels delegate to the inner `NativeBackend`. Among the inner
/// backend's ops, only GEMM and transpose honor the descriptor's `order`
/// field for both `RowMajor` and `ColumnMajor`; the decomposition family
/// (SVD, QR, LQ, eigh, eig, solve) is column-major only, so a descriptor
/// constructed with `order: MemoryOrder::RowMajor` and dispatched through
/// this wrapper returns `BackendError::InvalidArgument`. Callers must
/// pass descriptors and buffers consistent with the memory order they
/// expect — this backend only forces RM-branch selection in layout-aware
/// code that dispatches on `preferred_order()`.
pub(crate) struct RowMajorBackend {
    inner: NativeBackend,
}

impl RowMajorBackend {
    pub(crate) fn new() -> Self {
        Self {
            inner: NativeBackend::new(),
        }
    }
}

impl ComputeBackend for RowMajorBackend {
    fn name(&self) -> &'static str {
        "row-major-test"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn preferred_order(&self) -> MemoryOrder {
        MemoryOrder::RowMajor
    }

    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.gemm(desc)
    }

    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.transpose(desc)
    }

    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.svd(desc)
    }

    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.qr(desc)
    }

    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.lq(desc)
    }

    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.eigh(desc)
    }

    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.eig(desc)
    }

    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        self.inner.solve(desc)
    }
}

// `RecordingBackend` delegates every kernel to a full `NativeBackend`, so it
// genuinely supports both storage flavors and declares the capability exactly
// as an out-of-tree backend would (`OpsFor` is deliberately unsealed). This
// lets it be passed to the `OpsFor`-gated twins it exercises.
//
// `RowMajorBackend` deliberately does NOT declare `OpsFor`: it is a partial
// backend (its row-major decomposition paths are `todo!`), used only to force
// the row-major branch of the `pub(crate)` `*_dense` kernels, which take a plain
// `ComputeBackend`. It is never handed to a gated public twin, so claiming the
// capability would advertise support it does not have.
impl<T: Scalar> OpsFor<DenseStorage<T>> for RecordingBackend {}
impl<T: Scalar> OpsFor<BlockSparseStorage<T>> for RecordingBackend {}
