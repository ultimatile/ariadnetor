//! Authority-routing tests for the call-site-backend operation surface.
//!
//! Every MPS / MPO operation takes its compute backend explicitly at the
//! call site; the chain and its site tensors carry no backend. These
//! tests supply a [`CountingBackend`] at the call site and assert that
//! the kernels actually dispatch through it. That guards against a
//! regression where an operation ignores the supplied backend and
//! silently routes kernels through some other handle (e.g. a hardcoded
//! `Host`).
//!
//! Allocation-only operations (`diagonal_scale`, permute / fuse)
//! dispatch no counted kernel, so their routing is proven per twin by
//! the pointer-identity tests in the linalg crate, not here. `inner` /
//! `norm` / `braket` route their contractions through the supplied
//! backend and so are observable, but the kernel-dispatching operations
//! that historically derived authority from an operand tensor —
//! `canonicalize` / `truncate` (per-site decompositions) and `apply`
//! (the MPO-MPS contraction) — are the load-bearing cases and are the
//! ones exercised below.

use arnet_tensor::{ComputeBackendTensorExt, Host};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use arnet_core::backend::{
    BackendError, DeviceType, EigDescriptor, EighDescriptor, GemmDescriptor, LqDescriptor,
    MemoryOrder, QrDescriptor, SolveDescriptor, SvdDescriptor, TransposeDescriptor,
};
use arnet_core::{ComputeBackend, Scalar};
use arnet_linalg::TruncSvdParams;
use arnet_mps::{
    ApplyMethod, CanonicalForm, Mpo, Mps, TensorChain, TruncateParams, apply, apply_with_method,
    canonicalize, truncate,
};
use arnet_native::NativeBackend;
use arnet_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, DenseLayout,
    DenseStorage, DenseTensor, Direction, OpsFor, QNIndex, U1Sector,
};

// ---------------------------------------------------------------------------
// CountingBackend: NativeBackend delegate that counts kernel dispatches
// ---------------------------------------------------------------------------

/// Delegates every kernel to an inner [`NativeBackend`] and counts the
/// dispatches. Policy queries (`par_for_*`) carry no compute authority,
/// so they keep the trait defaults and are not counted.
struct CountingBackend {
    inner: NativeBackend,
    kernel_calls: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            inner: NativeBackend::new(),
            kernel_calls: AtomicUsize::new(0),
        }
    }

    fn count(&self) -> usize {
        self.kernel_calls.load(Ordering::SeqCst)
    }

    fn bump(&self) {
        self.kernel_calls.fetch_add(1, Ordering::SeqCst);
    }
}

impl ComputeBackend for CountingBackend {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn preferred_order(&self) -> MemoryOrder {
        self.inner.preferred_order()
    }

    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.gemm(desc)
    }

    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.transpose(desc)
    }

    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.svd(desc)
    }

    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.qr(desc)
    }

    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.lq(desc)
    }

    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.eigh(desc)
    }

    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.eig(desc)
    }

    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        self.bump();
        self.inner.solve(desc)
    }
}

// A test backend declares capability for the storage flavors it operates on,
// exactly as an out-of-tree backend would (`OpsFor` is deliberately unsealed).
impl<T: Scalar> OpsFor<DenseStorage<T>> for CountingBackend {}
impl<T: Scalar> OpsFor<BlockSparseStorage<T>> for CountingBackend {}

// ---------------------------------------------------------------------------
// Fixtures (backend-free — every chain carries no backend)
// ---------------------------------------------------------------------------

fn dense_site(shape: Vec<usize>) -> DenseTensor<f64> {
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (1..=len).map(|i| i as f64 * 0.1).collect();
    Host::shared().dense(data, shape)
}

/// 4-site dense MPS.
fn dense_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    Mps::from_sites(vec![
        dense_site(vec![1, 2, 4]),
        dense_site(vec![4, 2, 4]),
        dense_site(vec![4, 2, 3]),
        dense_site(vec![3, 2, 1]),
    ])
}

/// 4-site dense MPO (physical dim 2).
fn dense_mpo() -> Mpo<DenseStorage<f64>, DenseLayout> {
    Mpo::from_sites(vec![
        dense_site(vec![1, 2, 2, 2]),
        dense_site(vec![2, 2, 2, 2]),
        dense_site(vec![2, 2, 2, 2]),
        dense_site(vec![2, 2, 2, 1]),
    ])
}

type U1Sectors = Vec<(U1Sector, usize)>;

fn u1_site(
    left: U1Sectors,
    phys: U1Sectors,
    right: U1Sectors,
    counter: &mut f64,
) -> BlockSparseTensor<f64, U1Sector> {
    let left = QNIndex::new(left, Direction::Out);
    let phys = QNIndex::new(phys, Direction::Out);
    let right = QNIndex::new(right, Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, phys, right], U1Sector(0));
    fill_blocks(&mut site, counter);
    site
}

fn fill_blocks(t: &mut BlockSparseTensor<f64, U1Sector>, counter: &mut f64) {
    let coords: Vec<BlockCoord> = t
        .data()
        .layout()
        .block_metas()
        .iter()
        .map(|m| m.coord.clone())
        .collect();
    for coord in coords {
        let data = t.data_mut().block_data_mut(&coord).expect("allowed block");
        for slot in data.iter_mut() {
            *slot = *counter;
            *counter += 0.1;
        }
    }
}

/// 4-site U(1) MPS with multi-element sector blocks.
fn u1_mps() -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let mut counter = 0.1;
    let s0 = vec![(U1Sector(0), 1)];
    let p = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let b1 = vec![(U1Sector(0), 2), (U1Sector(1), 1)];
    let b2 = vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)];
    Mps::from_sites(vec![
        u1_site(s0.clone(), p.clone(), b1.clone(), &mut counter),
        u1_site(b1.clone(), p.clone(), b2.clone(), &mut counter),
        u1_site(b2, p.clone(), b1.clone(), &mut counter),
        u1_site(b1, p, s0, &mut counter),
    ])
}

/// 4-site U(1) MPO with single-sector dim-1/2 bonds and dense-filled
/// allowed blocks.
fn u1_mpo() -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let n = 4;
    let mut counter = 0.5;
    let mut sites = Vec::with_capacity(n);
    for j in 0..n {
        let left_dim = if j == 0 { 1 } else { 2 };
        let right_dim = if j == n - 1 { 1 } else { 2 };
        let left = QNIndex::new(vec![(U1Sector(0), left_dim)], Direction::Out);
        let ket = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::In);
        let bra = QNIndex::new(vec![(U1Sector(0), 1), (U1Sector(1), 1)], Direction::Out);
        let right = QNIndex::new(vec![(U1Sector(0), right_dim)], Direction::In);
        let mut site =
            BlockSparseTensor::<f64, U1Sector>::zeros(vec![left, ket, bra, right], U1Sector(0));
        fill_blocks(&mut site, &mut counter);
        sites.push(site);
    }
    Mpo::from_sites(sites)
}

fn trunc_params(chi_max: usize) -> TruncateParams {
    TruncateParams::from(TruncSvdParams {
        chi_max: Some(chi_max),
        target_trunc_err: None,
    })
}

// ---------------------------------------------------------------------------
// canonicalize
// ---------------------------------------------------------------------------

#[test]
fn canonicalize_dense_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let mut mps = dense_mps();

    canonicalize(&backend, &mut mps, 1);

    assert!(
        backend.count() > 0,
        "canonicalize (dense): call-site backend recorded no kernel dispatch",
    );
}

#[test]
fn canonicalize_bsp_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let mut mps = u1_mps();

    canonicalize(&backend, &mut mps, 1);

    assert!(
        backend.count() > 0,
        "canonicalize (block-sparse): call-site backend recorded no kernel dispatch",
    );
}

// ---------------------------------------------------------------------------
// truncate
// ---------------------------------------------------------------------------
//
// The chain is stamped as already canonical so `truncate` runs its SVD
// sweeps directly instead of auto-canonicalizing first.

#[test]
fn truncate_dense_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let mut mps = dense_mps();
    mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

    truncate(&backend, &mut mps, &trunc_params(2));

    assert!(
        backend.count() > 0,
        "truncate (dense): call-site backend recorded no kernel dispatch",
    );
}

#[test]
fn truncate_bsp_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let mut mps = u1_mps();
    mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

    truncate(&backend, &mut mps, &trunc_params(2));

    assert!(
        backend.count() > 0,
        "truncate (block-sparse): call-site backend recorded no kernel dispatch",
    );
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

#[test]
fn apply_dense_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let op = dense_mpo();
    let psi = dense_mps();

    let _ = apply(&backend, &op, &psi, None);

    assert!(
        backend.count() > 0,
        "apply (dense): call-site backend recorded no kernel dispatch",
    );
}

#[test]
fn apply_bsp_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let op = u1_mpo();
    let psi = u1_mps();

    let _ = apply(&backend, &op, &psi, None);

    assert!(
        backend.count() > 0,
        "apply (block-sparse): call-site backend recorded no kernel dispatch",
    );
}

// `apply(.., None)` keeps the forward sweep on its QR branch; the
// truncated-SVD branch needs both a `chi_max` and a `forward_cap`, so
// it gets its own pair of cases. `chi_max = 1` with `forward_cap = 1`
// makes the natural forward rank exceed the cap at interior sites.

fn forward_svd_method() -> ApplyMethod {
    ApplyMethod::StreamingNaive {
        forward_cap: Some(NonZeroUsize::new(1).expect("nonzero")),
    }
}

#[test]
fn apply_dense_svd_branch_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let op = dense_mpo();
    let psi = dense_mps();

    let _ = apply_with_method(
        &backend,
        &op,
        &psi,
        Some(&trunc_params(1)),
        forward_svd_method(),
    );

    assert!(
        backend.count() > 0,
        "apply (dense, SVD branch): call-site backend recorded no kernel dispatch",
    );
}

#[test]
fn apply_bsp_svd_branch_routes_kernels_to_call_site_backend() {
    let backend = CountingBackend::new();
    let op = u1_mpo();
    let psi = u1_mps();

    let _ = apply_with_method(
        &backend,
        &op,
        &psi,
        Some(&trunc_params(1)),
        forward_svd_method(),
    );

    assert!(
        backend.count() > 0,
        "apply (block-sparse, SVD branch): call-site backend recorded no kernel dispatch",
    );
}
