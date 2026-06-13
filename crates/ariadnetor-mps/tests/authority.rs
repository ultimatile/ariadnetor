//! Authority-routing tests for the explicit-backend call-graph migration.
//!
//! Each test splits the operand tensors across distinct
//! `CountingBackend` instances so the legacy authority sources and the
//! entry-derived handle are distinguishable: `canonicalize` /
//! `truncate` build the site tensors on one instance and hand the
//! chain constructor another, while `apply` uses three — the MPO on
//! one, the psi sites on a second, and the psi chain handle on a
//! third. The migrated operation paths must dispatch every kernel
//! through the entry-derived chain handle and never through any other
//! instance. Before the migration the legacy wrappers derived their
//! authority from the operand tensors, so these assertions fail on the
//! pre-migration code for every kernel-dispatching operation
//! (decompositions and contractions). Allocation-only operations
//! (`diagonal_scale`, permute / fuse) dispatch no counted kernel, so
//! their routing is invisible here; it is proven per twin by the
//! pointer-identity tests in the linalg crate.
//!
//! Only entry points whose legacy authority came from a tensor other
//! than the chain handle are tested — `canonicalize` / `truncate`
//! (per-site decompositions used each site's own instance) and `apply`
//! (contraction authority came from the MPO side). `inner` is excluded:
//! its legacy contractions already derived authority from the
//! chain-labeled accumulator, so no count assertion separates pre- from
//! post-migration routing.

use std::num::NonZeroUsize;
use std::sync::Arc;
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
    DenseStorage, DenseTensor, Direction, QNIndex, U1Sector,
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

// ---------------------------------------------------------------------------
// Fixtures (generic over the backend instance the sites are built on)
// ---------------------------------------------------------------------------

fn dense_site_on(
    shape: Vec<usize>,
    backend: &Arc<CountingBackend>,
) -> DenseTensor<f64, CountingBackend> {
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (1..=len).map(|i| i as f64 * 0.1).collect();
    DenseTensor::from_raw_parts(data, shape, Arc::clone(backend))
}

/// 4-site dense MPS: sites on `site_backend`, chain handle = `chain_backend`.
fn dense_mps_on(
    site_backend: &Arc<CountingBackend>,
    chain_backend: &Arc<CountingBackend>,
) -> Mps<DenseStorage<f64>, DenseLayout, CountingBackend> {
    let sites = vec![
        dense_site_on(vec![1, 2, 4], site_backend),
        dense_site_on(vec![4, 2, 4], site_backend),
        dense_site_on(vec![4, 2, 3], site_backend),
        dense_site_on(vec![3, 2, 1], site_backend),
    ];
    Mps::with_backend(sites, Arc::clone(chain_backend))
}

/// 4-site dense MPO (physical dim 2): sites and chain handle on `backend`.
fn dense_mpo_on(
    backend: &Arc<CountingBackend>,
) -> Mpo<DenseStorage<f64>, DenseLayout, CountingBackend> {
    let sites = vec![
        dense_site_on(vec![1, 2, 2, 2], backend),
        dense_site_on(vec![2, 2, 2, 2], backend),
        dense_site_on(vec![2, 2, 2, 2], backend),
        dense_site_on(vec![2, 2, 2, 1], backend),
    ];
    Mpo::with_backend(sites, Arc::clone(backend))
}

type U1Sectors = Vec<(U1Sector, usize)>;

fn u1_site_on(
    left: U1Sectors,
    phys: U1Sectors,
    right: U1Sectors,
    counter: &mut f64,
    backend: &Arc<CountingBackend>,
) -> BlockSparseTensor<f64, U1Sector, CountingBackend> {
    let left = QNIndex::new(left, Direction::Out);
    let phys = QNIndex::new(phys, Direction::Out);
    let right = QNIndex::new(right, Direction::In);
    let mut site = BlockSparseTensor::<f64, U1Sector, CountingBackend>::zeros_with_backend(
        vec![left, phys, right],
        U1Sector(0),
        Arc::clone(backend),
    );
    fill_blocks(&mut site, counter);
    site
}

fn fill_blocks(t: &mut BlockSparseTensor<f64, U1Sector, CountingBackend>, counter: &mut f64) {
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

/// 4-site U(1) MPS with multi-element sector blocks: sites on
/// `site_backend`, chain handle = `chain_backend`.
fn u1_mps_on(
    site_backend: &Arc<CountingBackend>,
    chain_backend: &Arc<CountingBackend>,
) -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>, CountingBackend> {
    let mut counter = 0.1;
    let s0 = vec![(U1Sector(0), 1)];
    let p = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let b1 = vec![(U1Sector(0), 2), (U1Sector(1), 1)];
    let b2 = vec![(U1Sector(0), 2), (U1Sector(1), 2), (U1Sector(2), 1)];
    let sites = vec![
        u1_site_on(
            s0.clone(),
            p.clone(),
            b1.clone(),
            &mut counter,
            site_backend,
        ),
        u1_site_on(
            b1.clone(),
            p.clone(),
            b2.clone(),
            &mut counter,
            site_backend,
        ),
        u1_site_on(b2, p.clone(), b1.clone(), &mut counter, site_backend),
        u1_site_on(b1, p, s0, &mut counter, site_backend),
    ];
    Mps::with_backend(sites, Arc::clone(chain_backend))
}

/// 4-site U(1) MPO with single-sector dim-1/2 bonds and dense-filled
/// allowed blocks: sites and chain handle on `backend`.
fn u1_mpo_on(
    backend: &Arc<CountingBackend>,
) -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>, CountingBackend> {
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
        let mut site = BlockSparseTensor::<f64, U1Sector, CountingBackend>::zeros_with_backend(
            vec![left, ket, bra, right],
            U1Sector(0),
            Arc::clone(backend),
        );
        fill_blocks(&mut site, &mut counter);
        sites.push(site);
    }
    Mpo::with_backend(sites, Arc::clone(backend))
}

fn trunc_params(chi_max: usize) -> TruncateParams {
    TruncateParams::from(TruncSvdParams {
        chi_max: Some(chi_max),
        target_trunc_err: None,
    })
}

fn assert_authority(entry: &str, chain: &Arc<CountingBackend>, silent: &[&Arc<CountingBackend>]) {
    assert!(
        chain.count() > 0,
        "{entry}: chain backend recorded no kernel dispatch",
    );
    for backend in silent {
        assert_eq!(
            backend.count(),
            0,
            "{entry}: a non-authority backend instance received kernel dispatch",
        );
    }
}

// ---------------------------------------------------------------------------
// canonicalize
// ---------------------------------------------------------------------------

#[test]
fn canonicalize_dense_routes_kernels_to_chain_backend() {
    let sites = Arc::new(CountingBackend::new());
    let chain = Arc::new(CountingBackend::new());
    let mut mps = dense_mps_on(&sites, &chain);

    canonicalize(&mut mps, 1);

    assert_authority("canonicalize (dense)", &chain, &[&sites]);
}

#[test]
fn canonicalize_bsp_routes_kernels_to_chain_backend() {
    let sites = Arc::new(CountingBackend::new());
    let chain = Arc::new(CountingBackend::new());
    let mut mps = u1_mps_on(&sites, &chain);

    canonicalize(&mut mps, 1);

    assert_authority("canonicalize (block-sparse)", &chain, &[&sites]);
}

// ---------------------------------------------------------------------------
// truncate
// ---------------------------------------------------------------------------
//
// The chain is stamped as already canonical so `truncate` runs its SVD
// sweeps directly on the A-built site tensors instead of auto-
// canonicalizing first (which would replace every site with a B-labeled
// intermediate and mask the truncation path's own routing).

#[test]
fn truncate_dense_routes_kernels_to_chain_backend() {
    let sites = Arc::new(CountingBackend::new());
    let chain = Arc::new(CountingBackend::new());
    let mut mps = dense_mps_on(&sites, &chain);
    mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

    truncate(&mut mps, &trunc_params(2));

    assert_authority("truncate (dense)", &chain, &[&sites]);
}

#[test]
fn truncate_bsp_routes_kernels_to_chain_backend() {
    let sites = Arc::new(CountingBackend::new());
    let chain = Arc::new(CountingBackend::new());
    let mut mps = u1_mps_on(&sites, &chain);
    mps.set_canonical_form(CanonicalForm::Mixed { center: 0 });

    truncate(&mut mps, &trunc_params(2));

    assert_authority("truncate (block-sparse)", &chain, &[&sites]);
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------
//
// The MPO lives entirely on one silent instance (the legacy MPO-MPS
// contraction derived its authority from the MPO site, the
// contraction's lhs) and the psi sites on another, so the assertions
// reject both the pre-migration MPO-side authority and a regression
// that re-derives the handle from a psi site instead of the chain.

#[test]
fn apply_dense_routes_kernels_to_psi_chain_backend() {
    let mpo_backend = Arc::new(CountingBackend::new());
    let psi_sites = Arc::new(CountingBackend::new());
    let psi_chain = Arc::new(CountingBackend::new());
    let op = dense_mpo_on(&mpo_backend);
    let psi = dense_mps_on(&psi_sites, &psi_chain);

    let _ = apply(&op, &psi, None);

    assert_authority("apply (dense)", &psi_chain, &[&mpo_backend, &psi_sites]);
}

#[test]
fn apply_bsp_routes_kernels_to_psi_chain_backend() {
    let mpo_backend = Arc::new(CountingBackend::new());
    let psi_sites = Arc::new(CountingBackend::new());
    let psi_chain = Arc::new(CountingBackend::new());
    let op = u1_mpo_on(&mpo_backend);
    let psi = u1_mps_on(&psi_sites, &psi_chain);

    let _ = apply(&op, &psi, None);

    assert_authority(
        "apply (block-sparse)",
        &psi_chain,
        &[&mpo_backend, &psi_sites],
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
fn apply_dense_svd_branch_routes_kernels_to_psi_chain_backend() {
    let mpo_backend = Arc::new(CountingBackend::new());
    let psi_sites = Arc::new(CountingBackend::new());
    let psi_chain = Arc::new(CountingBackend::new());
    let op = dense_mpo_on(&mpo_backend);
    let psi = dense_mps_on(&psi_sites, &psi_chain);

    let _ = apply_with_method(&op, &psi, Some(&trunc_params(1)), forward_svd_method());

    assert_authority(
        "apply (dense, SVD branch)",
        &psi_chain,
        &[&mpo_backend, &psi_sites],
    );
}

#[test]
fn apply_bsp_svd_branch_routes_kernels_to_psi_chain_backend() {
    let mpo_backend = Arc::new(CountingBackend::new());
    let psi_sites = Arc::new(CountingBackend::new());
    let psi_chain = Arc::new(CountingBackend::new());
    let op = u1_mpo_on(&mpo_backend);
    let psi = u1_mps_on(&psi_sites, &psi_chain);

    let _ = apply_with_method(&op, &psi, Some(&trunc_params(1)), forward_svd_method());

    assert_authority(
        "apply (block-sparse, SVD branch)",
        &psi_chain,
        &[&mpo_backend, &psi_sites],
    );
}
