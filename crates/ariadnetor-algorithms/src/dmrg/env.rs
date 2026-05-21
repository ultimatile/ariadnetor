//! DMRG L/R environment tensors and their incremental update.
//!
//! Each env slot carries a rank-3 tensor of shape `(top-bra-bond,
//! W-bond, bot-ket-bond)` matching the axis convention used by the
//! `arnet_mps::inner` braket family. Boundary slots (`left[0]` and
//! `right[N]`) hold the trivial 1×1×1 identity tensor; for the
//! BlockSparse variant they additionally carry QNIndex / direction /
//! flux metadata (`flux = S::identity()`).
//!
//! Index convention is **boundary-indexed**: `left(i)` is the L
//! tensor at the boundary just left of site `i` (sites `0..i` already
//! folded in), `right(j)` is the R tensor at the boundary just left
//! of site `j` (sites `j..N` folded from the right). A 2-site DMRG
//! step at sites `(i, i+1)` consumes `left(i)`, `W[i]`, `W[i+1]`, and
//! `right(i+2)`.
//!
//! Storage-specific dispatch is provided by [`DmrgEnvOps`], which is
//! implemented for [`DenseLayout`] in this module and for
//! [`BlockSparseLayout<S>`] in a sibling module. The two boundary
//! helpers fail loudly with [`DmrgEnvError::MalformedEdgeBond`] when a
//! chain's edge bonds violate the dim-1 single-sector contract required
//! by the BlockSparse boundary; for the Dense path the helpers always
//! succeed.

use std::sync::Arc;

use arnet::{
    ComputeBackend, DenseLayout, DenseStorage, LinalgError, MemoryOrder, NativeBackend, Scalar,
    Storage, StorageFor, Tensor, TensorLayout, contract,
};
use arnet_mps::{Mpo, Mps, TensorChain};

/// Errors raised by [`DmrgEnvs`] construction and advance operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum DmrgEnvError {
    /// MPS / MPO had zero sites.
    EmptyChain,
    /// MPS and MPO site counts differ.
    LengthMismatch { mps: usize, mpo: usize },
    /// `advance_*` was called with a site index outside `0..n_sites`.
    InvalidSite { index: usize, n_sites: usize },
    /// `advance_*` could not proceed because the predecessor env slot
    /// (`left[i]` for `advance_left(i)`, `right[j+1]` for
    /// `advance_right(j)`) is `None`. Indicates the caller advanced
    /// out of order or never built the initial envs.
    StaleNeighbor { side: &'static str, index: usize },
    /// An underlying `arnet::contract` call failed. The source is
    /// preserved so callers see the real cause (dimension mismatch,
    /// backend failure, etc.) rather than a panic.
    Contract(LinalgError),
    /// An MPS or MPO chain edge bond violated the dim-1 single-sector
    /// contract required by the BlockSparse boundary helper, or the
    /// chosen edge sectors yielded a flux-disallowed boundary block
    /// under `flux = S::identity()`. The `leg` field names the
    /// offending edge (`"mps_left"`, `"mpo_left"`, `"mps_right"`, or
    /// `"mpo_right"`).
    MalformedEdgeBond { leg: &'static str },
}

impl From<LinalgError> for DmrgEnvError {
    fn from(e: LinalgError) -> Self {
        DmrgEnvError::Contract(e)
    }
}

impl std::fmt::Display for DmrgEnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmrgEnvError::EmptyChain => write!(f, "MPS / MPO has zero sites"),
            DmrgEnvError::LengthMismatch { mps, mpo } => write!(
                f,
                "MPS and MPO site counts differ: mps = {mps}, mpo = {mpo}"
            ),
            DmrgEnvError::InvalidSite { index, n_sites } => write!(
                f,
                "site index {index} out of range for chain of length {n_sites}"
            ),
            DmrgEnvError::StaleNeighbor { side, index } => write!(
                f,
                "advance prerequisite {side} env at index {index} is stale (None); \
                 build the initial envs or advance in order"
            ),
            DmrgEnvError::Contract(_) => {
                write!(f, "contract failure during DMRG environment update")
            }
            DmrgEnvError::MalformedEdgeBond { leg } => {
                // MPS edges only need dim-1 / single-sector (any charge is
                // OK because env_leg0 and env_leg2 carry the same MPS sector
                // with opposite directions and cancel). MPO edges
                // additionally require an identity-fusing sector to land a
                // (0,0,0) boundary block under flux=identity.
                let detail = match *leg {
                    "mps_left" | "mps_right" => "must be dim-1 / single-sector",
                    "mpo_left" | "mpo_right" => {
                        "must be dim-1 / single-sector with sector fusing to identity flux"
                    }
                    _ => "must be dim-1 / single-sector",
                };
                write!(f, "malformed edge bond on {leg}: {detail}")
            }
        }
    }
}

impl std::error::Error for DmrgEnvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DmrgEnvError::Contract(err) => Some(err),
            _ => None,
        }
    }
}

/// Layout-keyed dispatch for DMRG env construction and per-site
/// updates.
///
/// The four trait methods are the only points at which storage type
/// matters; everything else in [`DmrgEnvs`] is dispatched generically
/// over `L: DmrgEnvOps<T>`. Boundary helpers receive the chain's edge
/// site tensors (rather than just the backend) so the BlockSparse
/// implementation can extract QNIndex / direction / flux metadata; the
/// Dense implementation ignores the site arguments and returns a
/// constant 1×1×1 tensor.
pub trait DmrgEnvOps<T: Scalar>: TensorLayout + Sized {
    /// Storage type paired with this layout (mirrors the
    /// `arnet_mps::MpsOps::Storage` association).
    type Storage: Storage + StorageFor<Self>;

    /// Build the trivial L boundary tensor sitting just left of site 0.
    fn trivial_left_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        mps_left_edge: &Tensor<Self::Storage, Self, B>,
        mpo_left_edge: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, DmrgEnvError>;

    /// Build the trivial R boundary tensor sitting just right of site
    /// `n_sites - 1`.
    fn trivial_right_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        mps_right_edge: &Tensor<Self::Storage, Self, B>,
        mpo_right_edge: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, DmrgEnvError>;

    /// Absorb one site into the L environment, advancing it by one
    /// step to the right.
    fn extend_left_step<B: ComputeBackend>(
        backend: &Arc<B>,
        env: &Tensor<Self::Storage, Self, B>,
        site: &Tensor<Self::Storage, Self, B>,
        mpo_site: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, LinalgError>;

    /// Absorb one site into the R environment, advancing it by one
    /// step to the left.
    fn extend_right_step<B: ComputeBackend>(
        backend: &Arc<B>,
        env: &Tensor<Self::Storage, Self, B>,
        site: &Tensor<Self::Storage, Self, B>,
        mpo_site: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, LinalgError>;
}

// ============================================================================
// DenseLayout implementation
// ============================================================================

impl<T: Scalar> DmrgEnvOps<T> for DenseLayout {
    type Storage = DenseStorage<T>;

    fn trivial_left_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        _mps_left_edge: &Tensor<Self::Storage, Self, B>,
        _mpo_left_edge: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, DmrgEnvError> {
        Ok(make_dense_one(backend))
    }

    fn trivial_right_boundary<B: ComputeBackend>(
        backend: &Arc<B>,
        _mps_right_edge: &Tensor<Self::Storage, Self, B>,
        _mpo_right_edge: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, DmrgEnvError> {
        Ok(make_dense_one(backend))
    }

    /// Per-site left extension for `DenseLayout`. Mirrors the loop
    /// body of `arnet_mps::inner::braket_dense`: bra = `site.conj()`,
    /// then a 3-step contraction `(env, bra) → (·, mpo) → (·, site)`.
    fn extend_left_step<B: ComputeBackend>(
        _backend: &Arc<B>,
        env: &Tensor<Self::Storage, Self, B>,
        site: &Tensor<Self::Storage, Self, B>,
        mpo_site: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, LinalgError> {
        let bra = site.conj();
        let t1 = contract(env, &bra, "abc,ade->bcde")?;
        let t2 = contract(&t1, mpo_site, "bcde,bfdg->cefg")?;
        contract(&t2, site, "cefg,cfh->egh")
    }

    /// Per-site right extension for `DenseLayout`.
    fn extend_right_step<B: ComputeBackend>(
        _backend: &Arc<B>,
        env: &Tensor<Self::Storage, Self, B>,
        site: &Tensor<Self::Storage, Self, B>,
        mpo_site: &Tensor<Self::Storage, Self, B>,
    ) -> Result<Tensor<Self::Storage, Self, B>, LinalgError> {
        let bra = site.conj();
        let t1 = contract(env, site, "egh,cfh->egcf")?;
        let t2 = contract(&t1, mpo_site, "egcf,bfdg->ecbd")?;
        contract(&t2, &bra, "ecbd,ade->abc")
    }
}

fn make_dense_one<T, B>(backend: &Arc<B>) -> Tensor<DenseStorage<T>, DenseLayout, B>
where
    T: Scalar,
    B: ComputeBackend,
{
    let order: MemoryOrder = backend.preferred_order();
    arnet::DenseTensor::<T, B>::from_raw_parts(
        vec![T::one()],
        vec![1, 1, 1],
        order,
        Arc::clone(backend),
    )
}

// ============================================================================
// DmrgEnvs<St, L, B>
// ============================================================================

/// L/R environment tensors for 2-site DMRG, with incremental update
/// operations for left-to-right and right-to-left sweeps.
///
/// Generic over the storage / layout pair, which the
/// [`DmrgEnvOps<T>`] trait pins together via its `type Storage`
/// association. The struct itself is layout-agnostic; per-site
/// extension and boundary-tensor construction route through the trait.
///
/// See the module-level docs for the index convention.
#[derive(Debug, Clone)]
pub struct DmrgEnvs<St, L, B = NativeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// `left[i]` for `i in 0..=n_sites`. `left[0]` is the trivial
    /// boundary; `left[i]` for `i > 0` carries sites `0..i`. `None`
    /// indicates the slot is stale relative to the current sweep
    /// position.
    left: Vec<Option<Tensor<St, L, B>>>,
    /// Mirror of `left` for the right sweep. `right[N]` is the
    /// trivial boundary; `right[j]` for `j < N` carries sites
    /// `j..N`.
    right: Vec<Option<Tensor<St, L, B>>>,
    n_sites: usize,
    backend: Arc<B>,
}

impl<St, L, B> DmrgEnvs<St, L, B>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
    B: ComputeBackend,
{
    /// Initial right-sweep build. Computes `right[N-1..=1]` from the
    /// trivial right boundary down through the chain, leaving only
    /// `left[0]` populated.
    pub fn build<T>(mps: &Mps<St, L, B>, mpo: &Mpo<St, L, B>) -> Result<Self, DmrgEnvError>
    where
        T: Scalar,
        L: DmrgEnvOps<T, Storage = St>,
    {
        let n_sites = mps.len();
        if n_sites == 0 {
            return Err(DmrgEnvError::EmptyChain);
        }
        if mpo.len() != n_sites {
            return Err(DmrgEnvError::LengthMismatch {
                mps: n_sites,
                mpo: mpo.len(),
            });
        }

        let backend: Arc<B> = mps.backend_arc().clone();
        let mut left: Vec<Option<Tensor<St, L, B>>> = (0..=n_sites).map(|_| None).collect();
        let mut right: Vec<Option<Tensor<St, L, B>>> = (0..=n_sites).map(|_| None).collect();

        // Trivial boundary tensors at the chain edges. For Dense these
        // are constant 1×1×1 ones; for BlockSparse they additionally
        // validate the dim-1 / single-sector edge-bond contract.
        left[0] = Some(<L as DmrgEnvOps<T>>::trivial_left_boundary(
            &backend,
            mps.site(0),
            mpo.site(0),
        )?);
        right[n_sites] = Some(<L as DmrgEnvOps<T>>::trivial_right_boundary(
            &backend,
            mps.site(n_sites - 1),
            mpo.site(n_sites - 1),
        )?);

        // Build right envs from the right edge down to right[1].
        for j in (1..=n_sites).rev() {
            // right[j] is defined; absorb site j-1 to produce right[j-1].
            // We stop at j == 1 (computing right[0] is unused: a 2-site
            // step at the leftmost block (0, 1) consumes right(2), not
            // right(0); right(0) would equal the global braket scalar
            // and provides no useful intermediate). Keep right[0] as
            // None to make that explicit — building it would just
            // discard work.
            if j == 1 {
                break;
            }
            let prev = right[j].as_ref().expect("just initialized or computed");
            let new = <L as DmrgEnvOps<T>>::extend_right_step(
                &backend,
                prev,
                mps.site(j - 1),
                mpo.site(j - 1),
            )?;
            right[j - 1] = Some(new);
        }

        Ok(Self {
            left,
            right,
            n_sites,
            backend,
        })
    }

    /// Number of MPS sites the env was built for.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// L tensor at the boundary just left of site `i`. Returns `None`
    /// when stale.
    pub fn left(&self, i: usize) -> Option<&Tensor<St, L, B>> {
        self.left.get(i).and_then(Option::as_ref)
    }

    /// R tensor at the boundary just left of site `j`.
    pub fn right(&self, j: usize) -> Option<&Tensor<St, L, B>> {
        self.right.get(j).and_then(Option::as_ref)
    }

    /// Absorb site `i` into the left environment.
    pub fn advance_left<T>(
        &mut self,
        mps: &Mps<St, L, B>,
        mpo: &Mpo<St, L, B>,
        i: usize,
    ) -> Result<(), DmrgEnvError>
    where
        T: Scalar,
        L: DmrgEnvOps<T, Storage = St>,
    {
        if i >= self.n_sites {
            return Err(DmrgEnvError::InvalidSite {
                index: i,
                n_sites: self.n_sites,
            });
        }
        if mpo.len() != self.n_sites || mps.len() != self.n_sites {
            return Err(DmrgEnvError::LengthMismatch {
                mps: mps.len(),
                mpo: mpo.len(),
            });
        }
        let prev = match &self.left[i] {
            Some(t) => t,
            None => {
                return Err(DmrgEnvError::StaleNeighbor {
                    side: "left",
                    index: i,
                });
            }
        };
        let new =
            <L as DmrgEnvOps<T>>::extend_left_step(&self.backend, prev, mps.site(i), mpo.site(i))?;
        self.left[i + 1] = Some(new);
        if i + 1 < self.n_sites {
            self.right[i + 1] = None;
        }
        Ok(())
    }

    /// Absorb site `j` into the right environment.
    pub fn advance_right<T>(
        &mut self,
        mps: &Mps<St, L, B>,
        mpo: &Mpo<St, L, B>,
        j: usize,
    ) -> Result<(), DmrgEnvError>
    where
        T: Scalar,
        L: DmrgEnvOps<T, Storage = St>,
    {
        if j >= self.n_sites {
            return Err(DmrgEnvError::InvalidSite {
                index: j,
                n_sites: self.n_sites,
            });
        }
        if mpo.len() != self.n_sites || mps.len() != self.n_sites {
            return Err(DmrgEnvError::LengthMismatch {
                mps: mps.len(),
                mpo: mpo.len(),
            });
        }
        let prev = match &self.right[j + 1] {
            Some(t) => t,
            None => {
                return Err(DmrgEnvError::StaleNeighbor {
                    side: "right",
                    index: j + 1,
                });
            }
        };
        let new =
            <L as DmrgEnvOps<T>>::extend_right_step(&self.backend, prev, mps.site(j), mpo.site(j))?;
        self.right[j] = Some(new);
        if j > 0 {
            self.left[j] = None;
        }
        Ok(())
    }
}
