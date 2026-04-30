//! DMRG L/R environment tensors and their incremental update.
//!
//! Each env slot carries a rank-3 tensor of shape `(top-bra-bond,
//! W-bond, bot-ket-bond)` matching the axis convention used by the
//! `arnet_mps::inner` braket family (`braket_dense` for [`Dense<T>`],
//! `braket_bsp` for `BlockSparse<T, S>`). Boundary slots (`left[0]`
//! and `right[N]`) hold the trivial 1×1×1 identity tensor; for the
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
//! implemented for [`Dense<T>`] in this module and for
//! `BlockSparse<T, S>` in a sibling module. The two boundary helpers
//! fail loudly with [`DmrgEnvError::MalformedEdgeBond`] when a chain's
//! edge bonds violate the dim-1 single-sector contract required by
//! the BlockSparse boundary; for `Dense<T>` the helpers always
//! succeed.

use std::sync::Arc;

use arnet_core::Scalar;
use arnet_core::backend::ComputeBackend;
use arnet_linalg::{LinalgError, contract};
use arnet_mps::{Mpo, Mps, TensorChain};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, Dense, TensorRepr};

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
    /// An underlying `arnet_linalg::contract` call failed. The source
    /// is preserved so callers see the real cause (dimension
    /// mismatch, backend failure, etc.) rather than a panic.
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
            DmrgEnvError::MalformedEdgeBond { leg } => write!(
                f,
                "malformed edge bond on {leg}: must be dim-1 / single-sector \
                 with sectors fusing to identity flux"
            ),
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

/// Storage-specific dispatch for DMRG env construction and per-site
/// updates.
///
/// The four trait methods are the only points at which storage type
/// matters; everything else in [`DmrgEnvs`] is dispatched generically
/// over `R: DmrgEnvOps`. Boundary helpers receive the chain's edge
/// site tensors (rather than just the backend) so the BlockSparse
/// implementation can extract QNIndex / direction / flux metadata; the
/// [`Dense<T>`] implementation ignores the site arguments and returns a
/// constant 1×1×1 tensor.
pub trait DmrgEnvOps: TensorRepr + Sized {
    /// Build the trivial L boundary tensor sitting just left of site 0.
    ///
    /// `mps_left_edge` and `mpo_left_edge` are the leftmost MPS / MPO
    /// site tensors; the BlockSparse implementation reads their leftmost
    /// (axis-0) bond metadata to construct the env's QNIndex /
    /// direction structure.
    fn trivial_left_boundary<B: ComputeBackend>(
        backend: &B,
        mps_left_edge: &Self,
        mpo_left_edge: &Self,
    ) -> Result<Self, DmrgEnvError>;

    /// Build the trivial R boundary tensor sitting just right of site
    /// `n_sites - 1`.
    ///
    /// `mps_right_edge` and `mpo_right_edge` are the rightmost MPS /
    /// MPO site tensors; the BlockSparse implementation reads their
    /// rightmost (axis-2) MPS bond and rightmost (axis-3) MPO bond
    /// metadata to construct the env's structure.
    fn trivial_right_boundary<B: ComputeBackend>(
        backend: &B,
        mps_right_edge: &Self,
        mpo_right_edge: &Self,
    ) -> Result<Self, DmrgEnvError>;

    /// Absorb one site into the L environment, advancing it by one
    /// step to the right.
    fn extend_left_step<B: ComputeBackend>(
        backend: &B,
        env: &Self,
        site: &Self,
        mpo_site: &Self,
    ) -> Result<Self, LinalgError>;

    /// Absorb one site into the R environment, advancing it by one
    /// step to the left.
    fn extend_right_step<B: ComputeBackend>(
        backend: &B,
        env: &Self,
        site: &Self,
        mpo_site: &Self,
    ) -> Result<Self, LinalgError>;
}

// ============================================================================
// Dense<T> implementation
// ============================================================================

impl<T: Scalar> DmrgEnvOps for Dense<T> {
    fn trivial_left_boundary<B: ComputeBackend>(
        backend: &B,
        _mps_left_edge: &Self,
        _mpo_left_edge: &Self,
    ) -> Result<Self, DmrgEnvError> {
        Ok(backend.make_tensor(vec![T::one()], vec![1, 1, 1]))
    }

    fn trivial_right_boundary<B: ComputeBackend>(
        backend: &B,
        _mps_right_edge: &Self,
        _mpo_right_edge: &Self,
    ) -> Result<Self, DmrgEnvError> {
        Ok(backend.make_tensor(vec![T::one()], vec![1, 1, 1]))
    }

    /// Per-site left extension for `Dense<T>`. Mirrors the loop body of
    /// `arnet_mps::inner::braket_dense`: bra = `site.conj()`, then a
    /// 3-step contraction `(env, bra) → (·, mpo) → (·, site)`.
    fn extend_left_step<B: ComputeBackend>(
        backend: &B,
        env: &Self,
        site: &Self,
        mpo_site: &Self,
    ) -> Result<Self, LinalgError> {
        let bra = site.conj();
        // env(a,b,c) × conj(A)(a,d,e) → t1(b,c,d,e)
        let t1 = contract(backend, env, &bra, "abc,ade->bcde")?;
        // t1(b,c,d,e) × W(b,f,d,g) → t2(c,e,f,g)
        let t2 = contract(backend, &t1, mpo_site, "bcde,bfdg->cefg")?;
        // t2(c,e,f,g) × A(c,f,h) → env'(e,g,h)
        contract(backend, &t2, site, "cefg,cfh->egh")
    }

    /// Per-site right extension for `Dense<T>`. Mirror of
    /// [`Self::extend_left_step`] with the contraction order reversed.
    fn extend_right_step<B: ComputeBackend>(
        backend: &B,
        env: &Self,
        site: &Self,
        mpo_site: &Self,
    ) -> Result<Self, LinalgError> {
        let bra = site.conj();
        // R(e,g,h) × A(c,f,h) → t1(e,g,c,f)
        let t1 = contract(backend, env, site, "egh,cfh->egcf")?;
        // t1(e,g,c,f) × W(b,f,d,g) → t2(e,c,b,d)
        let t2 = contract(backend, &t1, mpo_site, "egcf,bfdg->ecbd")?;
        // t2(e,c,b,d) × conj(A)(a,d,e) → R'(a,b,c)
        contract(backend, &t2, &bra, "ecbd,ade->abc")
    }
}

// ============================================================================
// DmrgEnvs<R, B>
// ============================================================================

/// L/R environment tensors for 2-site DMRG, with incremental
/// update operations for left-to-right and right-to-left sweeps.
///
/// Generic over the storage type `R: DmrgEnvOps`, which routes the
/// per-site extension and the boundary-tensor construction to either
/// the [`Dense<T>`] or `BlockSparse<T, S>` implementation. The struct
/// itself is storage-agnostic.
///
/// See the module-level docs for the index convention.
#[derive(Debug, Clone)]
pub struct DmrgEnvs<R: DmrgEnvOps, B: ComputeBackend = NativeBackend> {
    /// `left[i]` for `i in 0..=n_sites`. `left[0]` is the trivial
    /// boundary; `left[i]` for `i > 0` carries sites `0..i`. `None`
    /// indicates the slot is stale relative to the current sweep
    /// position.
    left: Vec<Option<R>>,
    /// Mirror of `left` for the right sweep. `right[N]` is the
    /// trivial boundary; `right[j]` for `j < N` carries sites
    /// `j..N`.
    right: Vec<Option<R>>,
    n_sites: usize,
    backend: Arc<B>,
}

impl<R: DmrgEnvOps, B: ComputeBackend> DmrgEnvs<R, B> {
    /// Initial right-sweep build. Computes `right[N-1..=1]` from the
    /// trivial right boundary down through the chain, leaving only
    /// `left[0]` populated. Caller is responsible for ensuring the
    /// MPS is in a canonical form compatible with the intended sweep
    /// direction (typically right-canonical for an L→R sweep that
    /// will follow); the wrapper does not assert this.
    pub fn build(mps: &Mps<R, B>, mpo: &Mpo<R, B>) -> Result<Self, DmrgEnvError> {
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
        let mut left: Vec<Option<R>> = vec![None; n_sites + 1];
        let mut right: Vec<Option<R>> = vec![None; n_sites + 1];

        // Trivial boundary tensors at the chain edges. For Dense these
        // are constant 1×1×1 ones; for BlockSparse they additionally
        // validate the dim-1 / single-sector edge-bond contract.
        left[0] = Some(R::trivial_left_boundary(
            &*backend,
            mps.storage(0),
            mpo.storage(0),
        )?);
        right[n_sites] = Some(R::trivial_right_boundary(
            &*backend,
            mps.storage(n_sites - 1),
            mpo.storage(n_sites - 1),
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
            let new =
                R::extend_right_step(&*backend, prev, mps.storage(j - 1), mpo.storage(j - 1))?;
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
    /// when stale (typical between an `advance_*` call that
    /// invalidated this slot and a subsequent advance that recomputes
    /// it).
    pub fn left(&self, i: usize) -> Option<&R> {
        self.left.get(i).and_then(Option::as_ref)
    }

    /// R tensor at the boundary just left of site `j`. See [`left`]
    /// for staleness semantics.
    pub fn right(&self, j: usize) -> Option<&R> {
        self.right.get(j).and_then(Option::as_ref)
    }

    /// Absorb site `i` into the left environment.
    ///
    /// Reads `mps.storage(i)` (assumed left-canonical at this point
    /// in a left-to-right sweep) and `mpo.storage(i)`, computes
    /// `left[i+1]` from `left[i]`, and **invalidates `right[i+1]`
    /// only when interior** (`i + 1 < n_sites`). The trivial
    /// `right[n_sites]` boundary is never invalidated.
    pub fn advance_left(
        &mut self,
        mps: &Mps<R, B>,
        mpo: &Mpo<R, B>,
        i: usize,
    ) -> Result<(), DmrgEnvError> {
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
        let new = R::extend_left_step(&*self.backend, prev, mps.storage(i), mpo.storage(i))?;
        self.left[i + 1] = Some(new);
        if i + 1 < self.n_sites {
            self.right[i + 1] = None;
        }
        Ok(())
    }

    /// Absorb site `j` into the right environment.
    ///
    /// Reads `mps.storage(j)` (assumed right-canonical at this point
    /// in a right-to-left sweep) and `mpo.storage(j)`, computes
    /// `right[j]` from `right[j+1]`, and **invalidates `left[j]` only
    /// when interior** (`j > 0`). The trivial `left[0]` boundary is
    /// never invalidated.
    pub fn advance_right(
        &mut self,
        mps: &Mps<R, B>,
        mpo: &Mpo<R, B>,
        j: usize,
    ) -> Result<(), DmrgEnvError> {
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
        let new = R::extend_right_step(&*self.backend, prev, mps.storage(j), mpo.storage(j))?;
        self.right[j] = Some(new);
        if j > 0 {
            self.left[j] = None;
        }
        Ok(())
    }
}
