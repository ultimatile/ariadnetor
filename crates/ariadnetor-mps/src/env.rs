//! Three-layer ⟨bra|W|ket⟩ environment tensors and their incremental
//! update — the per-bond partial contraction of an MPO `W` sandwiched
//! between two MPS, stored so a sweep can advance it one site at a time.
//! DMRG consumes it with bra = ket; the primitive itself is generic.
//!
//! Each env slot carries a rank-3 tensor of shape `(top-bra-bond,
//! W-bond, bot-ket-bond)` matching the axis convention used by the
//! `crate::inner` braket family. Boundary slots (`left[0]` and
//! `right[N]`) hold the trivial 1×1×1 identity tensor; for the
//! BlockSparse variant they additionally carry QNIndex / direction /
//! flux metadata (`flux = S::identity()`).
//!
//! Index convention is **boundary-indexed**: `left(i)` is the L
//! tensor at the boundary just left of site `i` (sites `0..i` already
//! folded in), `right(j)` is the R tensor at the boundary just left
//! of site `j` (sites `j..N` folded from the right). A 2-site sweep
//! step at sites `(i, i+1)` consumes `left(i)`, `W[i]`, `W[i+1]`, and
//! `right(i+2)`.
//!
//! Storage-specific dispatch is provided by [`BraketEnvOps`], which is
//! implemented for the Dense `BraketEnvs` chain in this module and for the
//! BlockSparse chain in a sibling module. The two boundary
//! helpers fail loudly with [`BraketEnvError::MalformedEdgeBond`] when a
//! chain's edge bonds violate the dim-1 single-sector contract required
//! by the BlockSparse boundary; for the Dense path the helpers always
//! succeed.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{LinalgError, contract};

use crate::{Mpo, Mps, TensorChain};
use ariadnetor_tensor::{
    DenseLayout, DenseStorage, Host, Storage, StorageFor, Tensor, TensorLayout,
};

/// Errors raised by [`BraketEnvs`] construction and advance operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BraketEnvError {
    /// bra / MPO / ket had zero sites.
    #[error("bra / MPO / ket has zero sites")]
    EmptyChain,
    /// bra, MPO, and ket site counts do not all agree.
    #[error("chain site counts differ: bra = {bra}, mpo = {mpo}, ket = {ket}")]
    LengthMismatch {
        /// Site count reported by the bra MPS.
        bra: usize,
        /// Site count reported by the MPO.
        mpo: usize,
        /// Site count reported by the ket MPS.
        ket: usize,
    },
    /// `advance_*` was called with a site index outside `0..n_sites`.
    #[error("site index {index} out of range for chain of length {n_sites}")]
    InvalidSite {
        /// The out-of-range site index.
        index: usize,
        /// Chain length the index was checked against.
        n_sites: usize,
    },
    /// `advance_*` could not proceed because the predecessor env slot
    /// (`left[i]` for `advance_left(i)`, `right[j+1]` for
    /// `advance_right(j)`) is `None`. Indicates the caller advanced
    /// out of order or never built the initial envs.
    #[error(
        "advance prerequisite {side} env at index {index} is stale (None); \
         build the initial envs or advance in order"
    )]
    StaleNeighbor {
        /// Which side the stale env is on (`"left"` / `"right"`).
        side: &'static str,
        /// Index of the stale (`None`) env slot.
        index: usize,
    },
    /// An underlying `ariadnetor_linalg::contract` call failed. The
    /// source is preserved so callers see the real cause (dimension
    /// mismatch, backend failure, etc.) rather than a panic.
    #[error("contract failure during braket environment update")]
    Contract(#[from] LinalgError),
    /// A bra / ket / MPO chain edge bond violated the dim-1
    /// single-sector contract required by the BlockSparse boundary
    /// helper, or the chosen edge sectors yielded a flux-disallowed
    /// boundary block under `flux = S::identity()`. The `leg` field
    /// names the offending edge (`"bra_left"`, `"mpo_left"`,
    /// `"ket_left"`, `"bra_right"`, `"mpo_right"`, or `"ket_right"`).
    #[error("malformed edge bond on {leg}: {detail}", detail = edge_bond_detail(.leg))]
    MalformedEdgeBond {
        /// Names the offending edge (`"bra_left"`, `"mpo_left"`,
        /// `"ket_left"`, `"bra_right"`, `"mpo_right"`, or
        /// `"ket_right"`).
        leg: &'static str,
    },
}

/// Per-edge well-formedness requirement rendered in
/// [`BraketEnvError::MalformedEdgeBond`]'s message. Every edge must be
/// dim-1 / single-sector (a structural check). The MPO leg's message
/// additionally names identity-flux fusion because the absent-`(0, 0, 0)`
/// -block case — the bra / MPO / ket edge charges failing to fuse to
/// identity — is attributed to the MPO leg (exact for the common case
/// where the MPS edges are the identity sector).
fn edge_bond_detail(leg: &str) -> &'static str {
    match leg {
        "bra_left" | "bra_right" | "ket_left" | "ket_right" => "must be dim-1 / single-sector",
        "mpo_left" | "mpo_right" => {
            "must be dim-1 / single-sector with sector fusing to identity flux"
        }
        _ => "must be dim-1 / single-sector",
    }
}

/// Crate-private seal for the chain-keyed [`BraketEnvOps`] dispatch
/// trait. Living in a private module, [`Sealed`](sealed::Sealed)'s name
/// is not reachable downstream, so [`BraketEnvOps`] cannot be
/// implemented outside this crate. It carries no associated surface, so
/// the public trait projects no storage / layout taxonomy through it.
mod sealed {
    use ariadnetor_core::Scalar;
    use ariadnetor_tensor::{
        BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, Sector,
    };

    use super::BraketEnvs;

    pub trait Sealed {}

    impl<T: Scalar> Sealed for BraketEnvs<DenseStorage<T>, DenseLayout> {}
    impl<T: Scalar, S: Sector> Sealed for BraketEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>> {}
}

/// Chain-keyed dispatch for BraKet env construction and per-site
/// updates.
///
/// Keyed on the [`BraketEnvs<St, L>`](BraketEnvs) chain and sealed (its
/// `sealed::Sealed` supertrait is crate-private), so the
/// storage / layout taxa are reachable only as the sealed associated types
/// rather than as free bounds on a public surface. The four trait methods
/// are the only points at which storage type matters; everything else in
/// [`BraketEnvs`] is dispatched generically. Boundary helpers receive the
/// chain's edge site tensors (rather than just the backend) so the
/// BlockSparse implementation can extract QNIndex / direction / flux
/// metadata; the Dense implementation ignores the site arguments and
/// returns a constant 1×1×1 tensor.
///
/// Every operation runs on the [`Host`] substrate — DMRG is host-resident
/// in the CPU-only Stage B scope — so the concrete impls obtain the
/// backend from [`Host::shared`] rather than receiving one through the
/// call site.
pub trait BraketEnvOps<T: Scalar>: sealed::Sealed {
    /// Layout type paired with this env chain.
    type Layout: TensorLayout;
    /// Storage type paired with this env chain (mirrors the
    /// `ariadnetor_mps::MpsOps::Storage` association).
    type Storage: Storage + StorageFor<Self::Layout>;

    /// Build the trivial L boundary tensor sitting just left of site 0.
    ///
    /// For BlockSparse the boundary's axis 0 carries the bra edge sector
    /// and axis 2 the ket edge sector; the boundary is valid iff the
    /// bra / MPO / ket edge charges fuse to identity flux (see the
    /// `env_block_sparse` module docs — an MPO edge can fuse a
    /// distinct-sector bra / ket pair).
    fn trivial_left_boundary(
        bra_left_edge: &Tensor<Self::Storage, Self::Layout>,
        mpo_left_edge: &Tensor<Self::Storage, Self::Layout>,
        ket_left_edge: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, BraketEnvError>;

    /// Build the trivial R boundary tensor sitting just right of site
    /// `n_sites - 1`.
    fn trivial_right_boundary(
        bra_right_edge: &Tensor<Self::Storage, Self::Layout>,
        mpo_right_edge: &Tensor<Self::Storage, Self::Layout>,
        ket_right_edge: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, BraketEnvError>;

    /// Absorb one site into the L environment, advancing it by one
    /// step to the right. The bra leg is conjugated internally; pass the
    /// un-conjugated bra site. For a self-overlap (`bra = ket`) pass the
    /// same tensor for `bra_site` and `ket_site`.
    fn extend_left_step(
        env: &Tensor<Self::Storage, Self::Layout>,
        bra_site: &Tensor<Self::Storage, Self::Layout>,
        mpo_site: &Tensor<Self::Storage, Self::Layout>,
        ket_site: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, LinalgError>;

    /// Absorb one site into the R environment, advancing it by one
    /// step to the left. The bra leg is conjugated internally.
    fn extend_right_step(
        env: &Tensor<Self::Storage, Self::Layout>,
        bra_site: &Tensor<Self::Storage, Self::Layout>,
        mpo_site: &Tensor<Self::Storage, Self::Layout>,
        ket_site: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, LinalgError>;
}

// ============================================================================
// Dense (BraketEnvs<DenseStorage<T>, DenseLayout>) implementation
// ============================================================================

impl<T: Scalar> BraketEnvOps<T> for BraketEnvs<DenseStorage<T>, DenseLayout> {
    type Layout = DenseLayout;
    type Storage = DenseStorage<T>;

    fn trivial_left_boundary(
        _bra_left_edge: &Tensor<Self::Storage, Self::Layout>,
        _mpo_left_edge: &Tensor<Self::Storage, Self::Layout>,
        _ket_left_edge: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, BraketEnvError> {
        Ok(make_dense_one())
    }

    fn trivial_right_boundary(
        _bra_right_edge: &Tensor<Self::Storage, Self::Layout>,
        _mpo_right_edge: &Tensor<Self::Storage, Self::Layout>,
        _ket_right_edge: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, BraketEnvError> {
        Ok(make_dense_one())
    }

    /// Per-site left extension for the Dense chain. Mirrors the loop
    /// body of `ariadnetor_mps::inner::braket_dense`: bra =
    /// `bra_site.conj()`, then a 3-step contraction
    /// `(env, bra) → (·, mpo) → (·, ket_site)`.
    fn extend_left_step(
        env: &Tensor<Self::Storage, Self::Layout>,
        bra_site: &Tensor<Self::Storage, Self::Layout>,
        mpo_site: &Tensor<Self::Storage, Self::Layout>,
        ket_site: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, LinalgError> {
        let backend = Host::shared();
        let bra = bra_site.conj();
        let t1 = contract(backend.as_ref(), env, &bra, "abc,ade->bcde")?;
        let t2 = contract(backend.as_ref(), &t1, mpo_site, "bcde,bfdg->cefg")?;
        contract(backend.as_ref(), &t2, ket_site, "cefg,cfh->egh")
    }

    /// Per-site right extension for the Dense chain.
    fn extend_right_step(
        env: &Tensor<Self::Storage, Self::Layout>,
        bra_site: &Tensor<Self::Storage, Self::Layout>,
        mpo_site: &Tensor<Self::Storage, Self::Layout>,
        ket_site: &Tensor<Self::Storage, Self::Layout>,
    ) -> Result<Tensor<Self::Storage, Self::Layout>, LinalgError> {
        let backend = Host::shared();
        let bra = bra_site.conj();
        let t1 = contract(backend.as_ref(), env, ket_site, "egh,cfh->egcf")?;
        let t2 = contract(backend.as_ref(), &t1, mpo_site, "egcf,bfdg->ecbd")?;
        contract(backend.as_ref(), &t2, &bra, "ecbd,ade->abc")
    }
}

fn make_dense_one<T>() -> Tensor<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
{
    ariadnetor_tensor::DenseTensor::<T>::ones(vec![1, 1, 1])
}

// ============================================================================
// BraketEnvs<St, L>
// ============================================================================

/// L/R ⟨bra|W|ket⟩ environment tensors, with incremental update
/// operations for left-to-right and right-to-left sweeps. Built from a
/// bra MPS, an MPO, and a (possibly distinct) ket MPS; DMRG uses it with
/// `bra = ket`, variational fitting with distinct bra / ket.
///
/// Generic over the storage / layout pair, which the
/// [`BraketEnvOps<T>`] trait pins together via its `type Storage`
/// association. The struct itself is layout-agnostic; per-site
/// extension and boundary-tensor construction route through the trait.
///
/// See the module-level docs for the index convention.
#[derive(Debug, Clone)]
pub struct BraketEnvs<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// `left[i]` for `i in 0..=n_sites`. `left[0]` is the trivial
    /// boundary; `left[i]` for `i > 0` carries sites `0..i`. `None`
    /// indicates the slot is stale relative to the current sweep
    /// position.
    left: Vec<Option<Tensor<St, L>>>,
    /// Mirror of `left` for the right sweep. `right[N]` is the
    /// trivial boundary; `right[j]` for `j < N` carries sites
    /// `j..N`.
    right: Vec<Option<Tensor<St, L>>>,
    n_sites: usize,
}

impl<St, L> BraketEnvs<St, L>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Initial right-sweep build of the ⟨bra|W|ket⟩ environment.
    /// Computes `right[N-1..=1]` from the trivial right boundary down
    /// through the chain, leaving only `left[0]` populated. For a
    /// self-overlap (DMRG) pass the same MPS as `bra` and `ket`.
    pub fn build<T>(
        bra: &Mps<St, L>,
        mpo: &Mpo<St, L>,
        ket: &Mps<St, L>,
    ) -> Result<Self, BraketEnvError>
    where
        T: Scalar,
        Self: BraketEnvOps<T, Storage = St, Layout = L>,
    {
        let n_sites = bra.len();
        if n_sites == 0 {
            return Err(BraketEnvError::EmptyChain);
        }
        if mpo.len() != n_sites || ket.len() != n_sites {
            return Err(BraketEnvError::LengthMismatch {
                bra: n_sites,
                mpo: mpo.len(),
                ket: ket.len(),
            });
        }

        let mut left: Vec<Option<Tensor<St, L>>> = (0..=n_sites).map(|_| None).collect();
        let mut right: Vec<Option<Tensor<St, L>>> = (0..=n_sites).map(|_| None).collect();

        // Trivial boundary tensors at the chain edges. For Dense these
        // are constant 1×1×1 ones; for BlockSparse they additionally
        // validate the dim-1 / single-sector edge-bond contract.
        left[0] = Some(<Self as BraketEnvOps<T>>::trivial_left_boundary(
            bra.site(0),
            mpo.site(0),
            ket.site(0),
        )?);
        right[n_sites] = Some(<Self as BraketEnvOps<T>>::trivial_right_boundary(
            bra.site(n_sites - 1),
            mpo.site(n_sites - 1),
            ket.site(n_sites - 1),
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
            let new = <Self as BraketEnvOps<T>>::extend_right_step(
                prev,
                bra.site(j - 1),
                mpo.site(j - 1),
                ket.site(j - 1),
            )?;
            right[j - 1] = Some(new);
        }

        Ok(Self {
            left,
            right,
            n_sites,
        })
    }

    /// Number of MPS sites the env was built for.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// L tensor at the boundary just left of site `i`. Returns `None`
    /// when stale.
    pub fn left(&self, i: usize) -> Option<&Tensor<St, L>> {
        self.left.get(i).and_then(Option::as_ref)
    }

    /// R tensor at the boundary just left of site `j`.
    pub fn right(&self, j: usize) -> Option<&Tensor<St, L>> {
        self.right.get(j).and_then(Option::as_ref)
    }

    /// Absorb site `i` into the left environment. For a self-overlap
    /// (DMRG) pass the same MPS as `bra` and `ket`.
    pub fn advance_left<T>(
        &mut self,
        bra: &Mps<St, L>,
        mpo: &Mpo<St, L>,
        ket: &Mps<St, L>,
        i: usize,
    ) -> Result<(), BraketEnvError>
    where
        T: Scalar,
        Self: BraketEnvOps<T, Storage = St, Layout = L>,
    {
        if i >= self.n_sites {
            return Err(BraketEnvError::InvalidSite {
                index: i,
                n_sites: self.n_sites,
            });
        }
        if mpo.len() != self.n_sites || bra.len() != self.n_sites || ket.len() != self.n_sites {
            return Err(BraketEnvError::LengthMismatch {
                bra: bra.len(),
                mpo: mpo.len(),
                ket: ket.len(),
            });
        }
        let prev = match &self.left[i] {
            Some(t) => t,
            None => {
                return Err(BraketEnvError::StaleNeighbor {
                    side: "left",
                    index: i,
                });
            }
        };
        let new = <Self as BraketEnvOps<T>>::extend_left_step(
            prev,
            bra.site(i),
            mpo.site(i),
            ket.site(i),
        )?;
        self.left[i + 1] = Some(new);
        if i + 1 < self.n_sites {
            self.right[i + 1] = None;
        }
        Ok(())
    }

    /// Absorb site `j` into the right environment. For a self-overlap
    /// (DMRG) pass the same MPS as `bra` and `ket`.
    pub fn advance_right<T>(
        &mut self,
        bra: &Mps<St, L>,
        mpo: &Mpo<St, L>,
        ket: &Mps<St, L>,
        j: usize,
    ) -> Result<(), BraketEnvError>
    where
        T: Scalar,
        Self: BraketEnvOps<T, Storage = St, Layout = L>,
    {
        if j >= self.n_sites {
            return Err(BraketEnvError::InvalidSite {
                index: j,
                n_sites: self.n_sites,
            });
        }
        if mpo.len() != self.n_sites || bra.len() != self.n_sites || ket.len() != self.n_sites {
            return Err(BraketEnvError::LengthMismatch {
                bra: bra.len(),
                mpo: mpo.len(),
                ket: ket.len(),
            });
        }
        let prev = match &self.right[j + 1] {
            Some(t) => t,
            None => {
                return Err(BraketEnvError::StaleNeighbor {
                    side: "right",
                    index: j + 1,
                });
            }
        };
        let new = <Self as BraketEnvOps<T>>::extend_right_step(
            prev,
            bra.site(j),
            mpo.site(j),
            ket.site(j),
        )?;
        self.right[j] = Some(new);
        if j > 0 {
            self.left[j] = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_edge_bond_mpo_detail_is_distinct() {
        // The mpo arm of `edge_bond_detail` adds the identity-flux clause that
        // the bra / ket / wildcard text omits; rendering an mpo edge must
        // surface it, so deleting that arm (collapsing to the wildcard) is
        // observable.
        let mpo = BraketEnvError::MalformedEdgeBond { leg: "mpo_left" }.to_string();
        assert!(mpo.contains("fusing to identity flux"));

        let bra = BraketEnvError::MalformedEdgeBond { leg: "bra_left" }.to_string();
        assert!(!bra.contains("fusing to identity flux"));
    }
}
