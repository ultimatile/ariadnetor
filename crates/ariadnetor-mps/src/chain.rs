//! Tensor chain accessor trait — common operations for MPS / MPO.
//!
//! [`TensorChain`] reads sites as
//! [`TensorData<St, L>`](TensorData) (backs [`Mps`] / [`Mpo`]).

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_tensor::{Storage, StorageFor, TensorData, TensorLayout};

use super::types::{CanonicalForm, Mpo, Mps};

// ============================================================================
// `TensorData<St, L>` trait — backs `Mps` / `Mpo`
// ============================================================================

/// Common operations for MPS / MPO tensor chains over a paired
/// [`Storage`] + [`TensorLayout`]. Each site is a
/// [`TensorData<St, L>`] with bonds read from the layout's
/// [`shape`](TensorLayout::shape).
pub trait TensorChain<St, L, B: ComputeBackend>
where
    St: Storage + StorageFor<L>,
    L: TensorLayout,
{
    /// Number of sites.
    fn len(&self) -> usize;

    /// Whether the chain has no sites.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reference to the site `TensorData` at a given index.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    fn site(&self, site: usize) -> &TensorData<St, L>;

    /// Mutable reference to the site `TensorData`.
    ///
    /// Resets canonical form to `Unknown` since the tensor data may be
    /// modified through this handle.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    fn site_mut(&mut self, site: usize) -> &mut TensorData<St, L>;

    /// Slice of all site `TensorData`s.
    fn sites(&self) -> &[TensorData<St, L>];

    /// Current canonical form.
    fn canonical_form(&self) -> &CanonicalForm;

    /// Set the canonical form.
    fn set_canonical_form(&mut self, form: CanonicalForm);

    /// Reference to the compute backend.
    fn backend(&self) -> &B;

    /// Shared reference to the backend Arc.
    fn backend_arc(&self) -> &Arc<B>;

    /// Bond dimension between site `bond` and site `bond + 1`.
    ///
    /// Reads the last logical mode dimension of site `bond` via its
    /// layout's [`shape`](TensorLayout::shape).
    ///
    /// # Panics
    ///
    /// Panics if `bond >= len() - 1`.
    fn bond_dim(&self, bond: usize) -> usize {
        let shape = self.site(bond).layout().shape();
        shape[shape.len() - 1]
    }

    /// All bond dimensions (length N-1 for N sites).
    fn bond_dims(&self) -> Vec<usize> {
        let n = self.len();
        if n <= 1 {
            return Vec::new();
        }
        (0..n - 1).map(|j| self.bond_dim(j)).collect()
    }

    /// Maximum bond dimension across all bonds.
    fn max_bond_dim(&self) -> usize {
        self.bond_dims().into_iter().max().unwrap_or(0)
    }
}

macro_rules! impl_tensor_chain {
    ($type:ident) => {
        impl<St, L, B> TensorChain<St, L, B> for $type<St, L, B>
        where
            St: Storage + StorageFor<L>,
            L: TensorLayout,
            B: ComputeBackend,
        {
            fn len(&self) -> usize {
                self.0.sites.len()
            }

            fn site(&self, site: usize) -> &TensorData<St, L> {
                &self.0.sites[site]
            }

            fn site_mut(&mut self, site: usize) -> &mut TensorData<St, L> {
                self.0.canonical_form = CanonicalForm::Unknown;
                &mut self.0.sites[site]
            }

            fn sites(&self) -> &[TensorData<St, L>] {
                &self.0.sites
            }

            fn canonical_form(&self) -> &CanonicalForm {
                &self.0.canonical_form
            }

            fn set_canonical_form(&mut self, form: CanonicalForm) {
                self.0.canonical_form = form;
            }

            fn backend(&self) -> &B {
                &self.0.backend
            }

            fn backend_arc(&self) -> &Arc<B> {
                &self.0.backend
            }
        }
    };
}

impl_tensor_chain!(Mps);
impl_tensor_chain!(Mpo);
