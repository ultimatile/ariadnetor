//! TensorChain trait — common operations for MPS/MPO tensor chains

use std::sync::Arc;

use arnet_core::backend::ComputeBackend;
use arnet_tensor::TensorStorage;

use super::types::{CanonicalForm, Mpo, Mps};

/// Common operations for MPS/MPO tensor chains.
///
/// Provides rank-independent accessors for site storages, bond dimensions,
/// canonical form tracking, and backend access.
pub trait TensorChain<T, B: ComputeBackend> {
    /// Number of sites.
    fn len(&self) -> usize;

    /// Whether the chain has no sites.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reference to the storage at a given site.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    fn storage(&self, site: usize) -> &TensorStorage<T>;

    /// Mutable reference to the storage at a given site.
    ///
    /// Resets canonical form to `Unknown` since the tensor data may be modified.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    fn storage_mut(&mut self, site: usize) -> &mut TensorStorage<T>;

    /// Slice of all site storages.
    fn storages(&self) -> &[TensorStorage<T>];

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
    /// This is the last mode dimension of site `bond`, which equals the
    /// first mode dimension of site `bond + 1`.
    ///
    /// # Panics
    ///
    /// Panics if `bond >= len() - 1`.
    fn bond_dim(&self, bond: usize) -> usize {
        let shape = self.storage(bond).shape();
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
        impl<T, B: ComputeBackend> TensorChain<T, B> for $type<T, B> {
            fn len(&self) -> usize {
                self.0.storages.len()
            }

            fn storage(&self, site: usize) -> &TensorStorage<T> {
                &self.0.storages[site]
            }

            fn storage_mut(&mut self, site: usize) -> &mut TensorStorage<T> {
                self.0.canonical_form = CanonicalForm::Unknown;
                &mut self.0.storages[site]
            }

            fn storages(&self) -> &[TensorStorage<T>] {
                &self.0.storages
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
