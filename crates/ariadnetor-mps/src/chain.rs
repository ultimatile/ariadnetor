//! TensorChain trait — common operations for MPS/MPO tensor chains.

use arnet_tensor::{Storage, StorageFor, Tensor, TensorLayout};

use super::types::{CanonicalForm, Mpo, Mps};

/// Common operations for MPS/MPO tensor chains.
///
/// Provides rank-independent accessors for site tensors, bond
/// dimensions, and canonical form tracking. The chain carries no
/// backend; operations receive one at their call site.
pub trait TensorChain<St, L>
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

    /// Reference to the site tensor at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len()`.
    fn site(&self, idx: usize) -> &Tensor<St, L>;

    /// Mutable reference to the site tensor at the given index.
    ///
    /// Resets canonical form to `Unknown` since the tensor data may be
    /// modified.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len()`.
    fn site_mut(&mut self, idx: usize) -> &mut Tensor<St, L>;

    /// Slice of all site tensors.
    fn sites(&self) -> &[Tensor<St, L>];

    /// Current canonical form.
    fn canonical_form(&self) -> &CanonicalForm;

    /// Set the canonical form.
    fn set_canonical_form(&mut self, form: CanonicalForm);

    /// Bond dimension between site `bond` and site `bond + 1`.
    ///
    /// This is the last mode dimension of site `bond`, which equals the
    /// first mode dimension of site `bond + 1`.
    ///
    /// # Panics
    ///
    /// Panics if `bond >= len() - 1`.
    fn bond_dim(&self, bond: usize) -> usize {
        let shape = self.site(bond).shape();
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
        impl<St, L> TensorChain<St, L> for $type<St, L>
        where
            St: Storage + StorageFor<L>,
            L: TensorLayout,
        {
            fn len(&self) -> usize {
                self.0.sites.len()
            }

            fn site(&self, idx: usize) -> &Tensor<St, L> {
                &self.0.sites[idx]
            }

            fn site_mut(&mut self, idx: usize) -> &mut Tensor<St, L> {
                self.0.canonical_form = CanonicalForm::Unknown;
                &mut self.0.sites[idx]
            }

            fn sites(&self) -> &[Tensor<St, L>] {
                &self.0.sites
            }

            fn canonical_form(&self) -> &CanonicalForm {
                &self.0.canonical_form
            }

            fn set_canonical_form(&mut self, form: CanonicalForm) {
                self.0.canonical_form = form;
            }
        }
    };
}

impl_tensor_chain!(Mps);
impl_tensor_chain!(Mpo);
