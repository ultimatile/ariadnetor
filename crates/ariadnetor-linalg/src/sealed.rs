//! Crate-private seal for the tensor-keyed dispatch traits
//! ([`LinalgDecompose`](crate::LinalgDecompose) /
//! [`LinalgContract`](crate::LinalgContract) /
//! [`LinalgScale`](crate::LinalgScale)).
//!
//! [`Sealed`] is the shared supertrait of the three dispatch traits; living in
//! a private module, its name is not reachable downstream, so none of the
//! traits can be implemented outside this crate. It carries no associated
//! surface, so the public traits project no storage / layout taxonomy through
//! it: their `Storage` associated type survives only behind this un-nameable
//! seal.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{
    BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, Sector, Tensor,
};

pub trait Sealed {}

impl<T: Scalar> Sealed for Tensor<DenseStorage<T>, DenseLayout> {}
impl<T: Scalar, S: Sector> Sealed for Tensor<BlockSparseStorage<T>, BlockSparseLayout<S>> {}
