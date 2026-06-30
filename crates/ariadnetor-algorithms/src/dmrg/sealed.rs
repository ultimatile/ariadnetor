//! Crate-private seal for the chain-keyed `DmrgEnvOps` dispatch trait.
//!
//! [`Sealed`] is the supertrait of [`super::DmrgEnvOps`]; living in a
//! private module, its name is not reachable downstream, so the trait
//! cannot be implemented outside this crate. It carries no associated
//! surface, so the public trait projects no storage / layout taxonomy
//! through it.
//!
//! [`super::DmrgOps`] is sealed separately — transitively — through its
//! `ariadnetor_mps::MpsOps` supertrait, which is itself sealed, so it needs no
//! impl here.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{BlockSparseLayout, BlockSparseStorage, DenseLayout, DenseStorage, Sector};

use super::env::DmrgEnvs;

pub trait Sealed {}

impl<T: Scalar> Sealed for DmrgEnvs<DenseStorage<T>, DenseLayout> {}
impl<T: Scalar, S: Sector> Sealed for DmrgEnvs<BlockSparseStorage<T>, BlockSparseLayout<S>> {}
