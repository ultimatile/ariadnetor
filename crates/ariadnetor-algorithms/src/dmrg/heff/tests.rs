//! In-crate white-box unit tests for the Dense `heff` per-step
//! primitives (`EffectiveHamiltonian2Site`, `dmrg_2site_step`). They
//! construct the operator and drive the per-step entry point directly
//! to assert on per-step outputs (matvec, eigenvalue, SVD split,
//! error variants), so they live next to the crate-internal code they
//! exercise.
//!
//! `product_state_mps` / `identity_mpo` are the minimal fixtures shared
//! across the submodules; richer per-test builders stay local to each.

use arnet_mps::{Mpo, Mps};
use arnet_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};

mod error_paths;
mod predicate_coverage;
mod step;

#[cfg(feature = "arpack")]
mod arpack;

/// Product state |0...0⟩ with bond dim 1 at every internal bond.
fn product_state_mps(n: usize, d: usize) -> Mps<DenseStorage<f64>, DenseLayout> {
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d];
            data[0] = 1.0;
            Host::shared().dense(data, vec![1, d, 1])
        })
        .collect();
    Mps::from_sites(sites)
}

/// Identity MPO at every site, bond dim 1.
fn identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let sites: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d * d];
            for k in 0..d {
                data[k + d * k] = 1.0;
            }
            Host::shared().dense(data, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(sites)
}
