//! Inner product, norm, and expectation value for MPS

use arnet_core::backend::ComputeBackend;
use arnet_core::scalar::Scalar;
use arnet_linalg::contract;
use arnet_tensor::ComputeBackendTensorExt;
use num_traits::{Float, One};

use super::chain::TensorChain;
use super::types::{CanonicalForm, Mpo, Mps};

/// Compute the inner product ⟨ψ|φ⟩ of two MPS via the transfer matrix method.
///
/// Contracts left-to-right, accumulating a (χ_ψ × χ_φ) environment tensor.
///
/// # Panics
///
/// Panics if the MPS lengths differ or either is empty.
pub fn inner<T, B>(psi: &Mps<T, B>, phi: &Mps<T, B>) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert!(n > 0, "MPS must have at least one site");

    let backend = psi.backend();

    // Environment: (χ_ψ, χ_φ), starts as 1×1 identity
    let mut env = backend.make_tensor(vec![T::one()], vec![1, 1]);

    for j in 0..n {
        let psi_j = psi.storage(j).conj();
        let phi_j = phi.storage(j);

        // env(a,b) × conj(ψ)(a,d,c) → temp(b,d,c)
        let temp = contract(backend, &env, &psi_j, "ab,adc->bdc")
            .expect("inner product contraction failed");

        // temp(b,d,c) × φ(b,d,e) → new_env(c,e)
        env = contract(backend, &temp, phi_j, "bdc,bde->ce")
            .expect("inner product contraction failed");
    }

    env.get(&[0, 0])
}

/// Compute the norm ‖ψ‖ = √⟨ψ|ψ⟩.
///
/// If the MPS is in canonical form, exploits the structure for O(1) computation:
/// - `Left`/`Right`: returns 1.0 (normalized by construction).
/// - `Mixed`: returns Frobenius norm of the orthogonality center tensor.
///
/// Otherwise computes the full inner product.
pub fn norm<T, B>(psi: &Mps<T, B>) -> T::Real
where
    T: Scalar,
    B: ComputeBackend,
{
    match psi.canonical_form() {
        CanonicalForm::Left | CanonicalForm::Right => T::Real::one(),
        CanonicalForm::Mixed { center } => psi.storage(*center).norm(),
        _ => {
            let overlap = inner(psi, psi);
            overlap.re().sqrt()
        }
    }
}

/// Compute ⟨ψ|A|φ⟩ — the MPO-inserted inner product (bra-ket with operator).
///
/// Contracts left-to-right with a (χ_ψ × χ_A × χ_φ) environment tensor.
/// When `psi == phi`, this is the expectation value of `A`.
///
/// # Panics
///
/// Panics if the MPS/MPO lengths differ or any is empty.
pub fn braket<T, B>(psi: &Mps<T, B>, op: &Mpo<T, B>, phi: &Mps<T, B>) -> T
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = psi.len();
    assert_eq!(n, phi.len(), "MPS lengths must match");
    assert_eq!(n, op.len(), "MPO length must match MPS length");
    assert!(n > 0, "must have at least one site");

    let backend = psi.backend();

    // Environment: (χ_ψ, χ_A, χ_φ), starts as 1×1×1
    let mut env = backend.make_tensor(vec![T::one()], vec![1, 1, 1]);

    for j in 0..n {
        let psi_j = psi.storage(j).conj(); // bra: (ψ_L, d_bra, ψ_R)
        let a_j = op.storage(j); // operator: (A_L, d_ket, d_bra, A_R)
        let phi_j = phi.storage(j); // ket: (φ_L, d_ket, φ_R)

        // Step 1: env(a,b,c) × conj(ψ)(a,d,e) → temp1(b,c,d,e)
        let temp1 = contract(backend, &env, &psi_j, "abc,ade->bcde")
            .expect("expect contraction step 1 failed");

        // Step 2: temp1(b,c,d,e) × A(b,f,d,g) → temp2(c,e,f,g)
        let temp2 = contract(backend, &temp1, a_j, "bcde,bfdg->cefg")
            .expect("expect contraction step 2 failed");

        // Step 3: temp2(c,e,f,g) × φ(c,f,h) → env_new(e,g,h)
        env = contract(backend, &temp2, phi_j, "cefg,cfh->egh")
            .expect("expect contraction step 3 failed");
    }

    env.get(&[0, 0, 0])
}
