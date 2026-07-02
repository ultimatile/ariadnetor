//! Tests for the DMRG 2-site local update (`heff` module).
//!
//! Strategy: build small chains (n=3-4, d=2, chi up to 3) with
//! Hermitian MPOs constructed as bond-1 local-product operators
//! (each site = a random Hermitian d×d matrix). Hermiticity of the
//! global H gives a Hermitian H_eff regardless of MPS canonical
//! form, so `eigh` of the dense H_eff matrix is a valid ground
//! truth for Lanczos.
//!
//! The dense H_eff is recovered by applying `EffectiveHamiltonian2Site`
//! to each standard basis vector — there is no `2^n × 2^n` global
//! Hamiltonian construction (H_eff lives in the local projected
//! subspace).

use super::{identity_mpo, product_state_mps};
use crate::dmrg::heff::{EffectiveHamiltonian2Site, dmrg_2site_step};
use crate::dmrg::{DmrgHeffError, LocalEigensolverParams};
use crate::krylov::{LanczosError, LanczosParams, LinearOp};
use approx::assert_abs_diff_eq;
use ariadnetor_core::Scalar;
use ariadnetor_linalg::{TruncSvdParams, contract, diagonal_scale, eigh_with_backend};
use ariadnetor_mps::BraketEnvs;
use ariadnetor_mps::{Mpo, Mps};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};
use num_complex::Complex;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Fixtures (`product_state_mps` / `identity_mpo` are shared from the parent
// `tests` module)
// ---------------------------------------------------------------------------

/// Random-but-seeded MPS with chi internal, d physical, n sites.
fn random_mps_f64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            Host::shared().dense(data, vec![l, d, r])
        })
        .collect();
    Mps::from_sites(storages)
}

fn random_mps_c64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<Complex<f64>>, DenseLayout> {
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<Complex<f64>>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<Complex<f64>> = (0..len)
                .map(|_| {
                    let re = rng.random_range(-0.5_f64..0.5);
                    let im = rng.random_range(-0.5_f64..0.5);
                    Complex::new(re, im)
                })
                .collect();
            Host::shared().dense(data, vec![l, d, r])
        })
        .collect();
    Mps::from_sites(storages)
}

/// Hermitian "local-product" MPO: bond dim 1 at every site, each
/// site stores a random Hermitian `d × d` operator. The global
/// H = h_0 ⊗ h_1 ⊗ ... ⊗ h_{n-1} is Hermitian for any choice of
/// per-site Hermitian h_i, so the projected H_eff is also Hermitian.
fn hermitian_local_mpo_f64(n: usize, d: usize, seed: u64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            // d×d Hermitian (real-symmetric) block: M = 0.5 * (R + R^T).
            let mut m = vec![0.0_f64; d * d];
            // Stored row-major: m[s * d + t].
            let mut r = vec![0.0_f64; d * d];
            for entry in r.iter_mut() {
                *entry = rng.random_range(-1.0_f64..1.0);
            }
            for s in 0..d {
                for t in 0..d {
                    m[s * d + t] = 0.5 * (r[s * d + t] + r[t * d + s]);
                }
            }
            // MPO axis order [W_l=1, d_ket=s, d_bra=t, W_r=1]. The
            // local Hermitian operator h satisfies h[s,t] = conj(h[t,s]),
            // and we identify W[0, s, t, 0] = h[s, t].
            Host::shared().dense(m, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(storages)
}

fn hermitian_local_mpo_c64(
    n: usize,
    d: usize,
    seed: u64,
) -> Mpo<DenseStorage<Complex<f64>>, DenseLayout> {
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<Complex<f64>>> = (0..n)
        .map(|_| {
            let mut m = vec![Complex::new(0.0, 0.0); d * d];
            let mut r = vec![Complex::new(0.0, 0.0); d * d];
            for entry in r.iter_mut() {
                let re = rng.random_range(-1.0_f64..1.0);
                let im = rng.random_range(-1.0_f64..1.0);
                *entry = Complex::new(re, im);
            }
            for s in 0..d {
                for t in 0..d {
                    // Hermitian: h[s, t] = 0.5 * (r[s, t] + conj(r[t, s])).
                    m[s * d + t] = (r[s * d + t] + r[t * d + s].conj()) * 0.5;
                }
            }
            Host::shared().dense(m, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(storages)
}

/// Build the dense matrix representation of the local effective
/// Hamiltonian by applying the operator to every standard basis
/// vector. Storage convention is column-major: entry `(i, k)`
/// (the `i`-th component of `H_eff @ e_k`) lives at flat index
/// `i + dim * k`.
fn build_heff_dense<T>(heff: &EffectiveHamiltonian2Site<'_, T>) -> DenseTensor<T>
where
    T: Scalar,
{
    let dim = heff.dim();
    let mut data = vec![T::zero(); dim * dim];
    for k in 0..dim {
        let mut e = vec![T::zero(); dim];
        e[k] = T::one();
        let e_dense = Host::shared().dense(e, vec![dim]);
        let col = heff.apply(&e_dense);
        let col_data = col.data_slice();
        for i in 0..dim {
            data[i + dim * k] = col_data[i];
        }
    }
    Host::shared().dense(data, vec![dim, dim])
}

/// Construct an `EffectiveHamiltonian2Site` borrowing a freshly built
/// envs at the requested two-site index. The returned operator borrows
/// the `envs` / `mps` / `mpo` arguments, which the caller keeps alive
/// so the references stay valid.
fn make_heff<'a, T>(
    envs: &'a BraketEnvs<DenseStorage<T>, DenseLayout>,
    mps: &'a Mps<DenseStorage<T>, DenseLayout>,
    mpo: &'a Mpo<DenseStorage<T>, DenseLayout>,
    site: usize,
) -> EffectiveHamiltonian2Site<'a, T>
where
    T: Scalar,
    T::Real: Scalar<Real = T::Real>,
{
    use ariadnetor_mps::TensorChain;
    let left = envs.left(site).expect("left(site)");
    let right = envs.right(site + 2).expect("right(site+2)");
    let w_i = mpo.site(site);
    let w_ip1 = mpo.site(site + 1);
    let mps_i = mps.site(site);
    let mps_ip1 = mps.site(site + 1);
    let chi_l = left.shape()[0];
    let chi_r = right.shape()[0];
    let d_i = mps_i.shape()[1];
    let d_ip1 = mps_ip1.shape()[1];

    EffectiveHamiltonian2Site::new(left, w_i, w_ip1, right, chi_l, d_i, d_ip1, chi_r)
}

// ---------------------------------------------------------------------------
// T1 — identity MPO + product state: matvec is identity, eigvalue = 1
// ---------------------------------------------------------------------------

#[test]
fn heff_identity_smoke() {
    let n = 4;
    let mps = product_state_mps(n, 2);
    let mpo = identity_mpo(n, 2);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    // Walk left envs up to site=1 so left(1) exists.
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    let result = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        1,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(7),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("step");
    assert_abs_diff_eq!(result.eigenvalue, 1.0, epsilon = 1e-9);
    assert!(result.converged, "Lanczos must converge on identity H_eff");
}

// ---------------------------------------------------------------------------
// T2 — matvec contract: dense H_eff via basis apply matches apply(random v)
// ---------------------------------------------------------------------------

#[test]
fn heff_matvec_matches_dense_apply() {
    let n = 4;
    let d = 2;
    let chi = 2;
    let mps = random_mps_f64(n, d, chi, 0xCAFE_F00D);
    let mpo = hermitian_local_mpo_f64(n, d, 0xBEEF_DEAD);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    let site = 1;
    let heff = make_heff(&envs, &mps, &mpo, site);
    let h_dense = build_heff_dense(&heff);
    let dim = heff.dim();

    // Hermitian check: H_dense[i, k] == conj(H_dense[k, i]) (real → equal).
    let h_data = h_dense.data_slice();
    for i in 0..dim {
        for k in 0..dim {
            let aij = h_data[i + dim * k];
            let aji = h_data[k + dim * i];
            assert_abs_diff_eq!(aij, aji, epsilon = 1e-12);
        }
    }

    // Apply on a random vector via the operator and compare to `H_dense @ v`.
    let mut rng = StdRng::seed_from_u64(0xABCD_1234);
    let v_data: Vec<f64> = (0..dim).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
    let v = Host::shared().dense(v_data.clone(), vec![dim]);

    let apply_out = heff.apply(&v);
    let apply_data = apply_out.data_slice();

    let mut dense_out = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            dense_out[i] += h_data[i + dim * k] * v_data[k];
        }
    }
    for i in 0..dim {
        assert_abs_diff_eq!(apply_data[i], dense_out[i], epsilon = 1e-9);
    }
}

// ---------------------------------------------------------------------------
// T3 — Lanczos eigenvalue matches eigh ground truth
// ---------------------------------------------------------------------------

#[test]
fn heff_lanczos_eigvalue_matches_eigh() {
    let n = 4;
    let d = 2;
    let chi = 2;
    let mps = random_mps_f64(n, d, chi, 0xCAFE_F00D);
    let mpo = hermitian_local_mpo_f64(n, d, 0xBEEF_DEAD);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    let site = 1;
    let heff = make_heff(&envs, &mps, &mpo, site);
    let h_dense = build_heff_dense(&heff);
    let (eigvals, _) = eigh_with_backend(&NativeBackend::new(), &h_dense, 1).expect("eigh");
    let reference = eigvals.data_slice()[0];

    let result = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        site,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(11),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("step");

    assert_abs_diff_eq!(result.eigenvalue, reference, epsilon = 1e-9);
    assert!(result.converged, "Lanczos must converge within budget");
    assert!(result.iters < 200, "iters must stay within max_iter budget");
}

// ---------------------------------------------------------------------------
// T4 — SVD canonical form: U^T U = I and Vt Vt^T = I
// ---------------------------------------------------------------------------

#[test]
fn heff_svd_split_is_canonical() {
    let n = 4;
    let d = 2;
    let chi = 2;
    let mps = random_mps_f64(n, d, chi, 0xCAFE_F00D);
    let mpo = hermitian_local_mpo_f64(n, d, 0xBEEF_DEAD);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    let result = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        1,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(11),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("step");

    // U^T U: contract `[chi_l, d, chi_new]` with itself, summing the
    // (chi_l, d) axes → Identity on the chi_new axis.
    let utu = contract(&NativeBackend::new(), &result.u, &result.u, "abc,abd->cd").expect("U^T U");
    let chi_new = result.u.shape()[2];
    assert_eq!(utu.shape(), &[chi_new, chi_new]);
    let utu_data = utu.data_slice();
    for i in 0..chi_new {
        for j in 0..chi_new {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(utu_data[i * chi_new + j], expected, epsilon = 1e-10);
        }
    }
    // Vt Vt^T: contract `[chi_new, d, chi_r]` with itself summing
    // the (d, chi_r) axes → Identity on the chi_new axis.
    let vvt =
        contract(&NativeBackend::new(), &result.vt, &result.vt, "abc,dbc->ad").expect("Vt Vt^T");
    assert_eq!(vvt.shape(), &[chi_new, chi_new]);
    let vvt_data = vvt.data_slice();
    for i in 0..chi_new {
        for j in 0..chi_new {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(vvt_data[i * chi_new + j], expected, epsilon = 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// T5 — SVD reconstruction: U · diag(S) · Vt = optimized 2-site block
//      (within trunc_err if any singular value was discarded)
// ---------------------------------------------------------------------------

#[test]
fn heff_svd_reconstruction_round_trips() {
    let n = 4;
    let d = 2;
    let chi = 2;
    let mps = random_mps_f64(n, d, chi, 0xCAFE_F00D);
    let mpo = hermitian_local_mpo_f64(n, d, 0xBEEF_DEAD);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    let site = 1;
    let lan_params = LanczosParams {
        seed: Some(11),
        ..LanczosParams::default()
    };
    let eigensolver = LocalEigensolverParams::Lanczos(lan_params.clone());
    let result = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        site,
        &eigensolver,
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("step");

    // No truncation requested → trunc_err must be zero (within float
    // slack) and `sum(s²) = ||psi||² = 1` (psi is unit-norm).
    assert_abs_diff_eq!(result.trunc_err, 0.0, epsilon = 1e-10);
    let s_sq_sum: f64 = result.s.data_slice().iter().map(|v| v * v).sum();
    assert_abs_diff_eq!(s_sq_sum, 1.0, epsilon = 1e-9);
    // Singular values are descending and non-negative.
    let s_data = result.s.data_slice().to_vec();
    for window in s_data.windows(2) {
        assert!(window[0] >= window[1] - 1e-12, "s descending");
    }
    for v in s_data.iter() {
        assert!(*v >= -1e-12, "s non-negative");
    }

    // Reconstruct U · diag(S) · Vt and verify it is a valid 2-site
    // block of the original eigenvector (re-running Lanczos with
    // the same seed is deterministic, so we can compare exactly).
    let us = diagonal_scale(&NativeBackend::new(), &result.u, result.s.data_slice(), 2)
        .expect("U·diag(S)");
    let recon = contract(&NativeBackend::new(), &us, &result.vt, "aik,kjb->aijb").expect("U·S·Vt");

    let heff = make_heff(&envs, &mps, &mpo, site);
    let dim = heff.dim();
    let lan = crate::krylov::lanczos_smallest::<f64, _>(&heff, dim, &lan_params).unwrap();
    let psi_4d = lan
        .eigenvector
        .reshape(vec![heff.chi_l, heff.d_i, heff.d_ip1, heff.chi_r]);

    let psi_data = psi_4d.data_slice();
    let recon_data = recon.data_slice();
    // Eigenvectors are determined up to sign; check the residual
    // for both ±recon and require the smaller one to be ~0.
    let frob_plus: f64 = psi_data
        .iter()
        .zip(recon_data.iter())
        .map(|(p, r)| (p - r).powi(2))
        .sum::<f64>()
        .sqrt();
    let frob_minus: f64 = psi_data
        .iter()
        .zip(recon_data.iter())
        .map(|(p, r)| (p + r).powi(2))
        .sum::<f64>()
        .sqrt();
    let frob = frob_plus.min(frob_minus);
    let tol = result.trunc_err + 1e-9;
    assert!(
        frob <= tol,
        "Frobenius residual {} exceeds trunc_err+slack {} (plus={}, minus={})",
        frob,
        tol,
        frob_plus,
        frob_minus
    );
}

// ---------------------------------------------------------------------------
// T6 — edge sites: chi_l = 1 and chi_r = 1 succeed
// ---------------------------------------------------------------------------

#[test]
fn heff_edge_sites_succeed() {
    let n = 3;
    let d = 2;
    let chi = 2;
    let mps = random_mps_f64(n, d, chi, 0xFEED_CAFE);
    let mpo = hermitian_local_mpo_f64(n, d, 0x0BAD_F00D);

    // site = 0 → left(0) trivial 1x1x1 boundary, chi_l = 1.
    // No advance needed; build seeds left(0) and right(2..=N).
    let envs0 = BraketEnvs::build(&mps, &mpo).expect("build site=0");
    let r0 = dmrg_2site_step(
        &envs0,
        &mps,
        &mpo,
        0,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(3),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("site=0");
    assert!(r0.converged);
    assert_eq!(r0.u.shape()[0], 1);

    // site = n - 2 → right(n) trivial boundary, chi_r = 1.
    // Need left(n-2) populated, so walk advance_left up to n-3.
    let mut envs1 = BraketEnvs::build(&mps, &mpo).expect("build site=n-2");
    for k in 0..(n - 2) {
        envs1.advance_left(&mps, &mpo, k).expect("advance_left");
    }
    let r1 = dmrg_2site_step(
        &envs1,
        &mps,
        &mpo,
        n - 2,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(5),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    )
    .expect("site=n-2");
    assert!(r1.converged);
    assert_eq!(r1.vt.shape()[2], 1);
}

// ---------------------------------------------------------------------------
// T8 — Complex<f64> matvec contract: catches conjugation regressions
// ---------------------------------------------------------------------------

#[test]
fn heff_matvec_matches_dense_apply_complex() {
    let n = 3;
    let d = 2;
    let chi = 2;
    let mps = random_mps_c64(n, d, chi, 0x1357_9BDF);
    let mpo = hermitian_local_mpo_c64(n, d, 0x2468_ACE0);
    let envs = BraketEnvs::build(&mps, &mpo).expect("build");

    let site = 0;
    let heff = make_heff(&envs, &mps, &mpo, site);
    let h_dense = build_heff_dense(&heff);
    let dim = heff.dim();
    let h_data = h_dense.data_slice();

    // Hermitian check (transpose-conj symmetry).
    for i in 0..dim {
        for k in 0..dim {
            let aij = h_data[i + dim * k];
            let aji = h_data[k + dim * i];
            assert_abs_diff_eq!(aij.re, aji.conj().re, epsilon = 1e-12);
            assert_abs_diff_eq!(aij.im, aji.conj().im, epsilon = 1e-12);
        }
    }

    // Apply contract pin.
    let mut rng = StdRng::seed_from_u64(0xFADE_D00D);
    let v_data: Vec<Complex<f64>> = (0..dim)
        .map(|_| {
            let re = rng.random_range(-0.5_f64..0.5);
            let im = rng.random_range(-0.5_f64..0.5);
            Complex::new(re, im)
        })
        .collect();
    let v = Host::shared().dense(v_data.clone(), vec![dim]);
    let apply_out = heff.apply(&v);
    let apply_data = apply_out.data_slice();
    let mut dense_out = vec![Complex::new(0.0, 0.0); dim];
    for i in 0..dim {
        for k in 0..dim {
            dense_out[i] += h_data[i + dim * k] * v_data[k];
        }
    }
    for i in 0..dim {
        assert_abs_diff_eq!(apply_data[i].re, dense_out[i].re, epsilon = 1e-9);
        assert_abs_diff_eq!(apply_data[i].im, dense_out[i].im, epsilon = 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Fallible-API contract: a non-finite local solve aborts the step
// ---------------------------------------------------------------------------

/// Identity-diagonal local MPO with a single NaN entry on `poison_site`,
/// so the 2-site step covering that site builds a non-finite `H_eff`.
/// Shapes / order stay valid (same `[1, d, d, 1]` blocks as
/// [`identity_mpo`]), so the step's pre-validation passes — it checks
/// shape / order, not value finiteness — and the NaN reaches the Lanczos
/// matvec.
fn nan_poisoned_mpo_f64(
    n: usize,
    d: usize,
    poison_site: usize,
) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|site| {
            let mut data = vec![0.0_f64; d * d];
            for k in 0..d {
                data[k + d * k] = 1.0;
            }
            if site == poison_site {
                data[0] = f64::NAN;
            }
            Host::shared().dense(data, vec![1, d, d, 1])
        })
        .collect();
    Mpo::from_sites(storages)
}

#[test]
fn dmrg_step_surfaces_nonfinite_lanczos_as_error() {
    let (n, d) = (4usize, 2usize);
    let mps = product_state_mps(n, d);
    // Poison site 1, which the step at site 1 contracts into H_eff.
    let mpo = nan_poisoned_mpo_f64(n, d, 1);
    let mut envs = BraketEnvs::build(&mps, &mpo).expect("build");
    envs.advance_left(&mps, &mpo, 0).expect("advance_left(0)");

    // The non-finite H_eff drives the Lanczos local solve non-finite. The
    // step must surface it as `DmrgHeffError::Lanczos(LanczosError::NonFinite)`
    // — not panic, and not return a silently-NaN `Ok` result that would poison
    // the sweep energy downstream.
    let result = dmrg_2site_step(
        &envs,
        &mps,
        &mpo,
        1,
        &LocalEigensolverParams::Lanczos(LanczosParams {
            seed: Some(7),
            ..LanczosParams::default()
        }),
        &TruncSvdParams {
            chi_max: None,
            target_trunc_err: None,
        },
    );

    match result {
        Err(DmrgHeffError::Lanczos(LanczosError::NonFinite { .. })) => {}
        Err(other) => panic!("expected Lanczos(NonFinite), got a different error: {other:?}"),
        Ok(_) => panic!("expected an error, got Ok — the NaN was not surfaced"),
    }
}
