//! ARPACK-arm coverage for the 2-site DMRG step. Gated under the
//! `arpack` feature.
//!
//! - **Happy-path equivalence**: `dmrg_2site` with
//!   `LocalEigensolverParams::Arpack(...)` must converge to the same
//!   ground-state energy as the Lanczos arm on the same Heisenberg
//!   MPO + random MPS, within a tolerance comparable to both
//!   solvers' configured `tol`. This exercises the new ARPACK
//!   match arm in `dmrg_2site_step{,_block_sparse}` end-to-end.
//! - **Error-path forwarding**: `dmrg_2site_step` with ARPACK at
//!   `max_iter = 1` on a non-trivial Heff must surface
//!   `Err(DmrgHeffError::Arpack(ArpackError::MaxIterReached { .. }))`,
//!   pinning the new `From<ArpackError>` impl + the
//!   `DmrgHeffError::Arpack` variant.
//!
//! The Heisenberg MPO + random MPS builders are inlined to mirror
//! the convention in `tests/dmrg_wrapper.rs` (each test file
//! self-contained).

#![cfg(feature = "arpack")]

use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arnet::TruncSvdParams;
use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, NativeBackend};
use arnet_algorithms::dmrg::{
    DmrgEnvs, DmrgHeffError, DmrgSweepParams, LocalEigensolverParams, dmrg_2site, dmrg_2site_step,
};
use arnet_algorithms::krylov::{ArpackError, ArpackParams, LanczosParams};
use arnet_mps::{Mpo, Mps};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const D: usize = 2;

// ---------------------------------------------------------------------------
// Heisenberg MPO + random MPS — inlined per existing test convention.
// ---------------------------------------------------------------------------

type Op = fn(usize, usize) -> f64;

fn op_id(k: usize, b: usize) -> f64 {
    if k == b { 1.0 } else { 0.0 }
}

fn op_sz(k: usize, b: usize) -> f64 {
    if k == b {
        if k == 0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

fn op_sm(k: usize, b: usize) -> f64 {
    if k == 0 && b == 1 { 1.0 } else { 0.0 }
}

fn build_mpo_site_f64(
    w_l_dim: usize,
    w_r_dim: usize,
    cells: &[(usize, usize, Op, f64)],
) -> DenseTensor<f64> {
    let backend = NativeBackend::shared();
    let len = w_l_dim * D * D * w_r_dim;
    let mut data = vec![0.0_f64; len];
    for &(vl, vr, op, scale) in cells {
        for k in 0..D {
            for b in 0..D {
                let idx = vl + w_l_dim * (k + D * (b + D * vr));
                data[idx] += scale * op(k, b);
            }
        }
    }
    DenseTensor::from_raw_parts(
        data,
        vec![w_l_dim, D, D, w_r_dim],
        backend.preferred_order(),
        Arc::clone(&backend),
    )
}

fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2, "heisenberg_mpo_f64 requires n >= 2");
    let mut sites = Vec::with_capacity(n);

    sites.push(build_mpo_site_f64(
        1,
        5,
        &[
            (0, 1, op_sm, 2.0 * j),
            (0, 2, op_sp, 2.0 * j),
            (0, 3, op_sz, j),
            (0, 4, op_id, 1.0),
        ],
    ));

    for _ in 1..n - 1 {
        sites.push(build_mpo_site_f64(
            5,
            5,
            &[
                (0, 0, op_id, 1.0),
                (1, 0, op_sp, 1.0),
                (2, 0, op_sm, 1.0),
                (3, 0, op_sz, 1.0),
                (4, 1, op_sm, 2.0 * j),
                (4, 2, op_sp, 2.0 * j),
                (4, 3, op_sz, j),
                (4, 4, op_id, 1.0),
            ],
        ));
    }

    sites.push(build_mpo_site_f64(
        5,
        1,
        &[
            (0, 0, op_id, 1.0),
            (1, 0, op_sp, 1.0),
            (2, 0, op_sm, 1.0),
            (3, 0, op_sz, 1.0),
        ],
    ));

    Mpo::from_sites(sites)
}

fn random_mps_unknown_f64(n: usize, chi: usize, seed: u64) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * D * r;
            let data: Vec<f64> = (0..len).map(|_| rng.random_range(-0.5_f64..0.5)).collect();
            DenseTensor::from_raw_parts(
                data,
                vec![l, D, r],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mps::from_sites(storages)
}

// ---------------------------------------------------------------------------
// Happy path: ARPACK and Lanczos must agree on the converged energy.
// ---------------------------------------------------------------------------

#[test]
fn dmrg_arpack_matches_lanczos_on_heisenberg_n6() {
    let n = 6;
    let chi = 8;
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let psi0 = random_mps_unknown_f64(n, chi, 0xDEAD_BEEF);

    let common = |eigensolver: LocalEigensolverParams| DmrgSweepParams {
        max_sweeps: 12,
        min_sweeps: 2,
        energy_tol: 1e-9,
        eigensolver,
        trunc: TruncSvdParams {
            chi_max: Some(chi),
            target_trunc_err: None,
        },
    };

    let p_lanczos = common(LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-10,
        seed: Some(0xACED),
    }));
    let p_arpack = common(LocalEigensolverParams::Arpack(ArpackParams {
        tol: 1e-10,
        max_iter: 300,
        ncv: None,
    }));

    let (res_lan, _) = dmrg_2site(&mpo, &psi0, &p_lanczos).expect("lanczos run");
    let (res_arp, _) = dmrg_2site(&mpo, &psi0, &p_arpack).expect("arpack run");

    assert!(
        res_lan.converged,
        "Lanczos arm must converge on n=6 Heisenberg"
    );
    assert!(
        res_arp.converged,
        "ARPACK arm must report convergence on n=6 Heisenberg \
         (step-level flag is `Ok` from arpack_smallest, i.e. ARPACK's \
         relative-tol stopping criterion fired)"
    );
    // 1e-7 absorbs the per-step solver tolerances + sweep-truncation
    // residual at chi_max=8 without being so loose it would mask a
    // wired-wrong arm.
    assert_abs_diff_eq!(res_lan.energy, res_arp.energy, epsilon = 1e-7);
}

// ---------------------------------------------------------------------------
// Error path: ARPACK upstream `MaxIterReached` must round-trip through
// the new `DmrgHeffError::Arpack(_)` variant.
// ---------------------------------------------------------------------------

#[test]
fn dmrg_arpack_max_iter_one_returns_arpack_error() {
    let n = 4;
    let chi = 4;
    let mut psi = random_mps_unknown_f64(n, chi, 0xCAFEBABE);
    arnet_mps::canonicalize(&mut psi, 0);
    let mpo = heisenberg_mpo_f64(n, 1.0);
    let mut envs = DmrgEnvs::build(&psi, &mpo).expect("envs build");
    // Walk the left env up so left(1) is populated for site=1.
    envs.advance_left(&psi, &mpo, 0).expect("advance_left(0)");

    // max_iter=1 with a tight relative tol is structurally
    // insufficient for a Heisenberg local Heff of dim ≥ 16 — ARPACK
    // returns `MaxIterReached` and the heff entry forwards it as
    // `DmrgHeffError::Arpack(_)` via the new `From<ArpackError>` impl.
    let bad = LocalEigensolverParams::Arpack(ArpackParams {
        tol: 1e-12,
        max_iter: 1,
        ncv: None,
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };

    let result = dmrg_2site_step(&envs, &psi, &mpo, 1, &bad, &trunc);
    assert!(
        matches!(
            result,
            Err(DmrgHeffError::Arpack(ArpackError::MaxIterReached { .. }))
        ),
        "expected DmrgHeffError::Arpack(ArpackError::MaxIterReached), got {result:?}",
    );

    // Contract: Display forwards the wrapped ArpackError so callers
    // who print the top-level error see ARPACK's own diagnostic
    // payload (iters / nconv / n_matvec for `MaxIterReached`)
    // without having to traverse `source()`. The wrapped error's
    // Display contains "ARPACK hit max_iter without convergence",
    // which must appear in the wrapping error's formatted output.
    let err = result.expect_err("error path verified above");
    let inner = match &err {
        DmrgHeffError::Arpack(inner) => format!("{inner}"),
        _ => unreachable!(),
    };
    let outer = format!("{err}");
    assert!(
        outer.contains(&inner),
        "DmrgHeffError::Arpack Display must forward the wrapped error: \
         outer = {outer:?}, inner = {inner:?}",
    );
}
