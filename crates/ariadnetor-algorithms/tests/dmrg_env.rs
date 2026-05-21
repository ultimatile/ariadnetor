//! Tests for DMRG L/R environment management (`DmrgEnvs`).
//!
//! Strategy: hand-chosen analytical inputs (small dim, identity MPO,
//! product-state MPS or random-but-seeded). The env contract is
//! pinned via cross-check against `arnet_mps::braket` ground truth.

use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arnet::contract;
use arnet::{ComputeBackend, DenseLayout, DenseStorage, DenseTensor, NativeBackend};
use arnet_algorithms::dmrg::{DmrgEnvError, DmrgEnvs};
use arnet_mps::{Mpo, Mps, TensorChain, braket};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Tiny MPS fixture: 4-site, physical dim d=2, all bond dim 1 (i.e. a
/// product state of single complex amplitudes per site). Each site
/// stores `[1, 0]` so the state is |0000⟩.
fn product_state_mps(n: usize) -> Mps<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            DenseTensor::from_raw_parts(
                vec![1.0_f64, 0.0],
                vec![1, 2, 1],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mps::from_sites(storages)
}

/// Identity MPO for d=2, n sites — every site is the rank-4 tensor
/// W[1, d_ket, d_bra, 1] with `W[0,k,k,0] = 1`, off-diagonal 0.
/// `<psi|H|psi> = <psi|psi> = 1` for any normalized MPS.
fn identity_mpo(n: usize, d: usize) -> Mpo<DenseStorage<f64>, DenseLayout> {
    let backend = NativeBackend::shared();
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|_| {
            let mut data = vec![0.0_f64; d * d];
            for k in 0..d {
                data[k + d * k] = 1.0;
            }
            DenseTensor::from_raw_parts(
                data,
                vec![1, d, d, 1],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mpo::from_sites(storages)
}

/// Random-but-seeded MPS for the cross-check tests. All bonds are
/// `chi`, physical `d`. Edge sites still have rank 3 with the outer
/// bond dim 1.
fn random_mps_f64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { chi };
            let r = if i + 1 == n { 1 } else { chi };
            let len = l * d * r;
            let data: Vec<f64> = (0..len)
                .map(|_| rand::Rng::random_range(&mut rng, -0.5_f64..0.5))
                .collect();
            DenseTensor::from_raw_parts(
                data,
                vec![l, d, r],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mps::from_sites(storages)
}

/// Random-but-seeded MPO for the cross-check tests. All MPO bonds
/// are `w`, physical `d`. Edge sites have outer bond dim 1.
fn random_mpo_f64(n: usize, d: usize, w: usize, seed: u64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let backend = NativeBackend::shared();
    let mut rng = StdRng::seed_from_u64(seed);
    let storages: Vec<DenseTensor<f64>> = (0..n)
        .map(|i| {
            let l = if i == 0 { 1 } else { w };
            let r = if i + 1 == n { 1 } else { w };
            let len = l * d * d * r;
            let data: Vec<f64> = (0..len)
                .map(|_| rand::Rng::random_range(&mut rng, -0.5_f64..0.5))
                .collect();
            DenseTensor::from_raw_parts(
                data,
                vec![l, d, d, r],
                backend.preferred_order(),
                Arc::clone(&backend),
            )
        })
        .collect();
    Mpo::from_sites(storages)
}

/// Fold an L env forward by absorbing sites `0..upto`. Stand-alone
/// helper that does not touch any `DmrgEnvs` state, so callers can
/// compute `left[upto]` independently and pair it with the existing
/// `env.right(upto)` produced by `build()`.
fn fold_left_to_boundary(
    mps: &Mps<DenseStorage<f64>, DenseLayout>,
    mpo: &Mpo<DenseStorage<f64>, DenseLayout>,
    initial: &DenseTensor<f64>,
    upto: usize,
) -> DenseTensor<f64> {
    let mut env = initial.clone();
    for i in 0..upto {
        let bra = mps.site(i).conj();
        let t1 = contract(&env, &bra, "abc,ade->bcde").expect("step 1");
        let t2 = contract(&t1, mpo.site(i), "bcde,bfdg->cefg").expect("step 2");
        env = contract(&t2, mps.site(i), "cefg,cfh->egh").expect("step 3");
    }
    env
}

/// Fold an L env all the way through the chain to a 1×1×1 scalar.
fn fold_left_to_scalar(
    mps: &Mps<DenseStorage<f64>, DenseLayout>,
    mpo: &Mpo<DenseStorage<f64>, DenseLayout>,
    initial: &DenseTensor<f64>,
) -> f64 {
    let env = fold_left_to_boundary(mps, mpo, initial, mps.len());
    env.data_slice()[0]
}

// ---------------------------------------------------------------------------
// Test 1 — boundaries are trivial
// ---------------------------------------------------------------------------

#[test]
fn env_boundaries_are_trivial() {
    let n = 4;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);

    let env = DmrgEnvs::build(&mps, &mpo).expect("build");

    let l0 = env.left(0).expect("left(0)");
    let rn = env.right(n).expect("right(N)");

    assert_eq!(l0.shape(), &[1, 1, 1]);
    assert_eq!(rn.shape(), &[1, 1, 1]);
    assert_abs_diff_eq!(l0.data_slice()[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rn.data_slice()[0], 1.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Test 2 — build consistency with braket
// ---------------------------------------------------------------------------

#[test]
fn env_build_consistency_with_braket() {
    let n = 6;
    let d = 2;
    let chi = 3;
    let w = 2;
    let mps = random_mps_f64(n, d, chi, 0xDEAD_BEEF);
    let mpo = random_mpo_f64(n, d, w, 0xC0FFEE);

    let env = DmrgEnvs::build(&mps, &mpo).expect("build");
    let l0 = env.left(0).expect("left(0)");
    let folded = fold_left_to_scalar(&mps, &mpo, l0);

    let reference = braket(&mps, &mpo, &mps);
    assert_abs_diff_eq!(folded, reference, epsilon = 1e-9);
}

// ---------------------------------------------------------------------------
// Contract — at every interior boundary j, <left(j), right(j)> reduced
// over (top, W, bot) equals the global braket scalar. This is the
// boundary-decomposition contract that DMRG sweeps rely on: a 2-site
// step at (i, i+1) consumes left(i) and right(i+2), and their product
// with the on-site tensors must reproduce the global expectation
// value. Parameterizing over every interior j pins build()'s entire
// right-loop, including the first iteration (right[N-1]) and the last
// (right[1] before the explicit break at j == 1).
// ---------------------------------------------------------------------------

#[test]
fn env_decomposition_holds_at_every_interior_boundary() {
    let n = 6;
    let d = 2;
    let chi = 3;
    let w = 2;
    let mps = random_mps_f64(n, d, chi, 0xDEAD_BEEF);
    let mpo = random_mpo_f64(n, d, w, 0xC0FFEE);

    let env = DmrgEnvs::build(&mps, &mpo).expect("build");
    let l0 = env.left(0).expect("left(0)");
    let reference = braket(&mps, &mpo, &mps);

    for j in 1..n {
        let l_at_j = fold_left_to_boundary(&mps, &mpo, l0, j);
        let r_at_j = env
            .right(j)
            .unwrap_or_else(|| panic!("right({j}) must be populated by build"));
        assert_eq!(
            l_at_j.shape(),
            r_at_j.shape(),
            "boundary {j}: shape mismatch"
        );
        let scalar: f64 = l_at_j
            .data_slice()
            .iter()
            .zip(r_at_j.data_slice().iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_abs_diff_eq!(scalar, reference, epsilon = 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Test 3 — interior advance_left invalidates right[i+1]
// ---------------------------------------------------------------------------

#[test]
fn env_advance_left_invalidates_right_interior() {
    let n = 4;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    // Right slot at index 2 is interior (2 < N = 4) and must be Some
    // after build.
    assert!(
        env.right(2).is_some(),
        "interior right slot must be Some after build"
    );

    // Walk the left envs forward so left[1] exists, then advance at
    // i = 1 (interior; i + 1 = 2 < 4).
    env.advance_left(&mps, &mpo, 0)
        .expect("advance_left(0) seeds left[1]");
    env.advance_left(&mps, &mpo, 1)
        .expect("advance_left interior");

    assert!(
        env.left(2).is_some(),
        "left[2] should be populated by advance_left(1)"
    );
    assert!(
        env.right(2).is_none(),
        "right[2] should be invalidated by advance_left(1)"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — edge advance_left preserves right[N] boundary
// ---------------------------------------------------------------------------

#[test]
fn env_advance_left_at_right_edge_preserves_boundary() {
    let n = 4;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    // Walk left envs all the way to N-1 so we can call advance_left(N-1).
    for i in 0..(n - 1) {
        env.advance_left(&mps, &mpo, i)
            .expect("advance_left interior");
    }

    let rn_before = env.right(n).expect("right(N) before edge advance").clone();
    env.advance_left(&mps, &mpo, n - 1)
        .expect("advance_left at right edge");

    let rn_after = env
        .right(n)
        .expect("right(N) must remain Some after edge advance");
    assert_eq!(rn_after.shape(), rn_before.shape());
    assert_abs_diff_eq!(
        rn_after.data_slice()[0],
        rn_before.data_slice()[0],
        epsilon = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Test 5 — edge advance_right preserves left[0] boundary
// ---------------------------------------------------------------------------

#[test]
fn env_advance_right_at_left_edge_preserves_boundary() {
    let n = 4;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    // After build, right[1..=N] are populated (right[0] is None by
    // construction); we can directly call advance_right(0).
    let l0_before = env.left(0).expect("left(0) before edge advance").clone();
    env.advance_right(&mps, &mpo, 0)
        .expect("advance_right at left edge");

    let l0_after = env
        .left(0)
        .expect("left(0) must remain Some after edge advance");
    assert_eq!(l0_after.shape(), l0_before.shape());
    assert_abs_diff_eq!(
        l0_after.data_slice()[0],
        l0_before.data_slice()[0],
        epsilon = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Test 6 — round trip sweep restores structure
// ---------------------------------------------------------------------------

#[test]
fn env_round_trip_sweep() {
    let n = 5;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    // Full L→R sweep: absorb sites 0..N-1 into left.
    for i in 0..n {
        env.advance_left(&mps, &mpo, i).expect("L→R");
    }

    // After the L→R sweep, all left slots are populated, all
    // interior right slots are stale.
    for i in 0..=n {
        assert!(env.left(i).is_some(), "left[{i}] should be Some after L→R");
    }
    for j in 1..n {
        assert!(
            env.right(j).is_none(),
            "interior right[{j}] should be None after L→R"
        );
    }
    assert!(env.right(n).is_some(), "right[N] boundary preserved");

    // Now reverse: R→L sweep absorbing sites N-1..0 into right.
    for j in (0..n).rev() {
        env.advance_right(&mps, &mpo, j).expect("R→L");
    }

    // After the R→L sweep, all right slots are populated except
    // right[0] (never built); all interior left slots are stale.
    for j in 1..=n {
        assert!(
            env.right(j).is_some(),
            "right[{j}] should be Some after R→L"
        );
    }
    for i in 1..n {
        assert!(
            env.left(i).is_none(),
            "interior left[{i}] should be None after R→L"
        );
    }
    assert!(env.left(0).is_some(), "left[0] boundary preserved");
}

// ---------------------------------------------------------------------------
// Test 7 — invalid site rejected
// ---------------------------------------------------------------------------

#[test]
fn env_invalid_site_rejected() {
    let n = 3;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    let result = env.advance_left(&mps, &mpo, n);
    assert!(matches!(
        result,
        Err(DmrgEnvError::InvalidSite { index, n_sites }) if index == n && n_sites == n
    ));

    let result = env.advance_right(&mps, &mpo, n + 5);
    assert!(matches!(result, Err(DmrgEnvError::InvalidSite { .. })));
}

// ---------------------------------------------------------------------------
// Test 8 — length mismatch surfaces as an error, not a panic
// ---------------------------------------------------------------------------

#[test]
fn env_length_mismatch_surfaces_as_error() {
    // Build mismatch: MPS has 3 sites, MPO has 4. Surfaces from
    // build's own length check (the upstream contract failure that
    // would otherwise come through `arnet_linalg::contract` as
    // `LinalgError`).
    let mps = product_state_mps(3);
    let mpo = identity_mpo(4, 2);

    let result = DmrgEnvs::build(&mps, &mpo);
    assert!(matches!(
        result,
        Err(DmrgEnvError::LengthMismatch { mps: 3, mpo: 4 })
    ));
}

// ---------------------------------------------------------------------------
// Test 9 — stale neighbor is surfaced as an error, not a panic
// ---------------------------------------------------------------------------

#[test]
fn env_stale_neighbor_surfaces_as_error() {
    let n = 4;
    let mps = product_state_mps(n);
    let mpo = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps, &mpo).expect("build");

    // After build, only left[0] is populated. advance_left(2) needs
    // left[2], which is None → StaleNeighbor("left", 2).
    let result = env.advance_left(&mps, &mpo, 2);
    assert!(matches!(
        result,
        Err(DmrgEnvError::StaleNeighbor {
            side: "left",
            index: 2
        })
    ));

    // Sweep all the way left so right[j+1] for interior j becomes
    // stale, then attempt advance_right at an interior site. Walk
    // L→R first so interior right slots get invalidated.
    for i in 0..n {
        env.advance_left(&mps, &mpo, i).expect("L→R sweep");
    }
    // Now right[1..n] are None (only right[n] survives). advance_right(0)
    // needs right[1] → StaleNeighbor("right", 1).
    let result = env.advance_right(&mps, &mpo, 0);
    assert!(matches!(
        result,
        Err(DmrgEnvError::StaleNeighbor {
            side: "right",
            index: 1
        })
    ));
}

// ---------------------------------------------------------------------------
// Test 10 — asymmetric length mismatch in advance_left / advance_right
// ---------------------------------------------------------------------------
//
// The length predicate is `mpo.len() != n_sites || mps.len() != n_sites`.
// When both are mismatched, the symmetric existing test passes under
// either `||` or `&&`. The asymmetric configuration (one matched, one
// mismatched) is what distinguishes the original `||` from the `&&`
// mutation: under `&&`, the mutated condition is `false && true = false`
// and the function continues past the length check, surfacing a
// different error or panic instead of `LengthMismatch`. Asserting
// the explicit `mps`/`mpo` values pinpoints the variant.

#[test]
fn env_advance_left_asymmetric_length_mismatch() {
    let n = 4;
    let mps_4 = product_state_mps(n);
    let mpo_4 = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps_4, &mpo_4).expect("build");

    let mps_3 = product_state_mps(3);
    let mpo_3 = identity_mpo(3, 2);

    // mpo matches n_sites = 4, mps does not (3).
    let result = env.advance_left(&mps_3, &mpo_4, 0);
    assert!(
        matches!(result, Err(DmrgEnvError::LengthMismatch { mps: 3, mpo: 4 })),
        "expected LengthMismatch {{ mps: 3, mpo: 4 }}, got {result:?}",
    );

    // mps matches n_sites = 4, mpo does not (3).
    let result = env.advance_left(&mps_4, &mpo_3, 0);
    assert!(
        matches!(result, Err(DmrgEnvError::LengthMismatch { mps: 4, mpo: 3 })),
        "expected LengthMismatch {{ mps: 4, mpo: 3 }}, got {result:?}",
    );
}

#[test]
fn env_advance_right_asymmetric_length_mismatch() {
    let n = 4;
    let mps_4 = product_state_mps(n);
    let mpo_4 = identity_mpo(n, 2);
    let mut env = DmrgEnvs::build(&mps_4, &mpo_4).expect("build");

    let mps_3 = product_state_mps(3);
    let mpo_3 = identity_mpo(3, 2);

    // advance_right targets `right[j+1]`. Use j = 0 so the read of
    // right[1] never happens (length check fires first).
    let result = env.advance_right(&mps_3, &mpo_4, 0);
    assert!(
        matches!(result, Err(DmrgEnvError::LengthMismatch { mps: 3, mpo: 4 })),
        "expected LengthMismatch {{ mps: 3, mpo: 4 }}, got {result:?}",
    );

    let result = env.advance_right(&mps_4, &mpo_3, 0);
    assert!(
        matches!(result, Err(DmrgEnvError::LengthMismatch { mps: 4, mpo: 3 })),
        "expected LengthMismatch {{ mps: 4, mpo: 3 }}, got {result:?}",
    );
}
