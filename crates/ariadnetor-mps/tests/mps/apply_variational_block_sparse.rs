//! Variational (fit) MPO-MPS apply tests (BlockSparse / U(1)).

use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mps, TensorChain, TruncSvdParams, TruncateParams,
    VariationalInit, inner,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::test_fixtures::legs;
use ariadnetor_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, U1Sector,
};

use super::helpers::{
    apply_ok, assert_block_sparse_close, bsp_mps_contract_full, make_3site_u1_mps_multipath_middle,
    make_4site_u1_mps, make_identity_u1_mpo, make_total_n_u1_mpo,
};

fn variational(init: VariationalInit, max_sweeps: usize) -> ApplyMethod {
    ApplyMethod::Variational {
        init,
        max_sweeps,
        tol: 1e-12,
    }
}

/// Applying an identity MPO and fitting returns the input state (up to gauge).
#[test]
fn variational_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let identity = make_identity_u1_mpo(4);

    let phi = apply_ok(
        &backend,
        &identity,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );

    let v_in = bsp_mps_contract_full(&psi);
    let v_out = bsp_mps_contract_full(&phi);
    assert_block_sparse_close(&v_in, &v_out, 1e-9);
}

/// At full bond the variational fit is lossless, so its contracted state
/// matches the streaming-naive baseline. `assert_block_sparse_close` compares
/// flux, leg directions, and sector layout, so this also pins the block-sparse
/// projection's leg-direction / flux wiring — the highest-risk detail, since
/// the `⟨φ|W|ψ⟩` env legs carry direction-flipped W / ket bonds.
#[test]
fn variational_lossless_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );
    let baseline = mps::apply(&backend, &op, &psi, None);

    let v_fit = bsp_mps_contract_full(&phi);
    let v_baseline = bsp_mps_contract_full(&baseline);
    assert_block_sparse_close(&v_fit, &v_baseline, 1e-10);
}

/// A `chi_max` cap bounds every bond of the fit result. The thick-middle
/// fixture has a dim-3 sector, so `chi_max = 2` genuinely truncates.
#[test]
fn variational_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = make_3site_u1_mps_multipath_middle();
    let op = make_total_n_u1_mpo(3);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        Some(&params),
        variational(VariationalInit::ZipUp, 10),
    );

    for d in phi.bond_dims() {
        assert!(d <= 2, "bond dim {d} exceeds chi_max=2");
    }
}

/// At a truncating bond the fit refines its seed: its overlap with the exact
/// product is at least the zip-up seed's.
#[test]
fn variational_refines_over_seed() {
    let backend = NativeBackend::new();
    let psi = make_3site_u1_mps_multipath_middle();
    let op = make_total_n_u1_mpo(3);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(2),
        target_trunc_err: None,
    });

    let exact = mps::apply(&backend, &op, &psi, None);
    let seed = apply_ok(&backend, &op, &psi, Some(&params), ApplyMethod::ZipUp);
    let fit = apply_ok(
        &backend,
        &op,
        &psi,
        Some(&params),
        variational(VariationalInit::ZipUp, 20),
    );

    let fidelity =
        |a: &Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
         b: &Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>| {
            let ab = inner(&backend, a, b);
            let aa = inner(&backend, a, a);
            let bb = inner(&backend, b, b);
            (ab * ab) / (aa * bb)
        };
    let f_seed = fidelity(&seed, &exact);
    let f_fit = fidelity(&fit, &exact);
    assert!(
        f_fit >= f_seed - 1e-9,
        "variational fidelity {f_fit} must not fall below the seed's {f_seed}",
    );
}

/// The fit ends with the R→L half-sweep, parking the center at site 0.
#[test]
fn variational_canonical_form() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

/// A U(1) tensor whose every leg carries only the charge-0 sector, so `data`
/// fills its single block.
fn charge_zero_tensor(
    dims: &[usize],
    directions: &[Direction],
    data: &[f64],
) -> BlockSparseTensor<f64, U1Sector> {
    let mut t = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs(
            dims.iter()
                .zip(directions)
                .map(|(&d, &dir)| (vec![(U1Sector(0), d)], dir)),
        ),
        U1Sector(0),
    );
    t.block_data_mut(&BlockCoord(vec![0; dims.len()]))
        .expect("the single charge-0 block")
        .copy_from_slice(data);
    t
}

type U1Mpo = mps::Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>;
type U1Mps = Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>;

/// Two-site product `Wψ` of rank 2 fitted at bond 1, all in the charge-0
/// sector. The zip-up seed truncates `W₀ψ₀` against the non-isometric
/// remainder `W₁ψ₁`, so it misses the optimal bond-1 state and the sweeps
/// raise the overlap over several cycles.
fn refining_fit_problem() -> (U1Mpo, U1Mps, TruncateParams) {
    use Direction::{In, Out};
    let op = mps::Mpo::from_sites(vec![
        charge_zero_tensor(
            &[1, 2, 2, 2],
            &[Out, In, Out, In],
            &[1.0, 0.2, -0.3, 0.6, 0.4, -0.5, 0.9, 0.1],
        ),
        charge_zero_tensor(
            &[2, 2, 2, 1],
            &[Out, In, Out, In],
            &[0.3, 0.8, -0.6, 0.2, 0.7, 0.1, -0.4, 1.1],
        ),
    ]);
    let psi = Mps::from_sites(vec![
        charge_zero_tensor(&[1, 2, 2], &[Out, Out, In], &[1.0, 0.3, -0.4, 0.8]),
        charge_zero_tensor(&[2, 2, 1], &[Out, Out, In], &[0.5, -0.2, 0.7, 0.9]),
    ]);
    let params = TruncateParams::from(TruncSvdParams {
        chi_max: Some(1),
        target_trunc_err: None,
    });
    (op, psi, params)
}

/// The sweep stops after the first cycle whose relative change of the center
/// overlap `‖P_center‖²` is at or below `tol`. A run with `tol = 0` and
/// `max_sweeps = k` yields the k-cycle iterate `φ_k`, and `‖φ_k‖²` is that
/// cycle's center overlap (the fit ends with site 0 freshly projected and the
/// rest right-isometric). With `tol` placed below cycle 3's change and above
/// cycle 4's, a run with ample `max_sweeps` must return exactly `φ_4`. `tol`
/// sits within a quarter of cycle 3's change, so measuring the change of
/// `‖P_center‖` (about half as large) instead would already stop at cycle 3.
#[test]
fn variational_stops_at_first_cycle_within_tol() {
    const STOP: usize = 4;
    let backend = NativeBackend::new();
    let (op, psi, params) = refining_fit_problem();
    let fit = |max_sweeps: usize, tol: f64| {
        apply_ok(
            &backend,
            &op,
            &psi,
            Some(&params),
            ApplyMethod::Variational {
                init: VariationalInit::ZipUp,
                max_sweeps,
                tol,
            },
        )
    };

    let iterates: Vec<_> = (1..=STOP + 1).map(|k| fit(k, 0.0)).collect();
    let overlaps: Vec<f64> = iterates
        .iter()
        .map(|phi| inner(&backend, phi, phi))
        .collect();
    for k in 1..overlaps.len() {
        assert!(
            overlaps[k] > overlaps[k - 1],
            "cycle {} must still raise the overlap at tol = 0: {} -> {}",
            k + 1,
            overlaps[k - 1],
            overlaps[k],
        );
    }
    // Relative change of the overlap over cycle `k` (1-based, k >= 2).
    let rel_change = |k: usize| (overlaps[k - 1] - overlaps[k - 2]) / overlaps[k - 1];
    let tol = 0.75 * rel_change(STOP - 1);
    for k in 2..STOP {
        assert!(
            rel_change(k) > tol,
            "cycle {k} must not converge at tol {tol}"
        );
    }
    assert!(
        rel_change(STOP) <= tol,
        "cycle {STOP}'s change {} must fall within tol {tol}",
        rel_change(STOP),
    );

    let stopped = fit(50, tol);
    assert_block_sparse_close(
        &bsp_mps_contract_full(&stopped),
        &bsp_mps_contract_full(&iterates[STOP - 1]),
        1e-12,
    );
}

/// The `DensityMatrix` seed reaches the same lossless fixed point at full bond,
/// exercising the block-sparse density-matrix seed path of the fit.
#[test]
fn variational_density_matrix_init_lossless() {
    let backend = NativeBackend::new();
    let psi = make_4site_u1_mps();
    let op = make_total_n_u1_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::DensityMatrix, 10),
    );
    let baseline = mps::apply(&backend, &op, &psi, None);

    let v_fit = bsp_mps_contract_full(&phi);
    let v_baseline = bsp_mps_contract_full(&baseline);
    assert_block_sparse_close(&v_fit, &v_baseline, 1e-10);
}
