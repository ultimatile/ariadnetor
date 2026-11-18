//! Variational (fit) MPO-MPS apply tests (Dense).

use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mps, TensorChain, TruncErrTarget, TruncSvdParams,
    TruncateParams, VariationalInit,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor};

use super::helpers::{
    apply_ok, cap, cm_dense_tensor, make_identity_mpo, make_total_n_dense_mpo, mps_to_dense,
    target_only,
};
use super::variational_helpers::variational_fit_at_first_converging_cycle;

/// 4-site bond-2 MPS with deterministic, genuinely entangled content, so that
/// applying a bond-2 MPO inflates the product bond to 4 and `chi_max = 2`
/// truncates.
fn test_mps() -> Mps<DenseStorage<f64>, DenseLayout> {
    Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 2]),
        cm_dense_tensor((1..=8).map(|i| i as f64 * 0.13).collect(), vec![2, 2, 2]),
        cm_dense_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ])
}

fn variational(init: VariationalInit, max_sweeps: usize) -> ApplyMethod {
    ApplyMethod::Variational {
        init,
        max_sweeps,
        tol: 1e-12,
    }
}

fn assert_dense_close(a: &DenseTensor<f64>, b: &DenseTensor<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for (i, (x, y)) in a.data_slice().iter().zip(b.data_slice().iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < tol, "elem {i} mismatch: {x} vs {y} (diff {diff})");
    }
}

/// `|⟨a|b⟩|² / (⟨a|a⟩⟨b|b⟩)`: overlap fidelity of two MPS.
fn fidelity(
    a: &Mps<DenseStorage<f64>, DenseLayout>,
    b: &Mps<DenseStorage<f64>, DenseLayout>,
) -> f64 {
    let backend = NativeBackend::new();
    let ab = mps::inner(&backend, a, b);
    let aa = mps::inner(&backend, a, a);
    let bb = mps::inner(&backend, b, b);
    (ab * ab) / (aa * bb)
}

/// Applying an identity MPO and fitting must return the input state (up to
/// gauge — the contracted statevector is gauge-invariant).
#[test]
fn variational_identity_preserves_state() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let identity = make_identity_mpo(4, 2);

    let phi = apply_ok(
        &backend,
        &identity,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );

    assert_dense_close(&mps_to_dense(&psi), &mps_to_dense(&phi), 1e-9);
}

/// At full bond (`params = None`) the fit is lossless: the seed already equals
/// the exact product, and the projection is its fixed point, so the contracted
/// state matches the streaming-naive baseline.
#[test]
fn variational_lossless_matches_streaming_naive() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );
    let baseline = mps::apply(&backend, &op, &psi, None);

    assert_dense_close(&mps_to_dense(&phi), &mps_to_dense(&baseline), 1e-9);
}

/// The `DensityMatrix` seed reaches the same lossless fixed point at full bond.
#[test]
fn variational_density_matrix_init_lossless() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::DensityMatrix, 10),
    );
    let baseline = mps::apply(&backend, &op, &psi, None);

    assert_dense_close(&mps_to_dense(&phi), &mps_to_dense(&baseline), 1e-9);
}

/// At a truncating bond the variational fit refines its seed: its overlap with
/// the exact product is at least the zip-up seed's (single-site sweeps only
/// decrease `‖φ − Wψ‖`).
#[test]
fn variational_refines_over_seed() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);
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

    let f_seed = fidelity(&seed, &exact);
    let f_fit = fidelity(&fit, &exact);
    assert!(
        f_fit >= f_seed - 1e-9,
        "variational fidelity {f_fit} must not fall below the seed's {f_seed}",
    );
}

/// A `chi_max` cap bounds every bond of the fit result (the bond is held at the
/// seed's).
#[test]
fn variational_truncates_bond_dim() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);
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

/// The fit ends with the R→L half-sweep, parking the center at site 0
/// (matching the DMRG sweep convention, unlike the other apply methods).
#[test]
fn variational_canonical_form() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );
    assert_eq!(*phi.canonical_form(), CanonicalForm::Mixed { center: 0 });
}

/// A single-site chain short-circuits to the (exact) seed product; no sweep is
/// meaningful.
#[test]
fn variational_single_site_short_circuits() {
    let backend = NativeBackend::new();
    let psi = Mps::from_sites(vec![cm_dense_tensor(vec![0.6, 0.8], vec![1, 2, 1])]);
    let op = make_total_n_dense_mpo(1);

    let phi = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );
    let baseline = mps::apply(&backend, &op, &psi, None);

    assert_dense_close(&mps_to_dense(&phi), &mps_to_dense(&baseline), 1e-9);
}

/// `target_trunc_err` is not consulted: only `chi_max` fixes the bond. A huge
/// cutoff that would collapse every bond to 1 if honored must leave the fit
/// identical to the lossless (`chi_max = None`) result. Guards the seed-param
/// stripping — a zip-up seed built from the raw params would otherwise leak
/// `target_trunc_err` into its truncation and shrink the fixed bond.
#[test]
fn variational_ignores_target_trunc_err() {
    let backend = NativeBackend::new();
    let psi = test_mps();
    let op = make_total_n_dense_mpo(4);

    let lossless = apply_ok(
        &backend,
        &op,
        &psi,
        None,
        variational(VariationalInit::ZipUp, 10),
    );

    // chi_max = None, but a target_trunc_err large enough to collapse every
    // bond to 1 if it were consulted.
    let params = TruncateParams::from(target_only(TruncErrTarget::Absolute(1e10)));
    let fit = apply_ok(
        &backend,
        &op,
        &psi,
        Some(&params),
        variational(VariationalInit::ZipUp, 10),
    );

    assert_eq!(
        fit.bond_dims(),
        lossless.bond_dims(),
        "target_trunc_err must not truncate the variational result",
    );
}

#[test]
#[should_panic(expected = "representable as a finite value")]
fn variational_f32_rejects_unrepresentable_tol() {
    let backend = NativeBackend::new();
    let psi: Mps<DenseStorage<f32>, DenseLayout> = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.5, 0.5], vec![1, 2, 2]),
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2, 1]),
    ]);
    let op = mps::Mpo::from_sites(vec![
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
        cm_dense_tensor(vec![1.0_f32, 0.0, 0.0, 1.0], vec![1, 2, 2, 1]),
    ]);

    // 1e300 is finite in f64 (passes the API's plain-f64 signature) but
    // overflows f32; the tolerance cast must reject it instead of casting to
    // infinity and reporting convergence after one sweep. Two sites are needed
    // because the single-site path returns before reaching the cast.
    let method = ApplyMethod::Variational {
        init: VariationalInit::ZipUp,
        max_sweeps: 10,
        tol: 1e300,
    };
    let _ = mps::apply_with_method(&backend, &op, &psi, None, method);
}

type DenseMpo = mps::Mpo<DenseStorage<f64>, DenseLayout>;
type DenseMps = Mps<DenseStorage<f64>, DenseLayout>;

/// Two-site product `Wψ` of rank 2 fitted at bond 1. The zip-up seed truncates
/// `W₀ψ₀` against the non-isometric remainder `W₁ψ₁`, so it misses the optimal
/// bond-1 state and the sweeps raise the overlap over several cycles.
fn refining_fit_problem() -> (DenseMpo, DenseMps, TruncateParams) {
    let op = mps::Mpo::from_sites(vec![
        cm_dense_tensor(
            vec![1.0, 0.2, -0.3, 0.6, 0.4, -0.5, 0.9, 0.1],
            vec![1, 2, 2, 2],
        ),
        cm_dense_tensor(
            vec![0.3, 0.8, -0.6, 0.2, 0.7, 0.1, -0.4, 1.1],
            vec![2, 2, 2, 1],
        ),
    ]);
    let psi = Mps::from_sites(vec![
        cm_dense_tensor(vec![1.0, 0.3, -0.4, 0.8], vec![1, 2, 2]),
        cm_dense_tensor(vec![0.5, -0.2, 0.7, 0.9], vec![2, 2, 1]),
    ]);
    let params = TruncateParams::from(cap(1));
    (op, psi, params)
}

/// The sweep stops after the first cycle whose relative change of the center
/// overlap `‖P_center‖²` is at or below `tol`. A run with `tol = 0` and
/// `max_sweeps = k` yields the k-cycle iterate `φ_k`, and `‖φ_k‖²` is that
/// cycle's center overlap (the fit ends with site 0 freshly projected and the
/// rest right-isometric). With `tol` placed below cycle 3's change and at or
/// above cycle 4's, a run with ample `max_sweeps` must return exactly `φ_4`. `tol`
/// sits within a quarter of cycle 3's change, so measuring the change of
/// `‖P_center‖` (about half as large) instead would already stop at cycle 3.
#[test]
fn variational_stops_at_first_cycle_within_tol() {
    let (op, psi, params) = refining_fit_problem();
    let (stopped, expected) = variational_fit_at_first_converging_cycle(&op, &psi, &params, 4);
    assert_dense_close(&mps_to_dense(&stopped), &mps_to_dense(&expected), 1e-12);
}
