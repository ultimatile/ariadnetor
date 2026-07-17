//! Variational (fit) MPO-MPS apply tests (BlockSparse / U(1)).

use ariadnetor_mps::{
    self as mps, ApplyMethod, CanonicalForm, Mps, TensorChain, TruncSvdParams, TruncateParams,
    VariationalInit, inner,
};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{BlockSparseLayout, BlockSparseStorage, U1Sector};

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
