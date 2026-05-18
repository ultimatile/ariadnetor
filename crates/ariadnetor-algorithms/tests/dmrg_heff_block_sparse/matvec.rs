//! Matvec / step / canonical-form / flux-propagation tests on the
//! n=2 fixture (boundary envs) and the n=3 fixture (one extended
//! env via `extend_right_step`).

use arnet_algorithms::dmrg::{
    DmrgEnvs, EffectiveHamiltonian2Site, EffectiveHamiltonian2SiteBlockSparse,
    LocalEigensolverParams, dmrg_2site_step_block_sparse,
};
use arnet_algorithms::krylov::{LanczosParams, LinearOp};
use arnet_linalg::{TruncSvdParams, eigh};
use arnet_mps::TensorChain;
use arnet_native::NativeBackend;
use arnet_tensor::{DenseTensorData, MemoryOrder, Sector, U1Sector, reorder};

use super::fixtures::{
    build_envs_n2_f64, make_n2_mpo_f64, make_n2_mps_f64, make_n3_mpo_f64, make_n3_mps_f64,
};
use super::helpers::{
    build_dense_psi_from_flat, dense_to_template_flat, densify_bsp_f64, template_block_offsets,
    template_from_mps_pair,
};

#[test]
fn bsp_heff_matvec_matches_dense_oracle() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);

    let backend = NativeBackend::shared();
    let bsp_heff = EffectiveHamiltonian2SiteBlockSparse::new(
        envs.left(0).expect("left env"),
        mpo.site(0),
        mpo.site(1),
        envs.right(2).expect("right env"),
        mps.site(0),
        mps.site(1),
        backend.clone(),
    );

    let left_d = densify_bsp_f64(envs.left(0).expect("left"));
    let right_d = densify_bsp_f64(envs.right(2).expect("right"));
    let w0_d = densify_bsp_f64(mpo.site(0));
    let w1_d = densify_bsp_f64(mpo.site(1));
    let chi_l = left_d.shape()[0];
    let d0 = mps.site(0).shape()[1];
    let d1 = mps.site(1).shape()[1];
    let chi_r = right_d.shape()[0];
    let dense_heff = EffectiveHamiltonian2Site::new(
        &left_d, &w0_d, &w1_d, &right_d, chi_l, d0, d1, chi_r, backend,
    );

    let dim = bsp_heff.dim();
    let test_inputs: Vec<Vec<f64>> = vec![
        (0..dim).map(|i| (i + 1) as f64 * 0.13).collect(),
        (0..dim).map(|i| ((i as i32 - 2) as f64) * 0.7).collect(),
        vec![1.0; dim],
    ];
    let template = template_from_mps_pair(mps.site(0), mps.site(1));
    for (case, v) in test_inputs.iter().enumerate() {
        let v_dense_flat =
            DenseTensorData::from_raw_parts(v.clone(), vec![dim], MemoryOrder::ColumnMajor);
        let bsp_out = bsp_heff.apply(&v_dense_flat);
        assert_eq!(bsp_out.shape(), &[dim], "BSP output shape");

        let psi_dense = build_dense_psi_from_flat(v, &template);
        let dense_out = dense_heff.apply(&psi_dense.reshape(vec![chi_l * d0 * d1 * chi_r]));
        let dense_out_4d = dense_out.reshape(vec![chi_l, d0, d1, chi_r]);
        let expected = dense_to_template_flat(&dense_out_4d, &template);

        for (i, (bsp_val, exp_val)) in bsp_out.data().iter().zip(expected.iter()).enumerate() {
            let diff = (bsp_val - exp_val).abs();
            assert!(
                diff <= 1e-10,
                "case {case} idx {i}: bsp={bsp_val}, dense={exp_val}, |diff|={diff}"
            );
        }
    }
}

#[test]
fn bsp_heff_matvec_matches_dense_oracle_n3_bulk() {
    let mps = make_n3_mps_f64();
    let mpo = make_n3_mpo_f64(1.5);
    let envs = DmrgEnvs::build(&mps, &mpo).expect("envs build");
    let backend = NativeBackend::shared();

    // `DmrgEnvs::build` populates `left[0]` (boundary) and
    // `right[1..=n_sites]` (extended down from the right edge).
    // Pick site=0 so we use envs.left(0) (boundary) AND
    // envs.right(2) (extended through MPS[2] + W[2]) — the latter
    // is the "bulk env" path the n=2 fixture cannot reach.
    let site = 0;
    let bsp_heff = EffectiveHamiltonian2SiteBlockSparse::new(
        envs.left(site).expect("left boundary"),
        mpo.site(site),
        mpo.site(site + 1),
        envs.right(site + 2).expect("right ext"),
        mps.site(site),
        mps.site(site + 1),
        backend.clone(),
    );

    let right_total_dim = envs.right(site + 2).expect("right").shape()[0];
    assert!(
        right_total_dim > 1,
        "n=3 site=0 must exercise an extended right env; \
         got total_dim={right_total_dim} (single-sector dim-1 = boundary path, not bulk)"
    );

    let left_d = densify_bsp_f64(envs.left(site).expect("left"));
    let right_d = densify_bsp_f64(envs.right(site + 2).expect("right"));
    let w_i_d = densify_bsp_f64(mpo.site(site));
    let w_ip1_d = densify_bsp_f64(mpo.site(site + 1));
    let chi_l = left_d.shape()[0];
    let d_i = mps.site(site).shape()[1];
    let d_ip1 = mps.site(site + 1).shape()[1];
    let chi_r = right_d.shape()[0];
    let dense_heff = EffectiveHamiltonian2Site::new(
        &left_d, &w_i_d, &w_ip1_d, &right_d, chi_l, d_i, d_ip1, chi_r, backend,
    );

    let template = template_from_mps_pair(mps.site(site), mps.site(site + 1));
    let dim = bsp_heff.dim();
    assert_eq!(dim, *template_block_offsets(&template).last().unwrap());

    let test_inputs: Vec<Vec<f64>> = vec![
        (0..dim).map(|i| (i + 1) as f64 * 0.21).collect(),
        (0..dim).map(|i| ((i as i32 - 1) as f64) * 0.3).collect(),
    ];
    for (case, v) in test_inputs.iter().enumerate() {
        let bsp_out = bsp_heff.apply(&DenseTensorData::from_raw_parts(
            v.clone(),
            vec![dim],
            MemoryOrder::ColumnMajor,
        ));
        let psi_dense = build_dense_psi_from_flat(v, &template);
        let dense_out = dense_heff.apply(&psi_dense.reshape(vec![chi_l * d_i * d_ip1 * chi_r]));
        let dense_out_4d = dense_out.reshape(vec![chi_l, d_i, d_ip1, chi_r]);
        let expected = dense_to_template_flat(&dense_out_4d, &template);

        for (i, (bsp_val, exp_val)) in bsp_out.data().iter().zip(expected.iter()).enumerate() {
            let diff = (bsp_val - exp_val).abs();
            assert!(
                diff <= 1e-10,
                "n=3 bulk case {case} idx {i}: bsp={bsp_val}, dense={exp_val}, |diff|={diff}"
            );
        }
    }

    let params = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(7),
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, site, &params, &trunc)
        .expect("step at site=0");
    assert!(result.converged);
    assert!(result.eigenvalue.is_finite());
}

#[test]
fn bsp_heff_step_eigenvalue_matches_eigh_on_bsp_flat() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let backend = NativeBackend::shared();
    let bsp_heff = EffectiveHamiltonian2SiteBlockSparse::new(
        envs.left(0).expect("left"),
        mpo.site(0),
        mpo.site(1),
        envs.right(2).expect("right"),
        mps.site(0),
        mps.site(1),
        backend.clone(),
    );
    let dim = bsp_heff.dim();
    assert!(
        dim >= 2,
        "fixture dim should be >= 2 for a meaningful eigh oracle"
    );

    // Build H_bsp_flat column-by-column (column-major:
    // data[i + dim*j] = H[i, j]).
    let mut h_data = vec![0.0_f64; dim * dim];
    for j in 0..dim {
        let mut e_j = vec![0.0_f64; dim];
        e_j[j] = 1.0;
        let out = bsp_heff.apply(&DenseTensorData::from_raw_parts(
            e_j,
            vec![dim],
            MemoryOrder::ColumnMajor,
        ));
        for i in 0..dim {
            h_data[i + dim * j] = out.data()[i];
        }
    }

    // Hermiticity assertion (real-symmetric for f64).
    for i in 0..dim {
        for j in (i + 1)..dim {
            let a = h_data[i + dim * j];
            let b = h_data[j + dim * i];
            assert!(
                (a - b).abs() <= 1e-10,
                "H_bsp_flat not symmetric at ({i},{j}): H[{i},{j}]={a}, H[{j},{i}]={b}"
            );
        }
    }

    let h_dense = DenseTensorData::from_raw_parts(h_data, vec![dim, dim], MemoryOrder::ColumnMajor);
    let (eigvals, _eigvecs) = eigh(backend.as_ref(), &h_dense, 1).expect("eigh");
    let eigh_smallest = eigvals.data()[0];

    let params = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(42),
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc).expect("step");
    assert!(result.converged, "Lanczos must converge");

    let diff = (result.eigenvalue - eigh_smallest).abs();
    assert!(
        diff <= 1e-10,
        "Lanczos eigenvalue {} vs eigh {} (|diff|={})",
        result.eigenvalue,
        eigh_smallest,
        diff
    );
}

#[test]
fn bsp_heff_step_uvt_canonical_form() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let params = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(42),
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc).expect("step");

    let u_d = densify_bsp_f64(&result.u);
    let u_shape = u_d.shape().to_vec();
    let m: usize = u_shape[0] * u_shape[1];
    let n: usize = u_shape[2];
    let u_2d = u_d.reshape(vec![m, n]);
    let u_2d_rm = reorder(&u_2d, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    let u_data = u_2d_rm.data();
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..m {
                acc += u_data[k * n + i] * u_data[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (acc - expected).abs();
            assert!(
                diff <= 1e-10,
                "U^†U at ({i},{j}) = {acc}, expected {expected} (|diff|={diff})"
            );
        }
    }

    let vt_d = densify_bsp_f64(&result.vt);
    let vt_shape = vt_d.shape().to_vec();
    let p: usize = vt_shape[0];
    let q: usize = vt_shape[1] * vt_shape[2];
    let vt_2d = vt_d.reshape(vec![p, q]);
    let vt_2d_rm = reorder(&vt_2d, MemoryOrder::ColumnMajor, MemoryOrder::RowMajor);
    let vt_data = vt_2d_rm.data();
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0_f64;
            for k in 0..q {
                acc += vt_data[i * q + k] * vt_data[j * q + k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (acc - expected).abs();
            assert!(
                diff <= 1e-10,
                "Vt Vt^† at ({i},{j}) = {acc}, expected {expected} (|diff|={diff})"
            );
        }
    }
}

#[test]
fn bsp_heff_step_flux_propagation() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);

    let psi_flux = mps.site(0).flux().fuse(mps.site(1).flux());
    assert_ne!(
        psi_flux,
        U1Sector::identity(),
        "fixture must have non-identity 2-site flux"
    );

    let params = LocalEigensolverParams::Lanczos(LanczosParams {
        max_iter: 200,
        tol: 1e-12,
        seed: Some(42),
    });
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc).expect("step");

    assert_eq!(
        result.u.layout().flux(),
        &U1Sector::identity(),
        "U.flux must be identity"
    );
    assert_eq!(
        result.vt.layout().flux(),
        &psi_flux,
        "Vt.flux must equal psi_flux"
    );
}

#[test]
fn bsp_heff_step_n2_edge_case() {
    let mps = make_n2_mps_f64();
    let mpo = make_n2_mpo_f64(1.5);
    let envs = build_envs_n2_f64(&mps, &mpo);
    let params = LocalEigensolverParams::Lanczos(LanczosParams::default());
    let trunc = TruncSvdParams {
        chi_max: None,
        target_trunc_err: None,
    };
    let result = dmrg_2site_step_block_sparse(&envs, &mps, &mpo, 0, &params, &trunc);
    assert!(result.is_ok(), "n=2 step must succeed at site=0");
    let r = result.unwrap();
    assert!(r.converged);
    assert!(r.eigenvalue.is_finite());
}
