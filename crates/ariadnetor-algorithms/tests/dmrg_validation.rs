//! End-to-end validation of the 2-site DMRG sweep driver against
//! exact diagonalization on the standard 1D spin-1/2 Hamiltonians:
//! transverse-field Ising (TFI) and antiferromagnetic Heisenberg
//! (XXX). Test-internal MPO builders and ED reference are inlined
//! here; no public API is added.

use arnet_algorithms::dmrg::{DmrgEnvs, DmrgSweepParams, LocalEigensolverParams, sweep_2site};
use arnet_algorithms::krylov::LanczosParams;
use arnet_linalg::{TruncSvdParams, eigh_with_backend};
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, canonicalize};
use arnet_native::NativeBackend;
use arnet_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Pauli matrix elements in computational basis (|0⟩=up, |1⟩=down).
//
// Matrix layout convention used throughout this file: every dense
// matrix is stored column-major, so `data[row + nrows * col]` is the
// (row, col) entry. For MPO sites with shape
// `[W_L, d_ket, d_bra, W_R]` (axis 1 = ket, axis 2 = bra per
// `heff.rs:12`), the column-major linear index is
// `vL + W_L * (k + d_ket * (b + d_bra * vR))`.
// ---------------------------------------------------------------------------

const D: usize = 2; // physical dim (spin-1/2)

/// (k_ket, b_bra) → matrix element of the operator.
type Op = fn(usize, usize) -> f64;

fn op_id(k: usize, b: usize) -> f64 {
    if k == b { 1.0 } else { 0.0 }
}

fn op_sx(k: usize, b: usize) -> f64 {
    if k != b { 1.0 } else { 0.0 }
}

fn op_sz(k: usize, b: usize) -> f64 {
    if k == b {
        if k == 0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

// σ⁺ raises: |down⟩ → |up⟩ in our bit convention.
// Single non-zero matrix element ⟨bra=0|σ⁺|ket=1⟩ = 1, so under
// the `(k_ket, b_bra)` indexing of `Op` this is op_sp(k=1, b=0) = 1.
fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

// σ⁻ lowers: |up⟩ → |down⟩.
// Single non-zero matrix element ⟨bra=1|σ⁻|ket=0⟩ = 1, so under
// the `(k_ket, b_bra)` indexing of `Op` this is op_sm(k=0, b=1) = 1.
fn op_sm(k: usize, b: usize) -> f64 {
    if k == 0 && b == 1 { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// MPO site builder — column-major fill of a rank-4 site tensor.
// `cells[(vL, vR)] = (op, scale)` populates the W matrix at virtual
// indices (vL, vR); cells absent from the map are zero. Shape is
// `[w_l_dim, D, D, w_r_dim]`.
// ---------------------------------------------------------------------------

fn build_mpo_site_f64(
    w_l_dim: usize,
    w_r_dim: usize,
    cells: &[(usize, usize, Op, f64)],
) -> DenseTensor<f64> {
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
    Host::shared().dense(data, vec![w_l_dim, D, D, w_r_dim])
}

// ---------------------------------------------------------------------------
// TFI MPO (bond dim 3): H = -J Σ σᶻᵢσᶻᵢ₊₁ − h Σ σˣᵢ
//
// Bulk W (row=vL, col=vR):
//   [[ I,        0,       0 ],
//    [ σᶻ,       0,       0 ],
//    [-h σˣ,    -J σᶻ,    I ]]
//
// Site 0 = bottom row of W with shape (1, D, D, 3).
// Site N-1 = first column of W with shape (3, D, D, 1).
// ---------------------------------------------------------------------------

fn tfi_mpo_f64(n: usize, j: f64, h: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2, "tfi_mpo_f64 requires n >= 2");
    let mut sites = Vec::with_capacity(n);

    // Site 0 — bottom row [-h σˣ, -J σᶻ, I], shape (1, D, D, 3).
    sites.push(build_mpo_site_f64(
        1,
        3,
        &[(0, 0, op_sx, -h), (0, 1, op_sz, -j), (0, 2, op_id, 1.0)],
    ));

    // Bulk — full 3×3 W, shape (3, D, D, 3).
    for _ in 1..n - 1 {
        sites.push(build_mpo_site_f64(
            3,
            3,
            &[
                (0, 0, op_id, 1.0),
                (1, 0, op_sz, 1.0),
                (2, 0, op_sx, -h),
                (2, 1, op_sz, -j),
                (2, 2, op_id, 1.0),
            ],
        ));
    }

    // Site N-1 — first column [I, σᶻ, -h σˣ]ᵀ, shape (3, D, D, 1).
    sites.push(build_mpo_site_f64(
        3,
        1,
        &[(0, 0, op_id, 1.0), (1, 0, op_sz, 1.0), (2, 0, op_sx, -h)],
    ));

    Mpo::from_sites(sites)
}

// ---------------------------------------------------------------------------
// Heisenberg MPO (bond dim 5) via S±/S∓ form:
//   H = J Σ (σᶻσᶻ + 2 σ⁺σ⁻ + 2 σ⁻σ⁺)
//
// Bulk W (row=vL, col=vR):
//   [[ I,    0,       0,       0,    0 ],
//    [ σ⁺,   0,       0,       0,    0 ],
//    [ σ⁻,   0,       0,       0,    0 ],
//    [ σᶻ,   0,       0,       0,    0 ],
//    [ 0,   2J σ⁻,   2J σ⁺,   J σᶻ,  I ]]
//
// Site 0 = bottom row, shape (1, D, D, 5).
// Site N-1 = first column, shape (5, D, D, 1).
// ---------------------------------------------------------------------------

fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
    assert!(n >= 2, "heisenberg_mpo_f64 requires n >= 2");
    let mut sites = Vec::with_capacity(n);

    // Site 0 — bottom row [0, 2J σ⁻, 2J σ⁺, J σᶻ, I], shape (1, D, D, 5).
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

    // Bulk — full 5×5 W, shape (5, D, D, 5).
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

    // Site N-1 — first column [I, σ⁺, σ⁻, σᶻ, 0]ᵀ, shape (5, D, D, 1).
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

// ---------------------------------------------------------------------------
// ED reference — dense Hamiltonian in computational basis.
//
// Convention: bit i of basis index `b` is site i (LSB = site 0).
// `bit = 0` → spin up (z = +1); `bit = 1` → spin down (z = -1).
// Output `H[row + dim * col]` (column-major) with row = bra (output)
// and col = ket (input).
//
// Off-diagonal write rule per the plan: iterate every basis state
// `b ∈ 0..2ⁿ`, and for each off-diagonal term write the directed
// entry `H[b'][b] += element` where `b'` is the state reached by
// applying the operator to `b`. Each symmetric pair is therefore
// visited twice (once from each side); do NOT also write the
// symmetric counterpart, that would double-count.
// ---------------------------------------------------------------------------

fn z_of_bit(bit: usize) -> f64 {
    if bit == 0 { 1.0 } else { -1.0 }
}

fn write_diag(h: &mut [f64], dim: usize, b: usize, value: f64) {
    h[b + dim * b] += value;
}

fn write_offdiag(h: &mut [f64], dim: usize, b_out: usize, b_in: usize, value: f64) {
    h[b_out + dim * b_in] += value;
}

fn tfi_ed_dense_f64(n: usize, j: f64, h_field: f64) -> DenseTensor<f64> {
    let dim = 1usize << n;
    let mut data = vec![0.0_f64; dim * dim];
    for b in 0..dim {
        // Diagonal: -J Σ z_i z_{i+1}.
        let mut diag = 0.0_f64;
        for i in 0..n - 1 {
            let zi = z_of_bit((b >> i) & 1);
            let zi1 = z_of_bit((b >> (i + 1)) & 1);
            diag += -j * zi * zi1;
        }
        write_diag(&mut data, dim, b, diag);

        // Off-diagonal: -h Σ σˣᵢ.
        for i in 0..n {
            let b_out = b ^ (1 << i);
            write_offdiag(&mut data, dim, b_out, b, -h_field);
        }
    }
    Host::shared().dense(data, vec![dim, dim])
}

fn heisenberg_ed_dense_f64(n: usize, j: f64) -> DenseTensor<f64> {
    let dim = 1usize << n;
    let mut data = vec![0.0_f64; dim * dim];
    for b in 0..dim {
        // Diagonal: +J Σ z_i z_{i+1}.
        let mut diag = 0.0_f64;
        for i in 0..n - 1 {
            let zi = z_of_bit((b >> i) & 1);
            let zi1 = z_of_bit((b >> (i + 1)) & 1);
            diag += j * zi * zi1;
        }
        write_diag(&mut data, dim, b, diag);

        // Off-diagonal: +2J for each (i, i+1) where bits differ —
        // covers σ⁺σ⁻ + σ⁻σ⁺ as a single "swap-bits" rule.
        for i in 0..n - 1 {
            let bi = (b >> i) & 1;
            let bi1 = (b >> (i + 1)) & 1;
            if bi != bi1 {
                let b_out = b ^ ((1 << i) | (1 << (i + 1)));
                write_offdiag(&mut data, dim, b_out, b, 2.0 * j);
            }
        }
    }
    Host::shared().dense(data, vec![dim, dim])
}

fn dense_min_eig_f64(h: &DenseTensor<f64>) -> f64 {
    let (eigvals, _v) = eigh_with_backend(&NativeBackend::new(), h, 1).expect("eigh");
    eigvals
        .data_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Random initial MPS in `Mixed { center: 0 }` form. Inlined here
// because the equivalent helper in `dmrg_sweep.rs` is not exported.
// ---------------------------------------------------------------------------

fn random_mps_center_zero_f64(
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
    let mut mps = Mps::from_sites(storages);
    canonicalize(&NativeBackend::new(), &mut mps, 0);
    mps
}

// ---------------------------------------------------------------------------
// Sweep parameters for the validation fixtures.
// ---------------------------------------------------------------------------

fn validation_params(chi_max: usize, lanczos_seed: u64) -> DmrgSweepParams {
    DmrgSweepParams {
        max_sweeps: 30,
        min_sweeps: 4,
        energy_tol: 1e-12,
        eigensolver: LocalEigensolverParams::Lanczos(LanczosParams {
            max_iter: 200,
            tol: 1e-12,
            seed: Some(lanczos_seed),
        }),
        trunc: TruncSvdParams {
            chi_max: Some(chi_max),
            target_trunc_err: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Run a DMRG validation case end-to-end and assert acceptance.
// ---------------------------------------------------------------------------

fn run_validation(
    name: &str,
    mpo: Mpo<DenseStorage<f64>, DenseLayout>,
    e_ed: f64,
    chi_max: usize,
    init_seed: u64,
    lanczos_seed: u64,
) {
    let n = mpo.len();
    let mut mps = random_mps_center_zero_f64(n, D, 4, init_seed);
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("envs build");
    let params = validation_params(chi_max, lanczos_seed);

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params)
        .unwrap_or_else(|e| panic!("{name}: sweep_2site failed: {e:?}"));

    let delta = (result.energy - e_ed).abs();
    assert!(
        delta <= 1e-8,
        "{name}: |E_DMRG - E_ED| = {delta:.3e} > 1e-8 (E_DMRG={:.10}, E_ED={:.10})",
        result.energy,
        e_ed,
    );
    assert!(
        result.converged,
        "{name}: result.converged = false (n_sweeps={}, last sweep_energy={:?})",
        result.n_sweeps,
        result.sweeps.last().map(|s| s.sweep_energy),
    );
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Mixed { center: 0 },
        "{name}: final canonical form mismatch"
    );
}

// ===========================================================================
// V1 — TFI N=6, J=1.0, h=1.0 (critical) vs ED.
// ===========================================================================
#[test]
fn v1_tfi_n6_critical_vs_ed() {
    let n = 6;
    let j = 1.0;
    let h = 1.0;
    let mpo = tfi_mpo_f64(n, j, h);
    let e_ed = dense_min_eig_f64(&tfi_ed_dense_f64(n, j, h));
    run_validation("v1_tfi_n6_critical", mpo, e_ed, 16, 0xD3F1, 0xA101);
}

// ===========================================================================
// V2 — TFI N=8, J=1.0, h=0.5 (FM-gapped side) vs ED.
// ===========================================================================
#[test]
fn v2_tfi_n8_fm_gapped_vs_ed() {
    let n = 8;
    let j = 1.0;
    let h = 0.5;
    let mpo = tfi_mpo_f64(n, j, h);
    let e_ed = dense_min_eig_f64(&tfi_ed_dense_f64(n, j, h));
    run_validation("v2_tfi_n8_fm_gapped", mpo, e_ed, 16, 0xD3F2, 0xA102);
}

// ===========================================================================
// V3 — TFI N=10, J=1.0, h=2.0 (paramagnetic) vs ED.
// ===========================================================================
#[test]
fn v3_tfi_n10_paramagnetic_vs_ed() {
    let n = 10;
    let j = 1.0;
    let h = 2.0;
    let mpo = tfi_mpo_f64(n, j, h);
    let e_ed = dense_min_eig_f64(&tfi_ed_dense_f64(n, j, h));
    run_validation("v3_tfi_n10_paramagnetic", mpo, e_ed, 16, 0xD3F3, 0xA103);
}

// ===========================================================================
// V4 — Heisenberg N=6, J=1.0 vs ED.
// ===========================================================================
#[test]
fn v4_heisenberg_n6_vs_ed() {
    let n = 6;
    let j = 1.0;
    let mpo = heisenberg_mpo_f64(n, j);
    let e_ed = dense_min_eig_f64(&heisenberg_ed_dense_f64(n, j));
    run_validation("v4_heisenberg_n6", mpo, e_ed, 32, 0xD3F4, 0xA104);
}

// ===========================================================================
// V5 — Heisenberg N=8, J=1.0 vs ED.
// ===========================================================================
#[test]
fn v5_heisenberg_n8_vs_ed() {
    let n = 8;
    let j = 1.0;
    let mpo = heisenberg_mpo_f64(n, j);
    let e_ed = dense_min_eig_f64(&heisenberg_ed_dense_f64(n, j));
    run_validation("v5_heisenberg_n8", mpo, e_ed, 32, 0xD3F5, 0xA105);
}

// ---------------------------------------------------------------------------
// MPO sanity check — contract a 2-site MPO into a 4×4 dense matrix
// and compare against the analytic 2-site Hamiltonian (built via the
// same ED reference).
//
// MPO sites: site_0 shape (1, D, D, χ), site_1 shape (χ, D, D, 1).
// Contract over the shared bond w:
//   H[k1, b1, k2, b2] = Σ_w site_0[0, k1, b1, w] · site_1[w, k2, b2, 0]
// Flatten to (D², D²) with row = bra-combined `b1 + D*b2`,
// col = ket-combined `k1 + D*k2`. This matches the ED-reference
// flattening (LSB = site 0 in basis index).
// ---------------------------------------------------------------------------

fn contract_2site_mpo_f64(mpo: &Mpo<DenseStorage<f64>, DenseLayout>) -> Vec<f64> {
    assert_eq!(mpo.len(), 2, "contract_2site_mpo_f64 requires N=2");
    let s0 = mpo.site(0);
    let s1 = mpo.site(1);
    let chi = s0.shape()[3];
    assert_eq!(s1.shape()[0], chi);

    let s0_data = s0.data_slice();
    let s1_data = s1.data_slice();
    // Column-major linear indices into the rank-4 site tensors.
    // site_0 has shape (1, D, D, χ), so vL=0, W_L=1 collapses to
    // `k1 + D * (b1 + D * w)`. site_1 has shape (χ, D, D, 1), so
    // vR=0 collapses to `w + χ * (k2 + D * b2)`.
    let s0_idx = |k1: usize, b1: usize, w: usize| k1 + D * (b1 + D * w);
    let s1_idx = |w: usize, k2: usize, b2: usize| w + chi * (k2 + D * b2);

    let dim = D * D;
    let mut out = vec![0.0_f64; dim * dim];
    for b1 in 0..D {
        for b2 in 0..D {
            for k1 in 0..D {
                for k2 in 0..D {
                    let mut acc = 0.0_f64;
                    for w in 0..chi {
                        acc += s0_data[s0_idx(k1, b1, w)] * s1_data[s1_idx(w, k2, b2)];
                    }
                    let row = b1 + D * b2;
                    let col = k1 + D * k2;
                    out[row + dim * col] = acc;
                }
            }
        }
    }
    out
}

fn assert_dense_close(actual: &[f64], expected: &DenseTensor<f64>, name: &str, tol: f64) {
    let exp = expected.data_slice();
    assert_eq!(actual.len(), exp.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(exp.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{name}: entry {i} differs by {diff:.3e} (got {a}, expected {e})",
        );
    }
}

// ===========================================================================
// V6 — TFI N=2 MPO contracts to the analytic 2-site Hamiltonian.
// ===========================================================================
#[test]
fn v6_tfi_n2_mpo_matches_analytic() {
    let j = 0.7;
    let h = 1.3;
    let mpo = tfi_mpo_f64(2, j, h);
    let actual = contract_2site_mpo_f64(&mpo);
    let expected = tfi_ed_dense_f64(2, j, h);
    assert_dense_close(&actual, &expected, "v6_tfi_n2_mpo_matches_analytic", 1e-12);
}

// ===========================================================================
// V7 — Heisenberg N=2 MPO contracts to the analytic 2-site Hamiltonian.
// ===========================================================================
#[test]
fn v7_heisenberg_n2_mpo_matches_analytic() {
    let j = 1.5;
    let mpo = heisenberg_mpo_f64(2, j);
    let actual = contract_2site_mpo_f64(&mpo);
    let expected = heisenberg_ed_dense_f64(2, j);
    assert_dense_close(
        &actual,
        &expected,
        "v7_heisenberg_n2_mpo_matches_analytic",
        1e-12,
    );
}
