//! Dense (non-symmetric) test fixtures: a minimal spin-1/2 Heisenberg
//! MPO builder, random Dense MPS generators, and random Hermitian
//! matrices, shared across the algorithms crate's integration tests and
//! the local-eigensolver benchmark.
//!
//! Heisenberg is preferred over TFI as the test Hamiltonian because the
//! planned BlockSparse / U(1) wrapper path reuses Heisenberg as its test
//! Hamiltonian (TFI is not U(1) symmetric); sharing one builder body
//! keeps the two test surfaces directly comparable when that follow-up
//! lands.

use ariadnetor_core::Scalar;
use ariadnetor_mps::{Mpo, Mps};
use ariadnetor_native::NativeBackend;
use ariadnetor_tensor::{ComputeBackendTensorExt, DenseLayout, DenseStorage, DenseTensor, Host};
use num_complex::Complex;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Physical (spin-1/2) dimension of the Heisenberg MPO toolkit below.
pub const D: usize = 2;

/// `(k_ket, b_bra)` → matrix element of a single-site operator.
pub type Op = fn(usize, usize) -> f64;

/// Identity single-site operator (Kronecker delta `δ_{k,b}`).
pub fn op_id(k: usize, b: usize) -> f64 {
    if k == b { 1.0 } else { 0.0 }
}

/// Diagonal `Sz`-type operator: `+1` on basis state 0, `-1` on basis state 1.
pub fn op_sz(k: usize, b: usize) -> f64 {
    if k == b {
        if k == 0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

/// σ⁺ raises (|down⟩ → |up⟩); single non-zero element at (k_ket=1, b_bra=0).
pub fn op_sp(k: usize, b: usize) -> f64 {
    if k == 1 && b == 0 { 1.0 } else { 0.0 }
}

/// σ⁻ lowers (|up⟩ → |down⟩); single non-zero element at (k_ket=0, b_bra=1).
pub fn op_sm(k: usize, b: usize) -> f64 {
    if k == 0 && b == 1 { 1.0 } else { 0.0 }
}

/// Fill a rank-4 MPO site tensor `[w_l_dim, D, D, w_r_dim]` column-major:
/// each `(vl, vr, op, scale)` cell adds `scale * op(k, b)` at virtual
/// bond indices `(vl, vr)`.
pub fn build_mpo_site_f64(
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

/// Spin-1/2 Heisenberg `H = J Σ S_i · S_{i+1}` as a bond-dim-5 MPO.
pub fn heisenberg_mpo_f64(n: usize, j: f64) -> Mpo<DenseStorage<f64>, DenseLayout> {
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

/// Random Dense MPS with physical dimensions `phys` and bond dimensions
/// `bonds` (boundaries included, so `bonds.len()` is `phys.len() + 1`) and
/// no canonical form set. Entries are drawn by `draw` site by site, each
/// site in storage order.
fn random_mps<T: Scalar>(
    phys: &[usize],
    bonds: &[usize],
    seed: u64,
    mut draw: impl FnMut(&mut StdRng) -> T,
) -> Mps<DenseStorage<T>, DenseLayout> {
    assert_eq!(bonds.len(), phys.len() + 1, "one bond per site boundary");
    let mut rng = StdRng::seed_from_u64(seed);
    let sites = (0..phys.len())
        .map(|j| {
            let shape = vec![bonds[j], phys[j], bonds[j + 1]];
            let data = (0..shape.iter().product::<usize>())
                .map(|_| draw(&mut rng))
                .collect();
            Host::shared().dense(data, shape)
        })
        .collect();
    Mps::from_sites(sites)
}

/// Random Dense MPS canonicalized with the orthogonality center at site 0.
pub fn random_mps_center_zero_f64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    let mut mps = random_mps_unknown_f64(n, d, chi, seed);
    mps.canonicalize(&NativeBackend::new(), 0);
    mps
}

/// Random real symmetric matrix `H = (A + Aᵀ) / 2` of size `n×n`, with the
/// data laid out column-major.
pub fn random_hermitian_f64(n: usize, seed: u64) -> DenseTensor<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let a_data = a.data_slice();
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = a_data[i + n * j];
            let aji = a_data[j + n * i];
            data[i + n * j] = 0.5 * (aij + aji);
        }
    }
    Host::shared().dense(data, vec![n, n])
}

/// Random complex Hermitian matrix `H = (A + A^H) / 2` of size `n×n`, with
/// the data laid out column-major.
pub fn random_hermitian_complex_f64(n: usize, seed: u64) -> DenseTensor<Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let real = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let imag = DenseTensor::<f64>::random(vec![n, n], &mut rng);
    let r = real.data_slice();
    let im = imag.data_slice();
    let mut data = vec![Complex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = Complex::new(r[i + n * j], im[i + n * j]);
            let aji = Complex::new(r[j + n * i], im[j + n * i]);
            data[i + n * j] = (aij + aji.conj()) * 0.5;
        }
    }
    Host::shared().dense(data, vec![n, n])
}

/// Random complex Dense MPS with per-site physical dimensions `phys` and
/// bond dimensions `bonds` (boundaries included, so `bonds.len()` is
/// `phys.len() + 1`), and no canonical form set.
pub fn random_mps_c64(
    phys: &[usize],
    bonds: &[usize],
    seed: u64,
) -> Mps<DenseStorage<Complex<f64>>, DenseLayout> {
    random_mps(phys, bonds, seed, |rng| {
        Complex::new(
            rng.random_range(-0.5_f64..0.5),
            rng.random_range(-0.5_f64..0.5),
        )
    })
}

/// One random complex Hermitian bond term per neighboring pair of `phys`,
/// with legs `(out_i, out_i+1, in_i, in_i+1)`. For a generic draw,
/// neighboring terms do not commute and no term is symmetric under
/// transposition.
pub fn random_bond_terms_c64(phys: &[usize], seed: u64) -> Vec<DenseTensor<Complex<f64>>> {
    phys.windows(2)
        .zip(seed..)
        .map(|(pair, s)| {
            random_hermitian_complex_f64(pair[0] * pair[1], s)
                .reshape_logical(vec![pair[0], pair[1], pair[0], pair[1]])
        })
        .collect()
}

/// Random Dense MPS with no canonical form set (`CanonicalForm::Unknown`);
/// the caller is responsible for canonicalizing.
pub fn random_mps_unknown_f64(
    n: usize,
    d: usize,
    chi: usize,
    seed: u64,
) -> Mps<DenseStorage<f64>, DenseLayout> {
    let bonds: Vec<usize> = (0..=n)
        .map(|j| if j == 0 || j == n { 1 } else { chi })
        .collect();
    random_mps(&vec![d; n], &bonds, seed, |rng| {
        rng.random_range(-0.5_f64..0.5)
    })
}
