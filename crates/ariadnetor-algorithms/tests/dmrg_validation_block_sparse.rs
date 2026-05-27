//! End-to-end validation of the BlockSparse / U(1) 2-site DMRG sweep
//! driver against exact diagonalization on the antiferromagnetic
//! Heisenberg (XXX) Hamiltonian
//!
//!   H = J Σᵢ (σᶻᵢσᶻᵢ₊₁ + 2 σ⁺ᵢσ⁻ᵢ₊₁ + 2 σ⁻ᵢσ⁺ᵢ₊₁)
//!
//! with even N (sites 6, 8, 10). Lieb-Mattis (1962) places the unique
//! ground state in the S = 0 sector, hence S_z = 0; under the
//! bit→sector convention `bit=0 → U1(0)`, `bit=1 → U1(+1)` this
//! corresponds to chain charge = N/2.
//!
//! ED reference is computed in the full Hilbert space (`2^N × 2^N`
//! dense `eigh`); the BlockSparse run pins chain charge = N/2 via the
//! terminal-site flux of the random initial MPS. Matrix-element
//! conventions (op_sp at (k_ket=1, b_bra=0), op_sm at (k=0, b=1))
//! match the Dense validation file (`dmrg_validation.rs`), since the
//! ED reference is reused.
//!
//! Test-internal helpers; no public API additions.

use std::sync::Arc;

use arnet::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, ComputeBackend,
    DenseTensor, Direction, NativeBackend, QNIndex, Sector, TruncSvdParams, U1Sector, eigh,
};
use arnet_algorithms::dmrg::{DmrgEnvs, DmrgSweepParams, LocalEigensolverParams, sweep_2site};
use arnet_algorithms::krylov::LanczosParams;
use arnet_mps::{CanonicalForm, Mpo, Mps, TensorChain, canonicalize};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const D: usize = 2; // physical dim (spin-1/2)

// ---------------------------------------------------------------------------
// Pauli matrix elements — bit=0=up, bit=1=down. Matches the Dense
// validation file's convention exactly: σ⁺ raises (|down⟩→|up⟩) so
// its non-zero matrix element ⟨b=0|σ⁺|k=1⟩ = 1 corresponds to
// op_sp(k=1, b=0); σ⁻ symmetrically at op_sm(k=0, b=1).
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

// ---------------------------------------------------------------------------
// Heisenberg ED reference, column-major (row=bra, col=ket). Off-diagonal
// term covers σ⁺σ⁻ + σ⁻σ⁺ as a single "swap-bits" rule for each
// nearest-neighbour pair where the two bits differ. Identical to the
// Dense validation file's reference; duplicated here so the BlockSparse
// validation file is self-contained.
// ---------------------------------------------------------------------------

fn heisenberg_ed_dense_f64(n: usize, j: f64) -> DenseTensor<f64> {
    let backend = NativeBackend::shared();
    let dim = 1usize << n;
    let mut data = vec![0.0_f64; dim * dim];
    for b in 0..dim {
        let mut diag = 0.0_f64;
        for i in 0..n - 1 {
            let zi = z_of_bit((b >> i) & 1);
            let zi1 = z_of_bit((b >> (i + 1)) & 1);
            diag += j * zi * zi1;
        }
        write_diag(&mut data, dim, b, diag);
        for i in 0..n - 1 {
            let bi = (b >> i) & 1;
            let bi1 = (b >> (i + 1)) & 1;
            if bi != bi1 {
                let b_out = b ^ ((1 << i) | (1 << (i + 1)));
                write_offdiag(&mut data, dim, b_out, b, 2.0 * j);
            }
        }
    }
    DenseTensor::from_raw_parts(
        data,
        vec![dim, dim],
        backend.preferred_order(),
        Arc::clone(&backend),
    )
}

fn dense_min_eig_f64(h: &DenseTensor<f64>) -> f64 {
    let (eigvals, _v) = eigh(h, 1).expect("eigh");
    eigvals
        .data_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// BlockSparse / U(1) Heisenberg MPO.
//
// Bulk W is the standard Heisenberg 5×5 (rows / cols indexed by the
// bond "channel"):
//
//   row 0 = identity propagator (charge 0, channel id-start)
//   row 1 = "σ⁻ was applied — expects σ⁺ next"          (charge +1)
//   row 2 = "σ⁺ was applied — expects σ⁻ next"          (charge -1)
//   row 3 = "σᶻ was applied — expects σᶻ next"          (charge 0, σᶻ chan)
//   row 4 = identity finished (charge 0, channel id-finish)
//
// BlockSparse bond QNIndex layout (total dim 5):
//   `[(U1(-1), 1), (U1(0), 3), (U1(+1), 1)]`
//
// Sub-channel ordering inside the U1(0) sector:
//   chan 0 = id-start, chan 1 = σᶻ-pending, chan 2 = id-finish.
//
// Site 0 collapses W_l from 5 channels to a single trivial(U1(0))
// channel that semantically equals the bulk's id-finish row 4 (the
// chain begins by "starting a new term"). Site N-1 collapses W_r to
// trivial(U1(0)) which equals the bulk id-start col 0 (terms
// terminate by closing back to identity).
//
// All site fluxes = identity. Per-cell encoding follows the Dense
// validation file's operator convention (op_sp at (k_ket=1, b_bra=0))
// so this MPO and `heisenberg_ed_dense_f64` agree element-wise on
// the same 2-site contraction.
// ---------------------------------------------------------------------------

// Sector / sub-channel constants.
const SEC_NEG: usize = 0; // U1(-1)
const SEC_ZERO: usize = 1; // U1( 0)
const SEC_POS: usize = 2; // U1(+1)

const ZCHAN_ID_START: usize = 0;
const ZCHAN_SZ_PEND: usize = 1;
const ZCHAN_ID_FINISH: usize = 2;

fn bond5() -> Vec<(U1Sector, usize)> {
    vec![(U1Sector(-1), 1), (U1Sector(0), 3), (U1Sector(1), 1)]
}

fn phys2() -> Vec<(U1Sector, usize)> {
    vec![(U1Sector(0), 1), (U1Sector(1), 1)]
}

fn trivial1() -> Vec<(U1Sector, usize)> {
    vec![(U1Sector(0), 1)]
}

// Column-major flat index inside a block of shape [d_l, d_ket, d_bra, d_r]
// at coords (chan_l, 0, 0, chan_r) with d_ket = d_bra = 1, matching
// `NativeBackend::preferred_order() == ColumnMajor`.
fn flat_within_phys1(d_l: usize, chan_l: usize, chan_r: usize) -> usize {
    chan_l + d_l * chan_r
}

fn heisenberg_mpo_bsp_f64(
    n: usize,
    j: f64,
) -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    assert!(n >= 2, "heisenberg_mpo_bsp_f64 requires n >= 2");
    let mut sites: Vec<BlockSparseTensor<f64, U1Sector>> = Vec::with_capacity(n);

    // ---- Site 0: W_l = trivial, W_r = bond5 ----
    // Acts as "identity-finished" (Dense row 4) projected to a single
    // W_l channel.
    let mut w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(trivial1(), Direction::Out),
            QNIndex::new(phys2(), Direction::In),
            QNIndex::new(phys2(), Direction::Out),
            QNIndex::new(bond5(), Direction::In),
        ],
        U1Sector::identity(),
    );
    // (vR=1, σ⁻, 2J) → block (0, ket=0, bra=1, W_r=U1(+1))
    w0.block_data_mut(&BlockCoord(vec![0, 0, 1, SEC_POS]))
        .expect("site0 σ⁻")[0] = 2.0 * j;
    // (vR=2, σ⁺, 2J) → block (0, ket=1, bra=0, W_r=U1(-1))
    w0.block_data_mut(&BlockCoord(vec![0, 1, 0, SEC_NEG]))
        .expect("site0 σ⁺")[0] = 2.0 * j;
    // (vR=3, σᶻ, J): writes into block (0, k, k, U1(0)) at chan_r=σᶻ-pending.
    // Block shape [1, 1, 1, 3], col-major flat = 0 + 1*0 + 1*0 + 1*chan_r = chan_r.
    {
        let blk = w0
            .block_data_mut(&BlockCoord(vec![0, 0, 0, SEC_ZERO]))
            .expect("site0 σᶻ k=0");
        blk[ZCHAN_SZ_PEND] += j; // +J · 1
        blk[ZCHAN_ID_FINISH] += 1.0; // (vR=4, I, 1) k=0
    }
    {
        let blk = w0
            .block_data_mut(&BlockCoord(vec![0, 1, 1, SEC_ZERO]))
            .expect("site0 σᶻ k=1");
        blk[ZCHAN_SZ_PEND] += -j; // +J · (-1)
        blk[ZCHAN_ID_FINISH] += 1.0; // (vR=4, I, 1) k=1
    }
    sites.push(w0);

    // ---- Bulk sites (1 ..= n-2): W_l = bond5, W_r = bond5 ----
    for _ in 1..n - 1 {
        let mut w = BlockSparseTensor::<f64, U1Sector>::zeros(
            vec![
                QNIndex::new(bond5(), Direction::Out),
                QNIndex::new(phys2(), Direction::In),
                QNIndex::new(phys2(), Direction::Out),
                QNIndex::new(bond5(), Direction::In),
            ],
            U1Sector::identity(),
        );
        // (0,0,I,1) ket=k bra=k → block (U1(0), k, k, U1(0))
        // (3,0,σᶻ,1) ket=k bra=k → same block, different chan_l/chan_r
        // (4,3,σᶻ,J) ket=k bra=k → same
        // (4,4,I,1) ket=k bra=k → same
        // Block shape [3, 1, 1, 3]; col-major flat = chan_l + 3·chan_r.
        for k in 0..D {
            let z = if k == 0 { 1.0 } else { -1.0 };
            let blk = w
                .block_data_mut(&BlockCoord(vec![SEC_ZERO, k, k, SEC_ZERO]))
                .expect("bulk diag U1(0)");
            // Dense (0,0)=I → (chan_l=id_start, chan_r=id_start)
            blk[flat_within_phys1(3, ZCHAN_ID_START, ZCHAN_ID_START)] += 1.0;
            // Dense (3,0)=σᶻ → (chan_l=σᶻ_pend, chan_r=id_start)
            blk[flat_within_phys1(3, ZCHAN_SZ_PEND, ZCHAN_ID_START)] += z;
            // Dense (4,3)=J·σᶻ → (chan_l=id_finish, chan_r=σᶻ_pend)
            blk[flat_within_phys1(3, ZCHAN_ID_FINISH, ZCHAN_SZ_PEND)] += j * z;
            // Dense (4,4)=I → (chan_l=id_finish, chan_r=id_finish)
            blk[flat_within_phys1(3, ZCHAN_ID_FINISH, ZCHAN_ID_FINISH)] += 1.0;
        }

        // (1,0,σ⁺,1): ket=1 bra=0; block (U1(+1), 1, 0, U1(0)). Shape [1,1,1,3]; flat = chan_r.
        w.block_data_mut(&BlockCoord(vec![SEC_POS, 1, 0, SEC_ZERO]))
            .expect("bulk σ⁺ close")[ZCHAN_ID_START] = 1.0;
        // (2,0,σ⁻,1): ket=0 bra=1; block (U1(-1), 0, 1, U1(0))
        w.block_data_mut(&BlockCoord(vec![SEC_NEG, 0, 1, SEC_ZERO]))
            .expect("bulk σ⁻ close")[ZCHAN_ID_START] = 1.0;
        // (4,1,σ⁻,2J): ket=0 bra=1; block (U1(0), 0, 1, U1(+1)). Shape [3,1,1,1]; flat = chan_l.
        w.block_data_mut(&BlockCoord(vec![SEC_ZERO, 0, 1, SEC_POS]))
            .expect("bulk σ⁻ start")[ZCHAN_ID_FINISH] = 2.0 * j;
        // (4,2,σ⁺,2J): ket=1 bra=0; block (U1(0), 1, 0, U1(-1))
        w.block_data_mut(&BlockCoord(vec![SEC_ZERO, 1, 0, SEC_NEG]))
            .expect("bulk σ⁺ start")[ZCHAN_ID_FINISH] = 2.0 * j;

        sites.push(w);
    }

    // ---- Site n-1: W_l = bond5, W_r = trivial ----
    let mut wn = BlockSparseTensor::<f64, U1Sector>::zeros(
        vec![
            QNIndex::new(bond5(), Direction::Out),
            QNIndex::new(phys2(), Direction::In),
            QNIndex::new(phys2(), Direction::Out),
            QNIndex::new(trivial1(), Direction::In),
        ],
        U1Sector::identity(),
    );
    // (vL=0, I, 1) ket=k bra=k: block (U1(0), k, k, 0); shape [3,1,1,1]; chan_l=id_start.
    // (vL=3, σᶻ, 1) same block; chan_l=σᶻ_pending.
    for k in 0..D {
        let z = if k == 0 { 1.0 } else { -1.0 };
        let blk = wn
            .block_data_mut(&BlockCoord(vec![SEC_ZERO, k, k, 0]))
            .expect("siteN diag U1(0)");
        blk[ZCHAN_ID_START] += 1.0;
        blk[ZCHAN_SZ_PEND] += z;
    }
    // (vL=1, σ⁺, 1): ket=1 bra=0; block (U1(+1), 1, 0, 0). Shape [1,1,1,1].
    wn.block_data_mut(&BlockCoord(vec![SEC_POS, 1, 0, 0]))
        .expect("siteN σ⁺ close")[0] = 1.0;
    // (vL=2, σ⁻, 1): ket=0 bra=1; block (U1(-1), 0, 1, 0).
    wn.block_data_mut(&BlockCoord(vec![SEC_NEG, 0, 1, 0]))
        .expect("siteN σ⁻ close")[0] = 1.0;

    sites.push(wn);

    Mpo::from_sites(sites)
}

// ---------------------------------------------------------------------------
// Random U(1) BlockSparse MPS pinned to chain charge = `total_charge`.
//
// Site flux: identity for sites 0..n-2, terminal site flux =
// `total_charge`. Internal bond carries the full ladder of sectors
// `0 ..= total_charge` so any partial-sum-of-bits-equals-q at
// site-cumulative-charge q has a non-empty allocation. Block
// fills use uniform `[-0.5, 0.5)` per-entry; `BlockSparseTensor::zeros`
// enforces fusion-legality so only allowed coords are written. The
// site is then handed to `canonicalize(_, 0)` to land in
// `Mixed { center: 0 }` form.
// ---------------------------------------------------------------------------

fn random_mps_bsp_center_zero_f64(
    n: usize,
    chi_internal: usize,
    total_charge: i32,
    seed: u64,
) -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    assert!(n >= 2, "random_mps_bsp_center_zero_f64 requires n >= 2");
    assert!(
        total_charge >= 0 && (total_charge as usize) <= n,
        "total_charge {} out of range [0, {}]",
        total_charge,
        n
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // Internal bond: full ladder 0 ..= total_charge with multiplicity
    // chi_internal each.
    let internal_legs: Vec<(U1Sector, usize)> = (0..=total_charge)
        .map(|q| (U1Sector(q), chi_internal))
        .collect();

    let mut storages: Vec<BlockSparseTensor<f64, U1Sector>> = Vec::with_capacity(n);

    for site in 0..n {
        let left_legs: Vec<(U1Sector, usize)> = if site == 0 {
            trivial1()
        } else {
            internal_legs.clone()
        };
        let right_legs: Vec<(U1Sector, usize)> = if site == n - 1 {
            trivial1()
        } else {
            internal_legs.clone()
        };
        let flux = if site == n - 1 {
            U1Sector(total_charge)
        } else {
            U1Sector::identity()
        };

        let mut s = BlockSparseTensor::<f64, U1Sector>::zeros(
            vec![
                QNIndex::new(left_legs, Direction::Out),
                QNIndex::new(phys2(), Direction::Out),
                QNIndex::new(right_legs, Direction::In),
            ],
            flux,
        );
        // Iterate every fusion-legal block and fill with random
        // `[-0.5, 0.5)` entries. `block_metas` returns the meta for
        // each allocated block; size = product of block dims.
        let coords: Vec<BlockCoord> = s.block_metas().iter().map(|m| m.coord.clone()).collect();
        for coord in coords {
            let blk = s.block_data_mut(&coord).expect("legal block");
            for v in blk.iter_mut() {
                *v = rng.random_range(-0.5_f64..0.5);
            }
        }
        storages.push(s);
    }

    let mut mps = Mps::from_sites(storages);
    canonicalize(&mut mps, 0);
    assert_eq!(
        *mps.canonical_form(),
        CanonicalForm::Mixed { center: 0 },
        "random_mps_bsp_center_zero_f64: post-canonicalize form mismatch"
    );
    mps
}

// ---------------------------------------------------------------------------
// Sweep parameters (mirroring the Dense validation file's
// `validation_params`).
// ---------------------------------------------------------------------------

fn validation_params_bsp(chi_max: usize, lanczos_seed: u64) -> DmrgSweepParams {
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
// Run a BlockSparse DMRG validation case end-to-end.
//
// `total_charge` pins the chain charge for the random initial MPS
// (= n/2 for Heisenberg even-N S_z=0 GS targeting). Acceptance is
// `|E_BSP - E_ED| ≤ tol`; primary `tol = 1e-8`. Caller can supply a
// relaxed tol for large-N gapless cases that converge slower.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_validation_bsp(
    name: &str,
    mpo: Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>>,
    e_ed: f64,
    chi_max: usize,
    chi_internal: usize,
    init_seed: u64,
    lanczos_seed: u64,
    total_charge: i32,
    tol: f64,
) {
    let n = mpo.len();
    let mut mps = random_mps_bsp_center_zero_f64(n, chi_internal, total_charge, init_seed);
    let mut envs = DmrgEnvs::build(&mps, &mpo).expect("envs build");
    let params = validation_params_bsp(chi_max, lanczos_seed);

    let result = sweep_2site(&mut envs, &mut mps, &mpo, &params)
        .unwrap_or_else(|e| panic!("{name}: sweep_2site failed: {e:?}"));

    let delta = (result.energy - e_ed).abs();
    assert!(
        delta <= tol,
        "{name}: |E_BSP - E_ED| = {delta:.3e} > {tol:.0e} (E_BSP={:.10}, E_ED={:.10})",
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
// V1 — Heisenberg N=6, J=1.0 vs ED.
// ===========================================================================
#[test]
fn v1_bsp_heisenberg_n6_vs_ed() {
    let n = 6;
    let j = 1.0;
    let mpo = heisenberg_mpo_bsp_f64(n, j);
    let e_ed = dense_min_eig_f64(&heisenberg_ed_dense_f64(n, j));
    run_validation_bsp(
        "v1_bsp_heisenberg_n6",
        mpo,
        e_ed,
        32,
        4,
        0xB6F1,
        0xB101,
        (n / 2) as i32,
        1e-8,
    );
}

// ===========================================================================
// V2 — Heisenberg N=8, J=1.0 vs ED.
// ===========================================================================
#[test]
fn v2_bsp_heisenberg_n8_vs_ed() {
    let n = 8;
    let j = 1.0;
    let mpo = heisenberg_mpo_bsp_f64(n, j);
    let e_ed = dense_min_eig_f64(&heisenberg_ed_dense_f64(n, j));
    run_validation_bsp(
        "v2_bsp_heisenberg_n8",
        mpo,
        e_ed,
        32,
        4,
        0xB6F2,
        0xB102,
        (n / 2) as i32,
        1e-8,
    );
}

// ===========================================================================
// V3 — Heisenberg N=10, J=1.0 vs ED.
// ===========================================================================
#[test]
fn v3_bsp_heisenberg_n10_vs_ed() {
    let n = 10;
    let j = 1.0;
    let mpo = heisenberg_mpo_bsp_f64(n, j);
    let e_ed = dense_min_eig_f64(&heisenberg_ed_dense_f64(n, j));
    run_validation_bsp(
        "v3_bsp_heisenberg_n10",
        mpo,
        e_ed,
        32,
        4,
        0xB6F3,
        0xB103,
        (n / 2) as i32,
        1e-8,
    );
}
