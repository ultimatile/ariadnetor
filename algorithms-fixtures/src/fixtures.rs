//! Test fixtures: U(1)-symmetric MPS / MPO chains carrying the XY
//! hopping interaction `H = J (S+_a S-_{a+1} + S-_a S+_{a+1})`. The
//! n=2 fixture places the bond at (0, 1) and encodes chain charge
//! U1(1) via `MPS[1].flux` so the 2-site psi at site=0 has flux
//! U1(1) ≠ identity. The n=3 fixture places the bond at (1, 2) with
//! chain charge in `MPS[2].right_bond = U1(1)` (per-site fluxes
//! identity) so an extended-env path runs without the flux check
//! overlapping. A complex variant of the n=2 fixture exercises the
//! `Scalar = Complex<f64>` matvec path.

use arnet_mps::{Mpo, Mps};
use arnet_tensor::test_fixtures::legs;
use arnet_tensor::{
    BlockCoord, BlockSparseLayout, BlockSparseStorage, BlockSparseTensor, Direction, Sector,
    U1Sector,
};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// n=2 f64
// ---------------------------------------------------------------------------

/// `n = 2` U(1) MPS fixture (`f64`); chain charge `U1(1)` is carried on
/// `MPS[1].flux` so the 2-site psi at site 0 has non-identity flux.
pub fn make_n2_mps_f64() -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let two_sec = vec![(U1Sector(0), 1), (U1Sector(1), 1)];

    let mut mps0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::Out),
            (two_sec.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps0.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .expect("(0,0,0)")[0] = 0.7;
    mps0.block_data_mut(&BlockCoord(vec![0, 1, 1]))
        .expect("(0,1,1)")[0] = 0.4;

    let mut mps1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (two_sec, Direction::Out),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector(1),
    );
    mps1.block_data_mut(&BlockCoord(vec![0, 1, 0]))
        .expect("(0,1,0)")[0] = 0.3;
    mps1.block_data_mut(&BlockCoord(vec![1, 0, 0]))
        .expect("(1,0,0)")[0] = -0.5;

    Mps::from_sites(vec![mps0, mps1])
}

/// `n = 2` XY-hopping MPO fixture (`f64`) with coupling `j`.
pub fn make_n2_mpo_f64(j: f64) -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];

    let mut w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::In),
            (phys.clone(), Direction::Out),
            (xy_bond.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    // J·S- at (W_l=0, ket=1, bra=0, W_r=U1(-1))
    w0.block_data_mut(&BlockCoord(vec![0, 1, 0, 0]))
        .expect("J·S-")[0] = j;
    // J·S+ at (W_l=0, ket=0, bra=1, W_r=U1(+1))
    w0.block_data_mut(&BlockCoord(vec![0, 0, 1, 1]))
        .expect("J·S+")[0] = j;

    let mut w1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (xy_bond, Direction::Out),
            (phys.clone(), Direction::In),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector::identity(),
    );
    // S+ at (W_l=U1(-1), ket=0, bra=1, W_r=0)
    w1.block_data_mut(&BlockCoord(vec![0, 0, 1, 0]))
        .expect("S+")[0] = 1.0;
    // S- at (W_l=U1(+1), ket=1, bra=0, W_r=0)
    w1.block_data_mut(&BlockCoord(vec![1, 1, 0, 0]))
        .expect("S-")[0] = 1.0;

    Mpo::from_sites(vec![w0, w1])
}

// ---------------------------------------------------------------------------
// n=3 f64 (bulk-env coverage)
// ---------------------------------------------------------------------------

/// `n = 3` U(1) MPS fixture (`f64`) for bulk-env coverage; chain charge
/// sits on `MPS[2].right_bond = U1(1)` with per-site fluxes identity.
pub fn make_n3_mps_f64() -> Mps<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let edge_left = vec![(U1Sector(0), 1)];
    let two_sec = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let edge_right = vec![(U1Sector(1), 1)];

    let mut mps0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (edge_left, Direction::Out),
            (phys.clone(), Direction::Out),
            (two_sec.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps0.block_data_mut(&BlockCoord(vec![0, 0, 0])).expect("a")[0] = 0.6;
    mps0.block_data_mut(&BlockCoord(vec![0, 1, 1])).expect("b")[0] = 0.35;

    let mut mps1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (two_sec.clone(), Direction::Out),
            (phys.clone(), Direction::Out),
            (two_sec.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps1.block_data_mut(&BlockCoord(vec![0, 0, 0])).expect("c")[0] = 0.4;
    mps1.block_data_mut(&BlockCoord(vec![0, 1, 1])).expect("d")[0] = -0.25;
    mps1.block_data_mut(&BlockCoord(vec![1, 0, 1])).expect("e")[0] = 0.5;

    let mut mps2 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (two_sec, Direction::Out),
            (phys, Direction::Out),
            (edge_right, Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps2.block_data_mut(&BlockCoord(vec![0, 1, 0])).expect("f")[0] = 0.3;
    mps2.block_data_mut(&BlockCoord(vec![1, 0, 0])).expect("g")[0] = -0.45;

    Mps::from_sites(vec![mps0, mps1, mps2])
}

/// `n = 3` XY-hopping MPO fixture (`f64`) with coupling `j`; `W[1]`
/// carries the multi-sector pair-start bond exercised by the bulk env.
pub fn make_n3_mpo_f64(j: f64) -> Mpo<BlockSparseStorage<f64>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];

    // W[0]: identity propagator (1×d×d×1).
    let mut w0 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::In),
            (phys.clone(), Direction::Out),
            (trivial.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    w0.block_data_mut(&BlockCoord(vec![0, 0, 0, 0]))
        .expect("I00")[0] = 1.0;
    w0.block_data_mut(&BlockCoord(vec![0, 1, 1, 0]))
        .expect("I11")[0] = 1.0;

    // W[1]: XY pair-start (1×d×d×2 with multi-sector W_r bond).
    let mut w1 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::In),
            (phys.clone(), Direction::Out),
            (xy_bond.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    w1.block_data_mut(&BlockCoord(vec![0, 1, 0, 0]))
        .expect("J·S-")[0] = j;
    w1.block_data_mut(&BlockCoord(vec![0, 0, 1, 1]))
        .expect("J·S+")[0] = j;

    // W[2]: XY pair-finish (2×d×d×1 with multi-sector W_l bond).
    let mut w2 = BlockSparseTensor::<f64, U1Sector>::zeros(
        legs([
            (xy_bond, Direction::Out),
            (phys.clone(), Direction::In),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector::identity(),
    );
    w2.block_data_mut(&BlockCoord(vec![0, 0, 1, 0]))
        .expect("S+")[0] = 1.0;
    w2.block_data_mut(&BlockCoord(vec![1, 1, 0, 0]))
        .expect("S-")[0] = 1.0;

    Mpo::from_sites(vec![w0, w1, w2])
}

// ---------------------------------------------------------------------------
// n=2 c64 (complex coverage)
// ---------------------------------------------------------------------------

/// Complex-`f64` variant of the `n = 2` MPS fixture; exercises the
/// `Scalar = Complex<f64>` matvec path.
pub fn make_n2_mps_c64() -> Mps<BlockSparseStorage<Complex<f64>>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let two_sec = vec![(U1Sector(0), 1), (U1Sector(1), 1)];

    let mut mps0 = BlockSparseTensor::<Complex<f64>, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::Out),
            (two_sec.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    mps0.block_data_mut(&BlockCoord(vec![0, 0, 0])).expect("a")[0] = Complex::new(0.7, 0.1);
    mps0.block_data_mut(&BlockCoord(vec![0, 1, 1])).expect("b")[0] = Complex::new(0.4, -0.2);

    let mut mps1 = BlockSparseTensor::<Complex<f64>, U1Sector>::zeros(
        legs([
            (two_sec, Direction::Out),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector(1),
    );
    mps1.block_data_mut(&BlockCoord(vec![0, 1, 0])).expect("c")[0] = Complex::new(0.3, 0.05);
    mps1.block_data_mut(&BlockCoord(vec![1, 0, 0])).expect("d")[0] = Complex::new(-0.5, 0.15);

    Mps::from_sites(vec![mps0, mps1])
}

/// Complex-`f64` variant of the `n = 2` XY-hopping MPO fixture with coupling `j`.
pub fn make_n2_mpo_c64(
    j: f64,
) -> Mpo<BlockSparseStorage<Complex<f64>>, BlockSparseLayout<U1Sector>> {
    let phys = vec![(U1Sector(0), 1), (U1Sector(1), 1)];
    let trivial = vec![(U1Sector(0), 1)];
    let xy_bond = vec![(U1Sector(-1), 1), (U1Sector(1), 1)];
    let jj = Complex::new(j, 0.0);
    let one = Complex::new(1.0, 0.0);

    let mut w0 = BlockSparseTensor::<Complex<f64>, U1Sector>::zeros(
        legs([
            (trivial.clone(), Direction::Out),
            (phys.clone(), Direction::In),
            (phys.clone(), Direction::Out),
            (xy_bond.clone(), Direction::In),
        ]),
        U1Sector::identity(),
    );
    w0.block_data_mut(&BlockCoord(vec![0, 1, 0, 0])).expect("a")[0] = jj;
    w0.block_data_mut(&BlockCoord(vec![0, 0, 1, 1])).expect("b")[0] = jj;

    let mut w1 = BlockSparseTensor::<Complex<f64>, U1Sector>::zeros(
        legs([
            (xy_bond, Direction::Out),
            (phys.clone(), Direction::In),
            (phys, Direction::Out),
            (trivial, Direction::In),
        ]),
        U1Sector::identity(),
    );
    w1.block_data_mut(&BlockCoord(vec![0, 0, 1, 0])).expect("c")[0] = one;
    w1.block_data_mut(&BlockCoord(vec![1, 1, 0, 0])).expect("d")[0] = one;

    Mpo::from_sites(vec![w0, w1])
}
