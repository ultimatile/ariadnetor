//! Targeted mutation-testing coverage for site_ops.rs.
//!
//! Verifies sign-sensitive elements that mutants would flip, including
//! SpinHalf::dim, SpinHalf::sz negation, and Qubit y/z/h sign patterns.

use approx::assert_abs_diff_eq;
use arnet::mps::{Qubit, SiteOps, SpinHalf};

// --------------------------------------------------------------------------
// SpinHalf::dim — catch mutant replacing 2 with 0 or 1
// --------------------------------------------------------------------------

#[test]
fn test_spin_half_dim_is_exactly_2() {
    let d = SpinHalf.dim();
    assert_eq!(d, 2);
    assert_ne!(d, 0);
    assert_ne!(d, 1);
}

// --------------------------------------------------------------------------
// SpinHalf::sz — signs on diagonal
// --------------------------------------------------------------------------

#[test]
fn test_sz_diagonal_signs() {
    let sz = SpinHalf.sz::<f64>();
    // (0,0) must be positive 0.5, not -0.5
    assert!(sz.get(&[0, 0]) > 0.0);
    assert_abs_diff_eq!(sz.get(&[0, 0]), 0.5, epsilon = 1e-15);
    // (1,1) must be negative -0.5, not +0.5
    assert!(sz.get(&[1, 1]) < 0.0);
    assert_abs_diff_eq!(sz.get(&[1, 1]), -0.5, epsilon = 1e-15);
    // trace = 0
    assert_abs_diff_eq!(sz.get(&[0, 0]) + sz.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_sz_off_diagonal_zero() {
    let sz = SpinHalf.sz::<f64>();
    assert_abs_diff_eq!(sz.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[1, 0]), 0.0, epsilon = 1e-15);
}

// --------------------------------------------------------------------------
// Qubit::dim — catch mutant replacing 2 with 0 or 1
// --------------------------------------------------------------------------

#[test]
fn test_qubit_dim_is_exactly_2() {
    let d = Qubit.dim();
    assert_eq!(d, 2);
    assert_ne!(d, 0);
    assert_ne!(d, 1);
}

// --------------------------------------------------------------------------
// Qubit::y — sign-sensitive imaginary elements
// --------------------------------------------------------------------------

#[test]
fn test_qubit_y_f64_sign_pattern() {
    // For real type T, from_real_imag drops imaginary part, so Y becomes
    // the zero matrix for f64. We verify this to catch sign mutants in the
    // real-type path.
    let y = Qubit.y::<f64>();
    assert_abs_diff_eq!(y.get(&[0, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_y_complex_signs() {
    use arnet_tensor::Complex;
    let y = Qubit.y::<Complex<f64>>();
    // (0,0) = 0
    assert_abs_diff_eq!(y.get(&[0, 0]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[0, 0]).im, 0.0, epsilon = 1e-15);
    // (0,1) = -i: real=0, imag=-1
    assert_abs_diff_eq!(y.get(&[0, 1]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[0, 1]).im, -1.0, epsilon = 1e-15);
    assert!(y.get(&[0, 1]).im < 0.0, "(0,1) imaginary must be negative");
    // (1,0) = +i: real=0, imag=+1
    assert_abs_diff_eq!(y.get(&[1, 0]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 0]).im, 1.0, epsilon = 1e-15);
    assert!(y.get(&[1, 0]).im > 0.0, "(1,0) imaginary must be positive");
    // (1,1) = 0
    assert_abs_diff_eq!(y.get(&[1, 1]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 1]).im, 0.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_y_squared_is_identity_complex() {
    // Y^2 = I — catches sign-flip mutants in both (0,1) and (1,0)
    use arnet_tensor::Complex;
    let backend = arnet_native::NativeBackend::new();
    let y = Qubit.y::<Complex<f64>>();
    let y2 = arnet_linalg::contract(&backend, &y, &y, "ij,jk->ik").unwrap();
    let id = Qubit.id::<Complex<f64>>();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(y2.get(&[i, j]).re, id.get(&[i, j]).re, epsilon = 1e-12);
            assert_abs_diff_eq!(y2.get(&[i, j]).im, 0.0, epsilon = 1e-12);
        }
    }
}

// --------------------------------------------------------------------------
// Qubit::z — sign on (1,1)
// --------------------------------------------------------------------------

#[test]
fn test_qubit_z_exact_diagonal() {
    let z = Qubit.z::<f64>();
    assert_abs_diff_eq!(z.get(&[0, 0]), 1.0, epsilon = 1e-15);
    assert!(z.get(&[0, 0]) > 0.0, "Z(0,0) must be positive");
    assert_abs_diff_eq!(z.get(&[1, 1]), -1.0, epsilon = 1e-15);
    assert!(z.get(&[1, 1]) < 0.0, "Z(1,1) must be negative");
    // off-diagonal zero
    assert_abs_diff_eq!(z.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(z.get(&[1, 0]), 0.0, epsilon = 1e-15);
    // trace = 0
    assert_abs_diff_eq!(z.get(&[0, 0]) + z.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_z_squared_is_identity() {
    // Z^2 = I — catches sign flip on (1,1)
    let backend = arnet_native::NativeBackend::new();
    let z = Qubit.z::<f64>();
    let z2 = arnet_linalg::contract(&backend, &z, &z, "ij,jk->ik").unwrap();
    let id = Qubit.id::<f64>();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(z2.get(&[i, j]), id.get(&[i, j]), epsilon = 1e-12);
        }
    }
}

// --------------------------------------------------------------------------
// Qubit::h — sign on (1,1) element
// --------------------------------------------------------------------------

#[test]
fn test_qubit_h_exact_signs() {
    let h = Qubit.h::<f64>();
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    // All four elements
    assert_abs_diff_eq!(h.get(&[0, 0]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[0, 1]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[1, 0]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[1, 1]), -inv_sqrt2, epsilon = 1e-15);
    // (1,1) must be negative
    assert!(h.get(&[1, 1]) < 0.0, "H(1,1) must be negative");
    // (0,0), (0,1), (1,0) must be positive
    assert!(h.get(&[0, 0]) > 0.0);
    assert!(h.get(&[0, 1]) > 0.0);
    assert!(h.get(&[1, 0]) > 0.0);
}

#[test]
fn test_qubit_h_is_unitary() {
    // H^T H = I for real Hadamard — catches any wrong sign
    let backend = arnet_native::NativeBackend::new();
    let h = Qubit.h::<f64>();
    let hth = arnet_linalg::contract(&backend, &h, &h, "ab,ac->bc").unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(hth.get(&[i, j]), expected, epsilon = 1e-12);
        }
    }
}

// --------------------------------------------------------------------------
// Qubit::id — structural sanity
// --------------------------------------------------------------------------

#[test]
fn test_qubit_id_exact() {
    let id = Qubit.id::<f64>();
    assert_abs_diff_eq!(id.get(&[0, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[1, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[1, 1]), 1.0, epsilon = 1e-15);
}
