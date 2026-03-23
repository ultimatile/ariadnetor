//! SiteOps tests for SpinHalf and Qubit.

use approx::assert_abs_diff_eq;
use arnet::mps::{Qubit, SiteOps, SpinHalf};

// ============================================================================
// SpinHalf tests
// ============================================================================

#[test]
fn test_spin_half_dim() {
    assert_eq!(SpinHalf.dim(), 2);
}

#[test]
fn test_spin_half_sz_f64() {
    let sz = SpinHalf.sz::<f64>();
    assert_eq!(sz.shape(), &[2, 2]);
    assert_abs_diff_eq!(sz.get(&[0, 0]), 0.5, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[1, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[1, 1]), -0.5, epsilon = 1e-15);
}

#[test]
fn test_spin_half_sp_f64() {
    let sp = SpinHalf.sp::<f64>();
    assert_eq!(sp.shape(), &[2, 2]);
    assert_abs_diff_eq!(sp.get(&[0, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sp.get(&[0, 1]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sp.get(&[1, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sp.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_spin_half_sm_f64() {
    let sm = SpinHalf.sm::<f64>();
    assert_abs_diff_eq!(sm.get(&[0, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sm.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sm.get(&[1, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sm.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_spin_half_id_f64() {
    let id = SpinHalf.id::<f64>();
    assert_abs_diff_eq!(id.get(&[0, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[0, 1]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[1, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(id.get(&[1, 1]), 1.0, epsilon = 1e-15);
}

#[test]
fn test_spin_half_sz_f32() {
    let sz = SpinHalf.sz::<f32>();
    assert_abs_diff_eq!(sz.get(&[0, 0]), 0.5f32, epsilon = 1e-6);
    assert_abs_diff_eq!(sz.get(&[1, 1]), -0.5f32, epsilon = 1e-6);
}

#[test]
fn test_spin_half_sz_complex_f64() {
    use arnet_tensor::Complex;
    let sz = SpinHalf.sz::<Complex<f64>>();
    assert_abs_diff_eq!(sz.get(&[0, 0]).re, 0.5, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[0, 0]).im, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[1, 1]).re, -0.5, epsilon = 1e-15);
    assert_abs_diff_eq!(sz.get(&[1, 1]).im, 0.0, epsilon = 1e-15);
}

#[test]
fn test_spin_half_commutation() {
    // [S+, S-] = 2*Sz
    let backend = arnet_native::NativeBackend::new();
    let sp = SpinHalf.sp::<f64>();
    let sm = SpinHalf.sm::<f64>();
    let sz = SpinHalf.sz::<f64>();

    let sp_sm = arnet_linalg::contract(&backend, &sp, &sm, "ij,jk->ik").unwrap();
    let sm_sp = arnet_linalg::contract(&backend, &sm, &sp, "ij,jk->ik").unwrap();

    // [S+, S-] = S+S- - S-S+
    for i in 0..2 {
        for j in 0..2 {
            let commutator = sp_sm.get(&[i, j]) - sm_sp.get(&[i, j]);
            let expected = 2.0 * sz.get(&[i, j]);
            assert_abs_diff_eq!(commutator, expected, epsilon = 1e-12);
        }
    }
}

// ============================================================================
// Qubit tests
// ============================================================================

#[test]
fn test_qubit_dim() {
    assert_eq!(Qubit.dim(), 2);
}

#[test]
fn test_qubit_x_f64() {
    let x = Qubit.x::<f64>();
    assert_eq!(x.shape(), &[2, 2]);
    assert_abs_diff_eq!(x.get(&[0, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(x.get(&[0, 1]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(x.get(&[1, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(x.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_y_complex() {
    use arnet_tensor::Complex;
    let y = Qubit.y::<Complex<f64>>();
    assert_abs_diff_eq!(y.get(&[0, 1]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[0, 1]).im, -1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 0]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(y.get(&[1, 0]).im, 1.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_z_f64() {
    let z = Qubit.z::<f64>();
    assert_abs_diff_eq!(z.get(&[0, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(z.get(&[1, 1]), -1.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_hadamard_f64() {
    let h = Qubit.h::<f64>();
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_abs_diff_eq!(h.get(&[0, 0]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[0, 1]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[1, 0]), inv_sqrt2, epsilon = 1e-15);
    assert_abs_diff_eq!(h.get(&[1, 1]), -inv_sqrt2, epsilon = 1e-15);
}

#[test]
fn test_qubit_s_complex() {
    use arnet_tensor::Complex;
    let s = Qubit.s::<Complex<f64>>();
    assert_abs_diff_eq!(s.get(&[0, 0]).re, 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(s.get(&[1, 1]).re, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(s.get(&[1, 1]).im, 1.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_t_complex() {
    use arnet_tensor::Complex;
    let t = Qubit.t::<Complex<f64>>();
    let angle = std::f64::consts::FRAC_PI_4;
    assert_abs_diff_eq!(t.get(&[0, 0]).re, 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(t.get(&[1, 1]).re, angle.cos(), epsilon = 1e-12);
    assert_abs_diff_eq!(t.get(&[1, 1]).im, angle.sin(), epsilon = 1e-12);
}

#[test]
fn test_qubit_proj0_f64() {
    let p = Qubit.proj0::<f64>();
    assert_abs_diff_eq!(p.get(&[0, 0]), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(p.get(&[1, 1]), 0.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_proj1_f64() {
    let p = Qubit.proj1::<f64>();
    assert_abs_diff_eq!(p.get(&[0, 0]), 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(p.get(&[1, 1]), 1.0, epsilon = 1e-15);
}

#[test]
fn test_qubit_x_squared_is_identity() {
    let backend = arnet_native::NativeBackend::new();
    let x = Qubit.x::<f64>();
    let x2 = arnet_linalg::contract(&backend, &x, &x, "ij,jk->ik").unwrap();
    let id = Qubit.id::<f64>();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(x2.get(&[i, j]), id.get(&[i, j]), epsilon = 1e-12);
        }
    }
}

#[test]
fn test_qubit_hadamard_squared_is_identity() {
    let backend = arnet_native::NativeBackend::new();
    let h = Qubit.h::<f64>();
    let h2 = arnet_linalg::contract(&backend, &h, &h, "ij,jk->ik").unwrap();
    let id = Qubit.id::<f64>();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(h2.get(&[i, j]), id.get(&[i, j]), epsilon = 1e-12);
        }
    }
}

#[test]
fn test_qubit_proj_completeness() {
    // proj0 + proj1 = identity
    let p0 = Qubit.proj0::<f64>();
    let p1 = Qubit.proj1::<f64>();
    let id = Qubit.id::<f64>();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(
                p0.get(&[i, j]) + p1.get(&[i, j]),
                id.get(&[i, j]),
                epsilon = 1e-15
            );
        }
    }
}
