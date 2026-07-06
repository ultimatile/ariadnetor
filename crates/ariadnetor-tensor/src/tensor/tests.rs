use std::fmt;
use std::ops::Mul;

use ariadnetor_core::Scalar;

use super::*;
// `Direction` / `U1Sector` are not surfaced by `use super::*` (the module's
// `use crate::{...}` block omits both); they are imported here once for the
// BlockSparse tests' direction assertions and sector literals below.
use crate::test_fixtures::square_legs;
use crate::{Direction, U1Sector};

/// Build a square U(1) BlockSparse tensor with an `Out` row leg and an `In`
/// column leg sharing one sector list. The square `Out`/`In` special case of
/// [`crate::test_fixtures::square_legs`], wrapped for the `BlockSparseTensor`
/// `zeros` surface.
fn u1_square_tensor<T>(
    sectors: Vec<(U1Sector, usize)>,
    flux: U1Sector,
) -> BlockSparseTensor<T, U1Sector>
where
    T: Clone + Zero,
{
    BlockSparseTensor::<T, U1Sector>::zeros(square_legs(sectors), flux)
}

/// Same square `Out`/`In` legs as [`u1_square_tensor`], but populated through
/// `from_block_fn` so tests can tag each stored block.
fn u1_square_tensor_from_block_fn<T, F>(
    sectors: Vec<(U1Sector, usize)>,
    flux: U1Sector,
    f: F,
) -> BlockSparseTensor<T, U1Sector>
where
    T: Clone + Zero,
    F: FnMut(&BlockCoord, &[usize]) -> Vec<T>,
{
    BlockSparseTensor::<T, U1Sector>::from_block_fn(square_legs(sectors), flux, f)
}

fn assert_tensor_mutation<S>(zero: S, val: S, fill_val: S, scale_factor: S)
where
    S: Scalar + PartialEq + fmt::Debug + Mul<S, Output = S>,
{
    let mut t = DenseTensor::<S>::zeros(vec![2, 3]);

    // set / get round-trip
    t.set([1, 2], val);
    assert_eq!(t.get([1, 2]), val);
    assert_eq!(t.get([0, 0]), zero);

    // fill overwrites all elements
    t.fill(fill_val);
    assert_eq!(t.get([0, 0]), fill_val);
    assert_eq!(t.get([1, 2]), fill_val);

    // data_slice_mut provides mutable access
    t.data_slice_mut()[0] = val;
    assert_eq!(t.get([0, 0]), val);

    // scale multiplies all elements
    t.fill(val);
    t.scale(scale_factor);
    assert_eq!(t.get([0, 0]), val * scale_factor);
}

#[test]
fn test_tensor_mutation() {
    assert_tensor_mutation(0.0f64, 42.0, 2.72, 3.0);
    assert_tensor_mutation(0.0f32, 42.0, 2.72, 3.0);
}

#[test]
fn scaled_out_of_place_preserves_original() {
    let mut a = DenseTensor::<f64>::zeros(vec![2, 2]);
    a.fill(3.0);
    let b = a.scaled(2.0);
    // a unchanged
    assert_eq!(a.get([0, 0]), 3.0);
    // b scaled
    assert_eq!(b.get([0, 0]), 6.0);
    assert_eq!(b.get([1, 1]), 6.0);
    assert_eq!(b.shape(), a.shape());
}

#[test]
fn norm_matches_frobenius_definition() {
    let mut t = DenseTensor::<f64>::zeros(vec![2, 2]);
    t.set([0, 0], 3.0);
    t.set([1, 1], 4.0);
    // sqrt(9 + 16) = 5
    let n = t.norm();
    assert!((n - 5.0).abs() < 1e-12, "expected 5.0, got {n}");
}

#[test]
fn norm_stays_finite_on_extreme_f32_magnitudes() {
    // Regression for https://github.com/ultimatile/ariadnetor/issues/189:
    // the public `.norm()` must reach the overflow-safe kernel. The naive
    // `sqrt(Σ |x|²)` returned inf here (1e20² overflows f32); exact-value
    // accuracy is covered by the kernel's own unit tests, so this only
    // confirms the delegation stays on the finite path.
    let mut t = DenseTensor::<f32>::zeros(vec![2]);
    t.set([0], 1e20);
    t.set([1], 2e20);
    let n = t.norm();
    assert!(n.is_finite(), "expected finite norm, got {n}");
    // Coarse sanity that the result is the real norm (~2.2e20), not a
    // degenerate value — the kernel unit test pins the exact magnitude.
    assert!(n > 1e20, "expected ~2.2e20, got {n}");
}

#[test]
fn normalize_in_place_returns_original_norm_and_unitizes() {
    let mut t = DenseTensor::<f64>::zeros(vec![2]);
    t.set([0], 3.0);
    t.set([1], 4.0);
    let n = t.normalize();
    assert!((n - 5.0).abs() < 1e-12, "returned norm {n}, expected 5");
    // post-normalize Frobenius norm is 1
    assert!((t.norm() - 1.0).abs() < 1e-12);
    // elements scaled by 1/5
    assert!((t.get([0]) - 0.6).abs() < 1e-12);
    assert!((t.get([1]) - 0.8).abs() < 1e-12);
}

#[test]
fn normalized_out_of_place_keeps_original_intact() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set([0], 3.0);
    a.set([1], 4.0);
    let (b, n) = a.normalized();
    assert!((n - 5.0).abs() < 1e-12);
    // original elements preserved
    assert_eq!(a.get([0]), 3.0);
    assert_eq!(a.get([1]), 4.0);
    // normalized copy has unit norm
    assert!((b.norm() - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Cannot normalize zero tensor")]
fn normalize_panics_on_zero_tensor() {
    let mut t = DenseTensor::<f64>::zeros(vec![3, 3]);
    t.normalize();
}

// A subnormal-magnitude element has a nonzero norm too small to reciprocate
// (`1 / norm` overflows to `+inf`); `DenseTensor::normalize` divides per
// element, so it must yield a finite unit-norm tensor, not `inf`.
#[test]
fn normalize_f32_subnormal_stays_finite() {
    let subnormal = f32::from_bits(1); // smallest positive subnormal (~1.4e-45)
    let mut t = DenseTensor::<f32>::zeros(vec![1]);
    t.set([0], subnormal);
    let n = t.normalize();
    assert_eq!(
        n, subnormal,
        "returned norm should be the pre-normalization value"
    );
    assert!(
        t.get([0]).is_finite(),
        "expected finite element, got {}",
        t.get([0])
    );
    assert_eq!(t.get([0]), 1.0f32);
    assert!((t.norm() - 1.0).abs() < 1e-6);
}

#[test]
fn linear_combine_sums_with_coefs() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set([0], 1.0);
    a.set([1], 2.0);
    let mut b = DenseTensor::<f64>::zeros(vec![2]);
    b.set([0], 10.0);
    b.set([1], 20.0);
    let r = crate::linear_combine(&[&a, &b], &[3.0, 4.0]).unwrap();
    // 3*1 + 4*10 = 43; 3*2 + 4*20 = 86
    assert_eq!(r.get([0]), 43.0);
    assert_eq!(r.get([1]), 86.0);
    assert_eq!(r.shape(), a.shape());
}

#[test]
fn add_all_sums_with_unit_coefs() {
    let mut a = DenseTensor::<f64>::zeros(vec![2]);
    a.set([0], 1.0);
    a.set([1], 2.0);
    let mut b = DenseTensor::<f64>::zeros(vec![2]);
    b.set([0], 10.0);
    b.set([1], 20.0);
    let r = crate::add_all(&[&a, &b]).unwrap();
    assert_eq!(r.get([0]), 11.0);
    assert_eq!(r.get([1]), 22.0);
}

#[test]
fn linear_combine_rejects_empty_list() {
    let err = crate::linear_combine::<f64>(&[], &[]).unwrap_err();
    assert!(err.to_string().contains("empty"), "got: {err}");
}

#[test]
fn linear_combine_rejects_length_mismatch() {
    let a = DenseTensor::<f64>::zeros(vec![2]);
    let b = DenseTensor::<f64>::zeros(vec![2]);
    let err = crate::linear_combine(&[&a, &b], &[1.0]).unwrap_err();
    assert!(err.to_string().contains("Mismatched lengths"), "got: {err}");
}

#[test]
fn linear_combine_rejects_shape_mismatch() {
    let a = DenseTensor::<f64>::zeros(vec![2]);
    let b = DenseTensor::<f64>::zeros(vec![3]);
    let err = crate::linear_combine(&[&a, &b], &[1.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("same shape"), "got: {err}");
}

#[test]
fn block_sparse_tensor_alias_resolves_and_basics_work() {
    let t = u1_square_tensor::<f64>(vec![(U1Sector(0), 2), (U1Sector(1), 3)], U1Sector(0));

    assert_eq!(t.shape(), &[5, 5]);
    assert_eq!(t.rank(), 2);
}

#[test]
fn dense_tensor_zeros_pins_order_to_host_preferred() {
    use ariadnetor_core::backend::ComputeBackend;
    use ariadnetor_native::NativeBackend;

    let expected_order = NativeBackend::shared().preferred_order();
    let t = DenseTensor::<f64>::zeros(vec![2, 3]);

    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.data().layout().order(), expected_order);
    assert!(t.data_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn dense_tensor_random_fills_in_flat_draw_order() {
    use rand::{RngExt, SeedableRng};

    // Pin the host `random` constructor against a direct sequential draw
    // from the same seed: it must draw once per element and store in flat
    // draw order. A construction path that reorders or skips a draw
    // diverges from `expected`; comparing two same-seed constructions
    // would not — both shift identically and mask the regression.
    let seed = 0xA11A;
    let mut direct = rand::rngs::StdRng::seed_from_u64(seed);
    let expected: Vec<f64> = (0..6).map(|_| direct.random()).collect();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let t = DenseTensor::<f64>::random(vec![2, 3], &mut rng);

    assert_eq!(t.data_slice(), expected.as_slice());
}

#[test]
fn block_sparse_tensor_zeros_pins_order_to_host_preferred() {
    use ariadnetor_core::backend::ComputeBackend;
    use ariadnetor_native::NativeBackend;

    let expected_order = NativeBackend::shared().preferred_order();
    let t = u1_square_tensor::<f64>(vec![(U1Sector(0), 2), (U1Sector(1), 1)], U1Sector(0));

    assert_eq!(t.rank(), 2);
    assert_eq!(t.data().layout().order(), expected_order);
}

#[test]
fn dense_tensor_conj_real_path_is_identity() {
    let mut t = DenseTensor::<f64>::zeros(vec![3, 3]);
    t.fill(2.5);
    let c = t.conj();

    assert_eq!(c.shape(), t.shape());
    assert_eq!(c.data_slice(), t.data_slice());
}

#[test]
fn block_sparse_tensor_dagger_is_involutive() {
    let t = u1_square_tensor::<f64>(vec![(U1Sector(0), 2)], U1Sector(0));

    let t_dd = t.dagger().dagger();

    assert_eq!(t_dd.shape(), t.shape());
    assert_eq!(t_dd.flux(), t.flux());
    for (a, b) in t.indices().iter().zip(t_dd.indices().iter()) {
        assert_eq!(a.direction(), b.direction());
    }
}

#[test]
fn block_sparse_tensor_dagger_conjugates_complex() {
    use num_complex::Complex;

    let mut t = u1_square_tensor::<Complex<f64>>(vec![(U1Sector(0), 2)], U1Sector(0));
    {
        let block = t
            .block_data_mut(&BlockCoord(vec![0, 0]))
            .expect("flux-allowed block");
        block[0] = Complex::new(1.0, 2.0);
        block[1] = Complex::new(3.0, -4.0);
    }

    let d = t.dagger();

    // Values conjugated.
    let d_block = d
        .block_data(&BlockCoord(vec![0, 0]))
        .expect("block present");
    assert_eq!(d_block[0], Complex::new(1.0, -2.0));
    assert_eq!(d_block[1], Complex::new(3.0, 4.0));

    // Leg directions flipped.
    assert_eq!(d.indices()[0].direction(), Direction::In);
    assert_eq!(d.indices()[1].direction(), Direction::Out);
}

#[test]
fn block_sparse_tensor_conj_keeps_directions_and_flux() {
    use num_complex::Complex;

    let mut t = u1_square_tensor::<Complex<f64>>(vec![(U1Sector(0), 2)], U1Sector(0));
    t.block_data_mut(&BlockCoord(vec![0, 0])).unwrap()[0] = Complex::new(2.0, 5.0);

    let c = t.conj();

    assert_eq!(c.flux(), t.flux());
    assert_eq!(c.indices()[0].direction(), Direction::Out);
    assert_eq!(c.indices()[1].direction(), Direction::In);
    assert_eq!(
        c.block_data(&BlockCoord(vec![0, 0])).unwrap()[0],
        Complex::new(2.0, -5.0)
    );
}

#[test]
fn block_sparse_tensor_scale_multiplies_each_stored_block() {
    // Tag each stored element by its flat position so the per-block
    // multiplication is checkable.
    let mut t = u1_square_tensor_from_block_fn::<f64, _>(
        vec![(U1Sector(0), 2), (U1Sector(1), 2)],
        U1Sector(0),
        |_coord, block_shape| {
            let len: usize = block_shape.iter().product();
            (0..len).map(|i| 1.0 + i as f64).collect()
        },
    );

    // Snapshot the stored blocks before scaling.
    let before00: Vec<f64> = t.block_data(&BlockCoord(vec![0, 0])).unwrap().to_vec();
    let before11: Vec<f64> = t.block_data(&BlockCoord(vec![1, 1])).unwrap().to_vec();

    t.scale(3.0);

    let after00 = t.block_data(&BlockCoord(vec![0, 0])).unwrap();
    let after11 = t.block_data(&BlockCoord(vec![1, 1])).unwrap();
    for (a, b) in after00.iter().zip(before00.iter()) {
        assert_eq!(*a, b * 3.0);
    }
    for (a, b) in after11.iter().zip(before11.iter()) {
        assert_eq!(*a, b * 3.0);
    }
}

#[test]
fn block_sparse_tensor_scaled_preserves_original() {
    let a = u1_square_tensor_from_block_fn::<f64, _>(
        vec![(U1Sector(0), 2)],
        U1Sector(0),
        |_coord, block_shape| {
            let len: usize = block_shape.iter().product();
            vec![2.0; len]
        },
    );

    let b = a.scaled(2.5);

    // Original untouched.
    assert!(
        a.block_data(&BlockCoord(vec![0, 0]))
            .unwrap()
            .iter()
            .all(|&x| x == 2.0)
    );
    // Out-of-place copy scaled, layout preserved.
    assert!(
        b.block_data(&BlockCoord(vec![0, 0]))
            .unwrap()
            .iter()
            .all(|&x| x == 5.0)
    );
    assert_eq!(b.shape(), a.shape());
}

#[test]
fn block_sparse_tensor_scale_handles_complex_factors() {
    use num_complex::Complex;

    let mut t = u1_square_tensor::<Complex<f64>>(vec![(U1Sector(0), 2)], U1Sector(0));
    {
        let block = t.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        block[0] = Complex::new(1.0, 2.0);
        block[1] = Complex::new(-3.0, 4.0);
    }

    // Real factor scales both parts.
    let real_scaled = t.scaled(2.0);
    let rs = real_scaled.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(rs[0], Complex::new(2.0, 4.0));
    assert_eq!(rs[1], Complex::new(-6.0, 8.0));

    // Complex factor rotates and scales (in-place).
    t.scale(Complex::new(0.0, 1.0));
    let d = t.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d[0], Complex::new(-2.0, 1.0));
    assert_eq!(d[1], Complex::new(-4.0, -3.0));
}

#[test]
fn test_get_set_accept_any_asref_coords() {
    // get/set take `impl AsRef<[usize]>`: an array literal (no borrow), a
    // borrowed slice, and a `&Vec` must all address the same element.
    let mut t = DenseTensor::<f64>::zeros(vec![2, 3]);

    t.set([1, 2], 7.0); // array literal, no `&`
    let coords = vec![1usize, 2];
    assert_eq!(t.get([1, 2]), 7.0); // array literal
    assert_eq!(t.get(&coords), 7.0); // &Vec (dynamic rank)
    assert_eq!(t.get(&coords[..]), 7.0); // slice

    // A write addressed via a dynamic &Vec is read back via an array literal.
    t.set(&coords, 9.0);
    assert_eq!(t.get([1, 2]), 9.0);
}

#[test]
fn tensor_len_and_is_empty() {
    // Exercises `Tensor::len` / `is_empty` directly. Existing tests touch
    // `DenseTensorData::len` (a distinct method), leaving the logical-shape
    // wrappers uncovered.
    let t = DenseTensor::<f64>::zeros(vec![2, 3]);
    assert_eq!(t.len(), 6); // product of the shape
    assert!(!t.is_empty());

    // A zero-sized dimension makes the shape product zero.
    let empty = DenseTensor::<f64>::zeros(vec![0]);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}
