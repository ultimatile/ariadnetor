//! End-to-end tests pinning the `lhs_trans_flag` / `rhs_trans_flag` /
//! `lhs_needs_physical_t` / `rhs_needs_physical_t` decision logic in
//! `contract_to_tensor`.
//!
//! Two axis configurations per side cover all observable mutations on
//! lines 303 / 304 / 313 / 314:
//!
//! * Config A — non-prefix contracted axes (`axes=[1,0]` for lhs,
//!   `axes=[2,1]` for rhs). Forces the physical-transpose path
//!   (`trans_flag=false`, `needs_physical=true`). Catches `&&`→`||`
//!   on the whole `lhs_trans_flag` / `rhs_trans_flag` expression and
//!   `delete !` on the `!*_trans_flag` operand of `*_needs_physical_t`.
//!
//! * Config B — natural prefix axes (`axes=[0,1]` for lhs,
//!   `axes=[1,2]` for rhs). Forces the GEMM trans-flag path
//!   (`trans_flag=true`, `needs_physical=false`). Catches `delete !`
//!   on the `!*_is_id` operand and `delete !` on `!*_trans_flag`
//!   (the latter would add a redundant physical permutation under a
//!   trans-flag GEMM, producing a wrong result).
//!
//! Note: under the dispatcher invariant `free_*` is auto-sorted ascending
//! (line 79-80 of `block_sparse_contract.rs`), so
//! `is_ascending_prefix(axes)` and `is_ascending_suffix(free, rank)` always
//! co-vary. The `&&` between those two predicate calls (column 53 on
//! lines 303 and 313) is therefore equivalent to `||` under the invariant
//! and cannot be killed end-to-end without a non-conforming caller.

use super::*;

fn lhs_rank3_data() -> Vec<f64> {
    // Conceptual RM-flat values for shape [2, 2, 2]: data[i,j,k] = 1.0 + i*4 + j*2 + k.
    (1..=8).map(|v| v as f64).collect()
}

fn rhs_rank3_data() -> Vec<f64> {
    // Conceptual RM-flat values for shape [2, 2, 2]: data[i,j,k] = 9.0 + i*4 + j*2 + k.
    (9..=16).map(|v| v as f64).collect()
}

/// Read element `data[i, j, k]` from a conceptual RM-flat slice for shape `[2, 2, 2]`.
fn at3(data: &[f64], i: usize, j: usize, k: usize) -> f64 {
    data[i * 4 + j * 2 + k]
}

/// Build a rank-3 BlockSparse over a single U(1) charge-0 sector with all
/// dims equal to 2 and the given conceptual RM-flat block data.
fn rank3_single_sector(directions: [Direction; 3], rm_data: &[f64]) -> BlockSparse<f64, U1Sector> {
    let indices: Vec<QNIndex<U1Sector>> = directions
        .iter()
        .map(|&d| QNIndex::new(vec![(U1Sector(0), 2)], d))
        .collect();
    let mut t = BlockSparse::<f64, U1Sector>::zeros(indices, U1Sector(0));
    t.block_data_mut(&BlockCoord(vec![0, 0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(rm_data, &[2, 2, 2]));
    t
}

#[test]
fn lhs_non_prefix_axes_pin_physical_transpose_path() {
    // Config A (LHS): axes_lhs=[1, 0] (non-prefix), free_lhs=[2] (auto).
    // Pairing: lhs axis 1 ↔ rhs axis 0; lhs axis 0 ↔ rhs axis 1.
    // RHS uses neutral natural axes (axes_rhs=[0, 1] with free_rhs=[2])
    // so any rhs-side mutant doesn't pollute the lhs decision under test.
    let lhs = rank3_single_sector(
        [Direction::Out, Direction::In, Direction::Out],
        &lhs_rank3_data(),
    );
    let rhs = rank3_single_sector(
        [Direction::Out, Direction::In, Direction::Out],
        &rhs_rank3_data(),
    );

    let result = contract_block_sparse(&b(), &lhs, &rhs, &[1, 0], &[0, 1]).unwrap();
    let out = match result {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // Output axes: [free_lhs[0]=lhs_axis_2, free_rhs[0]=rhs_axis_2].
    // out[k, m] = sum_{i, j} lhs[i, j, k] * rhs[j, i, m]
    //   (because axes_lhs[0]=1 ↔ axes_rhs[0]=0 means lhs.j == rhs.i,
    //          axes_lhs[1]=0 ↔ axes_rhs[1]=1 means lhs.i == rhs.j).
    let l = lhs_rank3_data();
    let r = rhs_rank3_data();
    let mut expected = vec![0.0; 4];
    for k in 0..2 {
        for m in 0..2 {
            let mut acc = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    acc += at3(&l, i, j, k) * at3(&r, j, i, m);
                }
            }
            expected[k * 2 + m] = acc;
        }
    }
    let expected_in_order = to_order(&expected, &[2, 2]);
    let actual = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (n, (&a, &e)) in actual.iter().zip(expected_in_order.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "lhs_non_prefix mismatch at idx {n}: actual={a}, expected={e}"
        );
    }
}

#[test]
fn lhs_prefix_axes_pin_trans_flag_path() {
    // Config B (LHS): axes_lhs=[0, 1] (prefix), free_lhs=[2] (auto).
    // lhs_is_id check: chain(free=[2], axes=[0,1])=[2,0,1]≠identity → !id=true.
    // prefix=true, suffix=true → trans_flag=true, needs_physical=false.
    // Pairing: lhs axis 0 ↔ rhs axis 0; lhs axis 1 ↔ rhs axis 1.
    let lhs = rank3_single_sector(
        [Direction::In, Direction::In, Direction::Out],
        &lhs_rank3_data(),
    );
    let rhs = rank3_single_sector(
        [Direction::Out, Direction::Out, Direction::Out],
        &rhs_rank3_data(),
    );

    let result = contract_block_sparse(&b(), &lhs, &rhs, &[0, 1], &[0, 1]).unwrap();
    let out = match result {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // out[k, m] = sum_{i, j} lhs[i, j, k] * rhs[i, j, m]
    let l = lhs_rank3_data();
    let r = rhs_rank3_data();
    let mut expected = vec![0.0; 4];
    for k in 0..2 {
        for m in 0..2 {
            let mut acc = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    acc += at3(&l, i, j, k) * at3(&r, i, j, m);
                }
            }
            expected[k * 2 + m] = acc;
        }
    }
    let expected_in_order = to_order(&expected, &[2, 2]);
    let actual = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (n, (&a, &e)) in actual.iter().zip(expected_in_order.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "lhs_prefix mismatch at idx {n}: actual={a}, expected={e}"
        );
    }
}

#[test]
fn rhs_non_prefix_axes_pin_physical_transpose_path() {
    // Config A_rhs: axes_rhs=[2, 1] (non-suffix in chain order),
    // free_rhs=[0]. rhs_is_id chain=[2,1,0]≠identity → !id=true.
    // prefix(free_rhs=[0])=true, suffix(axes_rhs=[2,1], 3) offset=1, (2==1)=false
    //   → rhs_trans_flag = true && true && false = false → physical-transpose path.
    // LHS uses neutral natural axes.
    let lhs = rank3_single_sector(
        [Direction::Out, Direction::Out, Direction::In],
        &lhs_rank3_data(),
    );
    // Pairing: lhs.1=Out ↔ rhs.2 (must be In); lhs.2=In ↔ rhs.1 (must be Out).
    let rhs = rank3_single_sector(
        [Direction::Out, Direction::Out, Direction::In],
        &rhs_rank3_data(),
    );

    let result = contract_block_sparse(&b(), &lhs, &rhs, &[1, 2], &[2, 1]).unwrap();
    let out = match result {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // axes_lhs[0]=1 ↔ axes_rhs[0]=2; axes_lhs[1]=2 ↔ axes_rhs[1]=1.
    // free_lhs=[0], free_rhs=[0]. out[i, m] = sum_{j, k} lhs[i, j, k] * rhs[m, k, j].
    let l = lhs_rank3_data();
    let r = rhs_rank3_data();
    let mut expected = vec![0.0; 4];
    for i in 0..2 {
        for m in 0..2 {
            let mut acc = 0.0;
            for j in 0..2 {
                for k in 0..2 {
                    acc += at3(&l, i, j, k) * at3(&r, m, k, j);
                }
            }
            expected[i * 2 + m] = acc;
        }
    }
    let expected_in_order = to_order(&expected, &[2, 2]);
    let actual = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (n, (&a, &e)) in actual.iter().zip(expected_in_order.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "rhs_non_prefix mismatch at idx {n}: actual={a}, expected={e}"
        );
    }
}

#[test]
fn rhs_suffix_axes_pin_trans_flag_path() {
    // Config B_rhs: axes_rhs=[1, 2] (suffix), free_rhs=[0]. rhs_is_id chain=[1,2,0]
    // ≠identity → !id=true. prefix(free=[0])=true, suffix(axes=[1,2], 3)=true →
    // rhs_trans_flag=true, needs_physical=false. LHS uses neutral natural axes.
    let lhs = rank3_single_sector(
        [Direction::Out, Direction::In, Direction::In],
        &lhs_rank3_data(),
    );
    let rhs = rank3_single_sector(
        [Direction::Out, Direction::Out, Direction::Out],
        &rhs_rank3_data(),
    );

    let result = contract_block_sparse(&b(), &lhs, &rhs, &[1, 2], &[1, 2]).unwrap();
    let out = match result {
        BlockSparseContractResult::Tensor(t) => t,
        _ => panic!("expected tensor"),
    };

    // axes_lhs[0]=1 ↔ axes_rhs[0]=1; axes_lhs[1]=2 ↔ axes_rhs[1]=2.
    // free_lhs=[0], free_rhs=[0]. out[i, m] = sum_{j, k} lhs[i, j, k] * rhs[m, j, k].
    let l = lhs_rank3_data();
    let r = rhs_rank3_data();
    let mut expected = vec![0.0; 4];
    for i in 0..2 {
        for m in 0..2 {
            let mut acc = 0.0;
            for j in 0..2 {
                for k in 0..2 {
                    acc += at3(&l, i, j, k) * at3(&r, m, j, k);
                }
            }
            expected[i * 2 + m] = acc;
        }
    }
    let expected_in_order = to_order(&expected, &[2, 2]);
    let actual = out.block_data(&BlockCoord(vec![0, 0])).unwrap();
    for (n, (&a, &e)) in actual.iter().zip(expected_in_order.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "rhs_suffix mismatch at idx {n}: actual={a}, expected={e}"
        );
    }
}
