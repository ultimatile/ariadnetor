//! Linear solve implementations via faer LU with partial pivoting for all supported scalar types

use arnet_core::backend::{BackendError, SolveDescriptor};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{
    factor::{PartialPivLuParams, lu_in_place, lu_in_place_scratch},
    solve::{solve_in_place, solve_in_place_scratch},
};
use faer::{MatRef, Spec};
use num_complex::Complex;

use crate::to_faer_par;

/// Linear solve AX = B for f64 via faer partial-pivoting LU
pub(crate) fn solve_f64(desc: SolveDescriptor<'_, f64>) -> Result<(), BackendError> {
    let SolveDescriptor {
        n,
        nrhs,
        a,
        b,
        x,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<PartialPivLuParams, f64> = Default::default();

    let mut a_owned = MatRef::from_column_major_slice(a, n, n).to_owned();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let req = lu_in_place_scratch::<usize, f64>(n, n, par, params)
        .or(solve_in_place_scratch::<usize, f64>(n, nrhs, par));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    let (_info, perm_ref) = lu_in_place(
        a_owned.as_mut(),
        &mut perm,
        &mut perm_inv,
        par,
        stack,
        params,
    );

    // Copy B into X buffer (solve_in_place writes the solution in place)
    let mut x_mat = MatRef::from_column_major_slice(b, n, nrhs).to_owned();
    // lu_in_place leaves L (strict-lower, unit-diagonal) and U (upper) in the
    // same matrix; solve_in_place reads each triangle separately.
    solve_in_place(
        a_owned.as_ref(),
        a_owned.as_ref(),
        perm_ref,
        x_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for f32 via faer partial-pivoting LU
pub(crate) fn solve_f32(desc: SolveDescriptor<'_, f32>) -> Result<(), BackendError> {
    let SolveDescriptor {
        n,
        nrhs,
        a,
        b,
        x,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<PartialPivLuParams, f32> = Default::default();

    let mut a_owned = MatRef::from_column_major_slice(a, n, n).to_owned();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let req = lu_in_place_scratch::<usize, f32>(n, n, par, params)
        .or(solve_in_place_scratch::<usize, f32>(n, nrhs, par));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    let (_info, perm_ref) = lu_in_place(
        a_owned.as_mut(),
        &mut perm,
        &mut perm_inv,
        par,
        stack,
        params,
    );

    let mut x_mat = MatRef::from_column_major_slice(b, n, nrhs).to_owned();
    // lu_in_place leaves L (strict-lower, unit-diagonal) and U (upper) in the
    // same matrix; solve_in_place reads each triangle separately.
    solve_in_place(
        a_owned.as_ref(),
        a_owned.as_ref(),
        perm_ref,
        x_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for Complex<f64> via faer partial-pivoting LU
pub(crate) fn solve_c64(desc: SolveDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let SolveDescriptor {
        n,
        nrhs,
        a,
        b,
        x,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<PartialPivLuParams, Complex<f64>> = Default::default();

    let mut a_owned = MatRef::from_column_major_slice(a, n, n).to_owned();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let req = lu_in_place_scratch::<usize, Complex<f64>>(n, n, par, params)
        .or(solve_in_place_scratch::<usize, Complex<f64>>(n, nrhs, par));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    let (_info, perm_ref) = lu_in_place(
        a_owned.as_mut(),
        &mut perm,
        &mut perm_inv,
        par,
        stack,
        params,
    );

    let mut x_mat = MatRef::from_column_major_slice(b, n, nrhs).to_owned();
    // lu_in_place leaves L (strict-lower, unit-diagonal) and U (upper) in the
    // same matrix; solve_in_place reads each triangle separately.
    solve_in_place(
        a_owned.as_ref(),
        a_owned.as_ref(),
        perm_ref,
        x_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for Complex<f32> via faer partial-pivoting LU
pub(crate) fn solve_c32(desc: SolveDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let SolveDescriptor {
        n,
        nrhs,
        a,
        b,
        x,
        order: _,
        policy,
    } = desc;
    let par = to_faer_par(policy);
    let params: Spec<PartialPivLuParams, Complex<f32>> = Default::default();

    let mut a_owned = MatRef::from_column_major_slice(a, n, n).to_owned();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let req = lu_in_place_scratch::<usize, Complex<f32>>(n, n, par, params)
        .or(solve_in_place_scratch::<usize, Complex<f32>>(n, nrhs, par));
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);

    let (_info, perm_ref) = lu_in_place(
        a_owned.as_mut(),
        &mut perm,
        &mut perm_inv,
        par,
        stack,
        params,
    );

    let mut x_mat = MatRef::from_column_major_slice(b, n, nrhs).to_owned();
    // lu_in_place leaves L (strict-lower, unit-diagonal) and U (upper) in the
    // same matrix; solve_in_place reads each triangle separately.
    solve_in_place(
        a_owned.as_ref(),
        a_owned.as_ref(),
        perm_ref,
        x_mat.as_mut(),
        par,
        stack,
    );

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}
