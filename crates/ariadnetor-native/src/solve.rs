//! Linear solve implementations via faer LU with partial pivoting for all supported scalar types

use arnet_core::backend::{BackendError, SolveDescriptor};
use faer::MatRef;
use faer::prelude::Solve;
use num_complex::Complex;

/// Linear solve AX = B for f64 via faer partial-pivoting LU
pub(crate) fn solve_f64(desc: SolveDescriptor<'_, f64>) -> Result<(), BackendError> {
    let SolveDescriptor { n, nrhs, a, b, x } = desc;

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let lu = a_mat.partial_piv_lu();
    let b_mat = MatRef::from_column_major_slice(b, n, nrhs);
    let x_mat = lu.solve(&b_mat);

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for f32 via faer partial-pivoting LU
pub(crate) fn solve_f32(desc: SolveDescriptor<'_, f32>) -> Result<(), BackendError> {
    let SolveDescriptor { n, nrhs, a, b, x } = desc;

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let lu = a_mat.partial_piv_lu();
    let b_mat = MatRef::from_column_major_slice(b, n, nrhs);
    let x_mat = lu.solve(&b_mat);

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for Complex<f64> via faer partial-pivoting LU
pub(crate) fn solve_c64(desc: SolveDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let SolveDescriptor { n, nrhs, a, b, x } = desc;

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let lu = a_mat.partial_piv_lu();
    let b_mat = MatRef::from_column_major_slice(b, n, nrhs);
    let x_mat = lu.solve(&b_mat);

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}

/// Linear solve AX = B for Complex<f32> via faer partial-pivoting LU
pub(crate) fn solve_c32(desc: SolveDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let SolveDescriptor { n, nrhs, a, b, x } = desc;

    let a_mat = MatRef::from_column_major_slice(a, n, n);
    let lu = a_mat.partial_piv_lu();
    let b_mat = MatRef::from_column_major_slice(b, n, nrhs);
    let x_mat = lu.solve(&b_mat);

    for i in 0..n {
        for j in 0..nrhs {
            x[j * n + i] = x_mat[(i, j)];
        }
    }

    Ok(())
}
