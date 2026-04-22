//! GEMM implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, GemmDescriptor, MemoryOrder};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef};
use num_complex::Complex;
use num_traits::{One, Zero};

use crate::to_faer_par;

/// GEMM for f64 via faer: C = alpha * op(A) * op(B) + beta * C
pub(crate) fn gemm_f64(desc: GemmDescriptor<'_, f64>) -> Result<(), BackendError> {
    let GemmDescriptor {
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
        trans_a,
        trans_b,
        order,
        policy,
    } = desc;
    let par = to_faer_par(policy);

    // Pre-scale C if beta ∉ {0, 1}; Accum::Replace handles beta == 0 (no read of C),
    // Accum::Add handles beta == 1 and the post-scaled case.
    let accum = if beta.is_zero() {
        Accum::Replace
    } else {
        if !beta.is_one() {
            for elem in c.iter_mut() {
                *elem *= beta;
            }
        }
        Accum::Add
    };

    match order {
        MemoryOrder::RowMajor => {
            let lhs = if trans_a {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
        MemoryOrder::ColumnMajor => {
            let lhs = if trans_a {
                MatRef::from_column_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_column_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_column_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_column_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_column_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
    }

    Ok(())
}

/// GEMM for f32 via faer: C = alpha * op(A) * op(B) + beta * C
pub(crate) fn gemm_f32(desc: GemmDescriptor<'_, f32>) -> Result<(), BackendError> {
    let GemmDescriptor {
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
        trans_a,
        trans_b,
        order,
        policy,
    } = desc;
    let par = to_faer_par(policy);

    let accum = if beta.is_zero() {
        Accum::Replace
    } else {
        if !beta.is_one() {
            for elem in c.iter_mut() {
                *elem *= beta;
            }
        }
        Accum::Add
    };

    match order {
        MemoryOrder::RowMajor => {
            let lhs = if trans_a {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
        MemoryOrder::ColumnMajor => {
            let lhs = if trans_a {
                MatRef::from_column_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_column_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_column_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_column_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_column_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
    }

    Ok(())
}

/// GEMM for Complex<f64> via faer: C = alpha * op(A) * op(B) + beta * C
pub(crate) fn gemm_c64(desc: GemmDescriptor<'_, Complex<f64>>) -> Result<(), BackendError> {
    let GemmDescriptor {
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
        trans_a,
        trans_b,
        order,
        policy,
    } = desc;
    let par = to_faer_par(policy);

    let accum = if beta.is_zero() {
        Accum::Replace
    } else {
        if !beta.is_one() {
            for elem in c.iter_mut() {
                *elem *= beta;
            }
        }
        Accum::Add
    };

    match order {
        MemoryOrder::RowMajor => {
            let lhs = if trans_a {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
        MemoryOrder::ColumnMajor => {
            let lhs = if trans_a {
                MatRef::from_column_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_column_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_column_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_column_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_column_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
    }

    Ok(())
}

/// GEMM for Complex<f32> via faer: C = alpha * op(A) * op(B) + beta * C
pub(crate) fn gemm_c32(desc: GemmDescriptor<'_, Complex<f32>>) -> Result<(), BackendError> {
    let GemmDescriptor {
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
        trans_a,
        trans_b,
        order,
        policy,
    } = desc;
    let par = to_faer_par(policy);

    let accum = if beta.is_zero() {
        Accum::Replace
    } else {
        if !beta.is_one() {
            for elem in c.iter_mut() {
                *elem *= beta;
            }
        }
        Accum::Add
    };

    match order {
        MemoryOrder::RowMajor => {
            let lhs = if trans_a {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
        MemoryOrder::ColumnMajor => {
            let lhs = if trans_a {
                MatRef::from_column_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_column_major_slice(a, m, k)
            };
            let rhs = if trans_b {
                MatRef::from_column_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_column_major_slice(b, k, n)
            };
            let c_mat = MatMut::from_column_major_slice_mut(c, m, n);
            matmul(c_mat, accum, lhs, rhs, alpha, par);
        }
    }

    Ok(())
}
