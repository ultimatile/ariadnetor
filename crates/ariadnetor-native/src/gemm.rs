//! GEMM implementations via faer for all supported scalar types

use arnet_core::backend::{BackendError, GemmDescriptor, MemoryOrder};
use faer::MatRef;
use num_complex::Complex;

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
    } = desc;

    match order {
        MemoryOrder::RowMajor => {
            let lhs: faer::Mat<f64> = if trans_a {
                let view = MatRef::from_row_major_slice(a, k, m);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<f64> = if trans_b {
                let view = MatRef::from_row_major_slice(b, n, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
        }
        MemoryOrder::ColumnMajor => {
            let lhs: faer::Mat<f64> = if trans_a {
                let view = MatRef::from_column_major_slice(a, m, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<f64> = if trans_b {
                let view = MatRef::from_column_major_slice(b, k, n);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = j * m + i;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
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
    } = desc;

    match order {
        MemoryOrder::RowMajor => {
            let lhs: faer::Mat<f32> = if trans_a {
                let view = MatRef::from_row_major_slice(a, k, m);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<f32> = if trans_b {
                let view = MatRef::from_row_major_slice(b, n, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
        }
        MemoryOrder::ColumnMajor => {
            let lhs: faer::Mat<f32> = if trans_a {
                let view = MatRef::from_column_major_slice(a, m, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<f32> = if trans_b {
                let view = MatRef::from_column_major_slice(b, k, n);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = j * m + i;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
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
    } = desc;

    match order {
        MemoryOrder::RowMajor => {
            let lhs: faer::Mat<Complex<f64>> = if trans_a {
                let view = MatRef::from_row_major_slice(a, k, m);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<Complex<f64>> = if trans_b {
                let view = MatRef::from_row_major_slice(b, n, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
        }
        MemoryOrder::ColumnMajor => {
            let lhs: faer::Mat<Complex<f64>> = if trans_a {
                let view = MatRef::from_column_major_slice(a, m, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<Complex<f64>> = if trans_b {
                let view = MatRef::from_column_major_slice(b, k, n);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = j * m + i;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
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
    } = desc;

    match order {
        MemoryOrder::RowMajor => {
            let lhs: faer::Mat<Complex<f32>> = if trans_a {
                let view = MatRef::from_row_major_slice(a, k, m);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<Complex<f32>> = if trans_b {
                let view = MatRef::from_row_major_slice(b, n, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_row_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
        }
        MemoryOrder::ColumnMajor => {
            let lhs: faer::Mat<Complex<f32>> = if trans_a {
                let view = MatRef::from_column_major_slice(a, m, k);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(a, m, k).to_owned()
            };

            let rhs: faer::Mat<Complex<f32>> = if trans_b {
                let view = MatRef::from_column_major_slice(b, k, n);
                view.transpose().to_owned()
            } else {
                MatRef::from_column_major_slice(b, k, n).to_owned()
            };

            let product = &lhs * &rhs;

            for i in 0..m {
                for j in 0..n {
                    let idx = j * m + i;
                    c[idx] = alpha * product[(i, j)] + beta * c[idx];
                }
            }
        }
    }

    Ok(())
}
