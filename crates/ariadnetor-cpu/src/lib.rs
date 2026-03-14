//! CPU compute backend for Ariadnetor
//!
//! Provides [`CpuBackend`] implementing `ComputeBackend` via:
//! - **GEMM**: faer (f64, f32)
//! - **Transpose**: HPTT when available (f64, f32, Complex), naive fallback

mod transpose;

use arnet_core::backend::{BackendError, ComputeBackend, DeviceType, GemmDescriptor, SvdDescriptor, TransposeDescriptor};
use arnet_core::scalar::Scalar;

/// CPU backend using faer for GEMM and HPTT for transpose.
///
/// This is the sole owner of faer and hptt-rs dependencies in the workspace.
/// Other crates access these capabilities through the `ComputeBackend` trait.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    /// GEMM: C = alpha * A * B + beta * C
    ///
    /// Dispatches to faer for f64/f32. Complex types are not yet supported.
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId. Reinterpret generic fields
            // to concrete f64 via pointer casts; layout is identical.
            let desc_f64 = unsafe { reinterpret_gemm_desc::<T, f64>(desc) };
            gemm_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_gemm_desc::<T, f32>(desc) };
            gemm_f32(desc_f32)
        } else {
            Err(BackendError::NotSupported(
                "GEMM is only supported for f64 and f32; Complex GEMM not yet implemented".into(),
            ))
        }
    }

    /// Transpose tensor axes according to permutation.
    ///
    /// Uses HPTT for f64/f32/Complex when the `hptt` feature is enabled,
    /// with a naive fallback for all types.
    fn transpose<T: Scalar>(&self, desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
        transpose::dispatch(desc)
    }

    /// Thin SVD via faer: A = U * diag(S) * Vt
    ///
    /// Dispatches to faer for f64/f32. Complex types are not yet supported.
    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_svd_desc::<T, f64>(desc) };
            svd_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_svd_desc::<T, f32>(desc) };
            svd_f32(desc_f32)
        } else {
            Err(BackendError::NotSupported(
                "SVD is only supported for f64 and f32; Complex SVD not yet implemented".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Generic → concrete type reinterpretation
// ---------------------------------------------------------------------------

/// Reinterpret `GemmDescriptor<T>` as `GemmDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_gemm_desc<'a, T, U>(
    desc: GemmDescriptor<'a, T>,
) -> GemmDescriptor<'a, U> {
    let GemmDescriptor { m, n, k, alpha, a, b, beta, c, trans_a, trans_b } = desc;
    unsafe {
        GemmDescriptor {
            m, n, k,
            alpha: std::ptr::read(&alpha as *const T as *const U),
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            b: std::slice::from_raw_parts(b.as_ptr() as *const U, b.len()),
            beta: std::ptr::read(&beta as *const T as *const U),
            c: std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut U, c.len()),
            trans_a, trans_b,
        }
    }
}

/// Reinterpret `SvdDescriptor<T>` as `SvdDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Real` and `U::Real` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_svd_desc<'a, T: Scalar, U: Scalar>(
    desc: SvdDescriptor<'a, T>,
) -> SvdDescriptor<'a, U> {
    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    unsafe {
        SvdDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            u: std::slice::from_raw_parts_mut(u.as_mut_ptr() as *mut U, u.len()),
            s: std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U::Real, s.len()),
            vt: std::slice::from_raw_parts_mut(vt.as_mut_ptr() as *mut U, vt.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM implementations (faer)
// ---------------------------------------------------------------------------

/// GEMM for f64 via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_f64(desc: GemmDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

    // Construct faer MatRef views from row-major flat slices.
    // faer's from_row_major_slice expects (data, nrows, ncols).
    // For transposed operands, swap dimensions.
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

    // C = alpha * product + beta * C
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = alpha * product[(i, j)] + beta * c[idx];
        }
    }

    Ok(())
}

/// GEMM for f32 via faer: C = alpha * op(A) * op(B) + beta * C
fn gemm_f32(desc: GemmDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let GemmDescriptor {
        m, n, k, alpha, a, b, beta, c, trans_a, trans_b,
    } = desc;

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

    Ok(())
}

// ---------------------------------------------------------------------------
// SVD implementations (faer)
// ---------------------------------------------------------------------------

/// Thin SVD for f64 via faer: A = U * diag(S) * Vt
fn svd_f64(desc: SvdDescriptor<'_, f64>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    // U (m×k, row-major)
    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    // Singular values (k)
    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    // Vt = V^T (k×n, row-major)
    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)];
        }
    }

    Ok(())
}

/// Thin SVD for f32 via faer: A = U * diag(S) * Vt
fn svd_f32(desc: SvdDescriptor<'_, f32>) -> Result<(), BackendError> {
    use faer::MatRef;

    let SvdDescriptor { m, n, a, u, s, vt } = desc;
    let k = m.min(n);

    let mat = MatRef::from_row_major_slice(a, m, n).to_owned();
    let thin = mat.thin_svd().map_err(|e| {
        BackendError::ExecutionFailed(format!("faer thin_svd failed: {e:?}"))
    })?;

    let u_mat = thin.U();
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_mat[(i, j)];
        }
    }

    let s_col = thin.S().column_vector();
    for i in 0..k {
        s[i] = s_col[i];
    }

    let v_mat = thin.V();
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arnet_core::backend::ComputeBackend;

    #[test]
    fn test_backend_metadata() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.device_type(), DeviceType::Cpu);
        assert!(backend.is_available());
    }

    // --- GEMM tests ---

    #[test]
    fn test_gemm_f64_identity() {
        let backend = CpuBackend::new();

        // A = [[1, 0], [0, 1]] (2x2 identity)
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_gemm_f64_basic() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_f64_alpha_beta() {
        let backend = CpuBackend::new();

        // C = 2.0 * A * B + 3.0 * C_init
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [1.0f64; 4]; // C_init = all ones

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 2.0, a: &a, b: &b,
            beta: 3.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        // C = 2 * [19, 22, 43, 50] + 3 * [1, 1, 1, 1] = [41, 47, 89, 103]
        assert_eq!(c, [41.0, 47.0, 89.0, 103.0]);
    }

    #[test]
    fn test_gemm_f64_rectangular() {
        let backend = CpuBackend::new();

        // A (2x3) * B (3x2) = C (2x2)
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f64; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 3,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64, 139, 154]
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        let desc = GemmDescriptor {
            m: 2, n: 2, k: 2,
            alpha: 1.0, a: &a, b: &b,
            beta: 0.0, c: &mut c,
            trans_a: false, trans_b: false,
        };
        backend.gemm(desc).unwrap();
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    // --- Transpose tests ---

    #[test]
    fn test_transpose_f64_2d() {
        let backend = CpuBackend::new();

        // 2x3 matrix → 3x2
        let input = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0f64; 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        // [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]]
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_f64_3d() {
        let backend = CpuBackend::new();

        // Shape [2,3,4], perm [1,0,2] → shape [3,2,4]
        let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let mut output = vec![0.0f64; 24];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3, 4],
            perm: &[1, 0, 2],
        };
        backend.transpose(desc).unwrap();

        // Verify a few elements: input[i][j][k] should equal output[j][i][k]
        // input[0][1][2] = 0*12 + 1*4 + 2 = 6 → output[1][0][2] = 1*8 + 0*4 + 2 = 10
        assert_eq!(output[10], 6.0);
        // input[1][0][3] = 1*12 + 0*4 + 3 = 15 → output[0][1][3] = 0*8 + 1*4 + 3 = 7
        assert_eq!(output[7], 15.0);
    }

    #[test]
    fn test_transpose_f32_2d() {
        let backend = CpuBackend::new();

        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0f32; 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_complex_f64_2d() {
        use num_complex::Complex;

        let backend = CpuBackend::new();

        let input = [
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0), Complex::new(9.0, 10.0), Complex::new(11.0, 12.0),
        ];
        let mut output = [Complex::new(0.0, 0.0); 6];

        let desc = TransposeDescriptor {
            input: &input,
            output: &mut output,
            shape: &[2, 3],
            perm: &[1, 0],
        };
        backend.transpose(desc).unwrap();
        assert_eq!(output[0], Complex::new(1.0, 2.0));
        assert_eq!(output[1], Complex::new(7.0, 8.0));
        assert_eq!(output[2], Complex::new(3.0, 4.0));
        assert_eq!(output[3], Complex::new(9.0, 10.0));
    }

    // --- SVD tests ---

    #[test]
    fn test_svd_f64_square() {
        let backend = CpuBackend::new();

        // A = [[1, 2], [3, 4]] (2x2)
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let mut u = [0.0f64; 4]; // 2x2
        let mut s = [0.0f64; 2]; // 2
        let mut vt = [0.0f64; 4]; // 2x2

        let desc = SvdDescriptor {
            m: 2, n: 2, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        // Singular values should be positive and in descending order
        assert!(s[0] > s[1]);
        assert!(s[1] >= 0.0);

        // Reconstruct: A ≈ U * diag(S) * Vt
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!((val - a[i * 2 + j]).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * 2 + j]);
            }
        }
    }

    #[test]
    fn test_svd_f64_rectangular() {
        let backend = CpuBackend::new();

        // A = [[1, 2, 3], [4, 5, 6]] (2x3), k = min(2,3) = 2
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n, k) = (2, 3, 2);
        let mut u = vec![0.0f64; m * k];
        let mut s = vec![0.0f64; k];
        let mut vt = vec![0.0f64; k * n];

        let desc = SvdDescriptor {
            m, n, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        // Reconstruct A ≈ U * diag(S) * Vt
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += u[i * k + l] * s[l] * vt[l * n + j];
                }
                assert!((val - a[i * n + j]).abs() < 1e-10,
                    "Reconstruction mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_svd_f32_basic() {
        let backend = CpuBackend::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut u = [0.0f32; 4];
        let mut s = [0.0f32; 2];
        let mut vt = [0.0f32; 4];

        let desc = SvdDescriptor {
            m: 2, n: 2, a: &a,
            u: &mut u, s: &mut s, vt: &mut vt,
        };
        backend.svd(desc).unwrap();

        assert!(s[0] > s[1]);

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0f32;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!((val - a[i * 2 + j]).abs() < 1e-4,
                    "Reconstruction mismatch at ({i},{j}): {val} vs {}", a[i * 2 + j]);
            }
        }
    }
}
