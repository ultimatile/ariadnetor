//! CPU compute backend for Ariadnetor
//!
//! Provides [`CpuBackend`] implementing `ComputeBackend` via:
//! - **GEMM**: faer (f64, f32, Complex<f64>, Complex<f32>)
//! - **SVD/QR/LQ/EIGH**: faer (f64, f32, Complex<f64>, Complex<f32>)
//! - **Transpose**: HPTT when available (f64, f32, Complex), naive fallback

mod eig;
mod eigh;
mod gemm;
mod lq;
mod qr;
mod solve;
mod svd;
mod transpose;

use arnet_core::backend::{BackendError, ComputeBackend, DeviceType, EigDescriptor, EighDescriptor, GemmDescriptor, LqDescriptor, QrDescriptor, SolveDescriptor, SvdDescriptor, TransposeDescriptor};
use arnet_core::scalar::Scalar;
use num_complex::Complex;

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
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn gemm<T: Scalar>(&self, desc: GemmDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId. Reinterpret generic fields
            // to concrete f64 via pointer casts; layout is identical.
            let desc_f64 = unsafe { reinterpret_gemm_desc::<T, f64>(desc) };
            gemm::gemm_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_gemm_desc::<T, f32>(desc) };
            gemm::gemm_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_gemm_desc::<T, Complex<f64>>(desc) };
            gemm::gemm_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_gemm_desc::<T, Complex<f32>>(desc) };
            gemm::gemm_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "GEMM is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
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
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    /// For complex types, Vt stores V^H (conjugate transpose).
    fn svd<T: Scalar>(&self, desc: SvdDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_svd_desc::<T, f64>(desc) };
            svd::svd_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_svd_desc::<T, f32>(desc) };
            svd::svd_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_svd_desc::<T, Complex<f64>>(desc) };
            svd::svd_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_svd_desc::<T, Complex<f32>>(desc) };
            svd::svd_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "SVD is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin QR via faer: A = Q * R
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn qr<T: Scalar>(&self, desc: QrDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_qr_desc::<T, f64>(desc) };
            qr::qr_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_qr_desc::<T, f32>(desc) };
            qr::qr_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_qr_desc::<T, Complex<f64>>(desc) };
            qr::qr_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_qr_desc::<T, Complex<f32>>(desc) };
            qr::qr_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "QR is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Thin LQ via faer: A = L * Q
    ///
    /// Internally computes QR of A^H (adjoint), then takes conjugate transposes.
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn lq<T: Scalar>(&self, desc: LqDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            // Safety: T is f64, verified by TypeId.
            let desc_f64 = unsafe { reinterpret_lq_desc::<T, f64>(desc) };
            lq::lq_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_lq_desc::<T, f32>(desc) };
            lq::lq_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            // Safety: T is Complex<f64>, verified by TypeId.
            let desc_c64 = unsafe { reinterpret_lq_desc::<T, Complex<f64>>(desc) };
            lq::lq_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_lq_desc::<T, Complex<f32>>(desc) };
            lq::lq_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "LQ is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Self-adjoint eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn eigh<T: Scalar>(&self, desc: EighDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_eigh_desc::<T, f64>(desc) };
            eigh::eigh_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_eigh_desc::<T, f32>(desc) };
            eigh::eigh_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_eigh_desc::<T, Complex<f64>>(desc) };
            eigh::eigh_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_eigh_desc::<T, Complex<f32>>(desc) };
            eigh::eigh_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "eigh is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// General eigenvalue decomposition via faer
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn eig<T: Scalar>(&self, desc: EigDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_eig_desc::<T, f64>(desc) };
            eig::eig_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_eig_desc::<T, f32>(desc) };
            eig::eig_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_eig_desc::<T, Complex<f64>>(desc) };
            eig::eig_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_eig_desc::<T, Complex<f32>>(desc) };
            eig::eig_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "eig is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }

    /// Linear solve via faer LU decomposition with partial pivoting
    ///
    /// Dispatches to faer for f64/f32/Complex<f64>/Complex<f32>.
    fn solve<T: Scalar>(&self, desc: SolveDescriptor<'_, T>) -> Result<(), BackendError> {
        use std::any::TypeId;

        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_solve_desc::<T, f64>(desc) };
            solve::solve_f64(desc_f64)
        } else if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_solve_desc::<T, f32>(desc) };
            solve::solve_f32(desc_f32)
        } else if tid == TypeId::of::<Complex<f64>>() {
            let desc_c64 = unsafe { reinterpret_solve_desc::<T, Complex<f64>>(desc) };
            solve::solve_c64(desc_c64)
        } else if tid == TypeId::of::<Complex<f32>>() {
            let desc_c32 = unsafe { reinterpret_solve_desc::<T, Complex<f32>>(desc) };
            solve::solve_c32(desc_c32)
        } else {
            Err(BackendError::NotSupported(
                "solve is only supported for f64, f32, Complex<f64>, Complex<f32>".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Generic -> concrete type reinterpretation
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

/// Reinterpret `QrDescriptor<T>` as `QrDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_qr_desc<'a, T, U>(
    desc: QrDescriptor<'a, T>,
) -> QrDescriptor<'a, U> {
    let QrDescriptor { m, n, a, q, r } = desc;
    unsafe {
        QrDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
            r: std::slice::from_raw_parts_mut(r.as_mut_ptr() as *mut U, r.len()),
        }
    }
}

/// Reinterpret `LqDescriptor<T>` as `LqDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_lq_desc<'a, T, U>(
    desc: LqDescriptor<'a, T>,
) -> LqDescriptor<'a, U> {
    let LqDescriptor { m, n, a, l, q } = desc;
    unsafe {
        LqDescriptor {
            m,
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            l: std::slice::from_raw_parts_mut(l.as_mut_ptr() as *mut U, l.len()),
            q: std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut U, q.len()),
        }
    }
}

/// Reinterpret `EighDescriptor<T>` as `EighDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Real` and `U::Real` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_eigh_desc<'a, T: Scalar, U: Scalar>(
    desc: EighDescriptor<'a, T>,
) -> EighDescriptor<'a, U> {
    let EighDescriptor { n, a, w, v } = desc;
    unsafe {
        EighDescriptor {
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            w: std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut U::Real, w.len()),
            v: std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut U, v.len()),
        }
    }
}

/// Reinterpret `EigDescriptor<T>` as `EigDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment,
/// and `T::Complex` and `U::Complex` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_eig_desc<'a, T: Scalar, U: Scalar>(
    desc: EigDescriptor<'a, T>,
) -> EigDescriptor<'a, U> {
    let EigDescriptor { n, a, w, v } = desc;
    unsafe {
        EigDescriptor {
            n,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            w: std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut U::Complex, w.len()),
            v: std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut U::Complex, v.len()),
        }
    }
}

/// Reinterpret `SolveDescriptor<T>` as `SolveDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
unsafe fn reinterpret_solve_desc<'a, T, U>(
    desc: SolveDescriptor<'a, T>,
) -> SolveDescriptor<'a, U> {
    let SolveDescriptor { n, nrhs, a, b, x } = desc;
    unsafe {
        SolveDescriptor {
            n,
            nrhs,
            a: std::slice::from_raw_parts(a.as_ptr() as *const U, a.len()),
            b: std::slice::from_raw_parts(b.as_ptr() as *const U, b.len()),
            x: std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut U, x.len()),
        }
    }
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

    // --- Transpose tests ---

    #[test]
    fn test_transpose_f64_2d() {
        let backend = CpuBackend::new();

        // 2x3 matrix -> 3x2
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

        // Shape [2,3,4], perm [1,0,2] -> shape [3,2,4]
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
        // input[0][1][2] = 0*12 + 1*4 + 2 = 6 -> output[1][0][2] = 1*8 + 0*4 + 2 = 10
        assert_eq!(output[10], 6.0);
        // input[1][0][3] = 1*12 + 0*4 + 3 = 15 -> output[0][1][3] = 0*8 + 1*4 + 3 = 7
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
}
