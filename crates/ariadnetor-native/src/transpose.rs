//! Transpose dispatch: HPTT for supported types, naive fallback otherwise

use arnet_core::Scalar;
use arnet_core::backend::{BackendError, ExecPolicy, MemoryOrder, TransposeDescriptor};

/// Dispatch transpose to the best available implementation.
///
/// With the `hptt` feature: f64/f32/Complex use HPTT, others use naive.
/// Without the `hptt` feature: all types use naive.
pub(crate) fn dispatch<T: Scalar>(desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f64>() {
            let desc_f64 = unsafe { reinterpret_transpose_desc::<T, f64>(desc) };
            return hptt_f64(desc_f64);
        }
        if tid == TypeId::of::<f32>() {
            let desc_f32 = unsafe { reinterpret_transpose_desc::<T, f32>(desc) };
            return hptt_f32(desc_f32);
        }
        if tid == TypeId::of::<num_complex::Complex<f64>>() {
            let desc_c64 =
                unsafe { reinterpret_transpose_desc::<T, num_complex::Complex<f64>>(desc) };
            return hptt_c64(desc_c64);
        }
        if tid == TypeId::of::<num_complex::Complex<f32>>() {
            let desc_c32 =
                unsafe { reinterpret_transpose_desc::<T, num_complex::Complex<f32>>(desc) };
            return hptt_c32(desc_c32);
        }
    }

    naive(desc)
}

// ---------------------------------------------------------------------------
// MemoryOrder conversion
// ---------------------------------------------------------------------------

#[cfg(feature = "hptt")]
fn to_hptt_order(order: MemoryOrder) -> hptt::MemoryOrder {
    match order {
        MemoryOrder::RowMajor => hptt::MemoryOrder::RowMajor,
        MemoryOrder::ColumnMajor => hptt::MemoryOrder::ColumnMajor,
    }
}

/// Map an [`ExecPolicy`] to HPTT's `num_threads` argument.
///
/// HPTT 0.4 rejects `num_threads == 0` with `Error::NumThreadsZero`, so
/// `Parallel(0)` (the "backend auto" convention from `ExecPolicy`) is
/// resolved here via `std::thread::available_parallelism()` to a positive
/// integer before crossing the FFI boundary. The return value is always
/// `>= 1`.
#[cfg(feature = "hptt")]
fn to_hptt_num_threads(policy: ExecPolicy) -> usize {
    match policy {
        ExecPolicy::Sequential => 1,
        ExecPolicy::Parallel(0) => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ExecPolicy::Parallel(n) => n,
    }
}

// ---------------------------------------------------------------------------
// Generic → concrete type reinterpretation
// ---------------------------------------------------------------------------

/// Reinterpret `TransposeDescriptor<T>` as `TransposeDescriptor<U>`.
///
/// # Safety
/// Caller must guarantee `T` and `U` have identical size and alignment
/// (typically verified via `TypeId::of::<T>() == TypeId::of::<U>()`).
#[cfg(feature = "hptt")]
unsafe fn reinterpret_transpose_desc<'a, T, U>(
    desc: TransposeDescriptor<'a, T>,
) -> TransposeDescriptor<'a, U> {
    let TransposeDescriptor {
        input,
        output,
        shape,
        perm,
        order,
        conj,
        policy,
    } = desc;
    unsafe {
        TransposeDescriptor {
            input: std::slice::from_raw_parts(input.as_ptr() as *const U, input.len()),
            output: std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut U, output.len()),
            shape,
            perm,
            order,
            conj,
            policy,
        }
    }
}

// ---------------------------------------------------------------------------
// HPTT implementations (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "hptt")]
fn hptt_f64(desc: TransposeDescriptor<'_, f64>) -> Result<(), BackendError> {
    hptt::transpose_f64(
        desc.perm,
        1.0,
        desc.input,
        desc.shape,
        0.0,
        desc.output,
        to_hptt_num_threads(desc.policy),
        to_hptt_order(desc.order),
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("HPTT transpose_f64: {e}")))?;
    Ok(())
}

#[cfg(feature = "hptt")]
fn hptt_f32(desc: TransposeDescriptor<'_, f32>) -> Result<(), BackendError> {
    hptt::transpose_f32(
        desc.perm,
        1.0,
        desc.input,
        desc.shape,
        0.0,
        desc.output,
        to_hptt_num_threads(desc.policy),
        to_hptt_order(desc.order),
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("HPTT transpose_f32: {e}")))?;
    Ok(())
}

#[cfg(feature = "hptt")]
fn hptt_c64(desc: TransposeDescriptor<'_, num_complex::Complex<f64>>) -> Result<(), BackendError> {
    let alpha = num_complex::Complex::new(1.0, 0.0);
    let beta = num_complex::Complex::new(0.0, 0.0);
    hptt::transpose_c64(
        desc.perm,
        alpha,
        desc.input,
        desc.shape,
        beta,
        desc.output,
        to_hptt_num_threads(desc.policy),
        desc.conj,
        to_hptt_order(desc.order),
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("HPTT transpose_c64: {e}")))?;
    Ok(())
}

#[cfg(feature = "hptt")]
fn hptt_c32(desc: TransposeDescriptor<'_, num_complex::Complex<f32>>) -> Result<(), BackendError> {
    let alpha = num_complex::Complex::new(1.0, 0.0);
    let beta = num_complex::Complex::new(0.0, 0.0);
    hptt::transpose_c32(
        desc.perm,
        alpha,
        desc.input,
        desc.shape,
        beta,
        desc.output,
        to_hptt_num_threads(desc.policy),
        desc.conj,
        to_hptt_order(desc.order),
    )
    .map_err(|e| BackendError::ExecutionFailed(format!("HPTT transpose_c32: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Naive fallback (always available)
// ---------------------------------------------------------------------------

/// Naive transpose for any Scalar type.
///
/// Iterates all elements, maps source coordinates through the permutation,
/// and writes to the destination buffer.
fn naive<T: Scalar>(desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
    let TransposeDescriptor {
        input,
        output,
        shape,
        perm,
        order,
        conj,
        ..
    } = desc;

    let rank = shape.len();
    let total: usize = shape.iter().product();

    if total == 0 {
        return Ok(());
    }

    let old_strides = compute_strides(shape, order);
    let new_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let new_strides = compute_strides(&new_shape, order);

    for (old_idx, &val) in input.iter().enumerate() {
        let old_coords = linear_to_coords(old_idx, &old_strides, rank, order);
        let mut new_idx = 0;
        for (axis, &p) in perm.iter().enumerate() {
            new_idx += old_coords[p] * new_strides[axis];
        }
        output[new_idx] = if conj { val.conj() } else { val };
    }

    Ok(())
}

/// Compute strides for a given shape and memory order.
fn compute_strides(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    match order {
        MemoryOrder::RowMajor => {
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

/// Convert a flat index to multi-dimensional coordinates.
///
/// Processes dimensions from largest stride to smallest:
/// row-major iterates forward, column-major iterates backward.
fn linear_to_coords(
    mut idx: usize,
    strides: &[usize],
    rank: usize,
    order: MemoryOrder,
) -> Vec<usize> {
    let mut coords = vec![0; rank];
    match order {
        MemoryOrder::RowMajor => {
            for i in 0..rank {
                coords[i] = idx / strides[i];
                idx %= strides[i];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in (0..rank).rev() {
                coords[i] = idx / strides[i];
                idx %= strides[i];
            }
        }
    }
    coords
}
