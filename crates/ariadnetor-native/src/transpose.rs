//! Transpose kernels.
//!
//! `NativeBackend::transpose` checks the descriptor with [`validate`] and
//! returns for a zero-size tensor before dispatching, so the four
//! `transpose_*` entry points — the per-scalar kernels the call-site
//! dispatcher (`DispatchScalar::dispatch_op` via the backend's
//! `ScalarKernels`) routes to once the type is concrete — take a valid,
//! nonempty descriptor. Each copies through strided-perm: a conjugating
//! transpose of a complex type through `strided_perm::copy_conj_into`, which
//! conjugates each element as it stores it, and every other transpose through
//! `strided_perm::copy_into`. For real types conjugation is the identity, so a
//! real `conj = true` transpose is a plain copy.
//!
//! strided-perm runs on a dedicated `threadteam` pool, not on the global Rayon
//! pool, sized from the global Rayon pool's width on first parallel use.

use std::sync::OnceLock;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{BackendError, ExecPolicy, MemoryOrder, TransposeDescriptor};
use strided_view::{ElementOpApply, StridedView, StridedViewMut};
use threadteam::{Exec, Pool};

pub(crate) fn transpose_f64(desc: TransposeDescriptor<'_, f64>) -> Result<(), BackendError> {
    strided(desc)
}

pub(crate) fn transpose_f32(desc: TransposeDescriptor<'_, f32>) -> Result<(), BackendError> {
    strided(desc)
}

pub(crate) fn transpose_c64(
    desc: TransposeDescriptor<'_, num_complex::Complex<f64>>,
) -> Result<(), BackendError> {
    transpose_complex(desc)
}

pub(crate) fn transpose_c32(
    desc: TransposeDescriptor<'_, num_complex::Complex<f32>>,
) -> Result<(), BackendError> {
    transpose_complex(desc)
}

/// A complex transpose: `copy_conj_into` when it conjugates, `copy_into`
/// otherwise.
fn transpose_complex<T: Scalar + ElementOpApply>(
    desc: TransposeDescriptor<'_, T>,
) -> Result<(), BackendError> {
    if desc.conj {
        strided_with(desc, strided_perm::copy_conj_into::<T>)
    } else {
        strided(desc)
    }
}

// ---------------------------------------------------------------------------
// Descriptor validation
// ---------------------------------------------------------------------------

/// Reject a descriptor no kernel can transpose correctly.
///
/// Every kernel indexes `input` and `output` through strides derived from
/// `shape` and `perm`, so each needs `perm` to be a permutation of
/// `0..shape.len()` and both buffers to hold exactly `product(shape)`
/// elements. Checked before any element is read or written; returns that
/// element count.
pub(crate) fn validate<T>(desc: &TransposeDescriptor<'_, T>) -> Result<usize, BackendError> {
    let rank = desc.shape.len();
    if desc.perm.len() != rank {
        return Err(BackendError::InvalidArgument(format!(
            "transpose: perm {:?} has {} entries for a rank-{rank} shape {:?}",
            desc.perm,
            desc.perm.len(),
            desc.shape
        )));
    }
    let mut seen = vec![false; rank];
    for &axis in desc.perm {
        if axis >= rank {
            return Err(BackendError::InvalidArgument(format!(
                "transpose: perm {:?} names axis {axis}, out of range for rank {rank}",
                desc.perm
            )));
        }
        if seen[axis] {
            return Err(BackendError::InvalidArgument(format!(
                "transpose: perm {:?} names axis {axis} more than once",
                desc.perm
            )));
        }
        seen[axis] = true;
    }
    // A zero extent makes the element count 0 whatever the other extents are;
    // only an all-nonzero shape can overflow.
    let total = if desc.shape.contains(&0) {
        Some(0)
    } else {
        desc.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }
    .ok_or_else(|| {
        BackendError::InvalidArgument(format!(
            "transpose: element count of shape {:?} overflows usize",
            desc.shape
        ))
    })?;
    if desc.input.len() != total || desc.output.len() != total {
        return Err(BackendError::InvalidArgument(format!(
            "transpose: shape {:?} has {total} elements, but input has {} and output has {}",
            desc.shape,
            desc.input.len(),
            desc.output.len()
        )));
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// strided-perm route
// ---------------------------------------------------------------------------

/// Copy `input` permuted by `perm` into `output` with `strided_perm::copy_into`.
fn strided<T: Scalar>(desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
    strided_with(desc, strided_perm::copy_into::<T>)
}

/// A strided-perm copy entry point: `copy_into` or `copy_conj_into`.
type CopyFn<T> =
    fn(&mut StridedViewMut<'_, T>, &StridedView<'_, T>, Exec<'_>) -> strided_view::Result<()>;

/// Copy `input` permuted by `perm` into `output` with `copy`.
///
/// The source view walks `input` in output-axis order: output axis `a` has
/// extent `shape[perm[a]]` and the stride input axis `perm[a]` has under
/// `order`. Copying it into a contiguous destination of the same `order`
/// writes `output[y] = input[x]` with `x[perm[a]] = y[a]`.
///
/// # Panics
///
/// On an empty `output`: a zero extent would let the stride products below
/// multiply the remaining extents past `usize::MAX`. `NativeBackend::transpose`
/// returns before dispatching for a zero-size tensor.
fn strided_with<T: Scalar>(
    desc: TransposeDescriptor<'_, T>,
    copy: CopyFn<T>,
) -> Result<(), BackendError> {
    let TransposeDescriptor {
        input,
        output,
        shape,
        perm,
        order,
        policy,
        ..
    } = desc;
    assert!(!output.is_empty(), "strided_with() takes a nonempty tensor");
    let input_strides = compute_strides(shape, order);
    let dims: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    let src_strides: Vec<isize> = perm
        .iter()
        .map(|&axis| stride_to_isize(input_strides[axis]))
        .collect();
    let dst_strides: Vec<isize> = compute_strides(&dims, order)
        .into_iter()
        .map(stride_to_isize)
        .collect();
    let src = StridedView::new(input, &dims, &src_strides, 0).map_err(strided_failure)?;
    let mut dst = StridedViewMut::new(output, &dims, &dst_strides, 0).map_err(strided_failure)?;
    copy(&mut dst, &src, exec_for(policy)).map_err(strided_failure)
}

/// Convert a stride to the `isize` strided-view takes.
///
/// With every extent at least 1, a stride is a product of extents and so at
/// most `product(shape)`, the slice length `validate` admitted, which a slice
/// keeps within `isize::MAX`.
fn stride_to_isize(stride: usize) -> isize {
    isize::try_from(stride).expect("a stride is at most the slice length")
}

fn strided_failure(e: impl std::fmt::Display) -> BackendError {
    BackendError::ExecutionFailed(format!("strided-perm transpose: {e}"))
}

/// The process-wide pool the strided-perm route runs parallel copies on,
/// built on first use. `None` when the pool could not be built; parallel
/// copies then run serially.
static POOL: OnceLock<Option<Pool>> = OnceLock::new();

/// The shared pool, sized from the global Rayon pool's width so that
/// `RAYON_NUM_THREADS` bounds this pool's width as it bounds the global
/// pool's. The two pools are separate thread sets.
///
/// `exec_in` reaches this only off Rayon workers, so `current_num_threads`
/// reads the global pool, not a caller's custom pool.
fn pool() -> Option<&'static Pool> {
    POOL.get_or_init(|| {
        assert!(
            !on_rayon_worker(),
            "the transpose pool must be sized off Rayon workers"
        );
        Pool::builder()
            .threads(rayon::current_num_threads())
            .build()
            .ok()
    })
    .as_ref()
}

/// The placement token for a strided-perm copy under `policy`, on the
/// shared pool.
fn exec_for(policy: ExecPolicy) -> Exec<'static> {
    exec_in(pool, policy)
}

/// Map `policy` to a token on the pool `pool` yields.
///
/// `pool` is called only once the serial cases are settled, so a call from a
/// Rayon worker neither runs on the pool nor initialises it.
///
/// * `Sequential` → serial.
/// * Any policy on a Rayon worker → serial. A Rayon worker that blocks on
///   another pool's `install` keeps stealing jobs from its own pool onto the
///   blocked stack, so parallel transposes issued from inside a `par_iter`
///   pile up on worker stacks until one overflows. Running such a call on
///   the worker itself avoids entering the pool at all.
/// * `Parallel(0)` → the pool's whole width; `Parallel(n)` → `n` slots,
///   clamped to the width by `Exec::new`. strided-perm narrows the width
///   further when the copy is too small to repay a fan-out.
/// * No pool → serial.
fn exec_in<'p>(pool: impl FnOnce() -> Option<&'p Pool>, policy: ExecPolicy) -> Exec<'p> {
    let n = match policy {
        ExecPolicy::Sequential => return Exec::serial(),
        ExecPolicy::Parallel(_) if on_rayon_worker() => return Exec::serial(),
        ExecPolicy::Parallel(n) => n,
    };
    let Some(pool) = pool() else {
        return Exec::serial();
    };
    // `Exec::new` fails only on a zero budget; the width is nonzero and the
    // other arm has `n > 0`.
    let budget = if n == 0 { pool.width().get() } else { n };
    Exec::new(pool, budget).expect("the budget is nonzero")
}

fn on_rayon_worker() -> bool {
    rayon::current_thread_index().is_some()
}

/// Compute contiguous strides for a given shape and memory order.
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

#[cfg(test)]
mod tests;
