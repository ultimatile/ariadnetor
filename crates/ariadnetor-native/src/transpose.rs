//! Transpose kernels.
//!
//! `NativeBackend::transpose` checks the descriptor with [`validate`] and
//! returns for a zero-size tensor before dispatching, so the four
//! `transpose_*` entry points — the per-scalar kernels the call-site
//! dispatcher (`DispatchScalar::dispatch_op` via the backend's
//! `ScalarKernels`) routes to once the type is concrete — take a valid,
//! nonempty descriptor. Each copies through `strided_perm::copy_into`, except
//! a conjugating transpose of a complex type: strided-perm only copies, so
//! that case goes to HPTT under the `hptt` feature and to the naive
//! output-driven kernel otherwise; for rank 0 its one element is
//! conjugated straight into `output`. For real types conjugation is the
//! identity, so a real `conj = true` transpose takes the strided-perm route
//! as well.
//!
//! strided-perm runs on a dedicated `threadteam` pool, not on the global Rayon
//! pool, sized from the global Rayon pool's width on first parallel use.

use std::sync::OnceLock;

use ariadnetor_core::Scalar;
use ariadnetor_core::backend::{BackendError, ExecPolicy, MemoryOrder, TransposeDescriptor};
#[cfg(not(feature = "hptt"))]
use rayon::prelude::*;
use strided_view::{StridedView, StridedViewMut};
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
    #[cfg(feature = "hptt")]
    let conj_kernel = hptt_c64;
    #[cfg(not(feature = "hptt"))]
    let conj_kernel = naive;
    transpose_complex(desc, conj_kernel)
}

pub(crate) fn transpose_c32(
    desc: TransposeDescriptor<'_, num_complex::Complex<f32>>,
) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    let conj_kernel = hptt_c32;
    #[cfg(not(feature = "hptt"))]
    let conj_kernel = naive;
    transpose_complex(desc, conj_kernel)
}

/// A complex transpose: strided-perm unless it conjugates, in which case
/// `conj_kernel` runs it.
///
/// HPTT rejects a rank-0 shape and a zero extent by terminating the process,
/// so neither reaches `conj_kernel`: `NativeBackend::transpose` has already
/// returned for a zero-size tensor, and a rank-0 tensor is its one element,
/// conjugated here.
fn transpose_complex<T: Scalar>(
    desc: TransposeDescriptor<'_, T>,
    conj_kernel: fn(TransposeDescriptor<'_, T>) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    if !desc.conj {
        return strided(desc);
    }
    if desc.shape.is_empty() {
        desc.output[0] = desc.input[0].conj();
        return Ok(());
    }
    conj_kernel(desc)
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

/// Copy `input` permuted by `perm` into `output` with strided-perm.
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
fn strided<T: Scalar>(desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
    let TransposeDescriptor {
        input,
        output,
        shape,
        perm,
        order,
        policy,
        ..
    } = desc;
    assert!(!output.is_empty(), "strided() takes a nonempty tensor");
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
    strided_perm::copy_into(&mut dst, &src, exec_for(policy)).map_err(strided_failure)
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

// ---------------------------------------------------------------------------
// HPTT kernels (feature-gated)
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
// Naive kernel (compiled when the `hptt` feature is off)
// ---------------------------------------------------------------------------

/// Naive transpose for any Scalar type.
///
/// Honors `desc.policy`:
/// * `Sequential` runs the output-driven kernel single-threaded on
///   the calling thread, paying no Rayon overhead.
/// * `Parallel(_)` dispatches the same output-driven kernel across
///   the current Rayon pool, where each chunk owns a disjoint
///   contiguous slice of `output` (race-free by construction).
///
/// Both paths share `new_to_old_idx` for the per-element index
/// mapping, so `Sequential` and `Parallel(_)` differ only in how the
/// output is split.
///
/// # Panics
///
/// On an empty `output`, which the parallel path could not split into
/// chunks. `NativeBackend::transpose` returns before dispatching for a
/// zero-size tensor.
#[cfg(not(feature = "hptt"))]
fn naive<T: Scalar>(desc: TransposeDescriptor<'_, T>) -> Result<(), BackendError> {
    let TransposeDescriptor {
        input,
        output,
        shape,
        perm,
        order,
        conj,
        policy,
    } = desc;

    let rank = shape.len();
    let total = output.len();
    assert!(total > 0, "naive() takes a nonempty tensor");

    let old_strides = compute_strides(shape, order);
    let new_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let new_strides = compute_strides(&new_shape, order);

    match policy {
        ExecPolicy::Sequential => {
            naive_sequential(
                input,
                output,
                &old_strides,
                &new_strides,
                perm,
                rank,
                order,
                conj,
            );
        }
        ExecPolicy::Parallel(n) => {
            naive_parallel(
                input,
                output,
                &old_strides,
                &new_strides,
                perm,
                rank,
                order,
                conj,
                n,
                total,
            );
        }
    }

    Ok(())
}

/// Single-threaded transpose using the same output-driven, no-alloc
/// kernel as the parallel path; differs only in that work runs on the
/// caller's thread without Rayon split.
#[cfg(not(feature = "hptt"))]
#[allow(clippy::too_many_arguments)]
fn naive_sequential<T: Scalar>(
    input: &[T],
    output: &mut [T],
    old_strides: &[usize],
    new_strides: &[usize],
    perm: &[usize],
    rank: usize,
    order: MemoryOrder,
    conj: bool,
) {
    for (new_idx, slot) in output.iter_mut().enumerate() {
        let old_idx = new_to_old_idx(new_idx, new_strides, old_strides, perm, rank, order);
        let val = input[old_idx];
        *slot = if conj { val.conj() } else { val };
    }
}

/// Output-driven Rayon kernel: each chunk owns a disjoint slice of
/// `output` so writes never race. Reads scatter across `input` via the
/// inverse mapping `old_idx = Σ new_coords[a] * old_strides[perm[a]]`.
///
/// Runs on the current Rayon pool (the global one unless the caller
/// installed another). `n` is interpreted as an advisory
/// target that influences chunk sizing, not a strict worker count:
/// `Parallel(0)` adopts `rayon::current_num_threads()`, `Parallel(n>0)`
/// splits work as if `n` workers were available. This mirrors faer's
/// `Par::rayon(n)` semantics; the pool's actual width is unchanged.
#[cfg(not(feature = "hptt"))]
#[allow(clippy::too_many_arguments)]
fn naive_parallel<T: Scalar>(
    input: &[T],
    output: &mut [T],
    old_strides: &[usize],
    new_strides: &[usize],
    perm: &[usize],
    rank: usize,
    order: MemoryOrder,
    conj: bool,
    n: usize,
    total: usize,
) {
    const MIN_CHUNK: usize = 4096;
    const TASKS_PER_THREAD: usize = 4;

    let target_threads = if n == 0 {
        rayon::current_num_threads().max(1)
    } else {
        n
    };
    let denom = target_threads.saturating_mul(TASKS_PER_THREAD).max(1);
    let chunk = total.div_ceil(denom).max(MIN_CHUNK).min(total);

    output
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, slot_chunk)| {
            let start = chunk_idx * chunk;
            for (i, slot) in slot_chunk.iter_mut().enumerate() {
                let new_idx = start + i;
                let old_idx = new_to_old_idx(new_idx, new_strides, old_strides, perm, rank, order);
                let val = input[old_idx];
                *slot = if conj { val.conj() } else { val };
            }
        });
}

/// Map an output linear index to the corresponding input linear index
/// without materializing the new-coordinate tuple.
///
/// Walks `new_strides` in descending-stride order (forward for
/// RowMajor, reverse for ColumnMajor), extracting one coordinate at a
/// time and accumulating
/// `old_idx += coord * old_strides[perm[new_axis]]`. Used by both the
/// sequential and parallel kernels — the sequential path runs it on
/// every output index inline, the parallel path runs it inside
/// disjoint Rayon chunks. Avoids the per-element `Vec<usize>` that
/// the previous coordinate helper allocated.
#[cfg(not(feature = "hptt"))]
fn new_to_old_idx(
    mut new_idx: usize,
    new_strides: &[usize],
    old_strides: &[usize],
    perm: &[usize],
    rank: usize,
    order: MemoryOrder,
) -> usize {
    let mut old_idx = 0;
    match order {
        MemoryOrder::RowMajor => {
            for new_axis in 0..rank {
                let s = new_strides[new_axis];
                let c = new_idx / s;
                new_idx -= c * s;
                old_idx += c * old_strides[perm[new_axis]];
            }
        }
        MemoryOrder::ColumnMajor => {
            for new_axis in (0..rank).rev() {
                let s = new_strides[new_axis];
                let c = new_idx / s;
                new_idx -= c * s;
                old_idx += c * old_strides[perm[new_axis]];
            }
        }
    }
    old_idx
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
