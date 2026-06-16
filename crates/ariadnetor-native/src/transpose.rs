//! Transpose kernels: HPTT per concrete type when the `hptt` feature is on,
//! a naive output-driven kernel otherwise.
//!
//! The four `transpose_*` entry points are the per-scalar kernels the
//! call-site dispatcher (`DispatchScalar::dispatch_op` via the backend's
//! `ScalarKernels`) routes to once the type is concrete. HPTT and the naive
//! kernel are mutually exclusive implementations selected at compile time by
//! the `hptt` feature, so the naive kernel is compiled only when HPTT is absent.

#[cfg(not(feature = "hptt"))]
use arnet_core::Scalar;
use arnet_core::backend::ExecPolicy;
use arnet_core::backend::{BackendError, MemoryOrder, TransposeDescriptor};
#[cfg(not(feature = "hptt"))]
use rayon::prelude::*;

pub(crate) fn transpose_f64(desc: TransposeDescriptor<'_, f64>) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    {
        hptt_f64(desc)
    }
    #[cfg(not(feature = "hptt"))]
    {
        naive(desc)
    }
}

pub(crate) fn transpose_f32(desc: TransposeDescriptor<'_, f32>) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    {
        hptt_f32(desc)
    }
    #[cfg(not(feature = "hptt"))]
    {
        naive(desc)
    }
}

pub(crate) fn transpose_c64(
    desc: TransposeDescriptor<'_, num_complex::Complex<f64>>,
) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    {
        hptt_c64(desc)
    }
    #[cfg(not(feature = "hptt"))]
    {
        naive(desc)
    }
}

pub(crate) fn transpose_c32(
    desc: TransposeDescriptor<'_, num_complex::Complex<f32>>,
) -> Result<(), BackendError> {
    #[cfg(feature = "hptt")]
    {
        hptt_c32(desc)
    }
    #[cfg(not(feature = "hptt"))]
    {
        naive(desc)
    }
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
// Naive fallback (compiled when the `hptt` feature is off)
// ---------------------------------------------------------------------------

/// Naive transpose for any Scalar type.
///
/// Honors `desc.policy`:
/// * `Sequential` runs the output-driven kernel single-threaded on
///   the calling thread, paying no Rayon overhead.
/// * `Parallel(_)` dispatches the same output-driven kernel across
///   the global Rayon pool, where each chunk owns a disjoint
///   contiguous slice of `output` (race-free by construction).
///
/// Both paths share `new_to_old_idx` for the per-element index
/// mapping so `Sequential` and `Parallel(_)` measurements are
/// comparable when calibrating `ThresholdTable::*.transpose`.
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
    let total: usize = shape.iter().product();

    if total == 0 {
        return Ok(());
    }

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
/// caller's thread without Rayon split. Sharing the inner kernel keeps
/// the Sequential / Parallel comparison fair when calibrating
/// `ThresholdTable::*.transpose` against the naive baseline.
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
/// Runs on the global Rayon pool. `n` is interpreted as an advisory
/// target that influences chunk sizing, not a strict worker count:
/// `Parallel(0)` adopts `rayon::current_num_threads()`, `Parallel(n>0)`
/// splits work as if `n` workers were available. This mirrors faer's
/// `Par::rayon(n)` semantics; the global pool's actual width is
/// unchanged.
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

/// Compute strides for a given shape and memory order.
#[cfg(not(feature = "hptt"))]
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
