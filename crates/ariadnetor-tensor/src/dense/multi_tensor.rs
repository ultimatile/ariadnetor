//! Multi-tensor operations: concatenate and stack.

use super::Dense;
use arnet_core::MemoryOrder;

impl<T> Dense<T>
where
    T: Clone,
{
    /// Concatenate tensors along an existing axis.
    ///
    /// All tensors must have the same rank and matching sizes on all axes
    /// except `axis`. The `order` parameter determines how flat data maps
    /// to multi-dimensional indices (provided by the compute backend).
    pub fn concatenate(tensors: &[&Dense<T>], axis: usize, order: MemoryOrder) -> Self {
        assert!(!tensors.is_empty(), "concatenate: empty tensor list");
        let rank = tensors[0].rank();
        assert!(
            axis < rank,
            "concatenate: axis {axis} out of range for rank {rank}"
        );

        let base_shape = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.rank(),
                rank,
                "concatenate: tensor {i} has rank {} but expected {rank}",
                t.rank()
            );
            for (d, (&ts, &bs)) in t.shape().iter().zip(base_shape).enumerate() {
                if d != axis {
                    assert_eq!(
                        ts, bs,
                        "concatenate: tensor {i} has size {ts} on axis {d} but expected {bs}",
                    );
                }
            }
        }

        let mut out_shape: Vec<usize> = base_shape.to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
        let out_total: usize = out_shape.iter().product();

        if out_total == 0 {
            return Self::new(Vec::new(), out_shape, order);
        }

        // Check if concatenation is along the outermost axis (block copy)
        let is_outermost = match order {
            MemoryOrder::RowMajor => axis == 0,
            MemoryOrder::ColumnMajor => axis == rank - 1,
        };

        let mut data = Vec::with_capacity(out_total);

        if is_outermost {
            for t in tensors {
                data.extend_from_slice(t.data());
            }
        } else {
            // Strip copy: iterate outer blocks, interleave strips from each input
            let (strip_len, outer_count) = match order {
                MemoryOrder::RowMajor => (
                    base_shape[axis + 1..].iter().product::<usize>(),
                    base_shape[..axis].iter().product::<usize>(),
                ),
                MemoryOrder::ColumnMajor => (
                    base_shape[..axis].iter().product::<usize>(),
                    base_shape[axis + 1..].iter().product::<usize>(),
                ),
            };

            for outer in 0..outer_count {
                for t in tensors {
                    let t_axis_size = t.shape()[axis];
                    let block_size = t_axis_size * strip_len;
                    let src_start = outer * block_size;
                    let src = &t.data()[src_start..src_start + block_size];
                    data.extend_from_slice(src);
                }
            }
        }

        Self::new(data, out_shape, order)
    }

    /// Stack tensors along a new axis.
    ///
    /// The `order` parameter determines memory layout interpretation.
    pub fn stack(tensors: &[&Dense<T>], axis: usize, order: MemoryOrder) -> Self {
        assert!(!tensors.is_empty(), "stack: empty tensor list");
        let base_shape = tensors[0].shape();
        let rank = tensors[0].rank();
        assert!(
            axis <= rank,
            "stack: axis {axis} out of range for rank {rank} (max {rank})"
        );

        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.shape(),
                base_shape,
                "stack: tensor {i} has shape {:?} but expected {base_shape:?}",
                t.shape()
            );
        }

        // Reshape each input to insert a size-1 axis, then concatenate.
        let mut new_shape = Vec::with_capacity(rank + 1);
        new_shape.extend_from_slice(&base_shape[..axis]);
        new_shape.push(1);
        new_shape.extend_from_slice(&base_shape[axis..]);

        let reshaped: Vec<Dense<T>> = tensors
            .iter()
            .map(|t| t.reshape(new_shape.clone()))
            .collect();
        let reshaped_refs: Vec<&Dense<T>> = reshaped.iter().collect();

        Self::concatenate(&reshaped_refs, axis, order)
    }
}
