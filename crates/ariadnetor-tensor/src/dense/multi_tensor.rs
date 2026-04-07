//! Multi-tensor operations: concatenate and stack.

use super::{Dense, MemoryOrder};

impl<T> Dense<T>
where
    T: Clone,
{
    /// Concatenate tensors along an existing axis.
    ///
    /// Output memory order matches the first tensor's order.
    /// Inputs may be any layout.
    pub fn concatenate(tensors: &[&Dense<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "concatenate: empty tensor list");
        let rank = tensors[0].rank();
        assert!(
            axis < rank,
            "concatenate: axis {axis} out of range for rank {rank}"
        );

        let base_shape = tensors[0].shape();
        let order = tensors[0].memory_order();
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
            return Self::from_data_with_order(Vec::new(), out_shape, order);
        }

        // Ensure all inputs are contiguous in the output order
        let contigs: Vec<Dense<T>> = tensors.iter().map(|t| t.to_contiguous(order)).collect();
        let contig_refs: Vec<&Dense<T>> = contigs.iter().collect();

        // Check if concatenation is along the outermost axis (block copy)
        let is_outermost = match order {
            MemoryOrder::RowMajor => axis == 0,
            MemoryOrder::ColumnMajor => axis == rank - 1,
        };

        let mut data = Vec::with_capacity(out_total);

        if is_outermost {
            // Block copy: each input's entire data maps to a contiguous output block
            for t in &contig_refs {
                data.extend_from_slice(t.data());
            }
        } else {
            // Strip copy: iterate outer blocks, interleave strips from each input
            //
            // For RowMajor: strip_len = product of shape[axis+1..rank]
            //               outer_count = product of shape[0..axis]
            // For ColumnMajor: strip_len = product of shape[0..axis]
            //                  outer_count = product of shape[axis+1..rank]
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
                for t in &contig_refs {
                    let t_axis_size = t.shape()[axis];
                    // Number of contiguous elements per outer block in this tensor
                    let block_size = t_axis_size * strip_len;
                    let src_start = outer * block_size;
                    let src = &t.data()[src_start..src_start + block_size];
                    data.extend_from_slice(src);
                }
            }
        }

        Self::from_data_with_order(data, out_shape, order)
    }

    /// Stack tensors along a new axis.
    ///
    /// Output memory order matches the first tensor's order.
    /// Inputs may be any layout.
    pub fn stack(tensors: &[&Dense<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "stack: empty tensor list");
        let base_shape = tensors[0].shape();
        let rank = tensors[0].rank();
        let order = tensors[0].memory_order();
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

        // Reshape each input to insert a size-1 axis, then delegate to concatenate.
        // reshape_view is zero-copy for contiguous tensors.
        let contigs: Vec<Dense<T>> = tensors.iter().map(|t| t.to_contiguous(order)).collect();

        let mut new_shape = Vec::with_capacity(rank + 1);
        new_shape.extend_from_slice(&base_shape[..axis]);
        new_shape.push(1);
        new_shape.extend_from_slice(&base_shape[axis..]);

        let reshaped: Vec<Dense<T>> = contigs
            .iter()
            .map(|t| {
                t.reshape_view(new_shape.clone())
                    .expect("reshape_view failed on contiguous tensor")
            })
            .collect();
        let reshaped_refs: Vec<&Dense<T>> = reshaped.iter().collect();

        Self::concatenate(&reshaped_refs, axis)
    }
}
