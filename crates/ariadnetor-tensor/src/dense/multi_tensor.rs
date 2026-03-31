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
        let mut data = Vec::with_capacity(out_total);
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..out_total {
            // Determine which input tensor and local coordinate for the concat axis
            let mut axis_pos = coords[axis];
            let mut src_tensor = None;
            for t in tensors {
                let t_size = t.shape()[axis];
                if axis_pos < t_size {
                    src_tensor = Some(t);
                    break;
                }
                axis_pos -= t_size;
            }
            let t = src_tensor.expect("concatenate: axis position out of range");
            let mut src_coords = coords.clone();
            src_coords[axis] = axis_pos;
            data.push(t.get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < out_shape[d] {
                    break;
                }
                coords[d] = 0;
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

        let n = tensors.len();
        let mut out_shape = Vec::with_capacity(rank + 1);
        out_shape.extend_from_slice(&base_shape[..axis]);
        out_shape.push(n);
        out_shape.extend_from_slice(&base_shape[axis..]);

        let out_total: usize = out_shape.iter().product();
        let out_rank = out_shape.len();
        let mut data = Vec::with_capacity(out_total);
        let mut coords = vec![0usize; out_rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..out_rank).collect(),
            MemoryOrder::ColumnMajor => (0..out_rank).rev().collect(),
        };

        for _ in 0..out_total {
            // The stacked axis at position `axis` indexes into tensors
            let t_idx = coords[axis];
            let mut src_coords: Vec<usize> = coords[..axis].to_vec();
            src_coords.extend_from_slice(&coords[axis + 1..]);
            data.push(tensors[t_idx].get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < out_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, out_shape, order)
    }
}
