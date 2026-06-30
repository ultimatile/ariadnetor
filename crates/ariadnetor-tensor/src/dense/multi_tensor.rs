//! Multi-tensor operations on `DenseTensorData<T>`: concatenate and stack.

use crate::DenseTensorData;
use ariadnetor_core::MemoryOrder;

impl<T> DenseTensorData<T>
where
    T: Clone,
{
    /// Concatenate tensors along an existing axis.
    ///
    /// All tensors must have the same rank, the same `order()`, and
    /// matching sizes on all axes except `axis`. The output preserves
    /// the shared `order()`.
    pub fn concatenate(tensors: &[&DenseTensorData<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "concatenate: empty tensor list");
        let rank = tensors[0].rank();
        assert!(
            axis < rank,
            "concatenate: axis {axis} out of range for rank {rank}"
        );

        let order = tensors[0].order();
        let base_shape = tensors[0].shape().to_vec();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.rank(),
                rank,
                "concatenate: tensor {i} has rank {} but expected {rank}",
                t.rank()
            );
            assert_eq!(
                t.order(),
                order,
                "concatenate: tensor {i} has order {:?} but expected {:?}",
                t.order(),
                order,
            );
            for (d, (&ts, &bs)) in t.shape().iter().zip(&base_shape).enumerate() {
                if d != axis {
                    assert_eq!(
                        ts, bs,
                        "concatenate: tensor {i} has size {ts} on axis {d} but expected {bs}",
                    );
                }
            }
        }

        let mut out_shape: Vec<usize> = base_shape.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
        let out_total: usize = out_shape.iter().product();

        if out_total == 0 {
            return DenseTensorData::from_raw_parts(Vec::new(), out_shape, order);
        }

        let is_outermost = match order {
            MemoryOrder::RowMajor => axis == 0,
            MemoryOrder::ColumnMajor => axis == rank - 1,
        };

        let mut data = Vec::with_capacity(out_total);
        if is_outermost {
            for t in tensors {
                data.extend_from_slice(t.storage().data());
            }
        } else {
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
                    let src = &t.storage().data()[src_start..src_start + block_size];
                    data.extend_from_slice(src);
                }
            }
        }

        DenseTensorData::from_raw_parts(data, out_shape, order)
    }

    /// Stack tensors along a new axis.
    ///
    /// All tensors must have the same shape and the same `order()`.
    /// The output preserves the shared `order()`.
    pub fn stack(tensors: &[&DenseTensorData<T>], axis: usize) -> Self {
        assert!(!tensors.is_empty(), "stack: empty tensor list");
        let base_shape = tensors[0].shape().to_vec();
        let rank = tensors[0].rank();
        assert!(
            axis <= rank,
            "stack: axis {axis} out of range for rank {rank} (max {rank})"
        );

        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                t.shape(),
                base_shape.as_slice(),
                "stack: tensor {i} has shape {:?} but expected {base_shape:?}",
                t.shape()
            );
        }

        let mut new_shape = Vec::with_capacity(rank + 1);
        new_shape.extend_from_slice(&base_shape[..axis]);
        new_shape.push(1);
        new_shape.extend_from_slice(&base_shape[axis..]);

        let reshaped: Vec<DenseTensorData<T>> = tensors
            .iter()
            .map(|t| t.reshape(new_shape.clone()))
            .collect();
        let reshaped_refs: Vec<&DenseTensorData<T>> = reshaped.iter().collect();

        Self::concatenate(&reshaped_refs, axis)
    }
}
