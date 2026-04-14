//! Slice, expand, and replace operations for Dense.

use num_traits::Zero;
use std::sync::Arc;

use super::{Dense, compute_strides_column_usize, compute_strides_usize};
use arnet_core::MemoryOrder;

/// Compute strides (usize) for the given shape and order.
fn strides_for(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    match order {
        MemoryOrder::RowMajor => compute_strides_usize(shape),
        MemoryOrder::ColumnMajor => compute_strides_column_usize(shape),
    }
}

impl<T> Dense<T>
where
    T: Clone,
{
    /// Extract a sub-tensor by specifying a range for each axis.
    ///
    /// Each range is `(start, end)` with exclusive end.
    /// The `order` parameter determines how flat data maps to axes.
    ///
    /// # Panics
    ///
    /// Panics if `ranges` length doesn't match rank, or any range is out of bounds.
    pub fn slice(&self, ranges: &[(usize, usize)], order: MemoryOrder) -> Self {
        let shape = self.shape();
        assert_eq!(
            ranges.len(),
            shape.len(),
            "slice: ranges length {} doesn't match rank {}",
            ranges.len(),
            shape.len()
        );
        for (i, &(start, end)) in ranges.iter().enumerate() {
            assert!(
                start <= end && end <= shape[i],
                "slice: range ({start}, {end}) out of bounds for axis {i} with size {}",
                shape[i]
            );
        }

        let new_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
        let new_total: usize = new_shape.iter().product();
        let rank = shape.len();

        if new_total == 0 {
            return Self::new(Vec::new(), new_shape);
        }

        let inner_axis = match order {
            MemoryOrder::RowMajor => rank - 1,
            MemoryOrder::ColumnMajor => 0,
        };

        let src_strides = strides_for(shape, order);
        let raw = self.data();
        let strip_len = new_shape[inner_axis];
        let num_strips = new_total / strip_len.max(1);

        let outer_axes: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank - 1).collect(),
            MemoryOrder::ColumnMajor => (1..rank).rev().collect(),
        };

        let mut data = Vec::with_capacity(new_total);
        let mut outer_coords = vec![0usize; rank];
        let strip_src_start: usize = ranges
            .iter()
            .zip(&src_strides)
            .map(|(&(s, _), &st)| s * st)
            .sum();

        let mut outer_flat = strip_src_start;

        for _ in 0..num_strips {
            data.extend_from_slice(&raw[outer_flat..outer_flat + strip_len]);

            for &d in outer_axes.iter().rev() {
                outer_coords[d] += 1;
                outer_flat += src_strides[d];
                if outer_coords[d] < new_shape[d] {
                    break;
                }
                outer_flat -= new_shape[d] * src_strides[d];
                outer_coords[d] = 0;
            }
        }

        Self::new(data, new_shape)
    }

    /// Expand tensor by adding zero-padding at the boundaries.
    ///
    /// The `order` parameter determines how flat data maps to axes.
    pub fn expand(&self, padding: &[(usize, usize)], order: MemoryOrder) -> Self
    where
        T: Zero,
    {
        let shape = self.shape();
        assert_eq!(
            padding.len(),
            shape.len(),
            "expand: padding length {} doesn't match rank {}",
            padding.len(),
            shape.len()
        );

        let new_shape: Vec<usize> = shape
            .iter()
            .zip(padding)
            .map(|(&s, &(before, after))| s + before + after)
            .collect();
        let new_total: usize = new_shape.iter().product();
        let dst_strides = strides_for(&new_shape, order);
        let rank = shape.len();
        let mut data = vec![T::zero(); new_total];

        let src_total = self.len();
        if src_total == 0 || rank == 0 {
            if src_total == 1 {
                data[0] = self.data()[0].clone();
            }
            return Self::new(data, new_shape);
        }

        let inner_axis = match order {
            MemoryOrder::RowMajor => rank - 1,
            MemoryOrder::ColumnMajor => 0,
        };

        let no_inner_pad = padding[inner_axis] == (0, 0);
        let src_strides = strides_for(shape, order);

        if no_inner_pad {
            // Strip copy: no padding on innermost axis
            let raw = self.data();
            let strip_len = shape[inner_axis];

            let outer_axes: Vec<usize> = match order {
                MemoryOrder::RowMajor => (0..rank - 1).collect(),
                MemoryOrder::ColumnMajor => (1..rank).rev().collect(),
            };

            let num_strips = src_total / strip_len.max(1);
            let mut src_offset = 0usize;
            let mut dst_flat: usize = (0..rank).map(|d| padding[d].0 * dst_strides[d]).sum();

            let mut outer_coords = vec![0usize; rank];

            for _ in 0..num_strips {
                data[dst_flat..dst_flat + strip_len]
                    .clone_from_slice(&raw[src_offset..src_offset + strip_len]);
                src_offset += strip_len;

                for &d in outer_axes.iter().rev() {
                    outer_coords[d] += 1;
                    dst_flat += dst_strides[d];
                    if outer_coords[d] < shape[d] {
                        break;
                    }
                    dst_flat -= shape[d] * dst_strides[d];
                    outer_coords[d] = 0;
                }
            }

            return Self::new(data, new_shape);
        }

        // General case: element-wise copy with dual index tracking
        let raw = self.data();
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        let mut src_flat: usize = 0;
        let mut dst_flat: usize = (0..rank).map(|d| padding[d].0 * dst_strides[d]).sum();

        for _ in 0..src_total {
            data[dst_flat] = raw[src_flat].clone();

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                src_flat += src_strides[d];
                dst_flat += dst_strides[d];
                if coords[d] < shape[d] {
                    break;
                }
                src_flat -= shape[d] * src_strides[d];
                dst_flat -= shape[d] * dst_strides[d];
                coords[d] = 0;
            }
        }

        Self::new(data, new_shape)
    }

    /// Write a sub-tensor into this tensor starting at the given position.
    ///
    /// The `order` parameter determines how flat data maps to axes.
    pub fn replace_slice(&mut self, sub: &Self, begin: &[usize], order: MemoryOrder) {
        let shape = self.shape().to_vec();
        let sub_shape = sub.shape();
        assert_eq!(
            sub_shape.len(),
            shape.len(),
            "replace_slice: sub rank {} doesn't match rank {}",
            sub_shape.len(),
            shape.len()
        );
        assert_eq!(
            begin.len(),
            shape.len(),
            "replace_slice: begin length {} doesn't match rank {}",
            begin.len(),
            shape.len()
        );
        for (d, (&b, &ss)) in begin.iter().zip(sub_shape).enumerate() {
            assert!(
                b + ss <= shape[d],
                "replace_slice: sub-tensor exceeds boundary on axis {d} ({b} + {ss} > {})",
                shape[d]
            );
        }

        let rank = shape.len();
        let sub_total = sub.len();
        if sub_total == 0 {
            return;
        }

        // Rank-0 (scalar): direct write
        if rank == 0 {
            Arc::make_mut(&mut self.data)[0] = sub.data()[0].clone();
            return;
        }

        let inner_axis = match order {
            MemoryOrder::RowMajor => rank - 1,
            MemoryOrder::ColumnMajor => 0,
        };

        let self_strides = strides_for(&shape, order);
        let sub_raw = sub.data().to_vec(); // snapshot before mutating self

        let strip_len = sub_shape[inner_axis];
        let num_strips = sub_total / strip_len.max(1);

        let dst_buf = Arc::make_mut(&mut self.data);

        let outer_axes: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank - 1).collect(),
            MemoryOrder::ColumnMajor => (1..rank).rev().collect(),
        };

        let mut src_offset = 0usize;
        let mut dst_flat: usize = begin.iter().zip(&self_strides).map(|(&b, &s)| b * s).sum();
        let mut outer_coords = vec![0usize; rank];

        for _ in 0..num_strips {
            dst_buf.as_mut_slice()[dst_flat..dst_flat + strip_len]
                .clone_from_slice(&sub_raw[src_offset..src_offset + strip_len]);
            src_offset += strip_len;

            for &d in outer_axes.iter().rev() {
                outer_coords[d] += 1;
                dst_flat += self_strides[d];
                if outer_coords[d] < sub_shape[d] {
                    break;
                }
                dst_flat -= sub_shape[d] * self_strides[d];
                outer_coords[d] = 0;
            }
        }
    }
}
