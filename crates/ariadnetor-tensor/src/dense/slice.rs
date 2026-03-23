//! Slice, expand, and replace operations for DenseTensor.

use num_traits::Zero;

use super::{DenseTensor, MemoryOrder, compute_strides_column_usize, compute_strides_usize};

impl<T> DenseTensor<T>
where
    T: Clone,
{
    /// Extract a sub-tensor by specifying a range for each axis.
    ///
    /// Each range is `(start, end)` with exclusive end.
    ///
    /// # Panics
    ///
    /// Panics if `ranges` length doesn't match rank, or any range is out of bounds.
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Self {
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
        let order = self.memory_order();
        let mut data = Vec::with_capacity(new_total);

        let rank = shape.len();
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..new_total {
            let src_coords: Vec<usize> = coords
                .iter()
                .zip(ranges)
                .map(|(&c, &(s, _))| c + s)
                .collect();
            data.push(self.get(&src_coords));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < new_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, new_shape, order)
    }

    /// Expand tensor by adding zero-padding at the boundaries.
    pub fn expand(&self, padding: &[(usize, usize)]) -> Self
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
        let order = self.memory_order();
        let dst_strides = match order {
            MemoryOrder::RowMajor => compute_strides_usize(&new_shape),
            MemoryOrder::ColumnMajor => compute_strides_column_usize(&new_shape),
        };
        let rank = shape.len();
        let mut data = vec![T::zero(); new_total];
        let mut coords = vec![0usize; rank];
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        let src_total = self.len();
        for _ in 0..src_total {
            let val = self.get(&coords);
            let dst_flat: usize = (0..rank)
                .map(|d| (coords[d] + padding[d].0) * dst_strides[d])
                .sum();
            data[dst_flat] = val;

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_order(data, new_shape, order)
    }

    /// Write a sub-tensor into this tensor starting at the given position.
    pub fn replace_slice(&mut self, sub: &Self, begin: &[usize]) {
        let shape = self.shape().to_vec();
        let sub_shape = sub.shape();
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
        let mut coords = vec![0usize; rank];

        for _ in 0..sub_total {
            let val = sub.get(&coords);
            let dst_coords: Vec<usize> = coords.iter().zip(begin).map(|(&c, &b)| c + b).collect();
            self.set(&dst_coords, val);

            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < sub_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
    }
}
