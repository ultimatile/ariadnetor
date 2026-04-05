//! Slice, expand, and replace operations for Dense.

use num_traits::Zero;

use super::{Dense, MemoryOrder, compute_strides_column_usize, compute_strides_usize};

impl<T> Dense<T>
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
        let rank = shape.len();

        if new_total == 0 {
            return Self::from_data_with_order(Vec::new(), new_shape, order);
        }

        // Identify the innermost axis (fastest-varying in memory)
        let inner_axis = match order {
            MemoryOrder::RowMajor => rank - 1,
            MemoryOrder::ColumnMajor => 0,
        };

        // Strip copy: when source is contiguous and innermost stride is 1,
        // copy each innermost strip in bulk via extend_from_slice.
        if self.is_contiguous() && self.strides[inner_axis] == 1 {
            let raw = &self.data.as_slice();
            let strip_len = new_shape[inner_axis];
            let num_strips = new_total / strip_len.max(1);

            // Outer axes: all axes except the innermost, in memory order
            let outer_axes: Vec<usize> = match order {
                MemoryOrder::RowMajor => (0..rank - 1).collect(),
                MemoryOrder::ColumnMajor => (1..rank).rev().collect(),
            };

            let mut data = Vec::with_capacity(new_total);
            let mut outer_coords = vec![0usize; rank];
            let strip_src_start: isize = ranges
                .iter()
                .zip(&self.strides)
                .map(|(&(s, _), &st)| s as isize * st)
                .sum::<isize>()
                + self.offset as isize;

            // Running offset tracks outer coordinate changes
            let mut outer_flat = strip_src_start;

            for _ in 0..num_strips {
                let src_start = outer_flat as usize;
                data.extend_from_slice(&raw[src_start..src_start + strip_len]);

                // Advance outer coordinates
                for &d in outer_axes.iter().rev() {
                    outer_coords[d] += 1;
                    outer_flat += self.strides[d];
                    if outer_coords[d] < new_shape[d] {
                        break;
                    }
                    outer_flat -= new_shape[d] as isize * self.strides[d];
                    outer_coords[d] = 0;
                }
            }

            return Self::from_data_with_order(data, new_shape, order);
        }

        // General case: incremental flat index (no per-element get() or Vec alloc)
        let raw = self.data.as_slice();
        let mut data = Vec::with_capacity(new_total);
        let mut coords = vec![0usize; rank];

        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        // Initialize src_flat to the flat index of the slice origin
        let mut src_flat: isize = self.offset as isize
            + ranges
                .iter()
                .zip(&self.strides)
                .map(|(&(s, _), &st)| s as isize * st)
                .sum::<isize>();

        for _ in 0..new_total {
            debug_assert!(src_flat >= 0 && (src_flat as usize) < raw.len());
            data.push(raw[src_flat as usize].clone());

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                src_flat += self.strides[d];
                if coords[d] < new_shape[d] {
                    break;
                }
                src_flat -= new_shape[d] as isize * self.strides[d];
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
